
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"


def normalize(text):
    t = str(text).lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)  # 'gggggg' -> 'gg', 'noooo' -> 'noo'
    return t


class TfidfEmbedder:
    def __init__(self, dims=64, seed=42):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import Normalizer
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=dims, random_state=seed)
        self.norm = Normalizer()

    def fit_transform(self, texts):
        return self.norm.transform(self.svd.fit_transform(self.vec.fit_transform(texts)))

    def transform(self, texts):
        return self.norm.transform(self.svd.transform(self.vec.transform(texts)))


class STEmbedder:
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def _enc(self, texts):
        return self.model.encode(list(texts), batch_size=256, normalize_embeddings=True, show_progress_bar=True)

    def fit_transform(self, texts):
        return self._enc(texts)

    def transform(self, texts):
        return self._enc(texts)


def nearest_centroid_labels(points, labels):
    """Assign every noise point (-1) to the nearest non-noise cluster centroid (in the reduced space)."""
    out = labels.copy()
    ids = np.unique(labels[labels >= 0])
    if len(ids) == 0 or not (labels < 0).any():
        return out, ids, None
    cents = np.vstack([points[labels == c].mean(axis=0) for c in ids])
    noise_idx = np.where(labels < 0)[0]
    d = ((points[noise_idx, None, :] - cents[None, :, :]) ** 2).sum(axis=2) if len(noise_idx) * len(ids) < 5e7 else None
    if d is None:  # chunked for large inputs
        assigned = np.empty(len(noise_idx), dtype=int)
        for s in range(0, len(noise_idx), 5000):
            chunk = points[noise_idx[s:s + 5000]]
            assigned[s:s + 5000] = ((chunk[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    else:
        assigned = d.argmin(axis=1)
    out[noise_idx] = ids[assigned]
    return out, ids, cents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(INTERIM_DIR / "clean_chat_messages_min10_bothteams.csv"))
    ap.add_argument("--top-n", type=int, default=100000, help="cluster the N most frequent unique messages")
    ap.add_argument("--embedder", choices=["st", "tfidf"], default="st")
    ap.add_argument("--st-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--method", choices=["hdbscan", "kmeans"], default="hdbscan")
    ap.add_argument("--k", type=int, default=40, help="kmeans: number of clusters")
    ap.add_argument("--min-cluster-size", type=int, default=30)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--selection", choices=["leaf", "eom"], default="leaf")
    ap.add_argument("--umap-dims", type=int, default=8)
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--noise-policy", choices=["keep", "nearest"], default="keep")
    ap.add_argument("--assign-rest", action="store_true", help="also assign unique messages outside the top-N")
    ap.add_argument("--rest-limit", type=int, default=None, help="cap the number of 'rest' messages (testing)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="msgclust")
    args = ap.parse_args()

    df = pd.read_csv(args.input, usecols=["text"])
    df["norm"] = df["text"].astype(str).map(normalize)
    df = df[df["norm"].str.len() > 0]
    total = len(df)
    vc = df["norm"].value_counts()
    uniq = pd.DataFrame({"message": vc.index, "count": vc.values})
    print(f"messages: {total:,} | unique: {len(uniq):,}")

    fit = uniq.iloc[: args.top_n].copy()
    rest = uniq.iloc[args.top_n:].copy()
    print(f"fitting on top {len(fit):,} unique messages = {fit['count'].sum() / total:.1%} of all messages")

    emb = TfidfEmbedder(seed=args.seed) if args.embedder == "tfidf" else STEmbedder(args.st_model)
    X = emb.fit_transform(fit["message"].tolist())

    import umap
    reducer = umap.UMAP(n_components=args.umap_dims, n_neighbors=args.neighbors, min_dist=0.0,
                        metric="cosine", random_state=args.seed)
    Z = reducer.fit_transform(X)
    print(f"UMAP reduced {X.shape[1]} -> {Z.shape[1]} dims")

    if args.method == "hdbscan":
        import hdbscan
        model = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
                                cluster_selection_method=args.selection, prediction_data=True)
        raw = model.fit_predict(Z)
    else:
        from sklearn.cluster import MiniBatchKMeans
        model = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, n_init=3, batch_size=4096)
        raw = model.fit_predict(Z)

    fit["raw_label"] = raw
    n_clusters = len(set(raw)) - (1 if -1 in raw else 0)
    noise_u = (raw == -1).mean()
    noise_w = fit.loc[fit["raw_label"] == -1, "count"].sum() / fit["count"].sum()
    print(f"\n[{args.method}] clusters: {n_clusters} | noise: {noise_u:.1%} of unique messages, "
          f"{noise_w:.1%} of message volume (within the fitted set)")

    if args.noise_policy == "nearest":
        final, ids, cents = nearest_centroid_labels(Z, raw)
        fit["cluster"] = final
        print("noise re-assigned to nearest cluster centroid")
    else:
        fit["cluster"] = raw
        cents = None

    fit["in_fit"] = True
    parts = [fit]

    if args.assign_rest and len(rest):
        if args.rest_limit:
            rest = rest.iloc[: args.rest_limit].copy()
        labels = []
        pts = []
        for s in range(0, len(rest), 20000):
            chunk = rest["message"].iloc[s:s + 20000].tolist()
            Zc = reducer.transform(emb.transform(chunk))
            pts.append(Zc)
            labels.append(hdbscan.approximate_predict(model, Zc)[0] if args.method == "hdbscan" else model.predict(Zc))
        Zr = np.vstack(pts)
        lr = np.concatenate(labels)
        rest["raw_label"] = lr
        if args.noise_policy == "nearest" and args.method == "hdbscan":
            # reuse the centroids learned on the fitted set
            ids = np.unique(raw[raw >= 0])
            cents = np.vstack([Z[raw == c].mean(axis=0) for c in ids])
            nz = np.where(lr < 0)[0]
            if len(nz):
                lr = lr.copy()
                lr[nz] = ids[((Zr[nz, None, :] - cents[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)]
        rest["cluster"] = lr
        rest["in_fit"] = False
        parts.append(rest)
        print(f"assigned {len(rest):,} additional unique messages "
              f"({rest['count'].sum() / total:.1%} of all messages)")

    out = pd.concat(parts, ignore_index=True)
    covered = out["count"].sum() / total
    nonnoise = out.loc[out["cluster"] >= 0, "count"].sum() / total
    print(f"messages with a cluster: {nonnoise:.1%} of ALL messages (covered by this run: {covered:.1%})")

    rng = np.random.RandomState(args.seed)
    rows = []
    for c, g in out[out["cluster"] >= 0].groupby("cluster"):
        g = g.sort_values("count", ascending=False)
        rows.append({
            "cluster": c, "n_unique": len(g), "n_messages": int(g["count"].sum()),
            "share_of_all_messages": g["count"].sum() / total,
            "top_messages": " | ".join(g["message"].head(10)),
            "random_messages": " | ".join(g["message"].sample(min(5, len(g)), random_state=int(rng.randint(1e6)))),
        })
    summ = pd.DataFrame(rows).sort_values("n_messages", ascending=False)

    out.to_csv(INTERIM_DIR / f"{args.out_prefix}_message_clusters.csv", index=False)
    summ.to_csv(INTERIM_DIR / f"{args.out_prefix}_cluster_summary.csv", index=False)

    print(f"\ncluster size (messages): median {summ['n_messages'].median():,.0f} | "
          f"largest cluster = {summ['share_of_all_messages'].max():.1%} of all messages")
    print("\nlargest clusters (most frequent messages shown):")
    for _, r in summ.head(12).iterrows():
        print(f"  c{int(r['cluster']):>3} {r['share_of_all_messages']:6.1%}  n_unique={r['n_unique']:>5}  {r['top_messages'][:95]}")
    print(f"\nSaved: {args.out_prefix}_message_clusters.csv, {args.out_prefix}_cluster_summary.csv")


if __name__ == "__main__":
    main()
