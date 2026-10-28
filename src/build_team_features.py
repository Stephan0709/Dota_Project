
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_messages import normalize
from vader_lexicon_override import get_analyzer_with_override

INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"
CUTOFFS = [("10min", 10 * 60), ("15min", 15 * 60), ("20min", 20 * 60), ("30min", 30 * 60), ("full", None)]


def score_texts(texts, cache_path):
    cache = {}
    if cache_path.exists():
        cache = pickle.load(open(cache_path, "rb"))
    todo = [t for t in texts if t not in cache]
    if todo:
        print(f"VADER scoring {len(todo):,} unique texts (cached: {len(cache):,})...")
        an = get_analyzer_with_override()
        for t in todo:
            cache[t] = an.polarity_scores(t)["compound"]
        pickle.dump(cache, open(cache_path, "wb"))
    return cache


def features_for_cutoff(df, types, fams, cutoff_s):
    d = df if cutoff_s is None else df[df["time"] <= cutoff_s]
    g = d.groupby(["match_id", "team"])
    agg = g.agg(n_msgs=("compound", "size"), sum_y=("compound", "sum"), sum_t=("time", "sum"),
                sum_tt=("t2", "sum"), sum_ty=("ty", "sum"), vol=("compound", "std"),
                team_won=("team_won", "first"))
    n = agg["n_msgs"].astype(float)
    den = n * agg["sum_tt"] - agg["sum_t"] ** 2          # = 0 when all timestamps are identical
    num = n * agg["sum_ty"] - agg["sum_t"] * agg["sum_y"]
    agg["vader_slope"] = np.where(den > 1e-9, num / den.where(den > 1e-9), np.nan)
    agg["vader_mean"] = agg["sum_y"] / n
    agg["vader_volatility"] = agg["vol"]                  # NaN when n < 2
    cnt = d.groupby(["match_id", "team", "type"]).size().unstack(fill_value=0).reindex(columns=types, fill_value=0)
    shares = cnt.div(n.reindex(cnt.index), axis=0)
    shares.columns = ["share_" + c for c in shares.columns]
    fcnt = d.groupby(["match_id", "team", "family"]).size().unstack(fill_value=0).reindex(columns=fams, fill_value=0)
    fshares = fcnt.div(n.reindex(fcnt.index), axis=0)
    fshares.columns = ["fam_" + c for c in fshares.columns]
    out = pd.concat([agg[["n_msgs", "vader_mean", "vader_slope", "vader_volatility", "team_won"]], shares, fshares], axis=1)
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(INTERIM_DIR / "clean_chat_messages_min10_bothteams.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if df["radiant_win"].dtype == object:
        df["radiant_win"] = df["radiant_win"].map({"True": True, "False": False})
    df["team_won"] = np.where(df["team"] == "radiant", df["radiant_win"], ~df["radiant_win"].astype(bool)).astype(bool)

    # --- sentiment (score the ORIGINAL text, as in RQ1-RQ3; only unique strings) ---
    texts = df["text"].astype(str)
    cache = score_texts(texts.unique().tolist(), INTERIM_DIR / "vader_scores_cache.pkl")
    df["compound"] = texts.map(cache)

    # --- message type via cluster -> type map ---
    mc = pd.read_csv(INTERIM_DIR / "msgclust_message_clusters.csv", usecols=["message", "cluster"])
    tmap = pd.read_csv(INTERIM_DIR / "cluster_type_map.csv")
    cl2type = dict(zip(tmap["cluster"], tmap["type"]))
    cl2fam = dict(zip(tmap["cluster"], tmap["family"]))
    msg2cluster = dict(zip(mc["message"], mc["cluster"]))
    df["cluster"] = texts.map(normalize).map(msg2cluster)
    df["type"] = df["cluster"].map(cl2type).fillna("unclustered")
    df["family"] = df["cluster"].map(cl2fam).fillna("unclustered")
    print(f"messages: {len(df):,} | typed (in a cluster): {(df['type'] != 'unclustered').mean():.1%} | "
          f"unclustered: {(df['type'] == 'unclustered').mean():.1%}")

    df["t2"] = df["time"].astype(float) ** 2
    df["ty"] = df["time"].astype(float) * df["compound"]
    types = sorted(set(tmap["type"])) + ["unclustered"]
    fams = sorted(set(tmap["family"])) + ["unclustered"]

    frames = []
    for name, secs in CUTOFFS:
        f = features_for_cutoff(df, types, fams, secs)
        f["cutoff"] = name
        f["is_radiant"] = (f["team"] == "radiant").astype(int)
        frames.append(f)
        print(f"cutoff {name:>4}: {len(f):6d} team-rows ({f['n_msgs'].ge(3).mean():.0%} with >=3 messages)")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(INTERIM_DIR / "team_features_by_cutoff.csv", index=False)
    print(f"\nSaved {len(out):,} rows to team_features_by_cutoff.csv")

    # --- quick leakage preview: do message-type shares already differ between winners and losers? ---
    share_cols = [c for c in out.columns if c.startswith("share_")]
    for name in ["15min", "full"]:
        f = out[(out["cutoff"] == name) & (out["n_msgs"] >= 5)]
        w = f[f["team_won"]][share_cols].mean()
        l = f[~f["team_won"]][share_cols].mean()
        diff = (w - l).sort_values(key=lambda s: s.abs(), ascending=False).head(6)
        print(f"\n[cutoff {name}, teams with >=5 msgs: {len(f):,}] largest winner-vs-loser gaps in mean message share:")
        for c, dv in diff.items():
            print(f"   {c[6:]:24s} winners {w[c]:6.2%}  losers {l[c]:6.2%}  gap {dv:+.2%}")


if __name__ == "__main__":
    main()
