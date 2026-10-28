

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")
INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"

CUTOFF_ORDER = ["10min", "15min", "20min", "30min", "full"]
SENT = ["vader_mean", "vader_slope", "vader_volatility"]
END_TYPES = ["share_closing_gg_wp", "share_gloating_ez", "share_surrender_end_requests", "share_outcome_statements"]


def build_match_table(feat, cutoff, min_msgs):
    d = feat[feat["cutoff"] == cutoff]
    r = d[d["team"] == "radiant"].set_index("match_id")
    a = d[d["team"] == "dire"].set_index("match_id")
    both = r.join(a, lsuffix="_r", rsuffix="_d", how="inner")
    both = both[(both["n_msgs_r"] >= min_msgs) & (both["n_msgs_d"] >= min_msgs)]
    cols = ["n_msgs"] + SENT + [c for c in r.columns if c.startswith("share_")]
    X = pd.DataFrame({"d_" + c: both[c + "_r"] - both[c + "_d"] for c in cols}).fillna(0.0)  # undefined slope -> 0
    y = both["team_won_r"].astype(int).values
    return X, y


def feature_sets(columns):
    types = [c for c in columns if c.startswith("d_share_")]
    types_noend = [c for c in types if c[2:] not in END_TYPES]
    sent = ["d_" + c for c in SENT]
    return {
        "volume": ["d_n_msgs"],
        "sentiment": sent,
        "types": types,
        "types_noend": types_noend,
        "sent+types": sent + types,
        "sent+types_noend": sent + types_noend,
    }


def make_dt(seed):
    return DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=seed)


def make_rf(seed, trees):
    return RandomForestClassifier(n_estimators=trees, max_depth=10, min_samples_leaf=50,
                                  max_features="sqrt", n_jobs=-1, random_state=seed)


def cv_oof(X, y, make_model, seed, folds):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        m = make_model().fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    acc = accuracy_score(y, oof > 0.5)
    ci = 1.96 * np.sqrt(acc * (1 - acc) / len(y))
    return acc, ci, roc_auc_score(y, oof)


def explain(X, y, cols, seed, trees, label):
    Xs = X[cols]
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=seed)
    rf = make_rf(seed, trees).fit(Xtr, ytr)
    auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
    pi = permutation_importance(rf, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=seed, n_jobs=1)
    top = sorted(zip(cols, pi.importances_mean, pi.importances_std), key=lambda t: -t[1])[:8]
    print(f"\n--- {label} | 80/20 split, held-out AUC {auc:.3f} ---")
    print("permutation importance (drop in held-out AUC when the feature is shuffled):")
    for c, m, s in top:
        print(f"   {c:34s} {m:+.4f} (+/- {s:.4f})")
    dt = make_dt(seed).fit(Xtr, ytr)
    print(f"depth-3 view of the decision tree (class 1 = Radiant wins; features are Radiant minus Dire):")
    print(export_text(dt, feature_names=list(cols), max_depth=3, decimals=3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(INTERIM_DIR / "team_features_by_cutoff.csv"))
    ap.add_argument("--cutoffs", nargs="+", default=CUTOFF_ORDER)
    ap.add_argument("--sets", nargs="+", default=["volume", "sentiment", "types", "types_noend", "sent+types", "sent+types_noend"])
    ap.add_argument("--min-msgs", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rf-trees", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-explain", action="store_true")
    args = ap.parse_args()

    feat = pd.read_csv(args.features)
    if feat["team_won"].dtype == object:
        feat["team_won"] = feat["team_won"].map({"True": True, "False": False})

    rows, tables = [], {}
    for cutoff in args.cutoffs:
        X, y = build_match_table(feat, cutoff, args.min_msgs)
        base = max(y.mean(), 1 - y.mean())
        tables[cutoff] = (X, y)
        print(f"\n=== cutoff {cutoff}: {len(y):,} matches (both teams >= {args.min_msgs} msgs) | baseline accuracy {base:.3f} ===", flush=True)
        for sname, cols in feature_sets(X.columns).items():
            if sname not in args.sets:
                continue
            for mname, mk in [("tree", lambda: make_dt(args.seed)), ("forest", lambda: make_rf(args.seed, args.rf_trees))]:
                acc, ci, auc = cv_oof(X[cols], y, mk, args.seed, args.folds)
                rows.append(dict(cutoff=cutoff, feature_set=sname, model=mname, n_matches=len(y), n_features=len(cols),
                                 baseline=base, accuracy=acc, acc_ci95=ci, auc=auc))
                print(f"   {sname:17s} {mname:6s} acc {acc:.3f} (+/-{ci:.3f})  AUC {auc:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(INTERIM_DIR / "model_results.csv", index=False)

    def pivot(model, value):
        p = res[res["model"] == model].pivot(index="feature_set", columns="cutoff", values=value)
        p = p.reindex(index=[s for s in args.sets if s in p.index], columns=[c for c in args.cutoffs if c in p.columns])
        return p

    base = res.drop_duplicates("cutoff").set_index("cutoff")[["n_matches", "baseline"]].T[args.cutoffs]
    pd.set_option("display.width", 200)
    print("\n\n######## SUMMARY ########")
    print("\nmatches & baseline accuracy:"); print(base.round(3).to_string())
    for model in ["forest", "tree"]:
        print(f"\n{model.upper()} accuracy:"); print(pivot(model, "accuracy").round(3).to_string())
    print("\nFOREST ROC-AUC:"); print(pivot("forest", "auc").round(3).to_string())

    # --- sanity check: shuffled labels must give baseline accuracy and AUC ~0.5 ---
    if "full" in tables and "sent+types" in args.sets:
        X, y = tables["full"]
        ys = np.random.RandomState(args.seed).permutation(y)
        cols = feature_sets(X.columns)["sent+types"]
        acc, ci, auc = cv_oof(X[cols], ys, lambda: make_rf(args.seed, args.rf_trees), args.seed, args.folds)
        print(f"\nSANITY (full match, sent+types, labels SHUFFLED): acc {acc:.3f} (baseline {max(ys.mean(), 1-ys.mean()):.3f}), AUC {auc:.3f}  -> expect ~baseline and ~0.5")

    if not args.no_explain:
        for cutoff, sname in [("full", "sent+types"), ("full", "sent+types_noend"), ("15min", "sent+types_noend")]:
            if cutoff in tables and sname in args.sets:
                X, y = tables[cutoff]
                explain(X, y, feature_sets(X.columns)[sname], args.seed, args.rf_trees, f"{cutoff} / {sname}")


if __name__ == "__main__":
    main()
