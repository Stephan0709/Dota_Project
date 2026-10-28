

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"

FEATURES = ["slope", "volatility", "phase_delta"]


def cohens_d(group1, group2):
    """d = (mean1 - mean2) / pooled_std. Positive d means group1 (winners)
    scores higher on this feature than group2 (losers)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def effect_size_label(d):
    if pd.isna(d):
        return "undefined"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


def analyze_feature(df, feature):
    valid = df.dropna(subset=[feature])
    winners = valid.loc[valid["team_won"] == True, feature]
    losers = valid.loc[valid["team_won"] == False, feature]

    if len(winners) < 2 or len(losers) < 2:
        return {
            "feature": feature,
            "n_winners": len(winners),
            "n_losers": len(losers),
            "note": "insufficient data for a valid test (need >=2 per group)",
        }

    u_stat, p_value = mannwhitneyu(winners, losers, alternative="two-sided")
    d = cohens_d(winners, losers)

    return {
        "feature": feature,
        "n_winners": len(winners),
        "n_losers": len(losers),
        "median_winners": float(winners.median()),
        "median_losers": float(losers.median()),
        "mean_winners": float(winners.mean()),
        "mean_losers": float(losers.mean()),
        "U_statistic": float(u_stat),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
        "cohens_d": float(d) if not pd.isna(d) else None,
        "effect_size": effect_size_label(d),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str,
        default=str(INTERIM_DIR / "trajectory_features.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    # CSV round-trips booleans as strings sometimes - normalize defensively
    if df["team_won"].dtype == object:
        df["team_won"] = df["team_won"].map({"True": True, "False": False})

    print("=" * 60)
    print("RQ1: Winning vs Losing Teams - Trajectory Feature Comparison")
    print("=" * 60)

    results = []
    for feature in FEATURES:
        result = analyze_feature(df, feature)
        results.append(result)

        print(f"\n--- {feature} ---")
        if "note" in result:
            print(f"  SKIPPED: {result['note']}")
            continue
        print(f"  n (winners / losers): {result['n_winners']} / {result['n_losers']}")
        print(f"  median (winners / losers): {result['median_winners']:.4f} / {result['median_losers']:.4f}")
        print(f"  mean   (winners / losers): {result['mean_winners']:.4f} / {result['mean_losers']:.4f}")
        print(f"  Mann-Whitney U = {result['U_statistic']:.1f}, p = {result['p_value']:.4f}"
              f" {'(significant at alpha=0.05)' if result['significant_at_0.05'] else '(not significant)'}")
        if result["cohens_d"] is not None:
            print(f"  Cohen's d = {result['cohens_d']:.4f} ({result['effect_size']} effect)")
        else:
            print("  Cohen's d: undefined (zero pooled variance)")

    out_path = INTERIM_DIR / "rq1_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()