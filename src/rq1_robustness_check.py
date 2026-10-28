

import argparse
from pathlib import Path

import pandas as pd

from rq1_analysis import analyze_feature, FEATURES

INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str,
        default=str(INTERIM_DIR / "trajectory_features.csv"),
    )
    parser.add_argument(
        "--thresholds", type=int, nargs="+",
        default=[1, 5, 10, 15, 20],
        help="Minimum n_messages thresholds to test (default: 1 5 10 15 20)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df["team_won"].dtype == object:
        df["team_won"] = df["team_won"].map({"True": True, "False": False})

    print("=" * 70)
    print("RQ1 ROBUSTNESS CHECK: effect of message-count threshold")
    print("=" * 70)
    print(
        "\nTesting whether the null result holds up as sparse, "
        "low-message trajectories are progressively excluded.\n"
    )

    all_rows = []
    for feature in FEATURES:
        print(f"\n{'#' * 70}")
        print(f"FEATURE: {feature}")
        print(f"{'#' * 70}")
        print(
            f"{'threshold':>10} {'n_win':>7} {'n_lose':>7} {'p_value':>9} "
            f"{'sig?':>6} {'cohens_d':>10} {'effect':>12}"
        )
        for threshold in args.thresholds:
            subset = df[df["n_messages"] >= threshold]
            result = analyze_feature(subset, feature)
            result["threshold"] = threshold
            all_rows.append(result)

            if "note" in result:
                print(
                    f"{threshold:>10} {'--':>7} {'--':>7} {'--':>9} "
                    f"{'--':>6} {'--':>10}  {result['note']}"
                )
                continue

            sig = "YES" if result["significant_at_0.05"] else "no"
            d = result["cohens_d"]
            d_str = f"{d:.4f}" if d is not None else "undefined"
            print(
                f"{threshold:>10} {result['n_winners']:>7} {result['n_losers']:>7} "
                f"{result['p_value']:>9.4f} {sig:>6} {d_str:>10} {result['effect_size']:>12}"
            )

    out_path = INTERIM_DIR / "rq1_robustness_check.csv"
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\n\nFull results saved to: {out_path}")

    print(
        "\nHow to read this: if p-values drop and Cohen's d grows in "
        "magnitude as the threshold increases, sparse trajectories were "
        "likely diluting a real effect. If nothing changes meaningfully "
        "across thresholds, the null result from rq1_analysis.py is "
        "likely robust, not a sparsity artifact - but note that sample "
        "sizes shrink at higher thresholds too, so p-values may also "
        "just get noisier rather than genuinely stronger."
    )


if __name__ == "__main__":
    main()
