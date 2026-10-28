
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"
PHASES = ["early", "mid", "late"]
FEATURES = ["slope", "volatility"]


def analyze(df, phase, feature):
    col = f"{phase}_{feature}"
    valid = df.dropna(subset=[col])
    if len(valid) < 10:
        return {"phase": phase, "feature": feature, "n": len(valid), "note": "insufficient data"}

    # team_won as 0/1 for correlation
    outcome = valid["team_won"].astype(int)
    rho, p_value = spearmanr(valid[col], outcome)

    return {
        "phase": phase,
        "feature": feature,
        "n": len(valid),
        "rho": rho,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(INTERIM_DIR / "phase_trajectory_features.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df["team_won"].dtype == object:
        df["team_won"] = df["team_won"].map({"True": True, "False": False})

    print("=" * 70)
    print("RQ3: Does trajectory-outcome correlation change across game phases?")
    print("=" * 70)

    results = []
    for feature in FEATURES:
        print(f"\n{'#' * 70}")
        print(f"FEATURE: {feature}")
        print(f"{'#' * 70}")
        print(f"{'phase':>8} {'n':>8} {'rho':>10} {'p_value':>12} {'significant?':>13}")
        for phase in PHASES:
            result = analyze(df, phase, feature)
            results.append(result)
            if "note" in result:
                print(f"{phase:>8} {result['n']:>8} {'--':>10} {'--':>12} {result['note']:>13}")
                continue
            sig = "YES" if result["significant_at_0.05"] else "no"
            print(f"{phase:>8} {result['n']:>8} {result['rho']:>10.4f} {result['p_value']:>12.2e} {sig:>13}")

    out_path = INTERIM_DIR / "rq3_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n\nFull results saved to: {out_path}")
    print(
        "\nReminder: the early-phase sample is smaller AND a non-random subset (only teams that\n"
        "happened to chat early are included) - treat early-vs-late comparisons with that caveat in mind."
    )


if __name__ == "__main__":
    main()
