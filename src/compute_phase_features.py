

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vader_lexicon_override import get_analyzer_with_override

INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"

PHASES = ["early", "mid", "late"]


def compute_slope(times, scores):
    # Require >=2 DISTINCT timestamps, not just >=2 messages - two
    # messages sent in the same second cannot support a trend line
    # (this was the source of the RankWarning noise seen in RQ1).
    if len(set(times)) < 2:
        return np.nan
    slope, _ = np.polyfit(times, scores, deg=1)
    return slope


def compute_volatility(scores):
    if len(scores) < 2:
        return np.nan
    return float(np.std(scores, ddof=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(INTERIM_DIR / "chat_with_phases.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    analyzer = get_analyzer_with_override()
    print(f"Scoring {len(df)} messages with VADER (+ domain lexicon override)...")
    df["compound"] = df["text"].astype(str).apply(lambda t: analyzer.polarity_scores(t)["compound"])

    if df["radiant_win"].dtype == object:
        df["radiant_win"] = df["radiant_win"].map({"True": True, "False": False})

    print("Computing per-phase trajectory features ...")
    rows = []
    for (match_id, team), group in df.groupby(["match_id", "team"]):
        radiant_win = bool(group["radiant_win"].iloc[0])
        team_won = (team == "radiant" and radiant_win) or (team == "dire" and not radiant_win)

        row = {"match_id": match_id, "team": team, "team_won": team_won}
        for phase in PHASES:
            phase_group = group[group["phase"] == phase].sort_values("time")
            times = phase_group["time"].values
            scores = phase_group["compound"].values
            row[f"{phase}_slope"] = compute_slope(times, scores)
            row[f"{phase}_volatility"] = compute_volatility(scores)
            row[f"{phase}_n"] = len(phase_group)
        rows.append(row)

    features_df = pd.DataFrame(rows)
    out_path = INTERIM_DIR / "phase_trajectory_features.csv"
    features_df.to_csv(out_path, index=False)

    n_total = len(features_df)
    print(f"\nSaved {n_total} team-trajectory rows to: {out_path}\n")

    print("Readiness per phase (how many team-trajectories have a computable slope/volatility in that phase):")
    for phase in PHASES:
        n_slope = features_df[f"{phase}_slope"].notna().sum()
        n_vol = features_df[f"{phase}_volatility"].notna().sum()
        median_n = features_df[f"{phase}_n"].median()
        print(f"  {phase:6s}: slope {n_slope:6d} / {n_total} ({n_slope/n_total*100:.1f}%)  "
              f"volatility {n_vol:6d} / {n_total} ({n_vol/n_total*100:.1f}%)  "
              f"median messages in phase: {median_n:.0f}")

    print(f"\nOutcome split (team_won): {features_df['team_won'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
