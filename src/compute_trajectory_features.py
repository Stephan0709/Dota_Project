

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vader_lexicon_override import get_analyzer_with_override

INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"


def compute_slope(times, scores):
    if len(times) < 2:
        return np.nan
    # np.polyfit fits a degree-1 polynomial (a line): returns [slope, intercept]
    slope, _ = np.polyfit(times, scores, deg=1)
    return slope


def compute_volatility(scores):
    if len(scores) < 2:
        return np.nan
    return float(np.std(scores, ddof=1))  # sample std, not population std


def compute_phase_delta(df_team):
    early_scores = df_team.loc[df_team["phase"] == "early", "compound"]
    late_scores = df_team.loc[df_team["phase"] == "late", "compound"]
    if len(early_scores) == 0 or len(late_scores) == 0:
        return np.nan
    return float(late_scores.mean() - early_scores.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str,
        default=str(INTERIM_DIR / "chat_with_phases.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    analyzer = get_analyzer_with_override()
    print(f"Scoring {len(df)} messages with VADER (+ domain lexicon override)...")
    df["compound"] = df["text"].astype(str).apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )

    print("Computing per-team, per-match trajectory features...")
    rows = []
    for (match_id, team), group in df.groupby(["match_id", "team"]):
        group = group.sort_values("time")
        times = group["time"].values
        scores = group["compound"].values

        radiant_win = group["radiant_win"].iloc[0]
        # Defensive: coerce to real bool even if CSV round-tripping ever
        # produced a string "True"/"False" instead of a native bool -
        # bool("False") is True in Python, which would silently corrupt
        # team_won if not handled explicitly.
        if isinstance(radiant_win, str):
            radiant_win = radiant_win.strip().lower() == "true"
        else:
            radiant_win = bool(radiant_win)
        team_won = (team == "radiant" and radiant_win) or (
            team == "dire" and not radiant_win
        )

        rows.append({
            "match_id": match_id,
            "team": team,
            "n_messages": len(group),
            "slope": compute_slope(times, scores),
            "volatility": compute_volatility(scores),
            "phase_delta": compute_phase_delta(group),
            "team_won": team_won,
            "boundary1_source": group["boundary1_source"].iloc[0],
            "boundary2_source": group["boundary2_source"].iloc[0],
        })

    features_df = pd.DataFrame(rows)
    out_path = INTERIM_DIR / "trajectory_features.csv"
    features_df.to_csv(out_path, index=False)

    n_total = len(features_df)
    n_slope_ready = features_df["slope"].notna().sum()
    n_volatility_ready = features_df["volatility"].notna().sum()
    n_phase_delta_ready = features_df["phase_delta"].notna().sum()

    print(f"\nSaved {n_total} team-trajectory rows to: {out_path}\n")
    print(f"Total team-trajectories (matches x 2 teams): {n_total}")
    print(f"  Slope computable:       {n_slope_ready} / {n_total}")
    print(f"  Volatility computable:  {n_volatility_ready} / {n_total}")
    print(f"  Phase delta computable: {n_phase_delta_ready} / {n_total}")

    print(f"\nOutcome split (team_won):")
    print(features_df["team_won"].value_counts())

    print("\nFeature summary statistics:")
    print(features_df[["slope", "volatility", "phase_delta", "n_messages"]].describe())


if __name__ == "__main__":
    main()