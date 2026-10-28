

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

from vader_lexicon_override import get_analyzer_with_override

INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"

CANDIDATE_PENALTIES = [0.25, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0]


def build_team_sequences(df):
    """Return a dict of (match_id, team) -> sorted (times, scores) arrays."""
    analyzer = get_analyzer_with_override()
    df = df.copy()
    df["compound"] = df["text"].astype(str).apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )

    sequences = {}
    for (match_id, team), group in df.groupby(["match_id", "team"]):
        group = group.sort_values("time")
        sequences[(match_id, team)] = (
            group["time"].values.astype(float),
            group["compound"].values.astype(float),
        )
    return sequences


def count_breakpoints(scores, penalty):
    """Run PELT on a single team's sentiment sequence, return the number
    of detected shift points (excluding the trivial final breakpoint
    ruptures always includes at the end of the series)."""
    if len(scores) < 4:
        return 0  # too short for PELT to say anything meaningful
    signal = scores.reshape(-1, 1)
    algo = rpt.Pelt(model="rbf").fit(signal)
    breakpoints = algo.predict(pen=penalty)
    # ruptures always includes len(signal) as the final "breakpoint" -
    # that's not a real shift point, just the end of the series
    real_breakpoints = [b for b in breakpoints if b < len(signal)]
    return len(real_breakpoints)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str,
        default=str(INTERIM_DIR / "chat_with_phases.csv"),
    )
    parser.add_argument("--sample-size", type=int, default=150,
                         help="Number of team-trajectories to hold out for this pilot")
    parser.add_argument("--min-messages", type=int, default=15,
                         help="Minimum messages for a team-trajectory to be eligible for the pilot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    print("Scoring messages and building per-team sequences...")
    all_sequences = build_team_sequences(df)

    eligible = {k: v for k, v in all_sequences.items() if len(v[1]) >= args.min_messages}
    print(f"Team-trajectories with >= {args.min_messages} messages: {len(eligible)} / {len(all_sequences)}")

    rng = np.random.RandomState(args.seed)
    keys = list(eligible.keys())
    sample_keys = rng.choice(len(keys), size=min(args.sample_size, len(keys)), replace=False)
    pilot_keys = [keys[i] for i in sample_keys]
    print(f"Held-out pilot sample: {len(pilot_keys)} team-trajectories\n")

    print(f"{'Penalty':>8} | {'Mean BPs':>9} | {'Median BPs':>10} | {'% with 1-3 BPs':>15} | {'% with 0 BPs':>13}")
    print("-" * 68)

    results = []
    for pen in CANDIDATE_PENALTIES:
        bp_counts = []
        for key in pilot_keys:
            _, scores = eligible[key]
            bp_counts.append(count_breakpoints(scores, pen))
        bp_counts = np.array(bp_counts)
        mean_bp = bp_counts.mean()
        median_bp = np.median(bp_counts)
        pct_1_to_3 = ((bp_counts >= 1) & (bp_counts <= 3)).mean() * 100
        pct_zero = (bp_counts == 0).mean() * 100
        results.append({
            "penalty": pen, "mean_bp": mean_bp, "median_bp": median_bp,
            "pct_1_to_3": pct_1_to_3, "pct_zero": pct_zero,
        })
        print(f"{pen:>8} | {mean_bp:>9.2f} | {median_bp:>10.1f} | {pct_1_to_3:>14.1f}% | {pct_zero:>12.1f}%")

    out_path = INTERIM_DIR / "rq2_pelt_penalty_pilot.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")
    print(
        "\nLook for the penalty where 'Mean BPs' lands around 1-3 and '% with 1-3 BPs' "
        "is high, without '% with 0 BPs' being too large (under-segmenting) or mean BPs "
        "climbing unboundedly (over-segmenting noise as signal)."
    )


if __name__ == "__main__":
    main()
