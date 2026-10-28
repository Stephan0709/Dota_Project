

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

from vader_lexicon_override import get_analyzer_with_override

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"

PROXIMITY_WINDOW_SECONDS = 120


def build_team_sequences(df, min_messages):
    analyzer = get_analyzer_with_override()
    df = df.copy()
    print(f"Scoring {len(df)} messages with VADER (+ domain lexicon override)...")
    df["compound"] = df["text"].astype(str).apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )

    sequences = {}
    for (match_id, team), group in df.groupby(["match_id", "team"]):
        if len(group) < min_messages:
            continue
        group = group.sort_values("time")
        sequences[(match_id, team)] = (
            group["time"].values.astype(float),
            group["compound"].values.astype(float),
        )
    return sequences


def detect_shift_points(times, scores, penalty):
    """Run PELT, return the actual TIMESTAMPS of detected shift points
    (not array indices - these get mapped back to real message times)."""
    if len(scores) < 4:
        return []
    signal = scores.reshape(-1, 1)
    algo = rpt.Pelt(model="rbf").fit(signal)
    breakpoints = algo.predict(pen=penalty)
    # Drop the trivial final "breakpoint" ruptures always adds (end of series)
    real_breakpoint_indices = [b for b in breakpoints if b < len(signal)]
    # Map array index -> the actual message time at that position
    return [times[idx] for idx in real_breakpoint_indices]


def load_objectives(match_ids):
    path = RAW_DIR / "objectives.csv"
    obj = pd.read_csv(path, usecols=["match_id", "subtype", "time"])
    return obj[obj["match_id"].isin(match_ids)]


def find_nearest_event(shift_time, match_events):
    """match_events: DataFrame of objectives rows for one match. Returns
    (subtype, time_diff) for the closest event within the proximity
    window, or (None, None) if nothing qualifies."""
    if match_events.empty:
        return None, None
    diffs = (match_events["time"] - shift_time).abs()
    within_window = diffs <= PROXIMITY_WINDOW_SECONDS
    if not within_window.any():
        return None, None
    nearest_idx = diffs[within_window].idxmin()
    return match_events.loc[nearest_idx, "subtype"], diffs.loc[nearest_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(INTERIM_DIR / "chat_with_phases.csv"))
    parser.add_argument("--penalty", type=float, default=0.55,
                         help="PELT penalty, selected via rq2_pelt_pilot.py")
    parser.add_argument("--min-messages", type=int, default=15)
    parser.add_argument("--sample", type=int, default=None,
                         help="If set, only process this many team-trajectories (for quick testing)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    sequences = build_team_sequences(df, args.min_messages)
    print(f"Team-trajectories with >= {args.min_messages} messages: {len(sequences)}")

    keys = list(sequences.keys())
    if args.sample:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(keys), size=min(args.sample, len(keys)), replace=False)
        keys = [keys[i] for i in idx]
        print(f"Processing a sample of {len(keys)} team-trajectories")

    print(f"\nDetecting shift points (penalty={args.penalty})...")
    shift_rows = []
    for match_id, team in keys:
        times, scores = sequences[(match_id, team)]
        shift_times = detect_shift_points(times, scores, args.penalty)
        for st in shift_times:
            shift_rows.append({"match_id": match_id, "team": team, "shift_time": st})

    shift_df = pd.DataFrame(shift_rows)
    print(f"Total shift points detected: {len(shift_df)} across {len(keys)} team-trajectories")
    print(f"Mean shift points per trajectory: {len(shift_df) / len(keys):.2f}")

    print("\nLoading objectives.csv and aligning shift points to nearest event...")
    match_ids_needed = shift_df["match_id"].unique()
    obj_df = load_objectives(match_ids_needed)
    obj_by_match = {mid: g for mid, g in obj_df.groupby("match_id")}

    nearest_subtypes = []
    time_diffs = []
    for _, row in shift_df.iterrows():
        match_events = obj_by_match.get(row["match_id"], pd.DataFrame(columns=["subtype", "time"]))
        subtype, diff = find_nearest_event(row["shift_time"], match_events)
        nearest_subtypes.append(subtype)
        time_diffs.append(diff)

    shift_df["nearest_event"] = nearest_subtypes
    shift_df["time_diff_seconds"] = time_diffs

    out_path = INTERIM_DIR / "rq2_shift_points.csv"
    shift_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(shift_df)} shift points to: {out_path}")

    matched = shift_df["nearest_event"].notna()
    print(f"\nShift points matched to a nearby event (within {PROXIMITY_WINDOW_SECONDS}s): {matched.sum()} / {len(shift_df)} ({matched.mean()*100:.1f}%)")
    print(f"Shift points with NO event nearby: {(~matched).sum()} ({(~matched).mean()*100:.1f}%)")

    print("\nEvent type ranking by RAW co-occurrence (proposal's stated metric):")
    ranking = shift_df.loc[matched, "nearest_event"].value_counts()
    ranking_pct = (ranking / matched.sum() * 100).round(1)
    for event, count in ranking.items():
        print(f"  {event:32s}: {count:6d} ({ranking_pct[event]}%)")

    # Supplementary: base-rate-adjusted association rate. Raw co-occurrence
    # counts are dominated by whichever event type simply happens most often
    # (towers fall far more often than Roshan is killed) - this doesn't
    # necessarily mean towers are MORE associated with sentiment shifts,
    # just that there are more opportunities for a tower kill to be nearby
    # ANY given point in time. Dividing by each event type's total
    # occurrence (within the same matches) controls for this.
    print("\nSupplementary: association RATE (% of each event type's total occurrences")
    print("that had a shift point within 2 minutes) - controls for base rate:")
    all_events_in_sample = obj_df[obj_df["match_id"].isin(shift_df["match_id"].unique())]
    total_by_type = all_events_in_sample["subtype"].value_counts()
    assoc_rate = (ranking / total_by_type * 100).dropna().sort_values(ascending=False)
    for event, rate in assoc_rate.items():
        print(f"  {event:32s}: {rate:.2f}% of {total_by_type[event]:,} total occurrences")


if __name__ == "__main__":
    main()
