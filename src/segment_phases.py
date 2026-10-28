

import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"

BOUNDARY1_FIXED_FALLBACK = 15 * 60
BOUNDARY2_FIXED_FALLBACK = 35 * 60


def load_objectives():
    path = RAW_DIR / "objectives.csv"
    if not path.exists():
        print(f"objectives.csv not found at {path}")
        return None
    return pd.read_csv(path, usecols=["match_id", "subtype", "time"])


def determine_boundaries_for_match(events, boundary1_mode="earliest"):
    roshan_times = sorted(events.loc[events["subtype"] == "CHAT_MESSAGE_ROSHAN_KILL", "time"].tolist())
    tower_times = sorted(events.loc[events["subtype"] == "CHAT_MESSAGE_TOWER_KILL", "time"].tolist())
    rax_times = sorted(events.loc[events["subtype"] == "CHAT_MESSAGE_BARRACKS_KILL", "time"].tolist())

    # Boundary 1
    if boundary1_mode == "earliest":
        candidates = []
        if roshan_times:
            candidates.append((roshan_times[0], "roshan"))
        if tower_times:
            candidates.append((tower_times[0], "tower"))
        if candidates:
            boundary1, boundary1_source = min(candidates, key=lambda c: c[0])
        else:
            boundary1, boundary1_source = BOUNDARY1_FIXED_FALLBACK, "fixed"
    else:  # cascade (original behaviour)
        if roshan_times:
            boundary1 = roshan_times[0]
            boundary1_source = "roshan"
        elif tower_times:
            boundary1 = tower_times[0]
            boundary1_source = "tower"
        else:
            boundary1 = BOUNDARY1_FIXED_FALLBACK
            boundary1_source = "fixed"

    # Boundary 2 - must be strictly after boundary 1
    candidates = []
    for t in rax_times:
        if t > boundary1:
            candidates.append((t, "barracks"))
            break
    if not candidates and len(roshan_times) >= 2:
        second_rosh = roshan_times[1]
        if second_rosh > boundary1:
            candidates.append((second_rosh, "second_roshan"))

    if candidates:
        boundary2, boundary2_source = candidates[0]
    else:
        boundary2 = BOUNDARY2_FIXED_FALLBACK
        boundary2_source = "fixed"
        if boundary2 <= boundary1:
            boundary2 = boundary1 + BOUNDARY2_FIXED_FALLBACK

    return {
        "boundary1": boundary1,
        "boundary1_source": boundary1_source,
        "boundary2": boundary2,
        "boundary2_source": boundary2_source,
    }


def assign_phase(time, boundary1, boundary2):
    if time < boundary1:
        return "early"
    elif time < boundary2:
        return "mid"
    else:
        return "late"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str,
        default=str(INTERIM_DIR / "clean_chat_messages_min10_bothteams.csv"),
    )
    parser.add_argument(
        "--boundary1-mode", type=str, default="earliest", choices=["cascade", "earliest"],
        help=(
            "'earliest' (default): compare actual Roshan and tower "
            "timestamps and use whichever happened first - adopted "
            "because real data showed Roshan-based boundaries averaging "
            "~27 minutes in, implausibly late for this genre. "
            "'cascade': always use Roshan if present anywhere in the "
            "match, regardless of timing (original behaviour)."
        ),
    )
    args = parser.parse_args()

    chat_df = pd.read_csv(args.input)
    match_ids = chat_df["match_id"].unique()
    print(f"Loading objectives.csv...")
    obj_df = load_objectives()
    if obj_df is None:
        return
    obj_df = obj_df[obj_df["match_id"].isin(match_ids)]

    print(f"Determining phase boundaries for {len(match_ids)} matches...")
    boundary_rows = []
    for match_id, events in obj_df.groupby("match_id"):
        boundaries = determine_boundaries_for_match(events, boundary1_mode=args.boundary1_mode)
        boundaries["match_id"] = match_id
        boundary_rows.append(boundaries)

    # Matches with chat but zero objectives rows at all (rare) still need
    # a fallback entry, otherwise they'd silently vanish from the merge.
    matches_with_no_objectives = set(match_ids) - set(obj_df["match_id"].unique())
    for match_id in matches_with_no_objectives:
        boundary_rows.append({
            "boundary1": BOUNDARY1_FIXED_FALLBACK, "boundary1_source": "fixed",
            "boundary2": BOUNDARY2_FIXED_FALLBACK, "boundary2_source": "fixed",
            "match_id": match_id,
        })
    if matches_with_no_objectives:
        print(f"Note: {len(matches_with_no_objectives)} matches had no objectives.csv rows at all - used fixed fallback for both boundaries")

    boundaries_df = pd.DataFrame(boundary_rows)
    boundaries_path = INTERIM_DIR / "match_phase_boundaries.csv"
    boundaries_df.to_csv(boundaries_path, index=False)
    print(f"Saved boundaries for {len(boundaries_df)} matches to: {boundaries_path}\n")

    print("Boundary 1 (early->mid) source breakdown:")
    print(boundaries_df["boundary1_source"].value_counts())
    print("\nBoundary 2 (mid->late) source breakdown:")
    print(boundaries_df["boundary2_source"].value_counts())

    both_fixed = (
        (boundaries_df["boundary1_source"] == "fixed")
        & (boundaries_df["boundary2_source"] == "fixed")
    ).sum()
    print(f"\nMatches using FIXED fallback for BOTH boundaries: {both_fixed} / {len(boundaries_df)}")

    print("\nBoundary 1 timing distribution (seconds), by source:")
    print(boundaries_df.groupby("boundary1_source")["boundary1"].describe()[["mean", "50%", "min", "max"]])
    print("\nBoundary 2 timing distribution (seconds), by source:")
    print(boundaries_df.groupby("boundary2_source")["boundary2"].describe()[["mean", "50%", "min", "max"]])

    merged = chat_df.merge(boundaries_df, on="match_id", how="inner")
    merged["phase"] = merged.apply(
        lambda row: assign_phase(row["time"], row["boundary1"], row["boundary2"]),
        axis=1,
    )

    out_path = INTERIM_DIR / "chat_with_phases.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved phase-tagged chat data to: {out_path}")
    print(f"Total messages: {len(merged)}")
    print(f"Messages by phase:\n{merged['phase'].value_counts()}")

    all_phases = ["early", "mid", "late"]
    all_teams = ["radiant", "dire"]
    merged["phase"] = pd.Categorical(merged["phase"], categories=all_phases)
    merged["team"] = pd.Categorical(merged["team"], categories=all_teams)

    coverage = merged.groupby(["match_id", "team", "phase"], observed=False).size().reset_index(name="count")
    pivot = coverage.pivot_table(index="match_id", columns=["team", "phase"], values="count", fill_value=0)

    full_coverage_mask = (pivot > 0).all(axis=1)
    print(f"\nMatches with >=1 message in ALL 3 phases for BOTH teams: {full_coverage_mask.sum()} / {len(pivot)}")
    print("(strictest possible bar - see feature-specific breakdown below)")

    print("\n" + "-" * 50)
    print("What each trajectory feature actually requires (proposal 2.5):")
    print(f"  Slope & volatility (whole-match, per team): {chat_df['match_id'].nunique()} / {chat_df['match_id'].nunique()} matches ready")

    early_late_pivot = pivot.loc[:, pivot.columns.get_level_values("phase").isin(["early", "late"])]
    phase_delta_ready = (early_late_pivot > 0).all(axis=1)
    print(f"  Phase delta (early + late only, per team): {phase_delta_ready.sum()} / {len(pivot)} matches ready")


if __name__ == "__main__":
    main()