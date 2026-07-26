

import argparse
import json
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"

BOUNDARY1_FIXED_FALLBACK = 15 * 60  
BOUNDARY2_FIXED_FALLBACK = 35 * 60  

TOWER_TIER_PATTERN = re.compile(r"tower(\d)")


def load_match_objectives(match_id):
    path = RAW_DIR / f"match_{match_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("objectives") or []


def extract_roshan_kill_times(objectives):
    times = [
        o["time"] for o in objectives
        if o.get("type") == "CHAT_MESSAGE_ROSHAN_KILL" and "time" in o
    ]
    return sorted(times)


def extract_tower_kill_times(objectives, tier):
    """tier: int (1-4). Matches key strings like 'npc_dota_goodguys_tower3_top'."""
    times = []
    for o in objectives:
        if o.get("type") != "building_kill":
            continue
        key = o.get("key", "")
        m = TOWER_TIER_PATTERN.search(key)
        if m and int(m.group(1)) == tier:
            times.append(o["time"])
    return sorted(times)


def determine_boundaries(objectives, boundary1_fallback_tier=1):
    roshan_times = extract_roshan_kill_times(objectives)
    fallback_tower_times = extract_tower_kill_times(objectives, boundary1_fallback_tier)
    tier3_times = extract_tower_kill_times(objectives, 3)
    tier4_times = extract_tower_kill_times(objectives, 4)

    # Boundary 1
    if roshan_times:
        boundary1 = roshan_times[0]
        boundary1_source = "roshan"
    elif fallback_tower_times:
        boundary1 = fallback_tower_times[0]
        boundary1_source = f"tower_tier{boundary1_fallback_tier}"
    else:
        boundary1 = BOUNDARY1_FIXED_FALLBACK
        boundary1_source = "fixed"

    # Boundary 2
    candidates = []
    for t in tier3_times:
        if t > boundary1:
            candidates.append((t, "tower_tier3"))
            break
    if not candidates:
        for t in tier4_times:
            if t > boundary1:
                candidates.append((t, "tower_tier4"))
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
        default=str(INTERIM_DIR / "clean_chat_messages_min6_bothteams.csv"),
    )
    parser.add_argument(
        "--boundary1-fallback-tier", type=int, default=1, choices=[1, 2],
        help=(
            "Which tower tier to use as the boundary-1 fallback when no "
            "Roshan kill exists (86%% of matches). Tier-1 towers often "
            "fall very early, potentially squeezing the early-game phase "
            "window too thin for phase-delta coverage. Try --boundary1-"
            "fallback-tier 2 to test whether a later fallback improves "
            "phase-delta readiness. Default: 1."
        ),
    )
    args = parser.parse_args()

    chat_df = pd.read_csv(args.input)
    match_ids = chat_df["match_id"].unique()
    print(f"Determining phase boundaries for {len(match_ids)} matches...")

    boundary_rows = []
    missing_objectives = 0
    for match_id in match_ids:
        objectives = load_match_objectives(match_id)
        if objectives is None:
            missing_objectives += 1
            continue
        boundaries = determine_boundaries(objectives, boundary1_fallback_tier=args.boundary1_fallback_tier)
        boundaries["match_id"] = match_id
        boundary_rows.append(boundaries)

    if missing_objectives:
        print(f"WARNING: {missing_objectives} matches had no raw JSON file found - skipped.")

    boundaries_df = pd.DataFrame(boundary_rows)
    boundaries_path = INTERIM_DIR / "match_phase_boundaries.csv"
    boundaries_df.to_csv(boundaries_path, index=False)
    print(f"Saved boundaries for {len(boundaries_df)} matches to: {boundaries_path}\n")

    print("Boundary 1 (early->mid) source breakdown:")
    print(boundaries_df["boundary1_source"].value_counts())
    print("\nBoundary 2 (mid->late) source breakdown:")
    print(boundaries_df["boundary2_source"].value_counts())

    print("\nBoundary 1 timing distribution (seconds), by source:")
    print(boundaries_df.groupby("boundary1_source")["boundary1"].describe()[["mean", "50%", "min", "max"]])

    both_fixed = (
        (boundaries_df["boundary1_source"] == "fixed")
        & (boundaries_df["boundary2_source"] == "fixed")
    ).sum()
    print(f"\nMatches using FIXED fallback for BOTH boundaries: {both_fixed} / {len(boundaries_df)}")
    print("(recommend flagging these as a separate stratum per proposal 4.5.1)")

    # Merge boundaries onto the chat data and assign phases
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
    pivot = coverage.pivot_table(
        index="match_id", columns=["team", "phase"], values="count", fill_value=0
    )
    
    expected_cols = len(all_teams) * len(all_phases)
    if pivot.shape[1] != expected_cols:
        print(
            f"WARNING: expected {expected_cols} team/phase columns, got "
            f"{pivot.shape[1]}. Coverage check below may be unreliable."
        )
    full_coverage_mask = (pivot > 0).all(axis=1)
    n_full_coverage = full_coverage_mask.sum()
    print(
        f"\nMatches with >=1 message in ALL 3 phases for BOTH teams: "
        f"{n_full_coverage} / {len(pivot)}"
    )
    print(
        "(this is the STRICTEST possible bar - useful as a ceiling "
        "reference, but NOT what most trajectory features actually "
        "require - see the breakdown below)"
    )

    
    print("\n" + "-" * 50)
    print("What each trajectory feature actually requires (proposal 2.5):")

    
    print(
        f"  Slope & volatility (whole-match, per team): "
        f"{chat_df['match_id'].nunique()} / {chat_df['match_id'].nunique()} "
        f"matches ready (already guaranteed by preprocessing.py filtering)"
    )

    # Phase delta: needs early AND late (not mid) present for BOTH teams
    early_late_pivot = pivot.loc[:, pivot.columns.get_level_values("phase").isin(["early", "late"])]
    phase_delta_ready = (early_late_pivot > 0).all(axis=1)
    n_phase_delta_ready = phase_delta_ready.sum()
    print(
        f"  Phase delta (early + late only, per team): "
        f"{n_phase_delta_ready} / {len(pivot)} matches ready"
    )


if __name__ == "__main__":
    main()