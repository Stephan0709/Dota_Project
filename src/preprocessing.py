

import argparse
import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"


def team_from_player_slot(player_slot):
    
    return "dire" if player_slot >= 128 else "radiant"


def load_match(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_usable_messages(match_data):
    match_id = match_data.get("match_id")
    radiant_win = match_data.get("radiant_win")

    rows = []
    for msg in match_data.get("chat", []):
        if msg.get("type") != "chat":
            continue  # drop chatwheel presets
        if msg.get("time", -1) < 0:
            continue  # drop pre-game banter

        player_slot = msg.get("player_slot")
        if player_slot is None:
            continue  # malformed entry, skip defensively

        rows.append(
            {
                "match_id": match_id,
                "time": msg["time"],
                "team": team_from_player_slot(player_slot),
                "player_slot": player_slot,
                "text": msg.get("key", ""),
                "radiant_win": radiant_win,
            }
        )
    return rows


def process_all_matches():
    raw_files = sorted(RAW_DIR.glob("match_*.json"))
    if not raw_files:
        print(f"No raw match files found in {RAW_DIR}")
        return None

    all_rows = []
    for path in raw_files:
        match_data = load_match(path)
        rows = extract_usable_messages(match_data)
        all_rows.extend(rows)
        print(f"{path.name}: {len(rows)} usable messages")

    df = pd.DataFrame(all_rows)
    return df


def summarize_match_stats(df):
    per_match = df.groupby("match_id").agg(
        total_messages=("text", "count"),
        teams_present=("team", lambda s: s.nunique()),
    )
    return per_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-messages", type=int, default=1,
        help=(
            "Minimum total usable messages a match needs to be included "
            "in the filtered output. Matches below this are still counted "
            "and reported, but excluded from the filtered CSV. Default: 1 "
            "(only removes matches with zero usable messages)."
        ),
    )
    parser.add_argument(
        "--require-both-teams", action="store_true",
        help=(
            "Also require at least one message from EACH team for a match "
            "to be included in the filtered output. A match can clear "
            "--min-messages while still having zero messages from one "
            "team, which breaks per-team trajectory construction just as "
            "badly as having too few messages overall."
        ),
    )
    args = parser.parse_args()

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    df = process_all_matches()
    if df is None or df.empty:
        print("No usable messages extracted. Nothing written.")
        return

    out_path = INTERIM_DIR / "clean_chat_messages.csv"
    df.to_csv(out_path, index=False)

    per_match = summarize_match_stats(df)
    total_matches = len(per_match)

    print("\n" + "=" * 50)
    print(f"Total matches processed: {total_matches}")
    print(f"Total usable messages:   {len(df)}")
    print(f"Messages by team:\n{df['team'].value_counts()}")
    print(f"\nSaved (unfiltered) to: {out_path}")

    print("\n" + "-" * 50)
    print("Matches surviving at different message-count thresholds (total messages, either team):")
    for t in [1, 4, 6, 10, 15, 20]:
        n = (per_match["total_messages"] >= t).sum()
        print(f"  >= {t:2d} messages: {n:4d} matches")

    both_teams = per_match["teams_present"] == 2
    print(f"\nMatches with BOTH teams represented (>=1 msg each): {both_teams.sum()} / {total_matches}")

    
    print("\nCOMBINED (>= threshold messages AND both teams present):")
    for t in [1, 4, 6, 10, 15, 20]:
        n = ((per_match["total_messages"] >= t) & both_teams).sum()
        print(f"  >= {t:2d} messages + both teams: {n:4d} matches")

    if args.min_messages > 1 or args.require_both_teams:
        qualifying = per_match["total_messages"] >= args.min_messages
        if args.require_both_teams:
            qualifying = qualifying & both_teams
        qualifying_ids = per_match[qualifying].index
        filtered_df = df[df["match_id"].isin(qualifying_ids)]

        suffix = f"min{args.min_messages}"
        if args.require_both_teams:
            suffix += "_bothteams"
        filtered_path = INTERIM_DIR / f"clean_chat_messages_{suffix}.csv"
        filtered_df.to_csv(filtered_path, index=False)

        criteria = f">= {args.min_messages} messages"
        if args.require_both_teams:
            criteria += " AND both teams present"
        print(f"\nFiltered to matches with {criteria}: {len(qualifying_ids)} matches, {len(filtered_df)} messages")
        print(f"Saved filtered set to: {filtered_path}")


if __name__ == "__main__":
    main()