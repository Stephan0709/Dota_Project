

import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"


def team_from_slot(slot):
    """This dataset's 'slot' is already the simple 0-9 index (not the
    Valve player_slot bitmask our old OpenDota pipeline had to handle)."""
    return "dire" if slot >= 5 else "radiant"


def load_and_clean_chat():
    match_path = RAW_DIR / "match.csv"
    chat_path = RAW_DIR / "chat.csv"

    if not match_path.exists() or not chat_path.exists():
        print(f"Expected files not found: {match_path} and {chat_path}")
        return None

    match_df = pd.read_csv(match_path, usecols=["match_id", "radiant_win"])
    chat_df = pd.read_csv(chat_path)

    before = len(chat_df)

    # Drop null text
    chat_df = chat_df.dropna(subset=["key"])

    # Drop pre-game messages (same rule as the old pipeline - these
    # precede the match clock and aren't reactions to any gameplay event)
    chat_df = chat_df[chat_df["time"] >= 0]

    # Defensive: drop any slot outside the valid 0-9 range. In practice
    # every such row observed so far already had time < 0 and was
    # dropped above, but this guards against that not always holding.
    chat_df = chat_df[(chat_df["slot"] >= 0) & (chat_df["slot"] <= 9)]

    after = len(chat_df)
    print(f"chat.csv: {before:,} raw rows -> {after:,} usable after cleaning")

    chat_df["team"] = chat_df["slot"].apply(team_from_slot)
    chat_df = chat_df.rename(columns={"key": "text"})

    # Join in match outcome
    chat_df = chat_df.merge(match_df, on="match_id", how="left")
    missing_outcome = chat_df["radiant_win"].isna().sum()
    if missing_outcome:
        print(f"WARNING: {missing_outcome} messages had no matching match_id in match.csv - dropping")
        chat_df = chat_df.dropna(subset=["radiant_win"])

    return chat_df[["match_id", "time", "team", "text", "radiant_win"]]


def summarize_match_stats(df):
    """Per-match total message count and whether both teams are present -
    matters because a match can clear a total-message threshold while
    still having zero messages from one team, which breaks per-team
    trajectory construction just as badly as having too few messages
    overall."""
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
            "in the filtered output. Default: 1 (only removes matches "
            "with zero usable messages)."
        ),
    )
    parser.add_argument(
        "--require-both-teams", action="store_true",
        help=(
            "Also require at least one message from EACH team for a "
            "match to be included in the filtered output."
        ),
    )
    args = parser.parse_args()

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_chat()
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
        print(f"  >= {t:2d} messages: {n:6d} matches")

    both_teams = per_match["teams_present"] == 2
    print(f"\nMatches with BOTH teams represented (>=1 msg each): {both_teams.sum()} / {total_matches}")

    print("\nCOMBINED (>= threshold messages AND both teams present):")
    for t in [1, 4, 6, 10, 15, 20]:
        n = ((per_match["total_messages"] >= t) & both_teams).sum()
        print(f"  >= {t:2d} messages + both teams: {n:6d} matches")

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