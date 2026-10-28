

import argparse
import re
from pathlib import Path

import pandas as pd

INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"
PROXIMITY_WINDOW_SECONDS = 120

NON_LATIN_PATTERN = re.compile(
    r'[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\u0600-\u06FF\u0E00-\u0E7F]'
)


def is_latin_script(text):
    text = str(text)
    if not text.strip():
        return True
    non_latin_chars = len(NON_LATIN_PATTERN.findall(text))
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars == 0:
        return True
    return (non_latin_chars / alpha_chars) < 0.3


def window_is_readable(window_texts):
    """An excerpt window is usable for manual coding only if the large
    majority of its messages are Latin-script - this dataset still has
    ~6% non-English (mostly Cyrillic) content, which would otherwise be
    unreadable to an English-speaking coder."""
    if len(window_texts) == 0:
        return False
    latin_count = sum(1 for t in window_texts if is_latin_script(t))
    return (latin_count / len(window_texts)) >= 0.7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift-points", type=str, default=str(INTERIM_DIR / "rq2_shift_points.csv"))
    parser.add_argument("--chat", type=str, default=str(INTERIM_DIR / "chat_with_phases.csv"))
    parser.add_argument("--per-event-type", type=int, default=15,
                         help="How many shift points to sample per event type (capped by availability for rare types)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shifts = pd.read_csv(args.shift_points)
    chat = pd.read_csv(args.chat)

    matched = shifts.dropna(subset=["nearest_event"]).copy()
    print(f"Matched shift points available: {len(matched)}")
    print("By event type:")
    print(matched["nearest_event"].value_counts())

    # Index chat by (match_id, team) for fast lookup
    chat_indexed = {
        key: group for key, group in chat.groupby(["match_id", "team"])
    }

    def get_window(row):
        key = (row["match_id"], row["team"])
        team_chat = chat_indexed.get(key)
        if team_chat is None:
            return None
        window = team_chat[
            (team_chat["time"] >= row["shift_time"] - PROXIMITY_WINDOW_SECONDS)
            & (team_chat["time"] <= row["shift_time"] + PROXIMITY_WINDOW_SECONDS)
        ].sort_values("time")
        return window

    # Stratified sample, filtering out excerpts that aren't readable
    # (majority non-Latin-script) - draw candidates and keep going until
    # the per-type target is met or the candidate pool is exhausted.
    sampled_rows = []
    shortfalls = {}
    for event_type, group in matched.groupby("nearest_event"):
        candidates = group.sample(frac=1, random_state=args.seed)  # shuffled
        kept = 0
        for _, row in candidates.iterrows():
            if kept >= args.per_event_type:
                break
            window = get_window(row)
            if window is None or window.empty:
                continue
            if not window_is_readable(window["text"].tolist()):
                continue
            sampled_rows.append((row, window))
            kept += 1
        if kept < args.per_event_type:
            shortfalls[event_type] = kept

    print(f"\nSampled {len(sampled_rows)} shift points for excerpt extraction "
          f"(target {args.per_event_type} per event type, after filtering non-readable excerpts)")
    if shortfalls:
        print("Event types that fell short of the target (ran out of readable candidates):")
        for et, n in shortfalls.items():
            print(f"  {et}: only {n} readable excerpts found")

    excerpt_rows = []
    for excerpt_id, (row, window) in enumerate(sampled_rows):
        for _, msg in window.iterrows():
            excerpt_rows.append({
                "excerpt_id": excerpt_id,
                "match_id": row["match_id"],
                "team": row["team"],
                "event_type": row["nearest_event"],
                "shift_time": row["shift_time"],
                "message_time": msg["time"],
                "time_offset_seconds": msg["time"] - row["shift_time"],
                "text": msg["text"],
                "category": "",  # empty - fill in during manual reading
            })

    excerpt_df = pd.DataFrame(excerpt_rows)
    out_path = INTERIM_DIR / "rq2_qualitative_excerpts.csv"
    excerpt_df.to_csv(out_path, index=False)

    print(f"\nSaved {len(excerpt_df)} individual messages across {len(sampled_rows)} excerpts to:")
    print(f"  {out_path}")
    print(
        "\nEach excerpt_id groups the messages surrounding one shift point, in\n"
        "chronological order, with time_offset_seconds showing how far before\n"
        "(negative) or after (positive) the detected shift each message was sent.\n"
        "The 'category' column is empty and intended for manual coding while reading\n"
        "(e.g. blame, encouragement, toxicity, celebration, tactical) - fill it in\n"
        "directly in Excel/Sheets."
    )


if __name__ == "__main__":
    main()
