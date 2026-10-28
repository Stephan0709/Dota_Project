

import argparse
import random
import re
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RAW_EXTERNAL_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "external" / "dota2_chat_messages.csv"
)
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "interim" / "validation_sample.csv"
)

GAMING_SLANG_TERMS = [
    "ez", "gg", "wp", "feed", "feeding", "throw", "throwing", "ff",
    "glhf", "rekt", "noob", "int", "inting", "reported", "bg", "trash",
]

CHUNK_SIZE = 200_000
RESERVOIR_CAP_PER_STRATUM = 5000


def guess_column(columns, candidates):
    """Case-insensitive match against a list of likely column names."""
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def peek(path):
    print(f"Peeking at {path} ...\n")
    df = pd.read_csv(path, nrows=10, encoding="utf-8", encoding_errors="replace")
    print("Detected columns:")
    print(list(df.columns))
    print("\nFirst 10 rows:")
    print(df.to_string())

    match_id_col = guess_column(df.columns, ["match_id", "match", "matchid", "game_id"])
    time_col = guess_column(df.columns, ["time", "timestamp", "game_time"])
    slot_col = guess_column(df.columns, ["slot", "player_slot", "playerslot"])
    text_col = guess_column(df.columns, ["text", "message", "key", "chat", "msg"])

    print("\nGuessed column mapping:")
    print(f"  match_id -> {match_id_col}")
    print(f"  time     -> {time_col}")
    print(f"  slot     -> {slot_col}")
    print(f"  text     -> {text_col}")
    


NON_LATIN_PATTERN = re.compile(
    r'[\u0400-\u04FF'   # Cyrillic
    r'\u4E00-\u9FFF'    # CJK
    r'\u3040-\u30FF'    # Hiragana/Katakana
    r'\uAC00-\uD7AF'    # Hangul
    r'\u0600-\u06FF'    # Arabic
    r'\u0E00-\u0E7F'    # Thai
    r']'
)


def is_latin_script(text):
    text = str(text)
    if not text.strip():
        return False
    non_latin_chars = len(NON_LATIN_PATTERN.findall(text))
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars == 0:
        return True  # no letters at all (e.g. "100%", "xd") - let it through
    return (non_latin_chars / alpha_chars) < 0.3


def classify_strata(compound_score, text):
    strata = []
    if compound_score < -0.5:
        strata.append("strongly_negative")
    if compound_score > 0.5:
        strata.append("strongly_positive")
    if -0.2 <= compound_score <= 0.2:
        strata.append("near_neutral")
    text_lower = str(text).lower()
    if any(term in text_lower for term in GAMING_SLANG_TERMS):
        strata.append("gaming_slang")
    return strata


def reservoir_add(reservoir, counts_seen, stratum, row):
    counts_seen[stratum] += 1
    bucket = reservoir[stratum]
    if len(bucket) < RESERVOIR_CAP_PER_STRATUM:
        bucket.append(row)
    else:
        j = random.randint(0, counts_seen[stratum] - 1)
        if j < RESERVOIR_CAP_PER_STRATUM:
            bucket[j] = row


def build_sample(path, match_id_col, time_col, slot_col, text_col, n_total):
    analyzer = SentimentIntensityAnalyzer()
    strata_names = ["strongly_negative", "strongly_positive", "near_neutral", "gaming_slang"]
    reservoir = {s: [] for s in strata_names}
    counts_seen = {s: 0 for s in strata_names}

    usecols = [c for c in [match_id_col, time_col, slot_col, text_col] if c]
    print(f"Streaming {path} in chunks of {CHUNK_SIZE} rows...\n")

    total_rows = 0
    non_latin_excluded = 0
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        encoding="utf-8",
        encoding_errors="replace",
        dtype={text_col: str},
    ):
        chunk = chunk.dropna(subset=[text_col])
        for _, row in chunk.iterrows():
            text = row[text_col]
            if not text or not str(text).strip():
                continue
            if not is_latin_script(text):
                non_latin_excluded += 1
                continue
            compound = analyzer.polarity_scores(str(text))["compound"]
            strata = classify_strata(compound, text)
            for stratum in strata:
                record = {
                    "match_id": row.get(match_id_col),
                    "time": row.get(time_col),
                    "slot": row.get(slot_col),
                    "text": text,
                    "vader_compound": compound,
                    "stratum": stratum,
                }
                reservoir_add(reservoir, counts_seen, stratum, record)
        total_rows += len(chunk)
        print(f"  processed {total_rows:,} rows so far...", end="\r")

    print(f"\n\nDone scanning. Total rows processed: {total_rows:,}\n")
    print(f"Excluded as non-Latin-script (likely non-English): {non_latin_excluded:,}")
    print("Candidates seen per stratum (before downsampling, English-script only):")
    for s in strata_names:
        print(f"  {s}: {counts_seen[s]:,} seen, {len(reservoir[s])} held in reservoir")

    per_stratum_target = n_total // len(strata_names)
    final_rows = []
    print(f"\nTarget per stratum: {per_stratum_target}")
    for s in strata_names:
        available = reservoir[s]
        take = min(per_stratum_target, len(available))
        if take < per_stratum_target:
            print(
                f"  WARNING: stratum '{s}' only has {len(available)} candidates, "
                f"fewer than target {per_stratum_target}. Taking all available."
            )
        sampled = random.sample(available, take) if available else []
        final_rows.extend(sampled)

    return pd.DataFrame(final_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(RAW_EXTERNAL_PATH))
    parser.add_argument("--peek", action="store_true", help="Just inspect columns, don't process")
    parser.add_argument("--n-total", type=int, default=200)
    parser.add_argument("--match-id-col", type=str, default=None)
    parser.add_argument("--time-col", type=str, default=None)
    parser.add_argument("--slot-col", type=str, default=None)
    parser.add_argument("--text-col", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}")
        print("Download dota2_chat_messages.csv from Kaggle and place it there first.")
        return

    if args.peek:
        peek(path)
        return

    header = pd.read_csv(path, nrows=0, encoding="utf-8", encoding_errors="replace")
    match_id_col = args.match_id_col or guess_column(header.columns, ["match_id", "match", "matchid", "game_id"])
    time_col = args.time_col or guess_column(header.columns, ["time", "timestamp", "game_time"])
    slot_col = args.slot_col or guess_column(header.columns, ["slot", "player_slot", "playerslot"])
    text_col = args.text_col or guess_column(header.columns, ["text", "message", "key", "chat", "msg"])

    if not text_col:
        print("Could not determine the text column. Run with --peek first and pass --text-col explicitly.")
        return

    print(f"Using columns -> match_id: {match_id_col}, time: {time_col}, slot: {slot_col}, text: {text_col}\n")

    df = build_sample(path, match_id_col, time_col, slot_col, text_col, args.n_total)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df["human_label"] = ""  
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 50)
    print(f"Final sample size: {len(df)}")
    print(df["stratum"].value_counts())
    print(f"\nSaved to: {OUTPUT_PATH}")
    print("Open this file and fill in 'human_label' (negative/neutral/positive) for each row.")


if __name__ == "__main__":
    main()