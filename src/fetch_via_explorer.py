

import argparse
import json
import os
import time
import urllib.parse
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENDOTA_API_KEY")
BASE_URL = "https://api.opendota.com/api"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
REQUEST_DELAY_SECONDS = 1.1


def explorer_query(sql):
    params = {"sql": sql}
    if API_KEY:
        params["api_key"] = API_KEY
    resp = requests.get(f"{BASE_URL}/explorer", params=params, timeout=30)
    if not resp.ok:
        print(f"HTTP {resp.status_code} error. Response body:")
        print(resp.text)
    resp.raise_for_status()
    return resp.json()


def get_matches_columns():
    sql = (
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'matches'"
    )
    result = explorer_query(sql)
    return [row["column_name"] for row in result.get("rows", [])]


def diagnose_counts(available_columns, days_back_start, days_back_end):
    now = time.time()
    window_start = int(now - (days_back_start * 86400))
    window_end = int(now - (days_back_end * 86400))

    has_leagueid = "leagueid" in available_columns

    
    where_with_filter = (
        f"chat IS NOT NULL AND start_time BETWEEN {window_start} AND {window_end}"
    )
    if has_leagueid:
        where_with_filter += " AND (leagueid IS NULL OR leagueid = 0)"
    sql_with = f"SELECT COUNT(*) FROM matches WHERE {where_with_filter}"

    
    sql_without = (
        f"SELECT COUNT(*) FROM matches "
        f"WHERE chat IS NOT NULL AND start_time BETWEEN {window_start} AND {window_end}"
    )

    print(f"Diagnostic: counting matches in window ({days_back_start}-{days_back_end} days ago)...")
    result_with = explorer_query(sql_with)
    count_with = result_with["rows"][0]["count"]
    print(f"  With pro-match filter (leagueid null/0): {count_with}")

    result_without = explorer_query(sql_without)
    count_without = result_without["rows"][0]["count"]
    print(f"  Without pro-match filter (any leagueid):  {count_without}\n")

    return count_with, count_without


def fetch_parsed_public_matches(n_matches, available_columns, days_back_start=14, days_back_end=7):
    now = time.time()
    window_start = now - (days_back_start * 86400)
    window_end = now - (days_back_end * 86400)

    desired = ["match_id", "radiant_win", "duration", "chat"]
    select_cols = [c for c in desired if c in available_columns]
    missing = [c for c in desired if c not in available_columns]
    if missing:
        print(f"Note: these expected columns aren't on the table, skipping them: {missing}")
    if "match_id" not in select_cols or "chat" not in select_cols:
        raise RuntimeError("match_id or chat column missing - cannot proceed safely")

    where_clauses = [
        "chat IS NOT NULL",
        f"start_time BETWEEN {int(window_start)} AND {int(window_end)}",
    ]
    if "leagueid" in available_columns:
        where_clauses.append("(leagueid IS NULL OR leagueid = 0)")
    else:
        print("Note: no leagueid column to filter out pro matches - results may include them")

    sql = (
        f"SELECT {', '.join(select_cols)} "
        f"FROM matches "
        f"WHERE {' AND '.join(where_clauses)} "
        f"LIMIT {n_matches}"
    )
    print(f"\nRunning query (window: {days_back_start}-{days_back_end} days ago):\n{sql}\n")
    return explorer_query(sql), select_cols


def has_usable_chat(chat_field):
    if not chat_field:
        return False
    real_messages = [m for m in chat_field if m.get("type") == "chat"]
    return len(real_messages) > 0


def save_match(row):
    match_id = row["match_id"]
    match_detail = {
        "match_id": match_id,
        "radiant_win": row.get("radiant_win"),
        "duration": row.get("duration"),
        "chat": row["chat"],
        "objectives": row.get("objectives"),
    }
    out_path = RAW_DIR / f"match_{match_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(match_detail, f)
    return out_path


def main(n_matches, days_back_start, days_back_end):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Checking matches table schema first...")
    columns = get_matches_columns()
    print(f"matches table has {len(columns)} columns:")
    print(columns)
    if "chat" not in columns:
        print("\nCannot proceed - no 'chat' column found. See column list above.")
        return
    print("\nConfirmed: 'chat' column exists.\n")

    count_with, count_without = diagnose_counts(columns, days_back_start, days_back_end)

    if count_without == 0:
        print(
            "Zero matches with chat at all in this window, pro or public. "
            "The window itself is likely the issue - try widening it "
            "(--days-back-start / --days-back-end) rather than proceeding."
        )
        return
    if count_with == 0 and count_without > 0:
        print(
            f"Found {count_without} matches with chat, but ALL of them are "
            f"pro/league matches (leagueid filter zeroed them out). This "
            f"suggests OpenDota's long-term parsed-chat retention in this "
            f"window may skew heavily toward pro matches specifically.\n"
        )
        return

    print(f"Querying for {n_matches} already-parsed public matches...")
    result, select_cols = fetch_parsed_public_matches(
        n_matches, columns, days_back_start, days_back_end
    )
    rows = result.get("rows", [])
    print(f"Query returned {len(rows)} matches.\n")

    saved = 0
    for row in rows:
        if not has_usable_chat(row.get("chat")):
            print(f"{row['match_id']} -> chat present but no real messages, skipping")
            continue
        path = save_match(row)
        n_messages = len([m for m in row["chat"] if m.get("type") == "chat"])
        print(f"{row['match_id']} -> saved ({n_messages} chat messages) -> {path.name}")
        saved += 1

    print("\n" + "=" * 50)
    print(f"Rows returned: {len(rows)}")
    print(f"Saved (usable chat): {saved}")
    print(f"\nSaved matches are in: {RAW_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-matches", type=int, default=30)
    parser.add_argument(
        "--days-back-start", type=int, default=14,
        help="Older edge of the search window, in days ago (default: 14)"
    )
    parser.add_argument(
        "--days-back-end", type=int, default=7,
        help="Newer edge of the search window, in days ago (default: 7)"
    )
    args = parser.parse_args()
    main(args.n_matches, args.days_back_start, args.days_back_end)