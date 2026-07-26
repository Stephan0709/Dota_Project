
import argparse
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENDOTA_API_KEY")
BASE_URL = "https://api.opendota.com/api"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Be polite to the free tier: 60 calls/minute is the documented limit.
REQUEST_DELAY_SECONDS = 1.1


def _get(endpoint, params=None):
    params = params or {}
    if API_KEY:
        params["api_key"] = API_KEY
    resp = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_public_match_ids(n_matches, min_age_seconds=600):
    
    match_ids = []
    less_than_match_id = None
    now = time.time()

    while len(match_ids) < n_matches:
        params = {}
        if less_than_match_id:
            params["less_than_match_id"] = less_than_match_id

        batch = _get("/publicMatches", params=params)
        if not batch:
            break

        for match in batch:
            end_time = match.get("start_time", 0) + match.get("duration", 0)
            age_seconds = now - end_time
            if age_seconds >= min_age_seconds:
                match_ids.append(match["match_id"])

        less_than_match_id = min(m["match_id"] for m in batch)
        time.sleep(REQUEST_DELAY_SECONDS)

    return match_ids[:n_matches]


def fetch_pro_match_ids(n_matches):
    
    match_ids = []
    less_than_match_id = None

    while len(match_ids) < n_matches:
        params = {}
        if less_than_match_id:
            params["less_than_match_id"] = less_than_match_id

        batch = _get("/proMatches", params=params)
        if not batch:
            break

        for match in batch:
            match_ids.append(match["match_id"])

        less_than_match_id = min(m["match_id"] for m in batch)
        time.sleep(REQUEST_DELAY_SECONDS)

    return match_ids[:n_matches]


def request_parse(match_id):
    
    params = {}
    if API_KEY:
        params["api_key"] = API_KEY
    resp = requests.post(
        f"{BASE_URL}/request/{match_id}", params=params, timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("job", {}).get("jobId") or data.get("jobId")


def check_job_status(job_id):
   
    return _get(f"/request/{job_id}")


def request_parse_batch(match_ids):
    
    queued = {}
    for match_id in match_ids:
        try:
            job_id = request_parse(match_id)
            queued[match_id] = job_id
        except requests.RequestException as e:
            print(f"  {match_id} -> parse request failed ({e})")
        time.sleep(REQUEST_DELAY_SECONDS)
    return queued


def poll_batch_for_chat(match_ids, max_wait_seconds=45, poll_interval=8):

    pending = set(match_ids)
    results = {}
    elapsed = 0

    while pending and elapsed < max_wait_seconds:
        for match_id in list(pending):
            try:
                detail = fetch_match_detail(match_id)
            except requests.RequestException as e:
                print(f"  {match_id} -> check failed ({e}), will retry via collect_pending.py later")
                time.sleep(REQUEST_DELAY_SECONDS)
                continue
            if has_usable_chat(detail):
                results[match_id] = detail
                pending.discard(match_id)
            time.sleep(REQUEST_DELAY_SECONDS)
        if pending:
            time.sleep(poll_interval)
            elapsed += poll_interval

    
    for match_id in pending:
        try:
            results[match_id] = fetch_match_detail(match_id)
        except requests.RequestException as e:
            print(f"  {match_id} -> final check failed ({e}), will retry via collect_pending.py later")
            results[match_id] = {}

    return results


def load_pending():
    path = RAW_DIR / "_pending_parses.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pending(pending):
    path = RAW_DIR / "_pending_parses.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pending, f, indent=2)


def fetch_match_detail(match_id):
    return _get(f"/matches/{match_id}")


def has_usable_chat(match_detail):
    chat = match_detail.get("chat")
    if not chat:
        return False
    
    real_messages = [m for m in chat if m.get("type") == "chat"]
    return len(real_messages) > 0


def save_match(match_detail):
    match_id = match_detail["match_id"]
    out_path = RAW_DIR / f"match_{match_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(match_detail, f)
    return out_path


def main(n_matches, source, request_parse_flag, min_age_seconds):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if source == "pro":
        print(f"Fetching {n_matches} recent pro match IDs (auto-parsed)...")
        match_ids = fetch_pro_match_ids(n_matches)
    else:
        print(
            f"Fetching {n_matches} public match IDs "
            f"(at least {min_age_seconds}s / {min_age_seconds/60:.1f}min old)..."
        )
        match_ids = fetch_public_match_ids(n_matches, min_age_seconds=min_age_seconds)
    print(f"Got {len(match_ids)} match IDs.\n")

    saved = 0
    skipped_no_chat = 0
    failed = 0

    already_parsed = {}
    needs_parsing = []
    for match_id in match_ids:
        try:
            detail = fetch_match_detail(match_id)
        except requests.RequestException as e:
            print(f"{match_id} -> FAILED fetching detail ({e})")
            failed += 1
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        if has_usable_chat(detail):
            already_parsed[match_id] = detail
        else:
            needs_parsing.append(match_id)
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"Already parsed: {len(already_parsed)} | Needs parsing: {len(needs_parsing)}\n")

    newly_parsed = {}
    if needs_parsing and source == "public" and request_parse_flag:
        print(f"Requesting parse for {len(needs_parsing)} matches (fired together)...")
        job_ids = request_parse_batch(needs_parsing)

        successfully_queued = [m for m in needs_parsing if job_ids.get(m) is not None]
        n_request_failed = len(needs_parsing) - len(successfully_queued)
        if n_request_failed:
            print(f"{n_request_failed} parse requests failed outright (see errors above) - not tracked as pending.")

        pending = load_pending()
        now = time.time()
        for match_id in successfully_queued:
            pending[str(match_id)] = {
                "requested_at": now,
                "job_id": job_ids.get(match_id),
            }
        save_pending(pending)
        print(f"Saved {len(successfully_queued)} requests to pending file (safe even if the next step fails).\n")

        print("Quick check for any fast successes (30s)...")
        newly_parsed = poll_batch_for_chat(successfully_queued, max_wait_seconds=30)
        n_succeeded = sum(1 for d in newly_parsed.values() if has_usable_chat(d))
        print(f"Quick check: {n_succeeded}/{len(successfully_queued)} already done\n")

        
        pending = load_pending()
        for match_id in successfully_queued:
            if has_usable_chat(newly_parsed.get(match_id, {})):
                pending.pop(str(match_id), None)
        save_pending(pending)

        n_still_pending = len(successfully_queued) - n_succeeded
        if n_still_pending:
            print(
                f"{n_still_pending} matches still parsing on OpenDota's "
                f"side (queue can take much longer than a few minutes - "
                f"their status page shows requests going through priority "
                f"queues, not instant processing).\n"
                f"Already saved to data/raw/_pending_parses.json.\n"
                f"Run 'python src/collect_pending.py' again later "
                f"(e.g. tomorrow) to check on them.\n"
            )

    # Save everything that ended up with usable chat
    all_results = {**already_parsed, **newly_parsed}
    for match_id in match_ids:
        detail = all_results.get(match_id)
        if detail is None or not has_usable_chat(detail):
            print(f"{match_id} -> no usable chat, skipping")
            skipped_no_chat += 1
            continue

        path = save_match(detail)
        n_chat_messages = len([m for m in detail["chat"] if m.get("type") == "chat"])
        print(f"{match_id} -> saved ({n_chat_messages} chat messages) -> {path.name}")
        saved += 1

    print("\n" + "=" * 50)
    print(f"Requested:        {len(match_ids)}")
    print(f"Saved (usable):   {saved}")
    print(f"Skipped (no chat):{skipped_no_chat}")
    print(f"Failed (errors):  {failed}")
    if len(match_ids) > 0:
        yield_rate = saved / len(match_ids) * 100
        print(f"Yield rate:       {yield_rate:.1f}%")

    print(f"\nSaved matches are in: {RAW_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-matches",
        type=int,
        default=20,
        help="Number of matches to attempt to pull (default: 20)",
    )
    parser.add_argument(
        "--source",
        choices=["pro", "public"],
        default="pro",
        help=(
            "Where to pull match IDs from. 'pro' matches are auto-parsed "
            "by OpenDota and reliably have chat data. 'public' matches "
            "are almost never parsed yet unless --request-parse is set "
            "(default: pro)"
        ),
    )
    parser.add_argument(
        "--request-parse",
        action="store_true",
        help=(
            "Only relevant with --source public. Actively requests "
            "OpenDota to parse each unparsed match and waits for it "
            "(slower, ~10-120s per match, but this is closer to what "
            "Task 4's full 500-match collection will require)."
        ),
    )
    parser.add_argument(
        "--min-age-minutes",
        type=float,
        default=10.0,
        help=(
            "Only relevant with --source public. Minimum minutes since "
            "a match ended before we attempt to parse it - Valve's "
            "replay files typically aren't downloadable until several "
            "minutes after a match finishes. (default: 10)"
        ),
    )
    args = parser.parse_args()
    main(
        args.n_matches,
        args.source,
        args.request_parse,
        min_age_seconds=int(args.min_age_minutes * 60),
    )