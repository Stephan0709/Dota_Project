
import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENDOTA_API_KEY")
BASE_URL = "https://api.opendota.com/api"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PENDING_PATH = RAW_DIR / "_pending_parses.json"
REQUEST_DELAY_SECONDS = 1.1


def _get(endpoint, params=None):
    params = params or {}
    if API_KEY:
        params["api_key"] = API_KEY
    resp = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


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


def check_job_status(job_id):
    return _get(f"/request/{job_id}")


def main():
    if not PENDING_PATH.exists():
        print(f"No pending file found at {PENDING_PATH}")
        print("Run data_collection.py with --source public --request-parse first.")
        return

    with open(PENDING_PATH, "r", encoding="utf-8") as f:
        pending = json.load(f)

    if not pending:
        print("Pending file is empty - nothing to check.")
        return

    print(f"Checking {len(pending)} pending matches...\n")

    now = time.time()
    still_pending = {}
    resolved_wait_times = []
    newly_saved = 0

    for match_id_str, info in pending.items():
        match_id = int(match_id_str)
        requested_at = info["requested_at"]
        job_id = info.get("job_id")
        age_minutes = (now - requested_at) / 60

        try:
            detail = _get(f"/matches/{match_id}")
        except requests.RequestException as e:
            print(f"{match_id} -> FAILED checking ({e}), will retry next run")
            still_pending[match_id_str] = info
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        if has_usable_chat(detail):
            path = save_match(detail)
            n_messages = len([m for m in detail["chat"] if m.get("type") == "chat"])
            wait_minutes = age_minutes
            resolved_wait_times.append(wait_minutes)
            print(
                f"{match_id} -> READY after {wait_minutes:.1f} min "
                f"({n_messages} chat messages) -> saved to {path.name}"
            )
            newly_saved += 1
        else:
            
            job_note = ""
            job_status = None
            if job_id:
                try:
                    job_status = check_job_status(job_id)
                except requests.RequestException as e:
                    job_note = f" | job status check failed: {e}"

            if job_id and job_status is None:
    
                print(
                    f"{match_id} -> DEAD ({age_minutes:.1f} min so far) - "
                    f"job no longer exists, parse likely failed permanently. Dropping."
                )
            else:
                job_note = f" | job status: {job_status}" if job_status else job_note
                print(f"{match_id} -> still pending ({age_minutes:.1f} min so far){job_note}")
                still_pending[match_id_str] = info

        time.sleep(REQUEST_DELAY_SECONDS)

    with open(PENDING_PATH, "w", encoding="utf-8") as f:
        json.dump(still_pending, f, indent=2)

    print("\n" + "=" * 50)
    print(f"Newly saved this run:  {newly_saved}")
    print(f"Still pending:         {len(still_pending)}")
    if resolved_wait_times:
        avg_wait = sum(resolved_wait_times) / len(resolved_wait_times)
        print(f"\nAvg real wait time for matches that succeeded: {avg_wait:.1f} min")
        print(f"Min/Max: {min(resolved_wait_times):.1f} / {max(resolved_wait_times):.1f} min")
    if still_pending:
        print(f"\nRun this script again later to keep checking on the remaining {len(still_pending)}.")


if __name__ == "__main__":
    main()