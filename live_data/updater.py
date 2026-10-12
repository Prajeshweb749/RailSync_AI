"""
live_data/updater.py
====================
Daily dataset updater.

Extracts COMPLETED station records from a live API response and appends
them to data/cleaned.csv (deduped on train_number + station_code + journey_date).

Usage (call once per day, e.g. via a scheduled task or /api/update-dataset):
    from live_data.updater import update_dataset_from_live
    rows_added = update_dataset_from_live("12301")
"""

import os
import pandas as pd
from datetime import datetime, date
from live_data.fetcher import fetch_live_status

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
CLEANED_CSV = os.path.join(DATA_DIR, "cleaned.csv")

# Columns that cleaned.csv uses
_REQUIRED_COLS = [
    "train_number", "station_code", "station_name",
    "arrival_delay", "departure_delay",
    "journey_date", "day_of_week",
    "scheduled_arrival", "scheduled_departure",
    "distance_km",
]


def update_dataset_from_live(train_number: str) -> int:
    """
    Fetches live data for train_number, extracts completed stations,
    and appends them to cleaned.csv.

    Returns:
        Number of new rows actually appended (0 if all already present).
    """
    live = fetch_live_status(train_number)
    today_str = date.today().isoformat()

    # ── Build rows from completed stations ────────────────────────
    rows = []
    for s in live["route"]:
        if s["status"] not in ("COMPLETED", "CURRENT"):
            continue
        if s["delay_arrival_min"] is None:
            continue  # no actual data yet — skip

        rows.append({
            "train_number":       live["train_number"],
            "station_code":       s["station_code"],
            "station_name":       s["station_name"],
            "arrival_delay":      s["delay_arrival_min"],
            "departure_delay":    s["delay_departure_min"] if s["delay_departure_min"] is not None else s["delay_arrival_min"],
            "journey_date":       today_str,
            "day_of_week":        datetime.today().weekday(),          # 0=Mon … 6=Sun
            "scheduled_arrival":  s["scheduled_arrival"] or "",
            "scheduled_departure":s["scheduled_departure"] or "",
            "distance_km":        s["distance_km"] or 0.0,
        })

    if not rows:
        return 0

    new_df = pd.DataFrame(rows)

    # ── Load existing cleaned.csv ─────────────────────────────────
    if os.path.exists(CLEANED_CSV):
        existing = pd.read_csv(CLEANED_CSV, low_memory=False)
    else:
        existing = pd.DataFrame(columns=_REQUIRED_COLS)

    # Ensure journey_date column is string for dedup
    existing["journey_date"] = existing["journey_date"].astype(str)

    # ── Deduplicate: drop rows already present for today ──────────
    key_cols = ["train_number", "station_code", "journey_date"]
    existing_keys = set(
        zip(
            existing["train_number"].astype(str),
            existing["station_code"].astype(str),
            existing["journey_date"].astype(str),
        )
    )
    new_df_filtered = new_df[
        ~new_df.apply(
            lambda r: (str(r["train_number"]), str(r["station_code"]), str(r["journey_date"])) in existing_keys,
            axis=1,
        )
    ]

    if new_df_filtered.empty:
        return 0

    # ── Append and save ───────────────────────────────────────────
    combined = pd.concat([existing, new_df_filtered], ignore_index=True)
    combined.to_csv(CLEANED_CSV, index=False)

    print(
        f"[updater] {live['train_number']} | {today_str} | "
        f"+{len(new_df_filtered)} rows appended → {len(combined)} total"
    )
    return len(new_df_filtered)
