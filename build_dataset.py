"""
build_dataset.py
================
Constructs real spatio-temporal network state tensors from PostgreSQL.

Temporal Structure:
  - 48 discrete 30-minute bins per 24-hour day.
  - T = 6 consecutive bins (3-hour history) -> predict bin t + 6.
  - Target mask captures all stations with active scheduled observations in bin t + 6.

Chronological Splits:
  - Train:      Aug 28, 2025 -> Jun 30, 2026
  - Validation: Jul 01, 2026 -> Jul 31, 2026
  - Test:       Aug 01, 2026 -> Aug 27, 2026

Output: historical_dataset_split.npz
"""

import json
import os
import sys
from datetime import date
import numpy as np
import psycopg2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "station_index.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "historical_dataset_split.npz")

if not os.path.exists(INDEX_PATH):
    INDEX_PATH = os.path.join(os.path.dirname(BASE_DIR), "station_index.json")

with open(INDEX_PATH, "r") as f:
    station_to_idx = json.load(f)

num_nodes = len(station_to_idx)
print(f"[*] Loaded index for {num_nodes} stations.", flush=True)

# PostgreSQL Credentials
DB_CONFIG = {
    "dbname": os.getenv("PGDATABASE", "SIH_Railway_DB"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "postgres"),
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", 5432))
}

print("[*] Connecting to PostgreSQL...", flush=True)
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# 1. Fetch available historical run dates
cur.execute("""
    SELECT DISTINCT run_date 
    FROM historical_station_observations 
    ORDER BY run_date ASC;
""")
dates = [r[0] for r in cur.fetchall()]
print(f"[+] Found {len(dates)} unique historical run dates ({dates[0]} to {dates[-1]}).", flush=True)

# 2. Extract scheduled 30-min time bins
# Bins: 0 (00:00-00:29) to 47 (23:30-23:59)
query = """
    SELECT 
        h.station_code,
        COALESCE(h.delay_minutes, 0) AS delay,
        EXTRACT(HOUR FROM COALESCE(
            NULLIF(s.arrival_time, '')::time, 
            NULLIF(s.departure_time, '')::time, 
            '00:00:00'::time
        )) * 2 + 
        FLOOR(EXTRACT(MINUTE FROM COALESCE(
            NULLIF(s.arrival_time, '')::time, 
            NULLIF(s.departure_time, '')::time, 
            '00:00:00'::time
        )) / 30) AS time_bin
    FROM historical_station_observations h
    JOIN train_stops s 
      ON h.train_number = s.train_number 
     AND h.station_code = s.station_code
    WHERE h.run_date = %s;
"""

splits = {
    "train": {"X": [], "Y": [], "Mask": []},
    "val":   {"X": [], "Y": [], "Mask": []},
    "test":  {"X": [], "Y": [], "Mask": []}
}

val_start = date(2026, 7, 1)
test_start = date(2026, 8, 1)

# Pre-compute diurnal cyclical angles
diurnal_sin = np.array([np.sin(2.0 * np.pi * t / 48.0) for t in range(48)], dtype=np.float32)
diurnal_cos = np.array([np.cos(2.0 * np.pi * t / 48.0) for t in range(48)], dtype=np.float32)

print("[*] Compiling daily temporal network states...", flush=True)

for d_idx, r_date in enumerate(dates):
    cur.execute(query, (r_date,))
    rows = cur.fetchall()
    if not rows:
        continue

    # Chronological partition routing
    if r_date < val_start:
        split_key = "train"
    elif r_date < test_start:
        split_key = "val"
    else:
        split_key = "test"

    # [48 bins, 4626 stations, 4 features]
    day_state = np.zeros((48, num_nodes, 4), dtype=np.float32)
    traffic_counts = np.zeros((48, num_nodes), dtype=np.float32)

    # Set diurnal features across all stations for every time bin
    for t in range(48):
        day_state[t, :, 2] = diurnal_sin[t]
        day_state[t, :, 3] = diurnal_cos[t]

    # Populate real observations into their scheduled time bin
    for stn_code, delay, t_bin in rows:
        if stn_code in station_to_idx and t_bin is not None:
            s_idx = station_to_idx[stn_code]
            b_idx = int(t_bin) % 48
            day_state[b_idx, s_idx, 0] += max(0.0, float(delay))
            traffic_counts[b_idx, s_idx] += 1.0

    # Average observed delays per station-bin and normalize
    active_mask = traffic_counts > 0
    day_state[active_mask, 0] /= (traffic_counts[active_mask] * 120.0)  # Normalized delay (120 min scale)
    day_state[:, :, 1] = np.log1p(traffic_counts)                      # Feature 1: Traffic density

    # 3-hour sliding lookback: 6 consecutive 30-min bins -> predict bin t + 6
    for t in range(0, 48 - 6):
        seq_x = day_state[t : t + 6]  # [6, Nodes, 4]

        # Target: delay at step t + 6
        target_counts = traffic_counts[t + 6]
        target_mask = (target_counts > 0).astype(np.float32)

        # Skip window if no trains were scheduled in the target bin
        if target_mask.sum() == 0:
            continue

        seq_y = np.zeros(num_nodes, dtype=np.float32)
        target_indices = np.where(target_counts > 0)[0]
        # Target in actual unnormalized minutes
        seq_y[target_indices] = day_state[t + 6, target_indices, 0] * 120.0

        splits[split_key]["X"].append(seq_x)
        splits[split_key]["Y"].append(seq_y)
        splits[split_key]["Mask"].append(target_mask)

    if (d_idx + 1) % 20 == 0 or d_idx == len(dates) - 1:
        print(f"    Processed {d_idx + 1}/{len(dates)} dates...", flush=True)

cur.close()
conn.close()

print("\n[*] Dataset Split Verification:")
for k in ["train", "val", "test"]:
    print(f"    - {k.upper():5s}: {len(splits[k]['X'])} continuous 3-hour temporal sequences")

np.savez_compressed(
    OUTPUT_PATH,
    X_train=np.array(splits["train"]["X"], dtype=np.float32),
    Y_train=np.array(splits["train"]["Y"], dtype=np.float32),
    Mask_train=np.array(splits["train"]["Mask"], dtype=np.float32),
    X_val=np.array(splits["val"]["X"], dtype=np.float32),
    Y_val=np.array(splits["val"]["Y"], dtype=np.float32),
    Mask_val=np.array(splits["val"]["Mask"], dtype=np.float32),
    X_test=np.array(splits["test"]["X"], dtype=np.float32),
    Y_test=np.array(splits["test"]["Y"], dtype=np.float32),
    Mask_test=np.array(splits["test"]["Mask"], dtype=np.float32),
)
print(f"[+] Serialized dataset to {OUTPUT_PATH} successfully!")
