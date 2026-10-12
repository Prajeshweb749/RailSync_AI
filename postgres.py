"""
unrestricted_network_crawler.py
===============================
Unrestricted Pan-India Network Crawler (Zero-Hang / Force-Complete Mode):
- Scans all stations in PostgreSQL.
- Ingests all trains on station boards across all schedules.
- Dynamically discovers and adds ANY new station found on any route.
- Marks each station COMPLETED even if individual errors occur (never stuck in PENDING).
- Deduplicates stations, trains, and stop sequences.
"""

import os
import time
from typing import Set, Dict, Any, Optional, List, Tuple

import psycopg2
from psycopg2.extras import execute_values
import requests

# ============================================================
# CONFIGURATION & CREDENTIALS
# ============================================================

# Set your new API key here or via the RAILRADAR_API_KEY environment variable
API_KEY = os.getenv("RAILRADAR_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError("CRITICAL: RAILRADAR_API_KEY is not set.")

BASE_URL = "https://api.railradar.in/v1"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

DB_CONFIG = {
    "dbname": os.getenv("PGDATABASE", "SIH_Railway_DB"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "postgres"),
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", 5432))
}

REQUEST_TIMEOUT = 35
BASE_REQUEST_DELAY = 3.0
MAX_RETRIES = 3
BACKOFF_DELAYS = [5.0, 15.0, 30.0]
MAX_CONSECUTIVE_STATION_FAILURES = 3

# ============================================================
# API CLIENT
# ============================================================

def api_get(url: str, params: Optional[dict] = None) -> Optional[dict]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                body = resp.json()
                if body.get("success"):
                    return body.get("data")
                return None
            elif resp.status_code == 429:
                wait_time = BACKOFF_DELAYS[attempt - 1]
                print(f"    [!] HTTP 429 (Rate Limited). Adaptive backoff: {wait_time}s ({attempt}/{MAX_RETRIES})...")
                time.sleep(wait_time)
            elif resp.status_code in (401, 403):
                print(f"    [!] HTTP {resp.status_code} (Authentication/Access Denied). Check API key.")
                return None
            else:
                return None
        except Exception as exc:
            wait_time = BACKOFF_DELAYS[attempt - 1]
            print(f"    [!] Network error: {exc}. Retrying in {wait_time}s ({attempt}/{MAX_RETRIES})...")
            time.sleep(wait_time)
    return None

def fetch_station_board(station_code: str) -> Optional[List[dict]]:
    url = f"{BASE_URL}/stations/{station_code}/trains"
    data = api_get(url, params={"includeIntermediate": "false"})
    if data is None:
        return None
    if isinstance(data, dict):
        return data.get("trains", [])
    elif isinstance(data, list):
        return data
    return []

def fetch_train_timetable(train_number: str) -> Optional[dict]:
    url = f"{BASE_URL}/trains/{train_number}"
    return api_get(url, params={"haltsOnly": "true"})

# ============================================================
# STATE TRACKING & QUEUE
# ============================================================

def setup_state_tracking():
    ddl = """
    CREATE TABLE IF NOT EXISTS station_crawl_status (
        station_code VARCHAR(10) PRIMARY KEY,
        status VARCHAR(20) DEFAULT 'PENDING',
        trains_found INT DEFAULT 0,
        all_schedule_trains INT DEFAULT 0,
        last_scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            cur.execute("""
                ALTER TABLE station_crawl_status 
                ADD COLUMN IF NOT EXISTS all_schedule_trains INT DEFAULT 0;
            """)
            cur.execute("""
                INSERT INTO station_crawl_status (station_code, status)
                SELECT station_code, 'PENDING'
                FROM stations
                ON CONFLICT (station_code) DO NOTHING;
            """)
        conn.commit()

def get_pending_stations() -> List[Tuple[str, str]]:
    query = """
    SELECT s.station_code, s.station_name
    FROM stations s
    JOIN station_crawl_status sc ON s.station_code = sc.station_code
    WHERE sc.status IN ('PENDING', 'IN_PROGRESS')
    ORDER BY s.station_code;
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return [(str(r[0]).strip().upper(), str(r[1]).strip()) for r in cur.fetchall()]

def set_station_status(code: str, status: str, total_trains: int = 0, all_trains: int = 0):
    query = """
    UPDATE station_crawl_status
    SET status = %s,
        trains_found = %s,
        all_schedule_trains = %s,
        last_scanned_at = CURRENT_TIMESTAMP
    WHERE station_code = %s;
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (status, total_trains, all_trains, code.upper()))
        conn.commit()

def enqueue_new_station(code: str):
    query = """
    INSERT INTO station_crawl_status (station_code, status)
    VALUES (%s, 'PENDING')
    ON CONFLICT (station_code) DO NOTHING;
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (code.strip().upper(),))
        conn.commit()

def get_existing_train_numbers() -> Set[str]:
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT train_number FROM trains;")
            return {str(r[0]).strip() for r in cur.fetchall()}

def get_existing_station_codes() -> Set[str]:
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT station_code FROM stations;")
            return {str(r[0]).strip().upper() for r in cur.fetchall()}

# ============================================================
# PERSISTENCE (PAN-INDIA UNRESTRICTED)
# ============================================================

def persist_train_and_route(
    train_meta: dict,
    timetable_data: dict,
    known_stations: Set[str]
) -> Tuple[List[str], int]:
    train = timetable_data.get("train", {}) or train_meta
    route = timetable_data.get("route", [])
    t_num = str(train.get("number") or train_meta.get("number", "")).strip()

    if not t_num or not route:
        return [], 0

    actual_origin_dep = route[0].get("departure") or route[0].get("arrival")
    actual_dest_arr = route[-1].get("arrival") or route[-1].get("departure")

    station_rows = []
    stop_rows = []
    newly_found_stations = []
    seen_in_route = set()

    for s in route:
        stn = s.get("station", {})
        code = str(stn.get("code") or s.get("stationCode", "")).strip().upper()
        name = str(stn.get("name") or s.get("stationName", "")).strip() or code
        state = stn.get("state")

        # Discovered a brand new station anywhere in India
        if code not in known_stations:
            known_stations.add(code)
            newly_found_stations.append(code)
            enqueue_new_station(code)

        if code not in seen_in_route:
            is_junc = "JN" in name.upper() or "JUNCTION" in name.upper()
            station_rows.append((
                code,
                name,
                state,
                s.get("lat"),
                s.get("lng"),
                is_junc
            ))
            seen_in_route.add(code)

        stop_rows.append((
            t_num,
            code,
            name,
            s.get("sequence"),
            s.get("arrival"),
            s.get("departure"),
            s.get("arrivalDay", 1),
            s.get("departureDay", 1),
            s.get("distance", 0),
            str(s.get("platform") or ""),
            s.get("isHalt", True)
        ))

    if not stop_rows:
        return [], 0

    train_record = (
        t_num,
        train.get("name"),
        train.get("type"),
        (train.get("source") or {}).get("code"),
        (train.get("destination") or {}).get("code"),
        actual_origin_dep,
        actual_dest_arr,
        train.get("duration"),
        train.get("distance"),
        (train.get("destination") or {}).get("name"),
        train.get("avgSpeed"),
        (train.get("source") or {}).get("name"),
        train.get("category")
    )

    sql_train = """
    INSERT INTO trains (
        train_number, train_name, train_type,
        from_station_code, to_station_code,
        departure_time, arrival_time, duration_minutes,
        distance_km, to_station_name, avg_speed,
        from_station_name, category
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (train_number) DO UPDATE SET
        train_name = EXCLUDED.train_name,
        train_type = EXCLUDED.train_type,
        from_station_code = EXCLUDED.from_station_code,
        to_station_code = EXCLUDED.to_station_code,
        departure_time = COALESCE(EXCLUDED.departure_time, trains.departure_time),
        arrival_time = COALESCE(EXCLUDED.arrival_time, trains.arrival_time),
        duration_minutes = COALESCE(EXCLUDED.duration_minutes, trains.duration_minutes),
        distance_km = COALESCE(EXCLUDED.distance_km, trains.distance_km),
        to_station_name = EXCLUDED.to_station_name,
        avg_speed = COALESCE(EXCLUDED.avg_speed, trains.avg_speed),
        from_station_name = EXCLUDED.from_station_name,
        category = EXCLUDED.category,
        updated_at = CURRENT_TIMESTAMP;
    """

    sql_stations = """
    INSERT INTO stations (station_code, station_name, state, latitude, longitude, is_junction)
    VALUES %s
    ON CONFLICT (station_code) DO UPDATE SET
        station_name = EXCLUDED.station_name,
        state = COALESCE(EXCLUDED.state, stations.state),
        latitude = COALESCE(EXCLUDED.latitude, stations.latitude),
        longitude = COALESCE(EXCLUDED.longitude, stations.longitude),
        is_junction = (stations.is_junction OR EXCLUDED.is_junction),
        updated_at = CURRENT_TIMESTAMP;
    """

    sql_stops = """
    INSERT INTO train_stops (
        train_number, station_code, station_name, stop_order,
        arrival_time, departure_time, arrival_day, departure_day,
        distance_from_source, platform, is_halt
    ) VALUES %s
    ON CONFLICT (train_number, stop_order) DO UPDATE SET
        station_code = EXCLUDED.station_code,
        station_name = EXCLUDED.station_name,
        arrival_time = EXCLUDED.arrival_time,
        departure_time = EXCLUDED.departure_time,
        arrival_day = EXCLUDED.arrival_day,
        departure_day = EXCLUDED.departure_day,
        distance_from_source = EXCLUDED.distance_from_source,
        platform = EXCLUDED.platform,
        is_halt = EXCLUDED.is_halt;
    """

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_train, train_record)
            if station_rows:
                execute_values(cur, sql_stations, station_rows)
            execute_values(cur, sql_stops, stop_rows)
        conn.commit()

    return newly_found_stations, len(stop_rows)

# ============================================================
# MAIN ORCHESTRATION PIPELINE
# ============================================================

def main():
    print("\n" + "=" * 75)
    print("RAILSYNC - UNRESTRICTED PAN-INDIA CRAWLER")
    print("Strategy: FORCE-COMPLETE (Errors never leave station as PENDING)")
    print("=" * 75)

    setup_state_tracking()
    known_stations = get_existing_station_codes()
    existing_trains = get_existing_train_numbers()
    processed_trains: Set[str] = set(existing_trains)

    while True:
        pending = get_pending_stations()
        if not pending:
            print("\n✓ Full network convergence reached. No pending stations remaining.")
            break

        total_pending = len(pending)
        print(f"\n[+] Active Queue: {total_pending} stations to scan.\n")

        consecutive_station_failures = 0

        for idx, (stn_code, stn_name) in enumerate(pending, start=1):
            print(f"[{idx}/{total_pending}] Scanning Station: {stn_code} ({stn_name})...")
            set_station_status(stn_code, "IN_PROGRESS")

            services = fetch_station_board(stn_code)
            
            # --- CRUCIAL CHANGE: On Error, mark COMPLETED and move on ---
            if services is None:
                print(f"    [!] Station {stn_code} board fetch failed/throttled. Force marking as COMPLETED to prevent hang.")
                set_station_status(stn_code, "COMPLETED", 0, 0)
                consecutive_station_failures += 1
                if consecutive_station_failures >= MAX_CONSECUTIVE_STATION_FAILURES:
                    print("\n" + "!" * 75)
                    print("[!] CRITICAL: 3 consecutive stations failed to respond.")
                    print("[!] API key is likely exhausted or invalid. Stopping crawler cleanly.")
                    print("!" * 75 + "\n")
                    return
                time.sleep(BASE_REQUEST_DELAY)
                continue

            consecutive_station_failures = 0

            if not services:
                print("    [-] Station board returned 0 services.")
                set_station_status(stn_code, "COMPLETED", 0, 0)
                time.sleep(BASE_REQUEST_DELAY)
                continue

            print(f"    ✓ Total board: {len(services)} services detected.")

            newly_ingested = 0
            for s in services:
                train_brief = s.get("train", {})
                t_num = str(train_brief.get("number", "")).strip()
                t_name = train_brief.get("name", "Unknown Train")

                if not t_num or t_num in processed_trains:
                    continue

                processed_trains.add(t_num)
                print(f"      → [{t_num}] {t_name} (Fetching full route...)")

                timetable = fetch_train_timetable(t_num)
                if not timetable:
                    print(f"         [-] Timetable for {t_num} unavailable/failed. Skipping train.")
                    time.sleep(BASE_REQUEST_DELAY)
                    continue

                try:
                    new_stns, stops_synced = persist_train_and_route(
                        train_brief, timetable, known_stations
                    )

                    if stops_synced > 0:
                        newly_ingested += 1
                        existing_trains.add(t_num)
                        print(f"         ✓ Synced {stops_synced} halts (Discovered {len(new_stns)} new stations)")
                except Exception as exc:
                    print(f"         [!] Database write error on train {t_num}: {exc}")

                time.sleep(BASE_REQUEST_DELAY)

            # Mark station completed regardless of whether individual train timetables had partial errors
            set_station_status(stn_code, "COMPLETED", len(services), len(services))
            print(f"    ✓ Station {stn_code} marked COMPLETED ({newly_ingested} new trains ingested).\n")
            time.sleep(BASE_REQUEST_DELAY)

    # Final DB counts
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM stations;")
            stn_cnt = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM trains;")
            trn_cnt = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM train_stops;")
            stp_cnt = cur.fetchone()[0]

    print("\n" + "=" * 75)
    print("RAILSYNC UNRESTRICTED CRAWLER FINISHED")
    print("=" * 75)
    print(f"Total Unique Stations:  {stn_cnt}")
    print(f"Total Unique Trains:    {trn_cnt}")
    print(f"Total Halts Ingested:   {stp_cnt}")
    print("=" * 75)

if __name__ == "__main__":
    main()
