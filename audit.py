"""
harvest_train_range_v2.py
=========================
Fast, rich-schema fleet ingestion for the 22xxx series (Vande Bharat & Superfast).
Populates timetable times, distance, day count, platform, and halt flags.
"""

import sys
import os
import re
import time
import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import execute_values

# Windows UTF-8 safety
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

START_TRAIN = 18000
END_TRAIN   = 18999
DB_CONFIG = {
    "dbname": os.getenv("PGDATABASE", "SIH_Railway_DB"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "postgres"),
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", 5432))
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def clean_station_name(raw_name: str) -> str:
    cleaned = re.split(r'\s+\d+\s*kms?', raw_name, flags=re.IGNORECASE)[0]
    cleaned = re.split(r'Platform:', cleaned, flags=re.IGNORECASE)[0]
    return cleaned.strip()

def parse_distance(raw_text: str):
    m = re.search(r'(\d+)\s*kms?', raw_text, re.IGNORECASE)
    return int(m.group(1)) if m else 0

def parse_time(raw_text: str):
    t = raw_text.strip()
    if re.match(r'^\d{1,2}:\d{2}$', t):
        return t
    return None

def parse_day(raw_text: str):
    m = re.search(r'(\d+)', raw_text)
    return int(m.group(1)) if m else 1

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("SELECT train_number FROM trains;")
    existing_fleet = {row[0] for row in cur.fetchall()}
    print(f"[*] Found {len(existing_fleet)} existing trains in DB.")
    print(f"[*] Starting Sweep: {START_TRAIN} -> {END_TRAIN} (16xxx Series)...")

    added_count = 0

    for t_int in range(START_TRAIN, END_TRAIN + 1):
        t_num = str(t_int).zfill(5)

        if t_num in existing_fleet:
            continue

        route_url = f"https://etrain.info/in?TRAIN={t_num}"
        try:
            resp = requests.get(route_url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            tables = soup.find_all("table")
            if not tables:
                continue

            table = max(tables, key=lambda t: len(t.find_all("tr")))
            rows = table.find_all("tr")

            raw_stops = []
            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 5:
                    continue

                first_cell = cols[0].get_text(" ", strip=True)
                tokens = first_cell.split()

                if len(tokens) >= 2 and tokens[0].isdigit():
                    s_code = tokens[-1].upper()
                    if not s_code.isalnum() or len(s_code) > 10:
                        continue

                    raw_sname_cell = cols[2].get_text(" ", strip=True)
                    s_name = clean_station_name(raw_sname_cell)
                    dist = parse_distance(raw_sname_cell)

                    arr_time = parse_time(cols[3].get_text(strip=True)) if len(cols) > 3 else None
                    dep_time = parse_time(cols[4].get_text(strip=True)) if len(cols) > 4 else None

                    # Halt determination
                    halt_text = cols[6].get_text(strip=True) if len(cols) > 6 else ""
                    is_halt = False if halt_text in ["--", "Pass", ""] else True

                    day_val = parse_day(cols[7].get_text(strip=True)) if len(cols) > 7 else 1
                    pf_val = cols[8].get_text(strip=True) if len(cols) > 8 else None
                    if pf_val in ["--", ""]:
                        pf_val = None

                    raw_stops.append({
                        "station_code": s_code,
                        "station_name": s_name,
                        "distance": dist,
                        "arr_time": arr_time,
                        "dep_time": dep_time,
                        "is_halt": is_halt,
                        "day": day_val,
                        "platform": pf_val
                    })

            if len(raw_stops) < 2:
                continue

            # Train name parsing
            t_name = f"Train {t_num}"
            title_tag = soup.find("title")
            if title_tag:
                m = re.search(r"Schedule of\s+(.*?)\s+\(" + t_num + r"\)", title_tag.text, re.IGNORECASE)
                if m:
                    t_name = m.group(1).strip()

            origin_code = raw_stops[0]["station_code"]
            dest_code = raw_stops[-1]["station_code"]

            # Stations Upsert
            stations_to_insert = [(s["station_code"], s["station_name"]) for s in raw_stops]
            execute_values(
                cur,
                "INSERT INTO stations (station_code, station_name) VALUES %s ON CONFLICT (station_code) DO NOTHING;",
                list({s[0]: s for s in stations_to_insert}.values())
            )

            # Train Upsert
            cur.execute(
                """
                INSERT INTO trains (train_number, train_name, from_station_code, to_station_code)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (train_number) DO UPDATE
                SET train_name = EXCLUDED.train_name,
                    from_station_code = EXCLUDED.from_station_code,
                    to_station_code = EXCLUDED.to_station_code;
                """,
                (t_num, t_name, origin_code, dest_code)
            )

            # Build enriched train_stops records
            stops_data = [
                (
                    t_num,
                    s["station_code"],
                    s["station_name"],
                    idx,
                    s["arr_time"],
                    s["dep_time"],
                    s["day"],
                    s["day"],
                    s["distance"],
                    s["platform"],
                    s["is_halt"]
                )
                for idx, s in enumerate(raw_stops, 1)
            ]

            execute_values(
                cur,
                """
                INSERT INTO train_stops (
                    train_number, station_code, station_name, stop_order,
                    arrival_time, departure_time, arrival_day, departure_day,
                    distance_from_source, platform, is_halt
                )
                VALUES %s
                ON CONFLICT (train_number, stop_order) DO UPDATE
                SET station_code = EXCLUDED.station_code,
                    station_name = EXCLUDED.station_name,
                    arrival_time = EXCLUDED.arrival_time,
                    departure_time = EXCLUDED.departure_time,
                    arrival_day = EXCLUDED.arrival_day,
                    departure_day = EXCLUDED.departure_day,
                    distance_from_source = EXCLUDED.distance_from_source,
                    platform = EXCLUDED.platform,
                    is_halt = EXCLUDED.is_halt;
                """,
                stops_data
            )

            conn.commit()
            existing_fleet.add(t_num)
            added_count += 1
            print(f"  [+] Ingested {t_num} ({t_name[:25]}) - {len(stops_data)} stops (dist: {raw_stops[-1]['distance']} km)")
            time.sleep(0.3)

        except Exception as e:
            conn.rollback()
            print(f"  [!] Failed on {t_num}: {e}")

    cur.close()
    conn.close()
    print(f"\n[OK] 22xxx Series sweep completed. Ingested {added_count} new trains.")

if __name__ == "__main__":
    main()
