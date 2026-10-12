"""
enrich_all_null_trains.py
=========================
Fixes NULL timetable values across train_stops by extracting arrival and departure
times directly from the dual <div> structure in etrain.info's timing column.
"""

import os
import re
import time
import requests
import psycopg2
from bs4 import BeautifulSoup

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

def get_trains_with_null_times(cur):
    query = """
        WITH route_bounds AS (
            SELECT 
                train_number, 
                MIN(stop_order) AS first_stop, 
                MAX(stop_order) AS last_stop
            FROM train_stops
            GROUP BY train_number
        )
        SELECT DISTINCT ts.train_number
        FROM train_stops ts
        JOIN route_bounds rb ON ts.train_number = rb.train_number
        WHERE ts.stop_order NOT IN (rb.first_stop, rb.last_stop)
          AND (ts.arrival_time IS NULL OR ts.departure_time IS NULL)
        ORDER BY ts.train_number ASC;
    """
    cur.execute(query)
    return [row[0] for row in cur.fetchall()]

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    target_trains = get_trains_with_null_times(cur)
    total_trains = len(target_trains)
    print(f"[*] Found {total_trains} trains needing timetable enrichment.")

    for idx, t_num in enumerate(target_trains, 1):
        url = f"https://etrain.info/in?TRAIN={t_num}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                print(f"[-] [{idx}/{total_trains}] Train {t_num}: HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            tables = soup.find_all("table")
            if not tables:
                print(f"[-] [{idx}/{total_trains}] Train {t_num}: No schedule table found.")
                continue

            table = max(tables, key=lambda t: len(t.find_all("tr")))
            updated_count = 0

            for row in table.find_all("tr"):
                cols = row.find_all("td")
                if len(cols) < 5:
                    continue

                # 1. Station Code (from cols[0])
                s_code_elem = cols[0].find("small")
                if s_code_elem:
                    s_code = s_code_elem.get_text(strip=True).upper()
                else:
                    match = re.search(r'([A-Z]{2,6})$', cols[0].get_text(strip=True))
                    s_code = match.group(1) if match else None

                if not s_code:
                    continue

                # 2. Distance and Platform (from cols[2])
                col2_text = cols[2].get_text(" ", strip=True)
                dist_match = re.search(r'(\d+)\s*kms?', col2_text, re.IGNORECASE)
                dist = int(dist_match.group(1)) if dist_match else 0

                pf_match = re.search(r'Platform:\s*(\d+)', col2_text)
                pf = pf_match.group(1) if pf_match else None

                # 3. Arrival and Departure Times (from cols[4])
                time_divs = cols[4].find_all("div")
                if len(time_divs) >= 2:
                    arr_raw = time_divs[0].get_text(strip=True)
                    dep_raw = time_divs[1].get_text(strip=True)

                    arr_m = re.search(r'(\d{1,2}:\d{2})', arr_raw)
                    dep_m = re.search(r'(\d{1,2}:\d{2})', dep_raw)

                    arr = arr_m.group(1) if arr_m else None
                    dep = dep_m.group(1) if dep_m else None

                    day_match = re.search(r'Day\s*(\d+)', cols[4].get_text())
                    day = int(day_match.group(1)) if day_match else 1
                else:
                    arr, dep, day = None, None, 1

                # 4. Update the record
                cur.execute("""
                    UPDATE train_stops
                    SET arrival_time = %s,
                        departure_time = %s,
                        arrival_day = %s,
                        departure_day = %s,
                        distance_from_source = %s,
                        platform = COALESCE(%s, platform)
                    WHERE train_number = %s AND station_code = %s;
                """, (arr, dep, day, day, dist, pf, t_num, s_code))
                updated_count += cur.rowcount

            conn.commit()
            print(f"[+] [{idx}/{total_trains}] Train {t_num}: updated {updated_count} halts.")
            time.sleep(0.3)

        except Exception as e:
            conn.rollback()
            print(f"[!] Error updating train {t_num}: {e}")
            time.sleep(1)

    cur.close()
    conn.close()
    print("\n[OK] Enrichment complete.")

if __name__ == "__main__":
    main()
