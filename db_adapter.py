"""
db_adapter.py
=============
PostgreSQL database interface for RailSync AI.
Connects directly to the local railway_db PostgreSQL database (5,209 trains, 4,600+ stations, 200,000+ train_stops).
"""

import os
import sys
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("db_adapter")

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

def get_db_connection():
    """Establishes connection to local PostgreSQL DB (railway_db)."""
    if not HAS_PSYCOPG2:
        return None

    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        logger.warning("DATABASE_URL environment variable is missing or empty.")
        return None

    try:
        return psycopg2.connect(db_url, cursor_factory=RealDictCursor)
    except Exception as e:
        logger.warning(f"PostgreSQL connection warning: {e}")
        return None


def get_all_trains_from_db(search_term: str = None, limit: int = 50):
    """
    Search across all 5,209 trains in PostgreSQL by train number or train name.
    Strictly filters out temporary/special trains (train numbers starting with '0').
    When search_term is empty, returns trains that are currently running on their schedule at local time.
    """
    conn = get_db_connection()
    if not conn:
        return []

    try:
        cur = conn.cursor()
        now = datetime.now()
        now_min = now.hour * 60 + now.minute

        if search_term and search_term.strip():
            term = f"%{search_term.strip()}%"
            sql = """
                SELECT t.train_number, t.train_name, t.train_type, t.from_station_code, t.to_station_code,
                       s_src.name AS source_name, s_dst.name AS dest_name,
                       src.departure_time::text AS src_dep, dst.arrival_time::text AS dst_arr,
                       (SELECT COUNT(*) FROM train_stops WHERE train_number = t.train_number) AS n_stations,
                       COALESCE((
                           SELECT ROUND(AVG(hso.arrival_delay_min)::numeric, 1)
                           FROM historical_runs hr
                           JOIN historical_station_observations hso ON hr.run_id = hso.run_id
                           WHERE hr.train_number = t.train_number
                       ), 8.5) AS avg_delay_min
                FROM trains t
                LEFT JOIN stations s_src ON t.from_station_code = s_src.code
                LEFT JOIN stations s_dst ON t.to_station_code = s_dst.code
                LEFT JOIN train_stops src ON t.train_number = src.train_number AND src.stop_order = 1
                LEFT JOIN train_stops dst ON t.train_number = dst.train_number AND dst.stop_order = (
                    SELECT MAX(stop_order) FROM train_stops WHERE train_number = t.train_number
                )
                WHERE TRIM(t.train_number) !~ '^0'
                  AND TRIM(t.train_number) NOT LIKE '0%%'
                  AND (t.train_number ILIKE %s OR t.train_name ILIKE %s)
                ORDER BY t.train_number ASC
                LIMIT %s;
            """
            cur.execute(sql, (term, term, limit))
        else:
            # Query trains currently running right now (departed before current time & arriving after current time based on journey duration)
            sql = """
                SELECT t.train_number, t.train_name, t.train_type, t.from_station_code, t.to_station_code,
                       s_src.name AS source_name, s_dst.name AS dest_name,
                       src.departure_time::text AS src_dep, dst.arrival_time::text AS dst_arr,
                       dst.day_number AS dst_day,
                       (SELECT COUNT(*) FROM train_stops WHERE train_number = t.train_number) AS n_stations,
                       COALESCE((
                           SELECT ROUND(AVG(hso.arrival_delay_min)::numeric, 1)
                           FROM historical_runs hr
                           JOIN historical_station_observations hso ON hr.run_id = hso.run_id
                           WHERE hr.train_number = t.train_number
                       ), 8.5) AS avg_delay_min
                FROM trains t
                JOIN train_stops src ON t.train_number = src.train_number AND src.stop_order = 1
                JOIN train_stops dst ON t.train_number = dst.train_number AND dst.stop_order = (
                    SELECT MAX(stop_order) FROM train_stops WHERE train_number = t.train_number
                )
                LEFT JOIN stations s_src ON t.from_station_code = s_src.code
                LEFT JOIN stations s_dst ON t.to_station_code = s_dst.code
                WHERE TRIM(t.train_number) !~ '^0'
                  AND TRIM(t.train_number) NOT LIKE '0%%'
                  AND (
                    -- Today's run active
                    (%s >= (EXTRACT(HOUR FROM src.departure_time)*60 + EXTRACT(MINUTE FROM src.departure_time))
                     AND %s - (EXTRACT(HOUR FROM src.departure_time)*60 + EXTRACT(MINUTE FROM src.departure_time)) <= (((dst.day_number - 1)*1440 + EXTRACT(HOUR FROM dst.arrival_time)*60 + EXTRACT(MINUTE FROM dst.arrival_time)) - (EXTRACT(HOUR FROM src.departure_time)*60 + EXTRACT(MINUTE FROM src.departure_time))))
                    OR
                    -- Yesterday's multi-day run active
                    (%s + 1440 >= (EXTRACT(HOUR FROM src.departure_time)*60 + EXTRACT(MINUTE FROM src.departure_time))
                     AND (%s + 1440) - (EXTRACT(HOUR FROM src.departure_time)*60 + EXTRACT(MINUTE FROM src.departure_time)) <= (((dst.day_number - 1)*1440 + EXTRACT(HOUR FROM dst.arrival_time)*60 + EXTRACT(MINUTE FROM dst.arrival_time)) - (EXTRACT(HOUR FROM src.departure_time)*60 + EXTRACT(MINUTE FROM src.departure_time))))
                    OR
                    -- 2-days-ago multi-day run active
                    (%s + 2880 >= (EXTRACT(HOUR FROM src.departure_time)*60 + EXTRACT(MINUTE FROM src.departure_time))
                     AND (%s + 2880) - (EXTRACT(HOUR FROM src.departure_time)*60 + EXTRACT(MINUTE FROM src.departure_time)) <= (((dst.day_number - 1)*1440 + EXTRACT(HOUR FROM dst.arrival_time)*60 + EXTRACT(MINUTE FROM dst.arrival_time)) - (EXTRACT(HOUR FROM src.departure_time)*60 + EXTRACT(MINUTE FROM src.departure_time))))
                  )
                ORDER BY 
                    CASE 
                        WHEN t.train_type ILIKE '%%Vande%%' OR t.train_type ILIKE '%%Rajdhani%%' OR t.train_type ILIKE '%%Shatabdi%%' THEN 1
                        WHEN TRIM(t.train_number) LIKE '2%%' THEN 2
                        WHEN TRIM(t.train_number) LIKE '12%%' THEN 3
                        WHEN TRIM(t.train_number) LIKE '1%%' THEN 4
                        WHEN TRIM(t.train_number) LIKE '5%%' THEN 5
                        ELSE 6
                    END,
                    t.train_number ASC
                LIMIT %s;
            """
            cur.execute(sql, (now_min, now_min, now_min, now_min, now_min, now_min, limit))

        rows = cur.fetchall()
        cur.close()
        conn.close()

        results = []
        for r in rows:
            tn = str(r["train_number"]).strip()
            if tn.startswith("0"):
                continue
            results.append({
                "train_number": tn,
                "name": r["train_name"] or f"Train {tn}",
                "source_code": r["from_station_code"] or "",
                "source_name": r["source_name"] or r["from_station_code"] or "",
                "dest_code": r["to_station_code"] or "",
                "dest_name": r["dest_name"] or r["to_station_code"] or "",
                "type": r["train_type"] or "Express",
                "color": "#1E40AF" if r["train_type"] == "SF" else "#065F46",
                "avg_delay_min": float(r["avg_delay_min"]) if r.get("avg_delay_min") is not None else 8.5,
                "n_stations": r["n_stations"] or 15,
            })
        return results
    except Exception as e:
        logger.error(f"Error querying trains from DB: {e}")
        if conn: conn.close()
        return []


def get_train_route_from_db(train_number: str):
    """
    Fetches complete route station sequence for a train from PostgreSQL train_stops.
    Calculates accurate proportional distance_from_source and marks commercial halt stations.
    """
    conn = get_db_connection()
    if not conn:
        return []

    try:
        cur = conn.cursor()
        sql = """
            SELECT ts.station_code, s.name AS station_name, ts.stop_order AS sequence,
                   ts.arrival_time::text AS scheduled_arrival,
                   ts.departure_time::text AS scheduled_departure,
                   t.distance_km AS total_train_distance,
                   (CASE WHEN ts.stop_order = 1 
                          OR (ts.arrival_time IS NOT NULL AND ts.departure_time IS NOT NULL AND ts.arrival_time != ts.departure_time)
                          OR ts.stop_order = (SELECT MAX(stop_order) FROM train_stops WHERE train_number = ts.train_number)
                         THEN true ELSE false END) AS is_halt
            FROM train_stops ts
            LEFT JOIN stations s ON ts.station_code = s.code
            LEFT JOIN trains t ON ts.train_number = t.train_number
            WHERE ts.train_number = %s
            ORDER BY ts.stop_order ASC;
        """
        cur.execute(sql, (str(train_number).strip(),))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return []

        total_dist = float(rows[0]["total_train_distance"] or 450)
        n_total = len(rows)

        route = []
        for r in rows:
            seq = int(r["sequence"])
            dist = round(((seq - 1) / max(n_total - 1, 1)) * total_dist, 1)
            route.append({
                "station_code": r["station_code"],
                "station_name": r["station_name"] or r["station_code"],
                "sequence": seq,
                "scheduled_arrival": r["scheduled_arrival"][:5] if r["scheduled_arrival"] else None,
                "scheduled_departure": r["scheduled_departure"][:5] if r["scheduled_departure"] else None,
                "distance_from_source": dist,
                "is_halt": bool(r["is_halt"]),
            })
        return route
    except Exception as e:
        logger.error(f"Error fetching route for train {train_number}: {e}")
        if conn: conn.close()
        return []


def get_train_metadata_from_db(train_number: str):
    """Fetches single train metadata dict from PostgreSQL trains table."""
    conn = get_db_connection()
    if not conn:
        return None

    try:
        cur = conn.cursor()
        sql = """
            SELECT t.train_number, t.train_name, t.train_type, t.from_station_code, t.to_station_code,
                   s_src.name AS source_name, s_dst.name AS dest_name
            FROM trains t
            LEFT JOIN stations s_src ON t.from_station_code = s_src.code
            LEFT JOIN stations s_dst ON t.to_station_code = s_dst.code
            WHERE t.train_number = %s
            LIMIT 1;
        """
        cur.execute(sql, (str(train_number).strip(),))
        r = cur.fetchone()
        cur.close()
        conn.close()

        if r:
            return {
                "name": r["train_name"] or f"Train {r['train_number']}",
                "source_name": r["source_name"] or r["from_station_code"] or "",
                "dest_name": r["dest_name"] or r["to_station_code"] or "",
                "type": r["train_type"] or "Express",
                "emoji": "🚆",
                "color": "#1E40AF" if r["train_type"] == "SF" else "#065F46"
            }
        return None
    except Exception as e:
        logger.error(f"Error fetching metadata for train {train_number}: {e}")
        if conn: conn.close()
        return None


def query_indirect_routes_sql(origin: str, destination: str, date_str: str = "2026-09-10"):
    """
    Executes SQL query to find direct & 1-transfer train itineraries from PostgreSQL.
    """
    conn = get_db_connection()
    if not conn:
        return None

    candidates = []
    try:
        cur = conn.cursor()

        # 1. Direct Trains Query
        direct_sql = """
            SELECT 
                ts1.train_number, t.train_name, t.train_type,
                ts1.station_code AS from_stn, ts1.departure_time::text AS sched_dep,
                ts2.station_code AS to_stn, ts2.arrival_time::text AS sched_arr,
                (ts2.stop_order - ts1.stop_order) AS num_stations
            FROM train_stops ts1
            JOIN train_stops ts2 ON ts1.train_number = ts2.train_number AND ts1.stop_order < ts2.stop_order
            JOIN trains t ON ts1.train_number = t.train_number
            WHERE ts1.station_code = %s AND ts2.station_code = %s
              AND TRIM(ts1.train_number) !~ '^0' AND TRIM(ts1.train_number) NOT LIKE '0%%'
            LIMIT 10;
        """
        cur.execute(direct_sql, (origin, destination))
        direct_rows = cur.fetchall()
        for row in direct_rows:
            tn_str = str(row['train_number']).strip()
            if tn_str.startswith('0'): continue
            candidates.append({
                "id": f"db_direct_{tn_str}",
                "name": f"{row['train_name'] or 'Direct Express'} ({tn_str})",
                "num_transfers": 0,
                "interchanges": [],
                "legs": [{
                    "train_number": tn_str,
                    "train_name": row['train_name'] or f"Train {tn_str}",
                    "from_station": origin,
                    "from_name": origin,
                    "to_station": destination,
                    "to_name": destination,
                    "sched_dep": str(row['sched_dep'])[:5] if row['sched_dep'] else "08:00",
                    "sched_arr": str(row['sched_arr'])[:5] if row['sched_arr'] else "16:00",
                    "num_stations": row['num_stations'],
                    "distance_km": 450.0,
                }]
            })

        # 2. 1-Transfer Indirect Query
        transfer_sql = """
            SELECT 
                t1.train_number AS tn1, tr1.train_name AS name1, t1.departure_time::text AS sched_dep1, t1_ic.arrival_time::text AS sched_arr1,
                t1_ic.station_code AS interchange, s_ic.name AS interchange_name,
                t2.train_number AS tn2, tr2.train_name AS name2, t2_ic.departure_time::text AS sched_dep2, t2.arrival_time::text AS sched_arr2
            FROM train_stops t1
            JOIN train_stops t1_ic ON t1.train_number = t1_ic.train_number AND t1.stop_order < t1_ic.stop_order
            JOIN train_stops t2_ic ON t1_ic.station_code = t2_ic.station_code
            JOIN train_stops t2 ON t2_ic.train_number = t2.train_number AND t2_ic.stop_order < t2.stop_order
            JOIN trains tr1 ON t1.train_number = tr1.train_number
            JOIN trains tr2 ON t2.train_number = tr2.train_number
            LEFT JOIN stations s_ic ON t1_ic.station_code = s_ic.code
            WHERE t1.station_code = %s AND t2.station_code = %s AND t1.train_number != t2.train_number
              AND TRIM(t1.train_number) !~ '^0' AND TRIM(t1.train_number) NOT LIKE '0%%'
              AND TRIM(t2.train_number) !~ '^0' AND TRIM(t2.train_number) NOT LIKE '0%%'
            LIMIT 15;
        """
        cur.execute(transfer_sql, (origin, destination))
        transfer_rows = cur.fetchall()
        for row in transfer_rows:
            ic = row['interchange']
            ic_name = row['interchange_name'] or ic
            candidates.append({
                "id": f"db_transfer_{row['tn1']}_{row['tn2']}_{ic}",
                "name": f"Via {ic_name} ({row['tn1']} → {row['tn2']})",
                "num_transfers": 1,
                "interchanges": [ic],
                "legs": [
                    {
                        "train_number": str(row['tn1']),
                        "train_name": row['name1'] or f"Train {row['tn1']}",
                        "from_station": origin,
                        "from_name": origin,
                        "to_station": ic,
                        "to_name": ic_name,
                        "sched_dep": str(row['sched_dep1'])[:5] if row['sched_dep1'] else "07:00",
                        "sched_arr": str(row['sched_arr1'])[:5] if row['sched_arr1'] else "11:00",
                        "num_stations": 4,
                        "distance_km": 250.0,
                    },
                    {
                        "train_number": str(row['tn2']),
                        "train_name": row['name2'] or f"Train {row['tn2']}",
                        "from_station": ic,
                        "from_name": ic_name,
                        "to_station": destination,
                        "to_name": destination,
                        "sched_dep": str(row['sched_dep2'])[:5] if row['sched_dep2'] else "12:15",
                        "sched_arr": str(row['sched_arr2'])[:5] if row['sched_arr2'] else "17:00",
                        "num_stations": 4,
                        "distance_km": 250.0,
                    }
                ]
            })

        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error querying Postgres for indirect routes: {e}")
        if conn: conn.close()
        return None

    return candidates if candidates else None


def log_prediction_to_db(
    journey_instance_id: str,
    train_number: str,
    journey_start_date: str,
    current_station: str,
    current_delay_min: float,
    target_station: str,
    scheduled_arrival: str | None = None,
    predicted_arrival: str | None = None,
    predicted_delay_min: float = 0.0,
    actual_arrival: str | None = None,
    actual_delay_min: float | None = None,
    model_source: str = "xgboost_hybrid"
) -> bool:
    """
    Persists a prediction record into prediction_log for automated live accuracy scoring.
    Safe and non-blocking: never raises an exception.
    """
    conn = get_db_connection()
    if not conn:
        return False

    try:
        cur = conn.cursor()
        err_min = (predicted_delay_min - actual_delay_min) if (actual_delay_min is not None) else None
        sql = """
            INSERT INTO prediction_log (
                journey_instance_id, train_number, journey_start_date,
                current_station, current_delay_min, target_station,
                scheduled_arrival, predicted_arrival, predicted_delay_min,
                actual_arrival, actual_delay_min, error_min, model_source
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        cur.execute(sql, (
            journey_instance_id,
            train_number,
            journey_start_date,
            current_station,
            current_delay_min,
            target_station,
            scheduled_arrival,
            predicted_arrival,
            predicted_delay_min,
            actual_arrival,
            actual_delay_min,
            err_min,
            model_source
        ))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.debug(f"Failed to log prediction: {e}")
        if conn:
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass
        return False
