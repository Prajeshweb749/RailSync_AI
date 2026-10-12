"""
prediction/connection_risk_service.py
======================================
RailSync AI — Connection Risk & Transfer Reliability Engine

Evaluates transfer buffer times and connection risk categories (SAFE, UNCERTAIN, RISKY)
for multi-leg / 1-transfer railway itineraries.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_adapter import get_db_connection

logger = logging.getLogger("connection_risk_service")

def _parse_time_min(t_str: str) -> float | None:
    """Parses time string 'HH:MM' or 'HH:MM:SS' into minutes from midnight."""
    if not t_str or str(t_str).strip() in ("None", "NaT", "nan", ""):
        return None
    try:
        parts = str(t_str).strip().split(":")
        hh = int(parts[0])
        mm = int(parts[1])
        return float(hh * 60 + mm)
    except Exception:
        return None


def calculate_connection_risk(
    incoming_train: str,
    outgoing_train: str,
    interchange_station: str,
    predicted_arr_delay_leg1: float,
    sched_arr_leg1_time: str = "11:00",
    sched_dep_leg2_time: str = "12:00"
) -> dict:
    """
    Evaluates connection risk between two connecting train legs at an interchange station.

    Parameters
    ----------
    incoming_train           : str   — Train number of incoming leg (Leg 1)
    outgoing_train           : str   — Train number of outgoing leg (Leg 2)
    interchange_station      : str   — Station code of transfer station
    predicted_arr_delay_leg1 : float — Predicted arrival delay of incoming train in minutes
    sched_arr_leg1_time      : str   — Scheduled arrival time 'HH:MM' of incoming train
    sched_dep_leg2_time      : str   — Scheduled departure time 'HH:MM' of outgoing train

    Returns
    -------
    dict with buffer minutes, risk level (SAFE/UNCERTAIN/RISKY), risk color, and recommendation.
    """
    t1 = str(incoming_train).strip()
    t2 = str(outgoing_train).strip()
    stn = str(interchange_station).strip().upper()

    t1_arr_min = _parse_time_min(sched_arr_leg1_time) or 660.0 # Default 11:00 AM
    t2_dep_min = _parse_time_min(sched_dep_leg2_time) or 720.0 # Default 12:00 PM

    # Account for next-day departures
    if t2_dep_min < t1_arr_min:
        sched_buffer = (t2_dep_min + 1440.0) - t1_arr_min
    else:
        sched_buffer = t2_dep_min - t1_arr_min

    # Effective buffer after accounting for predicted delay
    pred_delay = float(predicted_arr_delay_leg1 or 0.0)
    effective_buffer = sched_buffer - pred_delay

    # Categorize connection risk
    if effective_buffer >= 30.0:
        risk_level = "SAFE"
        risk_color = "#10B981" # Green
        connection_pct = min(99.0, max(85.0, 95.0 - (pred_delay * 0.2)))
        recommendation = f"Comfortable transfer margin ({effective_buffer:.1f} mins remaining). Connection is highly reliable."
    elif effective_buffer >= 10.0:
        risk_level = "UNCERTAIN"
        risk_color = "#F59E0B" # Amber
        connection_pct = min(84.0, max(50.0, 65.0 - ((30.0 - effective_buffer) * 1.5)))
        recommendation = f"Tight transfer margin ({effective_buffer:.1f} mins remaining). Monitor live running of Train {t1}."
    else:
        risk_level = "RISKY"
        risk_color = "#EF4444" # Red
        connection_pct = min(49.0, max(5.0, 25.0 + (effective_buffer * 2.0)))
        recommendation = f"High risk of missed transfer ({effective_buffer:.1f} mins buffer). Consider alternate connecting train."

    # Evidence & provenance lookup from PostgreSQL
    sample_count = 0
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(hso.observation_id) AS cnt
                FROM historical_runs hr
                JOIN historical_station_observations hso ON hr.run_id = hso.run_id
                WHERE hr.train_number IN (%s, %s) AND hso.station_code = %s;
            """, (t1, t2, stn))
            row = cur.fetchone()
            if row:
                sample_count = int(row["cnt"] or 0)
            cur.close()
            conn.close()
    except Exception as e:
        logger.warning(f"Error querying connection stats: {e}")

    return {
        "incoming_train": t1,
        "outgoing_train": t2,
        "interchange_station": stn,
        "scheduled_arr_leg1": str(sched_arr_leg1_time)[:5],
        "scheduled_dep_leg2": str(sched_dep_leg2_time)[:5],
        "scheduled_buffer_min": round(sched_buffer, 1),
        "predicted_incoming_delay_min": round(pred_delay, 1),
        "effective_buffer_min": round(effective_buffer, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "connection_probability_pct": round(connection_pct, 1),
        "confidence": "HIGH" if sample_count >= 20 else "MEDIUM",
        "recommendation": recommendation,
        "evidence": {
            "database": "railway_db (PostgreSQL)",
            "historical_observations_count": sample_count
        }
    }


if __name__ == "__main__":
    test_res = calculate_connection_risk("11020", "12801", "BBS", 15.0, "11:15", "12:00")
    print("Connection Risk Service Test Output:")
    print(test_res)
