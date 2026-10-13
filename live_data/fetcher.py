"""
live_data/fetcher.py
====================
Fetches and parses real-time live train data from the official RailRadar API.
Uses API key configured via RAILRADAR_API_KEY environment variable.
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# ── Load API key from environment ────────────────────────────────
load_dotenv()
_API_KEY  = os.getenv("RAILRADAR_API_KEY", "").strip()
_BASE_URL = os.getenv("RAILRADAR_BASE_URL", "https://api.railradar.in/v1").strip()

# ── Supported trains set (empty allows any train number) ──────────
SUPPORTED_TRAINS = set()


class RailRadarError(Exception):
    """Raised when the API returns an error or unsupported train."""


def fetch_live_status(train_number: str, journey_date: str | None = None) -> dict:
    """
    Calls RailRadar API for the given train number, optionally for a specific journey_date.
    Returns live train status, real-time current location, actual delays, route stops,
    and verified journey instance identity.
    """
    if not _API_KEY:
        raise RailRadarError("RAILRADAR_API_KEY environment variable is missing or empty.")
    tn = str(train_number).strip()
    url = f"{_BASE_URL}/trains/{tn}/live"
    headers = {
        "x-api-key": _API_KEY,
        "Authorization": f"Bearer {_API_KEY}"
    }
    params = {"date": str(journey_date).strip()} if journey_date and str(journey_date).strip() else None

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=3.5)
    except requests.RequestException as e:
        raise RailRadarError(f"Network error calling RailRadar API: {e}")

    if resp.status_code != 200:
        raise RailRadarError(
            f"RailRadar API returned HTTP {resp.status_code}: {resp.text[:300]}"
        )

    body = resp.json()
    if not body.get("success") or "data" not in body:
        raise RailRadarError(f"RailRadar API returned success=false: {body}")

    return _parse_response(body["data"], fallback_tn=tn)


# ── Internal parser ───────────────────────────────────────────────
def _parse_response(data: dict, fallback_tn: str = "") -> dict:
    """Converts raw RailRadar API JSON into a clean, normalised structure."""

    train = data.get("train", {})
    current = data.get("currentLocation") or {}
    prev_halt = data.get("previousHalt") or {}
    next_halt = data.get("nextHalt") or {}

    route_raw = data.get("route", [])
    route = [_parse_station(s) for s in route_raw]

    curr_dist = float(current.get("distanceFromOriginKm") or 0.0)
    prev_dist_raw = float(prev_halt.get("distance") or 0.0) if prev_halt else 0.0
    next_dist_raw = float(next_halt.get("distance") or 0.0) if next_halt else 0.0

    prev_rel_dist = round(abs(curr_dist - prev_dist_raw), 1) if (prev_halt and curr_dist > 0 and prev_dist_raw > 0) else prev_dist_raw
    next_rel_dist = round(abs(next_dist_raw - curr_dist), 1) if (next_halt and curr_dist > 0 and next_dist_raw > 0) else next_dist_raw

    tn_str = str(data.get("trainNumber") or train.get("number") or fallback_tn or "")
    start_date = data.get("startDate")
    journey_instance_id = f"{tn_str}_{start_date}" if (tn_str and start_date) else None

    return {
        # Train & Journey identity
        "train_number":         tn_str,
        "train_name":           data.get("trainName") or train.get("name") or "",
        "start_date":           start_date,
        "journey_instance_id":  journey_instance_id,
        "source_code":          (train.get("source") or {}).get("code", ""),
        "source_name":          (train.get("source") or {}).get("name", ""),
        "dest_code":            (train.get("destination") or {}).get("code", ""),
        "dest_name":            (train.get("destination") or {}).get("name", ""),

        # Live status
        "is_live":              bool(data.get("isLive", True)),
        "current_delay_minutes": float(data.get("delayMinutes") or current.get("delayMinutes") or 0.0),
        "last_updated_at":      data.get("lastUpdatedAt"),

        # Current position
        "current_station_code": current.get("stationCode"),
        "current_station_name": current.get("stationName"),
        "current_status":       current.get("status"),
        "speed_kmh":            float(current.get("speedKmh") or current.get("speed") or 65.0),
        "segment_progress":     float(current.get("segmentProgress") or 0.0),

        # Neighbouring halts
        "previous_halt": {
            "station_code": prev_halt.get("stationCode"),
            "station_name": prev_halt.get("stationName"),
            "distance_km":  prev_rel_dist,
        } if prev_halt and prev_halt.get("stationCode") else None,

        "next_halt": {
            "station_code": next_halt.get("stationCode"),
            "station_name": next_halt.get("stationName"),
            "distance_km":  next_rel_dist,
        } if next_halt and next_halt.get("stationCode") else None,

        # Full route
        "route": route,
        "fetched_at": datetime.now(timezone.utc).isoformat() if hasattr(timezone, 'utc') else datetime.utcnow().isoformat() + "Z",
    }


def _parse_station(s: dict) -> dict:
    """Normalises a single station entry from the route array."""
    return {
        "sequence":            int(s.get("sequence") or 0),
        "station_code":        s.get("stationCode", ""),
        "station_name":        s.get("stationName", ""),
        "status":              s.get("status", ""),   # e.g. "departed", "arrived", "upcoming"
        "is_halt":             bool(s.get("isHalt", True)),
        "scheduled_arrival":   s.get("scheduledArrival"),
        "scheduled_departure": s.get("scheduledDeparture"),
        "actual_arrival":      s.get("actualArrival"),
        "actual_departure":    s.get("actualDeparture"),
        "delay_arrival_min":   _safe_float(s.get("delayArrival")),
        "delay_departure_min": _safe_float(s.get("delayDeparture")),
        "platform":            s.get("platform"),
        "distance_km":         _safe_float(s.get("distance")),
    }


def _safe_float(v) -> float | None:
    """Returns float or None."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
