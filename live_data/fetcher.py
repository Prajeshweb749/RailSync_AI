"""
live_data/fetcher.py
====================
Fetches and parses real-time live train data from:
1. Primary Provider: RailYatri NTES Live Tracking Feed (Zero quota limits, real-time NTES integration, multi-day support)
2. Secondary Provider: Official RailRadar API (via RAILRADAR_API_KEY)
3. Fallback: PostgreSQL database static simulation
"""

import os
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# ── Load API key from environment ────────────────────────────────
load_dotenv()
_API_KEY  = os.getenv("RAILRADAR_API_KEY", "").strip()
_BASE_URL = os.getenv("RAILRADAR_BASE_URL", "https://api.railradar.in/v1").strip()

# ── Supported trains set (empty allows any train number) ──────────
SUPPORTED_TRAINS = set()


class RailRadarError(Exception):
    """Raised when live API providers return an error or train status is unavailable."""


def fetch_live_status(train_number: str, journey_date: str | None = None) -> dict:
    """
    Primary entrypoint for live train data.
    First tries RailYatri NTES live tracking feed (unlimited, accurate real-time data).
    If unavailable, falls back to RailRadar API.
    Raises RailRadarError if both fail so callers can seamlessly use PostgreSQL corridor fallback.
    """
    tn = str(train_number).strip()

    # 1. Primary Live Provider: RailYatri NTES feed
    try:
        ry_data = _fetch_railyatri_live(tn, journey_date=journey_date)
        if ry_data and ry_data.get("route"):
            return ry_data
    except Exception as e:
        # Fallback to secondary provider
        pass

    # 2. Secondary Live Provider: RailRadar API
    if _API_KEY:
        try:
            return _fetch_railradar_live(tn, journey_date=journey_date)
        except Exception as e:
            raise RailRadarError(f"RailRadar API error: {e}")

    raise RailRadarError(f"Live status currently unavailable for train {tn}")


# ── RailYatri NTES Scraper & Parser ──────────────────────────────
def _fetch_railyatri_live(train_number: str, journey_date: str | None = None) -> dict | None:
    """
    Fetches real-time NTES running status from RailYatri Next.js embedded state.
    Supports multi-day parallel train disambiguation via start_day parameter:
      start_day=0 -> Today's departed run
      start_day=1 -> Yesterday's departed run
      start_day=2 -> 2 days ago departed run
    """
    IST = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(IST)
    today_ist = now_ist.date()

    start_day = 0
    req_date_str = None
    if journey_date:
        req_date_str = str(journey_date).strip()[:10]
        try:
            target_dt = datetime.strptime(req_date_str, "%Y-%m-%d").date()
            diff = (today_ist - target_dt).days
            if 0 <= diff <= 3:
                start_day = diff
            else:
                # Outside active live NTES window (historical archive or future)
                return None
        except Exception:
            pass

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    url = f"https://www.railyatri.in/live-train-status/{train_number}?start_day={start_day}"

    try:
        resp = requests.get(url, headers=headers, timeout=5.0)
    except requests.RequestException:
        return None

    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    script = soup.find("script", id="__NEXT_DATA__")
    if not script or not script.string:
        return None

    try:
        page_data = json.loads(script.string)
    except Exception:
        return None

    lts = page_data.get("props", {}).get("pageProps", {}).get("ltsData", {})
    if not lts or not lts.get("success"):
        return None

    # Verify journey date if explicitly requested
    live_start_date = lts.get("train_start_date")
    if req_date_str and live_start_date and live_start_date != req_date_str:
        return None

    return _parse_railyatri_response(lts, train_number=train_number, fallback_date=req_date_str or str(today_ist))


def _parse_railyatri_response(lts: dict, train_number: str, fallback_date: str) -> dict:
    """Normalises RailYatri LTS payload into RailSync's unified live status format."""
    start_date_str = lts.get("train_start_date") or fallback_date
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()

    raw_prev = [s for s in lts.get("previous_stations", []) if s.get("station_code") and str(s.get("station_code")).strip()]
    raw_up = [s for s in lts.get("upcoming_stations", []) if s.get("station_code") and str(s.get("station_code")).strip()]

    route = []
    seq = 1

    def _make_iso(base_d, time_str):
        if not time_str or ":" not in time_str:
            return None
        t_clean = time_str.strip()[:5]
        return f"{base_d.isoformat()}T{t_clean}:00"

    # 1. Departed / Arrived Stations
    for s in raw_prev:
        a_day = int(s.get("a_day") or s.get("day") or 0)
        stn_date = start_date + timedelta(days=a_day)
        sta = s.get("sta")
        std = s.get("std")
        eta = s.get("eta")
        etd = s.get("etd")
        arr_delay = _safe_float(s.get("arrival_delay"))
        dep_delay = _safe_float(s.get("departure_delay"))
        if dep_delay is None:
            dep_delay = arr_delay
        if arr_delay is None:
            arr_delay = dep_delay or 0.0

        sch_arr = _make_iso(stn_date, sta)
        sch_dep = _make_iso(stn_date, std)
        if sta and std and std < sta and sch_dep:
            sch_dep = _make_iso(stn_date + timedelta(days=1), std)

        # Exact recorded actual arrival & departure
        act_arr = None
        if sch_arr and arr_delay is not None:
            act_arr = (datetime.fromisoformat(sch_arr) + timedelta(minutes=arr_delay)).strftime("%Y-%m-%dT%H:%M:%S")
        elif eta:
            act_arr = _make_iso(stn_date, eta)

        act_dep = None
        if sch_dep and dep_delay is not None:
            act_dep = (datetime.fromisoformat(sch_dep) + timedelta(minutes=dep_delay)).strftime("%Y-%m-%dT%H:%M:%S")
        elif etd:
            act_dep = _make_iso(stn_date, etd)

        route.append({
            "sequence":            seq,
            "station_code":        str(s["station_code"]).strip().upper(),
            "station_name":        s.get("station_name") or s["station_code"],
            "status":              "departed",
            "is_halt":             True,
            "scheduled_arrival":   sch_arr,
            "scheduled_departure": sch_dep,
            "actual_arrival":      act_arr,
            "actual_departure":    act_dep,
            "delay_arrival_min":   arr_delay,
            "delay_departure_min": dep_delay,
            "platform":            str(s.get("platform_number") or ""),
            "distance_km":         _safe_float(s.get("distance_from_source") or s.get("distance")),
        })
        seq += 1

    # 2. Upcoming Stations
    for s in raw_up:
        a_day = int(s.get("a_day") or s.get("day") or 0)
        stn_date = start_date + timedelta(days=a_day)
        sta = s.get("sta")
        std = s.get("std")
        arr_delay = _safe_float(s.get("arrival_delay"))
        dep_delay = _safe_float(s.get("departure_delay"))
        if arr_delay is None:
            arr_delay = _safe_float(lts.get("delay")) or 0.0
        if dep_delay is None:
            dep_delay = arr_delay

        sch_arr = _make_iso(stn_date, sta)
        sch_dep = _make_iso(stn_date, std)
        if sta and std and std < sta and sch_dep:
            sch_dep = _make_iso(stn_date + timedelta(days=1), std)

        route.append({
            "sequence":            seq,
            "station_code":        str(s["station_code"]).strip().upper(),
            "station_name":        s.get("station_name") or s["station_code"],
            "status":              "upcoming",
            "is_halt":             True,
            "scheduled_arrival":   sch_arr,
            "scheduled_departure": sch_dep,
            "actual_arrival":      None,
            "actual_departure":    None,
            "delay_arrival_min":   arr_delay,
            "delay_departure_min": dep_delay,
            "platform":            str(s.get("platform_number") or ""),
            "distance_km":         _safe_float(s.get("distance_from_source") or s.get("distance")),
        })
        seq += 1

    # Neighbouring halts
    prev_halt = None
    if raw_prev:
        prev_s = raw_prev[-1]
        prev_halt = {
            "station_code": str(prev_s["station_code"]).strip().upper(),
            "station_name": prev_s.get("station_name"),
            "distance_km":  _safe_float(prev_s.get("distance_from_source") or prev_s.get("distance")),
        }

    next_halt = None
    if raw_up:
        next_s = raw_up[0]
        next_halt = {
            "station_code": str(next_s["station_code"]).strip().upper(),
            "station_name": next_s.get("station_name"),
            "distance_km":  _safe_float(next_s.get("distance_from_source") or next_s.get("distance")),
        }

    # Current station & status
    curr_code = lts.get("current_station_code")
    curr_name = lts.get("current_station_name")
    if lts.get("at_dstn") or not raw_up:
        curr_status = "arrived"
    elif lts.get("at_src"):
        curr_status = "at_origin"
    else:
        curr_status = "in_transit"

    tn_str = str(lts.get("train_number") or train_number)
    journey_instance_id = f"{tn_str}_{start_date_str}"

    return {
        # Train & Journey identity
        "train_number":         tn_str,
        "train_name":           lts.get("train_name", ""),
        "start_date":           start_date_str,
        "journey_instance_id":  journey_instance_id,
        "source_code":          lts.get("source", ""),
        "source_name":          lts.get("source_stn_name", ""),
        "dest_code":            lts.get("destination", ""),
        "dest_name":            lts.get("dest_stn_name", ""),

        # Live status
        "is_live":              True,
        "current_delay_minutes": float(lts.get("delay") or 0.0),
        "last_updated_at":      lts.get("update_time") or lts.get("status_as_of"),

        # Current position
        "current_station_code": curr_code,
        "current_station_name": curr_name,
        "current_status":       curr_status,
        "speed_kmh":            float(lts.get("avg_speed") or 55.0),
        "segment_progress":     float(lts.get("ahead_distance") or 0.0),

        # Neighbouring halts
        "previous_halt":        prev_halt,
        "next_halt":            next_halt,

        # Full route
        "route":                route,
        "fetched_at":           datetime.now(timezone.utc).isoformat(),
        "provider":             "railyatri_ntes",
    }


# ── RailRadar API Secondary Provider ─────────────────────────────
def _fetch_railradar_live(train_number: str, journey_date: str | None = None) -> dict:
    """Calls RailRadar API for live status."""
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

    return _parse_railradar_response(body["data"], fallback_tn=tn)


def _parse_railradar_response(data: dict, fallback_tn: str = "") -> dict:
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
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "provider": "railradar",
    }


def _parse_station(s: dict) -> dict:
    """Normalises a single station entry from the RailRadar route array."""
    return {
        "sequence":            int(s.get("sequence") or 0),
        "station_code":        str(s.get("stationCode", "")).strip().upper(),
        "station_name":        s.get("stationName", ""),
        "status":              s.get("status", ""),   # e.g. "departed", "arrived", "upcoming"
        "is_halt":             bool(s.get("isHalt", True)),
        "scheduled_arrival":   s.get("scheduledArrival"),
        "scheduled_departure": s.get("scheduledDeparture"),
        "actual_arrival":      s.get("actualArrival"),
        "actual_departure":    s.get("actualDeparture"),
        "delay_arrival_min":   _safe_float(s.get("delayArrival")),
        "delay_departure_min": _safe_float(s.get("delayDeparture")),
        "platform":            str(s.get("platform") or ""),
        "distance_km":         _safe_float(s.get("distance")),
    }


def _safe_float(v) -> float | None:
    """Returns float or None."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
