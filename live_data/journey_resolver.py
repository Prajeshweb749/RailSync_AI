"""
live_data/journey_resolver.py
=============================
Temporal Data Association & Journey-Instance Resolution Engine for RailSync AI.

Principle:
A train number (e.g. '13006') identifies a scheduled service.
A journey instance identifies a specific dated run of that service:
    journey_instance_id = f"{train_number}_{journey_start_date}" (e.g. '13006_2026-10-10')

Responsibilities:
1. Resolves ambiguous observations for long-distance multi-day trains operating parallel active runs.
2. Accurately maps observations to the correct origin departure date, accounting for:
   - Day offsets (day_number 1, 2, 3...)
   - Midnight crossings
   - Delayed departures vs consecutive daily runs
3. Enforces journey-instance isolation across ingestion, prediction, and API layers.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import Optional, List, Dict, Any
import logging

from db_adapter import get_db_connection

logger = logging.getLogger("journey_resolver")


@dataclass
class JourneyInstance:
    journey_instance_id: str
    train_number: str
    journey_start_date: str          # YYYY-MM-DD representing origin departure date
    origin_station: str
    destination_station: str
    scheduled_origin_departure: Optional[datetime] = None
    scheduled_destination_arrival: Optional[datetime] = None
    day_of_journey: int = 1          # 1-indexed day of journey for current observation
    current_station: Optional[str] = None
    current_delay_min: float = 0.0
    is_active: bool = True
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    candidate_instances: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "journey_instance_id": self.journey_instance_id,
            "train_number": self.train_number,
            "journey_start_date": self.journey_start_date,
            "origin_station": self.origin_station,
            "destination_station": self.destination_station,
            "scheduled_origin_departure": self.scheduled_origin_departure.isoformat() if self.scheduled_origin_departure else None,
            "scheduled_destination_arrival": self.scheduled_destination_arrival.isoformat() if self.scheduled_destination_arrival else None,
            "day_of_journey": self.day_of_journey,
            "current_station": self.current_station,
            "current_delay_min": self.current_delay_min,
            "is_active": self.is_active,
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_reason": self.ambiguity_reason,
            "candidate_instances": self.candidate_instances,
            "confidence": self.confidence,
        }


def get_train_timetable_schedule(train_number: str) -> Dict[str, Any]:
    """Fetches origin, destination, day offsets and timings for a train from PostgreSQL."""
    conn = get_db_connection()
    if not conn:
        return {}

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT ts.stop_order, ts.station_code, ts.day_number, 
                   ts.arrival_time::text as arr_str, ts.departure_time::text as dep_str,
                   t.from_station_code, t.to_station_code, t.duration_minutes
            FROM train_stops ts
            JOIN trains t ON ts.train_number = t.train_number
            WHERE ts.train_number = %s
            ORDER BY ts.stop_order ASC;
        """, (str(train_number).strip(),))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return {}

        stops_by_code = {}
        for r in rows:
            stops_by_code[r["station_code"]] = {
                "stop_order": r["stop_order"],
                "day_number": r["day_number"] or 1,
                "arr_str": r["arr_str"][:5] if r["arr_str"] else None,
                "dep_str": r["dep_str"][:5] if r["dep_str"] else None,
            }

        first_stop = rows[0]
        last_stop = rows[-1]

        return {
            "train_number": str(train_number).strip(),
            "origin": first_stop["from_station_code"] or first_stop["station_code"],
            "destination": last_stop["to_station_code"] or last_stop["station_code"],
            "origin_dep_str": first_stop["dep_str"][:5] if first_stop["dep_str"] else "08:00",
            "dest_arr_str": last_stop["arr_str"][:5] if last_stop["arr_str"] else "18:00",
            "total_days": last_stop["day_number"] or 1,
            "duration_minutes": first_stop.get("duration_minutes") or 1440,
            "stops": stops_by_code,
        }
    except Exception as e:
        logger.warning(f"Error reading timetable for train {train_number}: {e}")
        return {}


def get_active_journey_instances(train_number: str, reference_time: Optional[datetime] = None) -> List[JourneyInstance]:
    """
    Identifies all currently active or scheduled daily journey instances for a train.
    For a train spanning 3 calendar days (e.g. 13006), up to 3 instances may be active/valid.
    """
    ref_time = reference_time or datetime.now()
    ref_date = ref_time.date()

    tt = get_train_timetable_schedule(train_number)
    if not tt:
        # Fallback to single daily instance
        inst_id = f"{train_number}_{ref_date.isoformat()}"
        return [JourneyInstance(
            journey_instance_id=inst_id,
            train_number=train_number,
            journey_start_date=ref_date.isoformat(),
            origin_station="ORIGIN",
            destination_station="DEST"
        )]

    total_days = tt.get("total_days", 1)
    dep_h, dep_m = map(int, tt["origin_dep_str"].split(":")[:2])
    arr_h, arr_m = map(int, tt["dest_arr_str"].split(":")[:2])

    instances = []
    # Check start dates over the span of the train's journey duration
    for days_back in range(total_days + 1):
        start_date = ref_date - timedelta(days=days_back)
        sched_dep = datetime(start_date.year, start_date.month, start_date.day, dep_h, dep_m)
        dest_date = start_date + timedelta(days=total_days - 1)
        sched_arr = datetime(dest_date.year, dest_date.month, dest_date.day, arr_h, arr_m)

        # Allow 6-hour pre-departure and 12-hour post-arrival operational window
        window_start = sched_dep - timedelta(hours=6)
        window_end = sched_arr + timedelta(hours=14)

        is_active = (window_start <= ref_time <= window_end)
        day_of_journey = max(1, min((ref_time.date() - start_date).days + 1, total_days))

        inst_id = f"{train_number}_{start_date.isoformat()}"
        instances.append(JourneyInstance(
            journey_instance_id=inst_id,
            train_number=train_number,
            journey_start_date=start_date.isoformat(),
            origin_station=tt["origin"],
            destination_station=tt["destination"],
            scheduled_origin_departure=sched_dep,
            scheduled_destination_arrival=sched_arr,
            day_of_journey=day_of_journey,
            is_active=is_active,
        ))

    return instances


def resolve_journey_instance(
    train_number: str,
    observation_station: Optional[str] = None,
    observation_time: Optional[datetime] = None,
    journey_date: Optional[str] = None,
    live_status: Optional[Dict[str, Any]] = None,
) -> JourneyInstance:
    """
    Core Journey Resolver: Disambiguates and associates an observation with the correct journey instance.
    
    Resolution hierarchy:
    1. Explicit journey_date or live_status['startDate'] from provider API.
    2. Station sequence & day offset evidence (e.g. Lucknow is Day 2 -> started yesterday).
    3. Observation timestamp vs timetable schedules.
    4. Ambiguity flag when two candidate runs could both plausibly occupy the observation window.
    """
    tn = str(train_number).strip()
    obs_time = observation_time or datetime.now()
    tt = get_train_timetable_schedule(tn)

    origin = tt.get("origin", "ORIGIN") if tt else "ORIGIN"
    dest = tt.get("destination", "DEST") if tt else "DEST"

    # Case 1: Explicit journey_date passed by client or payload
    if journey_date and str(journey_date).strip():
        clean_date = str(journey_date).strip()
        # Handle format: '13006_2026-10-10' or '2026-10-10'
        if "_" in clean_date:
            clean_date = clean_date.split("_")[-1]
        
        inst_id = f"{tn}_{clean_date}"
        total_days = tt.get("total_days", 1) if tt else 1
        try:
            start_d_obj = datetime.strptime(clean_date, "%Y-%m-%d").date()
            day_of_journey = max(1, min((obs_time.date() - start_d_obj).days + 1, total_days))
        except Exception:
            day_of_journey = 1

        return JourneyInstance(
            journey_instance_id=inst_id,
            train_number=tn,
            journey_start_date=clean_date,
            origin_station=origin,
            destination_station=dest,
            day_of_journey=day_of_journey,
            is_active=True,
            confidence=1.0,
            is_ambiguous=False,
        )

    # Case 2: Live status from RailRadar provides canonical startDate
    if live_status and live_status.get("startDate"):
        start_d = str(live_status["startDate"]).strip()
        inst_id = f"{tn}_{start_d}"
        curr_stn = (live_status.get("currentLocation") or {}).get("stationCode") or live_status.get("current_station_code")
        curr_del = float(live_status.get("delayMinutes") or live_status.get("current_delay_minutes") or 0.0)

        return JourneyInstance(
            journey_instance_id=inst_id,
            train_number=tn,
            journey_start_date=start_d,
            origin_station=origin,
            destination_station=dest,
            current_station=curr_stn,
            current_delay_min=curr_del,
            confidence=0.98,
            is_ambiguous=False,
        )

    # Case 3: Disambiguation using observation station and timestamp
    if observation_station and tt and "stops" in tt:
        stn = observation_station.strip().upper()
        stop_info = tt["stops"].get(stn)
        if stop_info:
            day_num = stop_info["day_number"]
            # A station visited on day_num = 2 on obs_date departed on (obs_date - (day_num - 1))
            inferred_start_date = obs_time.date() - timedelta(days=day_num - 1)
            inst_id = f"{tn}_{inferred_start_date.isoformat()}"

            # Check if this could be an extreme delay (e.g. yesterday's train running 18+ hours late)
            candidate_ids = [inst_id]
            earlier_start = inferred_start_date - timedelta(days=1)
            candidate_ids.append(f"{tn}_{earlier_start.isoformat()}")

            return JourneyInstance(
                journey_instance_id=inst_id,
                train_number=tn,
                journey_start_date=inferred_start_date.isoformat(),
                origin_station=origin,
                destination_station=dest,
                day_of_journey=day_num,
                current_station=stn,
                confidence=0.95,
                is_ambiguous=False,
            )

    # Case 4: Ambiguous / Fallback — evaluate all active instances
    active_instances = get_active_journey_instances(tn, reference_time=obs_time)
    if len(active_instances) == 1:
        return active_instances[0]
    elif len(active_instances) > 1:
        # Multiple candidate journeys active simultaneously!
        candidate_ids = [inst.journey_instance_id for inst in active_instances]
        primary = active_instances[0]
        primary.is_ambiguous = True
        primary.ambiguity_reason = f"Multiple active journey instances ({', '.join(candidate_ids)}) running concurrently."
        primary.candidate_instances = candidate_ids
        primary.confidence = 0.50
        return primary

    # Default fallback
    def_date = obs_time.date().isoformat()
    return JourneyInstance(
        journey_instance_id=f"{tn}_{def_date}",
        train_number=tn,
        journey_start_date=def_date,
        origin_station=origin,
        destination_station=dest,
        confidence=0.70,
    )
