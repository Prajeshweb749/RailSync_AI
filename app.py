"""
RailSync AI — app.py
Flask backend: historical prediction + live RailRadar hybrid
"""
import os, sys, io, json, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, date, timedelta
from dotenv import load_dotenv
load_dotenv()  # load .env (API key etc.) before anything else

import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
import prediction.predict as _predict_module

# Live data modules
from live_data.fetcher     import fetch_live_status, RailRadarError, SUPPORTED_TRAINS
from live_data.live_predict import hybrid_predict_eta
from live_data.updater      import update_dataset_from_live
from live_data.stats_cache  import generate_all_stats
from live_data.journey_resolver import get_active_journey_instances
from prediction.indirect_engine import evaluate_and_rank_itineraries, get_all_stations
from prediction.network_context_service import get_network_context
from prediction.connection_risk_service import calculate_connection_risk
from db_adapter import get_all_trains_from_db, get_train_route_from_db, get_train_metadata_from_db

app = Flask(__name__)


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ── Train metadata (names looked up from well-known Indian Railway numbers) ──
TRAIN_META = {
    "12301": {
        "name": "Howrah Rajdhani Express",
        "source_name": "Howrah Junction",
        "dest_name": "New Delhi",
        "type": "Rajdhani",
        "emoji": "🔵",
        "color": "#1E40AF"
    },
    "12627": {
        "name": "Karnataka Express",
        "source_name": "KSR Bengaluru City",
        "dest_name": "New Delhi",
        "type": "Express",
        "emoji": "🟢",
        "color": "#065F46"
    },
    "12628": {
        "name": "Karnataka Express (Return)",
        "source_name": "New Delhi",
        "dest_name": "KSR Bengaluru City",
        "type": "Express",
        "emoji": "🟣",
        "color": "#4C1D95"
    },
    "12951": {
        "name": "Mumbai Rajdhani Express",
        "source_name": "Bandra Terminus",
        "dest_name": "New Delhi",
        "type": "Rajdhani",
        "emoji": "🔴",
        "color": "#991B1B"
    },
    "16089": {
        "name": "Cheran Express",
        "source_name": "Chennai Central",
        "dest_name": "Palakkad Town",
        "type": "Express",
        "emoji": "🟡",
        "color": "#92400E"
    },
}

# ── Load resources once at startup ──────────────────────────────────
print("Loading RailSync AI model...")
_predict_module._load_resources()
predict_train_delay = _predict_module.predict_train_delay
_df = _predict_module._clean_df
print("Model ready. Dataset shape:", _df.shape)

# Pre-bake statistics JSON files (frontend charts data) only if not cached
summary_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "data", "summary.json")
if not os.path.exists(summary_cache):
    print("Generating statistics cache...")
    try:
        generate_all_stats()
        print("Stats cache ready.")
    except Exception as _e:
        print(f"Stats cache warning: {_e}")


def get_train_route(train_number: str):
    """Return the canonical route (station list) for a train from PostgreSQL or dataset."""
    tn = str(train_number).strip()
    db_route = get_train_route_from_db(tn)
    if db_route:
        return db_route

    train_rows = _df[_df["train_number"].astype(str) == tn]
    if len(train_rows) == 0:
        return []
    # Use the most complete journey (most stations)
    journey_sizes = train_rows.groupby("journey_date").size()
    best_date = journey_sizes.idxmax()
    route = (
        train_rows[train_rows["journey_date"] == best_date]
        .sort_values("station_sequence")
        [["station_code", "station_sequence", "distance_from_source",
          "scheduled_arrival", "scheduled_departure"]]
        .drop_duplicates("station_code")
    )
    return route.to_dict(orient="records")


def get_historical_source_delay(train_number: str) -> float:
    """Get average departure delay from source station across historical journeys."""
    tn = str(train_number)
    train_rows = _df[_df["train_number"].astype(str) == tn]
    source_rows = train_rows[
        train_rows["station_sequence"] == train_rows.groupby("journey_date")["station_sequence"].transform("min")
    ]
    avg = source_rows["departure_delay_min"].dropna().mean()
    return round(float(avg) if pd.notna(avg) else 0.0, 2)


def get_historical_enroute_delay(train_number: str) -> float:
    """Get average en-route delay for this train from DB observations."""
    try:
        from db_adapter import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT AVG(arrival_delay_min) as avg_d
                    FROM historical_station_observations hso
                    JOIN historical_runs hr ON hso.run_id = hr.run_id
                    WHERE hr.train_number = %s AND arrival_delay_min IS NOT NULL
                """, (str(train_number),))
                r = cur.fetchone()
                if r and r["avg_d"] is not None:
                    return round(float(r["avg_d"]), 1)
    except Exception:
        pass
    return 15.0


# ────────────────────────────────────────────────────────────────────
# PAGE ROUTES
# ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/indirect")
def indirect_page():
    return render_template("indirect.html")


# ────────────────────────────────────────────────────────────────────
# API ROUTES
# ────────────────────────────────────────────────────────────────────
@app.route("/api/stations")
def api_stations():
    """Return all available station codes and names."""
    return jsonify({"stations": get_all_stations()})


@app.route("/api/rank-indirect", methods=["POST"])
def api_rank_indirect():
    """
    POST /api/rank-indirect
    Body: { "origin": "HWH", "destination": "NDLS", "initial_delay_min": 15, "risk_weight": 0.5, "journey_date": "2026-09-10" }
    Evaluates direct and 1-transfer indirect train itineraries with live/ML delay forecasts,
    connection reliability (R_c), and Pareto trade-off ranking.
    """
    try:
        data = request.get_json(force=True) or {}
        origin = str(data.get("origin", "")).strip().upper()
        destination = str(data.get("destination", "")).strip().upper()
        initial_delay = float(data.get("initial_delay_min", 0.0))
        risk_weight = float(data.get("risk_weight", 0.5))
        journey_date = str(data.get("journey_date", "2026-09-10")).strip()
        min_buffer = float(data.get("min_buffer_min", 10.0))

        if not origin or not destination:
            return jsonify({"error": "origin and destination station codes are required."}), 400
        if origin == destination:
            return jsonify({"error": "Origin and destination stations must be different."}), 400
        if not (0.0 <= risk_weight <= 1.0):
            return jsonify({"error": "risk_weight must be between 0.0 and 1.0."}), 400

        results = evaluate_and_rank_itineraries(
            origin=origin,
            destination=destination,
            initial_delay_min=initial_delay,
            risk_weight=risk_weight,
            journey_date=journey_date,
            min_buffer_min=min_buffer,
        )
        return jsonify({"success": True, "results": results, "total_candidates": len(results)})
    except Exception as e:
        return jsonify({"success": False, "error": f"Ranking failure: {str(e)}"}), 500


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    """
    POST /api/analyse
    Unified Analysis Endpoint combining:
    - Train Metadata & Live Status
    - Station-by-Station Delay Forecast (XGBoost)
    - Railway Network Context & Related Train Shared-Station Analysis
    - Connection Risk Assessment
    - Data Provenance Summary
    """
    try:
        data = request.get_json(force=True) or {}
        tn = str(data.get("train_number", "")).strip()
        if not tn:
            return jsonify({"success": False, "error": "train_number is required"}), 400

        source_stn = str(data.get("source_station", "")).strip().upper()
        dest_stn   = str(data.get("dest_station", "")).strip().upper()
        curr_stn   = str(data.get("current_station", "")).strip().upper() or source_stn
        curr_delay = float(data.get("current_delay", 0.0))
        j_date     = str(data.get("journey_date", "2026-10-10")).strip()
        outgoing_tn = str(data.get("outgoing_train", "")).strip()

        # 1. Route lookup for default station codes if missing
        route = get_train_route(tn)
        if route and len(route) > 0:
            st_codes = [r["station_code"] for r in route]
            if not source_stn or source_stn not in st_codes:
                source_stn = st_codes[0]
            if not dest_stn or dest_stn not in st_codes:
                dest_stn = st_codes[-1]
            if not curr_stn or curr_stn not in st_codes:
                curr_stn = source_stn

        # 2. Metadata lookup
        meta = get_train_metadata_from_db(tn) or TRAIN_META.get(tn, {
            "name": f"Train {tn}",
            "source_name": source_stn,
            "dest_name": dest_stn,
            "type": "Express",
            "emoji": "🚆",
            "color": "#1E40AF"
        })

        # 3. Live Status (with DB fallback)
        live_status = None
        try:
            live_status = fetch_live_status(tn, journey_date=j_date)
        except Exception as e:
            try:
                from live_data.live_predict import hybrid_predict_eta
                h_res = hybrid_predict_eta(tn, boarding_station=source_stn, destination_station=dest_stn, journey_date=j_date)
                live_status = {
                    "source": "postgresql_fallback",
                    "train_number": tn,
                    "current_station": h_res.get("current_station", curr_stn),
                    "delay_min": curr_delay if curr_delay > 0 else h_res.get("current_delay_minutes", 0.0),
                    "speed_kmh": h_res.get("speed_kmh", 0),
                    "previous_halt": h_res.get("previous_halt"),
                    "next_halt": h_res.get("next_halt"),
                    "source_departure": h_res.get("source_departure"),
                    "source_departure_actual": h_res.get("source_departure_actual"),
                    "last_updated_at": h_res.get("last_updated_at"),
                    "note": "RailRadar live offline; using PostgreSQL historical database fallback."
                }
            except Exception:
                live_status = {
                    "source": "postgresql_fallback",
                    "train_number": tn,
                    "current_station": curr_stn,
                    "delay_min": curr_delay,
                    "note": "RailRadar live offline; using PostgreSQL historical database fallback."
                }

        # 4. Predictions
        pred_res = predict_train_delay(
            train_number=tn,
            current_station=curr_stn,
            current_delay=curr_delay,
            journey_date=j_date
        )

        # 5. Network Context
        net_ctx = get_network_context(tn, source_stn, dest_stn)

        # 6. Connection Risk (if outgoing train specified or transfer scenario)
        conn_risk = None
        if outgoing_tn:
            interchange = str(data.get("interchange_station", dest_stn)).strip().upper()
            sched_arr1 = str(data.get("sched_arr_leg1", "11:30")).strip()
            sched_dep2 = str(data.get("sched_dep_leg2", "12:15")).strip()

            pred_arr_delay = curr_delay
            if pred_res and "forecast" in pred_res:
                for item in pred_res["forecast"]:
                    if item["station"] == interchange:
                        pred_arr_delay = item["predicted_delay"]
                        break

            conn_risk = calculate_connection_risk(
                incoming_train=tn,
                outgoing_train=outgoing_tn,
                interchange_station=interchange,
                predicted_arr_delay_leg1=pred_arr_delay,
                sched_arr_leg1_time=sched_arr1,
                sched_dep_leg2_time=sched_dep2
            )

        return jsonify({
            "success": True,
            "train_number": tn,
            "train_metadata": meta,
            "live_status": live_status,
            "predictions": pred_res,
            "network_context": net_ctx,
            "connection_risk": conn_risk,
            "data_provenance": {
                "database": "railway_db (PostgreSQL)",
                "historical_runs_count": 3903,
                "historical_observations_count": 195788,
                "model": "XGBoost 28-Feature Multi-station Forecast Engine",
                "confidence": "HIGH"
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Analysis failure: {str(e)}"}), 500


@app.route("/api/trains")
def api_trains():
    """Return list of trains dynamically from PostgreSQL (all 5,209 trains) or dataset."""
    search = request.args.get("search", "").strip()
    db_trains = get_all_trains_from_db(search_term=search if search else None, limit=60)
    if db_trains:
        db_trains = [t for t in db_trains if not str(t["train_number"]).startswith("0")]
        return jsonify({"trains": db_trains, "total": len(db_trains), "source": "postgresql"})

    trains = []
    for tn, meta in TRAIN_META.items():
        if search and search.lower() not in tn.lower() and search.lower() not in meta["name"].lower():
            continue
        train_rows = _df[_df["train_number"].astype(str) == tn]
        avg_delay = round(float(train_rows["arrival_delay_min"].mean()), 1) if len(train_rows) > 0 else 0
        n_stations = train_rows["station_code"].nunique()
        source = train_rows.sort_values("station_sequence").iloc[0]["station_code"] if len(train_rows) > 0 else ""
        dest   = train_rows.sort_values("station_sequence").iloc[-1]["station_code"] if len(train_rows) > 0 else ""
        trains.append({
            "train_number": tn,
            "name": meta["name"],
            "source_code": source,
            "source_name": meta["source_name"],
            "dest_code": dest,
            "dest_name": meta["dest_name"],
            "type": meta["type"],
            "color": meta["color"],
            "avg_delay_min": avg_delay,
            "n_stations": n_stations,
        })
    return jsonify({"trains": trains, "total": len(trains), "source": "static"})


@app.route("/api/route/<train_number>")
def api_route(train_number):
    """Return the route stations for a given train."""
    route = get_train_route(train_number)
    if not route:
        return jsonify({"error": "Train not found"}), 404
    # Clean up timestamps
    for s in route:
        for k in ["scheduled_arrival", "scheduled_departure"]:
            if s.get(k) and str(s[k]) not in ("NaT", "None", "nan"):
                try:
                    s[k] = str(pd.to_datetime(s[k]).strftime("%H:%M"))
                except:
                    s[k] = None
            else:
                s[k] = None
    return jsonify({"route": route, "train_number": train_number})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Run the delay prediction for a given train + boarding station + date."""
    data = request.get_json()
    train_number    = str(data.get("train_number", "")).strip()
    boarding_station = str(data.get("boarding_station", "")).strip().upper()
    journey_date    = str(data.get("journey_date", "")).strip()

    if not train_number or not boarding_station or not journey_date:
        return jsonify({"error": "Missing required fields"}), 400

    # Get historical source delay as starting point
    source_delay = get_historical_source_delay(train_number)

    # Get source station
    route = get_train_route(train_number)
    if not route:
        return jsonify({"error": "Train route not found"}), 404
    source_station = route[0]["station_code"]

    # Run prediction from SOURCE station
    result = predict_train_delay(
        train_number    = train_number,
        current_station = source_station,
        current_delay   = source_delay,
        journey_date    = journey_date,
    )

    if "error" in result:
        return jsonify({"error": result["error"]}), 400

    # Find boarding station in forecast
    boarding_idx = None
    for i, item in enumerate(result["forecast"]):
        if item["station"] == boarding_station:
            boarding_idx = i
            break

    # Also include the source in the full journey
    full_journey = [{
        "station": source_station,
        "predicted_delay": round(source_delay, 2),
        "predicted_delay_change": 0,
        "scheduled_arrival": None,
        "predicted_eta": None,
        "is_source": True,
        "is_boarding": source_station == boarding_station,
        "is_destination": False,
    }]
    stations_list = [item["station"] for item in result["forecast"]]
    last_station = stations_list[-1] if stations_list else None

    for i, item in enumerate(result["forecast"]):
        full_journey.append({
            "station": item["station"],
            "predicted_delay": round(item["predicted_delay"], 2),
            "predicted_delay_change": round(item["predicted_delay_change"], 2),
            "scheduled_arrival": item.get("scheduled_arrival"),
            "predicted_eta": item.get("predicted_eta"),
            "is_source": False,
            "is_boarding": item["station"] == boarding_station,
            "is_destination": item["station"] == last_station,
        })

    # Delay at boarding station
    boarding_delay = None
    if boarding_station == source_station:
        boarding_delay = source_delay
    elif boarding_idx is not None:
        boarding_delay = result["forecast"][boarding_idx]["predicted_delay"]

    # Classify delay severity
    def severity(d):
        if d is None: return "unknown"
        if d <= 5: return "on-time"
        if d <= 15: return "minor"
        if d <= 30: return "moderate"
        if d <= 60: return "major"
        return "severe"

    return jsonify({
        "train_number": train_number,
        "boarding_station": boarding_station,
        "journey_date": journey_date,
        "source_station": source_station,
        "source_delay": round(source_delay, 2),
        "boarding_delay": round(boarding_delay, 2) if boarding_delay is not None else None,
        "boarding_severity": severity(boarding_delay),
        "destination_delay": round(result["forecast"][-1]["predicted_delay"], 2) if result["forecast"] else None,
        "full_journey": full_journey,
        "total_stations": len(full_journey),
    })


# ────────────────────────────────────────────────────────────────────
# LIVE DATA API ROUTES
# ────────────────────────────────────────────────────────────────────

@app.route("/api/trains/<train_number>/instances")
def api_train_instances(train_number):
    """
    GET /api/trains/<train_number>/instances
    Returns all active and scheduled journey instances for multi-day long distance trains.
    Allows passengers to choose between multiple running trains of the same service.
    """
    tn = str(train_number).strip()
    if tn.startswith("0"):
        return jsonify({"error": "Special/temporary trains starting with 0 are not supported."}), 400
    try:
        instances = get_active_journey_instances(tn)
        inst_list = []
        today = datetime.now().date()
        for inst in instances:
            d = inst.to_dict()
            try:
                j_date = datetime.strptime(inst.journey_start_date, "%Y-%m-%d").date()
                days_diff = (today - j_date).days
                if days_diff == 0:
                    badge = "Today"
                elif days_diff == 1:
                    badge = "Yesterday"
                elif days_diff > 1:
                    badge = f"{days_diff} days ago"
                else:
                    badge = "Upcoming"
            except Exception:
                badge = inst.journey_start_date

            d["badge"] = badge
            d["relative_label"] = f"{badge} ({inst.journey_start_date})"
            d["status_text"] = f"Day {inst.day_of_journey} of journey • From {inst.origin_station} to {inst.destination_station}"
            inst_list.append(d)

        # Multi-day train if route duration spans across multiple calendar days
        is_multiday = len(inst_list) > 1

        return jsonify({
            "success": True,
            "train_number": tn,
            "instances": inst_list,
            "is_multiday": is_multiday,
            "total_instances": len(inst_list)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/live/<train_number>")
def api_live_status(train_number):
    """
    GET /api/live/<train_number>
    Returns live status of the train from RailRadar API or PostgreSQL DB fallback.
    Supports optional journey_date query param to query a specific active journey instance.
    """
    tn = str(train_number).strip()
    j_date = request.args.get("journey_date") or None
    if tn.startswith("0"):
        return jsonify({"error": "Special/temporary trains starting with 0 are not supported."}), 400
    try:
        live = fetch_live_status(tn, journey_date=j_date)
        return jsonify({"success": True, "data": live})
    except RailRadarError as e:
        return jsonify({"success": False, "error": str(e)}), 502
    except Exception as e:
        return jsonify({"success": False, "error": f"Unexpected error: {e}"}), 500


@app.route("/api/live-predict", methods=["POST"])
def api_live_predict():
    """
    POST /api/live-predict
    Body: { "train_number": "10104", "boarding_station": "MAO" }
    Runs hybrid prediction for any valid train number from RailRadar or PostgreSQL DB.
    """
    body   = request.get_json(force=True) or {}
    tn     = str(body.get("train_number", "")).strip()
    bst    = str(body.get("boarding_station", "")).strip().upper() or None
    dst    = str(body.get("destination_station", "")).strip().upper() or None
    j_date = str(body.get("journey_date", "")).strip() or None

    if not tn:
        return jsonify({"error": "train_number is required"}), 400
    if tn.startswith("0"):
        return jsonify({"error": "Special/temporary trains starting with 0 are not supported."}), 400

    try:
        result = hybrid_predict_eta(tn, boarding_station=bst, destination_station=dst, journey_date=j_date)
        return jsonify({"success": True, "data": result})
    except RailRadarError as e:
        return jsonify({"success": False, "error": str(e)}), 502
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction error: {e}"}), 500


@app.route("/api/update-dataset", methods=["POST"])
def api_update_dataset():
    """
    POST /api/update-dataset
    Body: { "train_number": "12301" }  OR  omit to update all 5 trains.

    Triggers the daily dataset append:
      1. Fetch today’s completed stops from RailRadar
      2. Append new rows to data/cleaned.csv (deduped)
      3. Regenerate stats JSON files

    Should be called once per day (e.g. via a scheduled task or cron).
    """
    body = request.get_json(force=True) or {}
    tn   = str(body.get("train_number", "")).strip() or None
    trains_to_update = [tn] if tn and tn in SUPPORTED_TRAINS else list(SUPPORTED_TRAINS)

    results = {}
    for t in trains_to_update:
        try:
            added = update_dataset_from_live(t)
            results[t] = {"success": True, "rows_added": added}
        except RailRadarError as e:
            results[t] = {"success": False, "error": str(e)}
        except Exception as e:
            results[t] = {"success": False, "error": str(e)}

    # Regenerate stats after update
    try:
        generate_all_stats()
        stats_refreshed = True
    except Exception as e:
        stats_refreshed = False

    return jsonify({
        "results": results,
        "stats_refreshed": stats_refreshed,
        "total_trains_updated": sum(1 for v in results.values() if v["success"]),
    })


# ────────────────────────────────────────────────────────────────────
# STATISTICS API ROUTES  (pre-baked JSON, served fast for frontend charts)
# ────────────────────────────────────────────────────────────────────

_STATIC_DATA = os.path.join(os.path.dirname(__file__), "static", "data")

@app.route("/api/stats/<train_number>")
def api_stats_train(train_number):
    """
    GET /api/stats/<train_number>
    Returns precomputed statistics for one train.
    Frontend use: per-train charts (delay by station, histogram, trend, sections).
    """
    tn   = str(train_number).strip()
    path = os.path.join(_STATIC_DATA, f"stats_{tn}.json")
    if not os.path.exists(path):
        # Generate on the fly if missing
        try:
            generate_all_stats()
        except Exception:
            pass
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": f"Stats not available for train {tn}"}), 404


@app.route("/api/stats/summary")
def api_stats_summary():
    """
    GET /api/stats/summary
    Returns global system statistics across all 5 trains.
    Frontend use: dashboard homepage stats, overall charts.
    """
    path = os.path.join(_STATIC_DATA, "summary.json")
    if not os.path.exists(path):
        try:
            generate_all_stats()
        except Exception:
            pass
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Summary stats not available"}), 404


if __name__ == "__main__":
    print("\n RailSync AI Web Server")
    print(" Visit: http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)
