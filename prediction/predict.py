"""
RailSync AI — prediction/predict.py
Steps 10, 14: Recursive journey forecast + reusable prediction function
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "models", "saved")
DATA_DIR  = os.path.join(ROOT_DIR, "data")
VIZ_DIR   = os.path.join(ROOT_DIR, "visualization")
os.makedirs(VIZ_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Load model + supporting data
# ─────────────────────────────────────────────
_model     = None
_encoders  = None
_feat_cols = None
_meta      = None
_clean_df  = None   # full cleaned dataset for route lookups


def _load_resources():
    global _model, _encoders, _feat_cols, _meta, _clean_df
    if _encoders is None:
        try:
            _encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
        except Exception:
            _encoders = {}
        try:
            with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
                _meta = json.load(f)
            _feat_cols = _meta.get("feature_cols", [])
        except Exception:
            _feat_cols = []

    if _model is None:
        if HAS_XGB and os.path.exists(os.path.join(MODEL_DIR, "xgb_model.ubj")):
            try:
                _model = xgb.XGBRegressor()
                _model.load_model(os.path.join(MODEL_DIR, "xgb_model.ubj"))
            except Exception:
                _model = None
        if _model is None and os.path.exists(os.path.join(MODEL_DIR, "rf_model.joblib")):
            try:
                _model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
            except Exception:
                _model = None

    if _clean_df is None:
        _clean_df = pd.read_csv(os.path.join(DATA_DIR, "cleaned.csv"))
        _clean_df["journey_date"] = pd.to_datetime(_clean_df["journey_date"])
        for col in ["scheduled_arrival", "actual_arrival", "scheduled_departure", "actual_departure"]:
            _clean_df[col] = pd.to_datetime(_clean_df[col], errors="coerce")


def _safe_encode(le, value):
    known = set(le.classes_)
    if str(value) in known:
        return le.transform([str(value)])[0]
    return -1


def _get_route(train_number: str, journey_date_str: str) -> pd.DataFrame | None:
    """
    Look up the route for a train on a given date.
    Queries PostgreSQL railway_db first, then falls back to cleaned.csv dataset.
    Returns DataFrame with columns: station_code, station_sequence, distance_from_source, distance_to_destination, scheduled_arrival, scheduled_departure, departure_delay_min, arrival_delay_min.
    """
    _load_resources()
    tn = str(train_number).strip()

    # 1. Try PostgreSQL DB route
    try:
        from db_adapter import get_db_connection
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT ts.station_code, ts.stop_order AS station_sequence,
                       ts.day_number,
                       ts.arrival_time::text AS scheduled_arrival_str,
                       ts.departure_time::text AS scheduled_departure_str,
                       t.distance_km AS total_train_dist
                FROM train_stops ts
                LEFT JOIN trains t ON ts.train_number = t.train_number
                WHERE ts.train_number = %s
                ORDER BY ts.stop_order ASC;
            """, (tn,))
            rows = cur.fetchall()
            cur.close()
            conn.close()

            if rows:
                n_total = len(rows)
                tot_dist = float(rows[0]["total_train_dist"] or 450)
                records = []
                j_date = pd.to_datetime(journey_date_str)
                for r in rows:
                    seq = int(r["station_sequence"])
                    dist_src = round(((seq - 1) / max(n_total - 1, 1)) * tot_dist, 1)
                    dist_rem = round(tot_dist - dist_src, 1)
                    day_offset = max(int(r.get("day_number") or 1) - 1, 0)

                    sched_arr = None
                    if r["scheduled_arrival_str"]:
                        try:
                            t_str = r["scheduled_arrival_str"][:5]
                            sched_arr = pd.to_datetime(f"{journey_date_str} {t_str}") + pd.Timedelta(days=day_offset)
                        except Exception:
                            sched_arr = pd.NaT

                    sched_dep = None
                    if r["scheduled_departure_str"]:
                        try:
                            t_str = r["scheduled_departure_str"][:5]
                            sched_dep = pd.to_datetime(f"{journey_date_str} {t_str}") + pd.Timedelta(days=day_offset)
                        except Exception:
                            sched_dep = pd.NaT

                    records.append({
                        "train_number": tn,
                        "journey_date": j_date,
                        "station_code": r["station_code"],
                        "station_sequence": seq,
                        "distance_from_source": dist_src,
                        "distance_to_destination": dist_rem,
                        "scheduled_arrival": sched_arr,
                        "scheduled_departure": sched_dep,
                        "arrival_delay_min": np.nan,
                        "departure_delay_min": np.nan,
                    })
                return pd.DataFrame(records)
    except Exception as e:
        pass

    # 2. Fall back to cleaned.csv
    df = _clean_df
    target_date = pd.to_datetime(journey_date_str).date()

    mask = (df["train_number"].astype(str) == str(train_number)) & \
           (df["journey_date"].dt.date == target_date)
    route = df[mask].sort_values("station_sequence").reset_index(drop=True)
    if len(route) > 0:
        return route

    train_mask = df["train_number"].astype(str) == str(train_number)
    train_df = df[train_mask]
    if len(train_df) == 0:
        return None
    latest_date = train_df["journey_date"].max().date()
    mask2 = (df["train_number"].astype(str) == str(train_number)) & \
            (df["journey_date"].dt.date == latest_date)
    route = df[mask2].sort_values("station_sequence").reset_index(drop=True)
    return route if len(route) > 0 else None


def _build_historical_stats_for_train(train_number: str, journey_date_str: str) -> dict:
    """Build leakage-free historical stats from PostgreSQL database or cleaned.csv at prediction time."""
    tn = str(train_number).strip()

    # 1. Try querying PostgreSQL DB for rich historical station and section stats
    try:
        from db_adapter import get_db_connection
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            # Overall train stats
            cur.execute("""
                SELECT 
                    AVG(hso.arrival_delay_min) AS avg_delay,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY hso.arrival_delay_min) AS median_delay,
                    STDDEV(hso.arrival_delay_min) AS std_delay,
                    AVG(CASE WHEN hso.arrival_delay_min <= 5 THEN 1.0 ELSE 0.0 END) AS on_time_rate
                FROM historical_runs hr
                JOIN historical_station_observations hso ON hr.run_id = hso.run_id
                WHERE hr.train_number = %s;
            """, (tn,))
            t_row = cur.fetchone()

            # Station-specific stats
            cur.execute("""
                SELECT 
                    hso.station_code,
                    AVG(hso.arrival_delay_min) AS hist_avg_station_delay,
                    STDDEV(hso.arrival_delay_min) AS hist_std_station_delay
                FROM historical_runs hr
                JOIN historical_station_observations hso ON hr.run_id = hso.run_id
                WHERE hr.train_number = %s
                GROUP BY hso.station_code;
            """, (tn,))
            st_rows = cur.fetchall()
            station_stats_df = pd.DataFrame(st_rows) if st_rows else pd.DataFrame()

            cur.close()
            conn.close()

            if t_row and t_row["avg_delay"] is not None:
                return {
                    "hist_avg_train_delay":    float(t_row["avg_delay"]),
                    "hist_median_train_delay": float(t_row["median_delay"] or 5.0),
                    "hist_std_train_delay":    float(t_row["std_delay"] or 4.0),
                    "hist_on_time_rate_train": float(t_row["on_time_rate"] or 0.8),
                    "station_stats":           station_stats_df,
                    "section_df":              pd.DataFrame(),
                }
    except Exception:
        pass

    # 2. Fall back to cleaned.csv
    _load_resources()
    df = _clean_df.copy()
    journey_date = pd.to_datetime(journey_date_str)
    past_df = df[(df["train_number"].astype(str) == tn) &
                 (df["journey_date"] < journey_date)]

    if len(past_df) == 0:
        return {}

    hist_avg_train     = past_df["arrival_delay_min"].mean()
    hist_median_train  = past_df["arrival_delay_min"].median()
    hist_std_train     = past_df["arrival_delay_min"].std()
    hist_on_time_train = (past_df["arrival_delay_min"] <= 5).mean()

    station_stats = past_df.groupby("station_code")["arrival_delay_min"].agg(
        hist_avg_station_delay="mean",
        hist_std_station_delay="std"
    ).reset_index()

    past_df_sorted = past_df.sort_values(["journey_date", "station_sequence"])
    past_df_sorted["next_station_code"] = past_df_sorted.groupby("journey_date")["station_code"].shift(-1)
    past_df_sorted["delay_change"] = past_df_sorted.groupby("journey_date")["arrival_delay_min"].diff().shift(-1)
    section_df = past_df_sorted.dropna(subset=["next_station_code", "delay_change"])[["station_code", "next_station_code", "delay_change"]]

    return {
        "hist_avg_train_delay":    hist_avg_train,
        "hist_median_train_delay": hist_median_train,
        "hist_std_train_delay":    hist_std_train,
        "hist_on_time_rate_train": hist_on_time_train,
        "station_stats":           station_stats,
        "section_df":              section_df,
    }


def _build_feature_vector(
    train_number, station_code, next_station_code,
    current_arrival_delay, current_departure_delay,
    prev_delay, prev2_delay,
    scheduled_arrival, scheduled_departure,
    scheduled_next_arrival,
    distance_from_source, distance_remaining,
    next_distance_from_source,
    journey_date, hist_stats
) -> dict:
    """Assemble a single feature vector dict for prediction."""
    journey_date = pd.to_datetime(journey_date)

    sched_arr_hour  = scheduled_arrival.hour  if pd.notna(scheduled_arrival)  else np.nan
    sched_dep_hour  = scheduled_departure.hour if pd.notna(scheduled_departure) else np.nan
    day_of_week     = journey_date.dayofweek
    month           = journey_date.month

    total_dist = (distance_from_source + distance_remaining
                  if pd.notna(distance_from_source) and pd.notna(distance_remaining) else np.nan)
    progress = distance_from_source / total_dist if total_dist and total_dist > 0 else np.nan
    segment_distance = (next_distance_from_source - distance_from_source
                        if pd.notna(next_distance_from_source) else np.nan)

    delay_trend  = current_arrival_delay - prev_delay  if pd.notna(prev_delay) else np.nan
    delay_trend2 = prev_delay - prev2_delay             if pd.notna(prev_delay) and pd.notna(prev2_delay) else np.nan

    # Dwell time (not available at prediction time → NaN)
    dwell_time = np.nan

    sched_dwell = np.nan
    if pd.notna(scheduled_departure) and pd.notna(scheduled_arrival):
        sched_dwell = (scheduled_departure - scheduled_arrival).total_seconds() / 60

    sched_section_travel = np.nan
    if pd.notna(scheduled_next_arrival) and pd.notna(scheduled_departure):
        sched_section_travel = (scheduled_next_arrival - scheduled_departure).total_seconds() / 60

    # Historical stats
    h_avg_train     = hist_stats.get("hist_avg_train_delay",    np.nan)
    h_med_train     = hist_stats.get("hist_median_train_delay", np.nan)
    h_std_train     = hist_stats.get("hist_std_train_delay",    np.nan)
    h_on_time_train = hist_stats.get("hist_on_time_rate_train", np.nan)

    h_avg_station = h_std_station = np.nan
    if "station_stats" in hist_stats and len(hist_stats["station_stats"]) > 0:
        st = hist_stats["station_stats"][hist_stats["station_stats"]["station_code"] == station_code]
        if not st.empty:
            h_avg_station = st["hist_avg_station_delay"].values[0]
            h_std_station = st["hist_std_station_delay"].values[0]

    h_avg_section = h_std_section = np.nan
    if "section_df" in hist_stats and len(hist_stats["section_df"]) > 0:
        sc = hist_stats["section_df"][
            (hist_stats["section_df"]["station_code"] == station_code) &
            (hist_stats["section_df"]["next_station_code"] == next_station_code)
        ]
        if not sc.empty:
            h_avg_section = sc["delay_change"].mean()
            h_std_section = sc["delay_change"].std()

    train_enc   = _safe_encode(_encoders["train"],        str(train_number))
    station_enc = _safe_encode(_encoders["station"],      str(station_code))
    next_enc    = _safe_encode(_encoders["next_station"], str(next_station_code))

    return {
        "sched_arr_hour":          sched_arr_hour,
        "sched_dep_hour":          sched_dep_hour,
        "day_of_week":             day_of_week,
        "month":                   month,
        "distance_from_source":    distance_from_source,
        "distance_remaining":      distance_remaining,
        "progress_fraction":       progress,
        "segment_distance":        segment_distance,
        "current_arrival_delay":   current_arrival_delay,
        "current_departure_delay": current_departure_delay,
        "prev_station_delay":      prev_delay,
        "prev_2_station_delay":    prev2_delay,
        "delay_trend":             delay_trend,
        "delay_trend2":            delay_trend2,
        "dwell_time":              dwell_time,
        "sched_dwell_time":        sched_dwell,
        "sched_section_travel":    sched_section_travel,
        "hist_avg_train_delay":       h_avg_train,
        "hist_median_train_delay":    h_med_train,
        "hist_std_train_delay":       h_std_train,
        "hist_on_time_rate_train":    h_on_time_train,
        "hist_avg_station_delay":     h_avg_station,
        "hist_std_station_delay":     h_std_station,
        "hist_avg_section_delay_chg": h_avg_section,
        "hist_std_section_delay_chg": h_std_section,
        "train_number_enc":           train_enc,
        "station_code_enc":           station_enc,
        "next_station_code_enc":      next_enc,
    }


def _predict_single(fvec: dict) -> float:
    _load_resources()
    if _model is None:
        # Fallback to historical section average or 0.5 min delay accumulation
        h_sec = fvec.get("hist_avg_section_delay_chg")
        if h_sec is not None and not np.isnan(h_sec):
            return float(h_sec)
        return 0.5
    X = pd.DataFrame([{c: fvec.get(c, np.nan) for c in _feat_cols}])
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(0)
    return float(_model.predict(X)[0])


# ─────────────────────────────────────────────
# STEP 14: MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_train_delay(
    train_number: str,
    current_station: str,
    current_delay: float,
    journey_date: str,
) -> dict:
    """
    Recursively forecast delay across remaining stations.

    Parameters
    ----------
    train_number    : str   — e.g. "12601"
    current_station : str   — station code where current delay is observed, e.g. "AJJ"
    current_delay   : float — current observed arrival delay in minutes
    journey_date    : str   — ISO date string, e.g. "2026-08-15"

    Returns
    -------
    dict with keys:
      train_number, current_station, current_delay, forecast (list of dicts)
    """
    _load_resources()
    current_station = str(current_station).upper().strip()
    train_number    = str(train_number).strip()

    route = _get_route(train_number, journey_date)
    if route is None or len(route) == 0:
        return {"error": f"Train {train_number} not found in dataset"}

    # Find starting position
    route_stations = route["station_code"].tolist()
    if current_station not in route_stations:
        return {"error": f"Station {current_station} not found in route of train {train_number}"}

    start_idx = route_stations.index(current_station)
    remaining = route.iloc[start_idx:].reset_index(drop=True)

    if len(remaining) < 2:
        return {"error": "Current station is the final station — no remaining journey to forecast"}

    # Build historical stats
    hist_stats = _build_historical_stats_for_train(train_number, journey_date)

    # Recursive forecast
    forecast = []
    delays = [current_delay]          # delay at each station (index 0 = current_station)
    dep_delay = float(remaining.iloc[0]["departure_delay_min"]) \
                if pd.notna(remaining.iloc[0]["departure_delay_min"]) else current_delay

    for i in range(len(remaining) - 1):
        cur_row = remaining.iloc[i]
        nxt_row = remaining.iloc[i + 1]

        cur_delay_val = delays[i]
        prev_delay    = delays[i - 1] if i >= 1 else np.nan
        prev2_delay   = delays[i - 2] if i >= 2 else np.nan

        fvec = _build_feature_vector(
            train_number      = train_number,
            station_code      = cur_row["station_code"],
            next_station_code = nxt_row["station_code"],
            current_arrival_delay   = cur_delay_val,
            current_departure_delay = dep_delay,
            prev_delay   = prev_delay,
            prev2_delay  = prev2_delay,
            scheduled_arrival      = cur_row["scheduled_arrival"],
            scheduled_departure    = cur_row["scheduled_departure"],
            scheduled_next_arrival = nxt_row["scheduled_arrival"],
            distance_from_source  = cur_row["distance_from_source"],
            distance_remaining    = cur_row["distance_to_destination"],
            next_distance_from_source = nxt_row["distance_from_source"],
            journey_date = journey_date,
            hist_stats   = hist_stats,
        )

        pred_change = _predict_single(fvec)

        # Scale by hop distance if distance is present (model trained on ~40km inter-station blocks)
        cur_d = cur_row.get("distance_from_source")
        nxt_d = nxt_row.get("distance_from_source")
        if pd.notna(cur_d) and pd.notna(nxt_d) and abs(nxt_d - cur_d) > 0:
            hop_km = abs(nxt_d - cur_d)
            if hop_km < 40.0:
                pred_change = pred_change * (hop_km / 40.0)

        # Clamp extreme predictions (sanity constraint)
        pred_change = float(np.clip(pred_change, -60, 120))

        next_delay = cur_delay_val + pred_change
        delays.append(next_delay)
        dep_delay = next_delay  # best proxy for departure delay at next station

        # ETA calculation
        sched_arr = nxt_row["scheduled_arrival"]
        if pd.notna(sched_arr):
            eta = sched_arr + pd.Timedelta(minutes=next_delay)
        else:
            eta = None

        forecast.append({
            "station":                nxt_row["station_code"],
            "station_sequence":       int(nxt_row["station_sequence"]),
            "distance_from_source":   float(nxt_row["distance_from_source"])
                                      if pd.notna(nxt_row["distance_from_source"]) else None,
            "predicted_delay_change": round(pred_change, 2),
            "predicted_delay":        round(next_delay, 2),
            "scheduled_arrival":      str(sched_arr) if pd.notna(sched_arr) else None,
            "predicted_eta":          str(eta) if eta is not None else None,
        })

    return {
        "train_number":    train_number,
        "current_station": current_station,
        "current_delay":   current_delay,
        "journey_date":    journey_date,
        "forecast":        forecast,
    }


def print_forecast(result: dict) -> None:
    """Pretty-print the forecast result."""
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("\n" + "=" * 70)
    print(f"  RailSync AI — Journey Delay Forecast")
    print(f"  Train: {result['train_number']}   Date: {result['journey_date']}")
    print("=" * 70)
    print(f"  {'Station':<10} {'Sched Arrival':<22} {'Predicted ETA':<22} {'Delay':>8}  {'Change':>8}")
    print("-" * 70)
    print(f"  {'→ '+result['current_station']:<10} {'—':<22} {'—':<22} {result['current_delay']:>7.1f}m  {'(current)':>8}")
    for item in result["forecast"]:
        sched = item["scheduled_arrival"][:19] if item["scheduled_arrival"] else "N/A"
        eta   = item["predicted_eta"][:19]     if item["predicted_eta"]     else "N/A"
        chg   = item["predicted_delay_change"]
        chg_str = f"{'+' if chg >= 0 else ''}{chg:.1f}m"
        print(f"  {'  '+item['station']:<10} {sched:<22} {eta:<22} {item['predicted_delay']:>7.1f}m  {chg_str:>8}")
    print("=" * 70)


# ─────────────────────────────────────────────
# STEP 15: DELAY TRAJECTORY PLOT
# ─────────────────────────────────────────────
def plot_delay_trajectory(result: dict, actual_delays: list = None, save: bool = True) -> None:
    """
    Visualize predicted delay trajectory across the journey.
    Optionally overlay actual delays if provided.
    """
    if "error" in result:
        print(f"Cannot plot: {result['error']}")
        return

    stations  = [result["current_station"]] + [f["station"] for f in result["forecast"]]
    pred_delays = [result["current_delay"]] + [f["predicted_delay"] for f in result["forecast"]]

    fig, ax = plt.subplots(figsize=(max(10, len(stations) * 0.8), 5))

    # Color segments: red if delay increased, green if recovered
    for i in range(len(pred_delays) - 1):
        color = "#E84C4C" if pred_delays[i + 1] > pred_delays[i] else "#4CAF50"
        ax.plot([i, i + 1], [pred_delays[i], pred_delays[i + 1]],
                color=color, linewidth=2.5, alpha=0.9)
        ax.fill_between([i, i + 1], [pred_delays[i], pred_delays[i + 1]],
                        alpha=0.07, color=color)

    ax.scatter(range(len(pred_delays)), pred_delays,
               color=["#1565C0"] + ["#E84C4C" if d > pred_delays[j - 1] else "#4CAF50"
                                     for j, d in enumerate(pred_delays[1:], 1)],
               s=80, zorder=5, edgecolors="white", linewidths=1.5)

    # Annotate each station with delay
    for i, (s, d) in enumerate(zip(stations, pred_delays)):
        ax.annotate(f"{d:.1f}m", (i, d), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8.5,
                    color="#333333", fontweight="bold")

    # Actual delays overlay
    if actual_delays and len(actual_delays) == len(stations):
        ax.plot(range(len(actual_delays)), actual_delays,
                color="#FF6B00", linewidth=2, linestyle="--",
                marker="s", markersize=6, label="Actual delay", alpha=0.85)
        legend_items = [
            mpatches.Patch(color="#E84C4C", alpha=0.8, label="Delay accumulation"),
            mpatches.Patch(color="#4CAF50", alpha=0.8, label="Delay recovery"),
            plt.Line2D([0], [0], color="#FF6B00", linewidth=2,
                       linestyle="--", marker="s", markersize=6, label="Actual delay"),
        ]
        ax.legend(handles=legend_items, fontsize=9, loc="upper left")
    else:
        legend_items = [
            mpatches.Patch(color="#E84C4C", alpha=0.8, label="Delay accumulation"),
            mpatches.Patch(color="#4CAF50", alpha=0.8, label="Delay recovery"),
        ]
        ax.legend(handles=legend_items, fontsize=9, loc="upper left")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(stations)))
    ax.set_xticklabels(stations, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Predicted Delay (min)", fontsize=12)
    ax.set_xlabel("Station", fontsize=12)
    ax.set_title(
        f"RailSync AI — Train {result['train_number']} Delay Trajectory\n"
        f"From {result['current_station']} (delay = {result['current_delay']} min)  |  {result['journey_date']}",
        fontsize=13, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        fname = os.path.join(VIZ_DIR, f"delay_trajectory_{result['train_number']}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: {fname}")
    else:
        plt.show()


# ─────────────────────────────────────────────
# DEMO RUN (Step 10)
# ─────────────────────────────────────────────
def run_demo():
    _load_resources()
    print("\n" + "=" * 70)
    print("STEP 10 — RECURSIVE JOURNEY FORECAST DEMO")
    print("=" * 70)

    # Pick sample trains from the dataset
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample_journeys = (
        test_df.groupby(["train_number", "journey_date"])
        .size()
        .reset_index(name="n_stations")
    )
    sample_journeys = sample_journeys[sample_journeys["n_stations"] >= 5].head(3)

    for _, row in sample_journeys.iterrows():
        tn   = str(row["train_number"])
        jd   = str(row["journey_date"])[:10]
        route = _get_route(tn, jd)
        if route is None or len(route) < 3:
            continue

        # Start from 2nd station
        start_station = route.iloc[1]["station_code"]
        actual_start_delay = route.iloc[1]["arrival_delay_min"]
        if pd.isna(actual_start_delay):
            actual_start_delay = 0.0

        print(f"\n--- Demo: Train {tn}  |  Date: {jd}  |  Starting at: {start_station} ---")
        result = predict_train_delay(tn, start_station, float(actual_start_delay), jd)
        print_forecast(result)

        # Collect actual delays for overlay (using integer iloc)
        station_list = route["station_code"].tolist()
        if start_station in station_list:
            start_pos = station_list.index(start_station)
            route_remaining = route.iloc[start_pos:].reset_index(drop=True)
        else:
            route_remaining = pd.DataFrame()
        actuals = route_remaining["arrival_delay_min"].tolist()

        plot_delay_trajectory(result, actual_delays=actuals if len(actuals) == len(result["forecast"]) + 1 else None)

    return result


if __name__ == "__main__":
    run_demo()
