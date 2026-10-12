"""
live_data/stats_cache.py
========================
Precomputes per-train and system-wide statistics from cleaned.csv
and writes them to static/data/ as JSON files ready for the frontend.

Generates:
  static/data/stats_<train_number>.json  — per-train stats
  static/data/summary.json              — global system summary

Run manually or call generate_all_stats() from app startup.

Frontend chart data format notes
─────────────────────────────────
delay_by_station     → horizontal bar chart / heatmap
delay_by_day         → grouped bar chart
delay_over_time      → line chart (trend)
delay_distribution   → histogram
worst_sections       → ranked bar chart (delay added)
best_sections        → ranked bar chart (delay recovered)
"""

import os
import json
import numpy as np
import pandas as pd

_ROOT      = os.path.dirname(os.path.dirname(__file__))
DATA_DIR   = os.path.join(_ROOT, "data")
STATIC_DIR = os.path.join(_ROOT, "static", "data")
CLEANED    = os.path.join(DATA_DIR, "cleaned.csv")
META_PATH  = os.path.join(_ROOT, "models", "saved", "metadata.json")

TRAIN_NAMES = {
    "12301": "Howrah Rajdhani Express",
    "12627": "Karnataka Express",
    "12628": "Karnataka Express (Return)",
    "12951": "Mumbai Rajdhani Express",
    "16089": "Cheran Express",
}

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _ensure_static_dir():
    os.makedirs(STATIC_DIR, exist_ok=True)


def _load_cleaned() -> pd.DataFrame:
    df = pd.read_csv(CLEANED, low_memory=False)
    df["journey_date"]      = pd.to_datetime(df["journey_date"], errors="coerce")
    df["arrival_delay_min"] = pd.to_numeric(df["arrival_delay_min"], errors="coerce")
    df["departure_delay_min"] = pd.to_numeric(df["departure_delay_min"], errors="coerce")
    df["train_number"]      = df["train_number"].astype(str)
    return df


def _load_meta() -> dict:
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────
# Per-train statistics
# ─────────────────────────────────────────────────────────────────
def compute_train_stats(df: pd.DataFrame, train_number: str) -> dict:
    tdf = df[df["train_number"] == train_number].copy()
    if tdf.empty:
        return {"error": f"No data for train {train_number}"}

    # ── Delay by station ─────────────────────────────────────────
    by_station = (
        tdf.groupby("station_code")["arrival_delay_min"]
        .agg(avg="mean", std="std", count="count", min="min", max="max")
        .reset_index()
        .sort_values("avg", ascending=False)
    )
    delay_by_station = [
        {
            "station": row["station_code"],
            "avg_delay": round(float(row["avg"]), 2),
            "std_delay": round(float(row["std"] if not np.isnan(row["std"]) else 0), 2),
            "count": int(row["count"]),
            "min_delay": round(float(row["min"]), 2),
            "max_delay": round(float(row["max"]), 2),
        }
        for _, row in by_station.iterrows()
    ]

    # ── Delay by day of week ──────────────────────────────────────
    if "day_of_week" in tdf.columns:
        by_day = (
            tdf.groupby("day_of_week")["arrival_delay_min"]
            .mean()
            .reindex(range(7), fill_value=0.0)
        )
        delay_by_day = [
            {"day": DAY_NAMES[i], "avg_delay": round(float(v), 2)}
            for i, v in enumerate(by_day)
        ]
    else:
        delay_by_day = []

    # ── Delay over time (daily avg) ───────────────────────────────
    tdf_valid = tdf.dropna(subset=["journey_date"])
    if not tdf_valid.empty:
        by_date = (
            tdf_valid.groupby(tdf_valid["journey_date"].dt.date)["arrival_delay_min"]
            .mean()
            .reset_index()
        )
        delay_over_time = [
            {"date": str(row["journey_date"]), "avg_delay": round(float(row["arrival_delay_min"]), 2)}
            for _, row in by_date.iterrows()
        ]
    else:
        delay_over_time = []

    # ── Delay distribution (histogram bins) ──────────────────────
    valid_delays = tdf["arrival_delay_min"].dropna()
    if len(valid_delays) > 0:
        counts, bin_edges = np.histogram(valid_delays.clip(-20, 300), bins=30)
        delay_distribution = {
            "bins":   [round(float(e), 1) for e in bin_edges],
            "counts": [int(c) for c in counts],
        }
    else:
        delay_distribution = {"bins": [], "counts": []}

    # ── Section-level delay change ────────────────────────────────
    tdf_sorted = tdf.sort_values(["journey_date", "station_code"])
    worst_sections, best_sections = _compute_sections(tdf)

    # ── Summary scalars ───────────────────────────────────────────
    avg_delay   = float(tdf["arrival_delay_min"].mean())
    on_time_pct = float((tdf["arrival_delay_min"] <= 5).mean() * 100)
    late_pct    = float((tdf["arrival_delay_min"] > 30).mean() * 100)
    n_journeys  = int(tdf["journey_date"].nunique()) if "journey_date" in tdf else 0
    n_records   = len(tdf)

    return {
        "train_number":       train_number,
        "train_name":         TRAIN_NAMES.get(train_number, ""),
        "n_records":          n_records,
        "n_journey_days":     n_journeys,
        "avg_delay_minutes":  round(avg_delay, 2),
        "on_time_pct":        round(on_time_pct, 1),
        "late_pct":           round(late_pct, 1),
        # Frontend chart data
        "delay_by_station":   delay_by_station,
        "delay_by_day":       delay_by_day,
        "delay_over_time":    delay_over_time,
        "delay_distribution": delay_distribution,
        "worst_sections":     worst_sections,
        "best_sections":      best_sections,
    }


def _compute_sections(tdf: pd.DataFrame) -> tuple[list, list]:
    """
    Computes average delay CHANGE between consecutive stations.
    Returns (worst_sections, best_sections) — top 10 each.
    """
    results = []
    tdf = tdf.copy()
    tdf["journey_date"] = pd.to_datetime(tdf["journey_date"], errors="coerce")

    # Group by journey (train + date), sort by... station order if available
    for (_, date_), grp in tdf.groupby(["train_number", tdf["journey_date"].dt.date]):
        grp = grp.sort_values("station_sequence") if "station_sequence" in grp.columns else grp
        codes  = grp["station_code"].tolist()
        delays = grp["arrival_delay_min"].tolist()
        for i in range(1, len(codes)):
            if pd.notna(delays[i]) and pd.notna(delays[i - 1]):
                results.append({
                    "from":   codes[i - 1],
                    "to":     codes[i],
                    "change": delays[i] - delays[i - 1],
                })

    if not results:
        return [], []

    df_sec = pd.DataFrame(results)
    agg = df_sec.groupby(["from", "to"])["change"].agg(avg="mean", count="count").reset_index()
    agg = agg[agg["count"] >= 3]   # at least 3 observations

    worst = agg.nlargest(10, "avg")[["from", "to", "avg", "count"]].copy()
    best  = agg.nsmallest(10, "avg")[["from", "to", "avg", "count"]].copy()

    def to_list(df_):
        return [
            {"from": r["from"], "to": r["to"],
             "avg_change": round(float(r["avg"]), 2), "count": int(r["count"])}
            for _, r in df_.iterrows()
        ]

    return to_list(worst), to_list(best)


# ─────────────────────────────────────────────────────────────────
# Summary statistics (global / all trains)
# ─────────────────────────────────────────────────────────────────
def compute_summary_stats(df: pd.DataFrame, meta: dict) -> dict:
    trains_summary = []
    for tn in TRAIN_NAMES:
        tdf = df[df["train_number"] == tn]
        if tdf.empty:
            continue
        trains_summary.append({
            "train_number":      tn,
            "train_name":        TRAIN_NAMES[tn],
            "n_records":         len(tdf),
            "avg_delay_minutes": round(float(tdf["arrival_delay_min"].mean()), 2),
            "on_time_pct":       round(float((tdf["arrival_delay_min"] <= 5).mean() * 100), 1),
        })

    # Overall delay distribution (all trains combined)
    valid = df["arrival_delay_min"].dropna()
    counts, edges = np.histogram(valid.clip(-20, 300), bins=40)
    overall_distribution = {
        "bins":   [round(float(e), 1) for e in edges],
        "counts": [int(c) for c in counts],
    }

    # Delay trend over time (all trains)
    df_valid = df.dropna(subset=["journey_date"])
    if not df_valid.empty:
        by_date = (
            df_valid.groupby(df_valid["journey_date"].dt.date)["arrival_delay_min"]
            .mean().reset_index()
        )
        overall_trend = [
            {"date": str(r["journey_date"]), "avg_delay": round(float(r["arrival_delay_min"]), 2)}
            for _, r in by_date.iterrows()
        ]
    else:
        overall_trend = []

    return {
        "total_records":      len(df),
        "total_trains":       len(trains_summary),
        "total_stations":     int(df["station_code"].nunique()),
        "date_range": {
            "from": str(df["journey_date"].min().date()) if not df["journey_date"].isna().all() else "",
            "to":   str(df["journey_date"].max().date()) if not df["journey_date"].isna().all() else "",
        },
        "model_accuracy": {
            "val_mae":   meta.get("val_mae"),
            "test_mae":  meta.get("test_mae"),
            "test_rmse": meta.get("test_rmse"),
            "test_r2":   meta.get("test_r2"),
        },
        "trains":                trains_summary,
        "overall_distribution":  overall_distribution,
        "overall_trend":         overall_trend,
        "generated_at":          pd.Timestamp.utcnow().isoformat() + "Z",
    }


# ─────────────────────────────────────────────────────────────────
# Entry point — generate all stat files
# ─────────────────────────────────────────────────────────────────
def generate_all_stats():
    """
    Generates and writes all stats JSON files to static/data/.
    Call once at server startup and after each daily dataset update.
    """
    _ensure_static_dir()
    df   = _load_cleaned()
    meta = _load_meta()

    # Per-train files
    for tn in TRAIN_NAMES:
        stats = compute_train_stats(df, tn)
        out   = os.path.join(STATIC_DIR, f"stats_{tn}.json")
        with open(out, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"[stats_cache] Written: {out}")

    # Summary file
    summary = compute_summary_stats(df, meta)
    out = os.path.join(STATIC_DIR, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[stats_cache] Written: {out}")

    print("[stats_cache] All stats generated successfully.")
    return summary


if __name__ == "__main__":
    generate_all_stats()
