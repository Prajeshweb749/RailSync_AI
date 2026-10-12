"""
retrain_and_evaluate_xgboost.py
===============================
RailSync AI — Phase 9: Model Training, Chronological Holdout Evaluation & Model Registry

Constructs 28-feature vectors from PostgreSQL railway_db historical observations,
splits dataset strictly chronologically (train/validation/test), trains XGBoost model,
evaluates MAE, RMSE, and R2 against baseline models, and updates saved model artifacts.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_adapter import get_db_connection

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model_trainer")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "saved")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "sched_arr_hour", "sched_dep_hour", "day_of_week", "month",
    "distance_from_source", "distance_remaining", "progress_fraction", "segment_distance",
    "current_arrival_delay", "current_departure_delay", "prev_station_delay", "prev_2_station_delay",
    "delay_trend", "delay_trend2", "dwell_time", "sched_dwell_time", "sched_section_travel",
    "hist_avg_train_delay", "hist_median_train_delay", "hist_std_train_delay", "hist_on_time_rate_train",
    "hist_avg_station_delay", "hist_std_station_delay", "hist_avg_section_delay_chg", "hist_std_section_delay_chg",
    "train_number_enc", "station_code_enc", "next_station_code_enc"
]

def load_dataset_from_db():
    """Fetches full historical observations and route information from PostgreSQL railway_db."""
    conn = get_db_connection()
    if not conn:
        logger.error("Could not connect to PostgreSQL database.")
        sys.exit(1)

    cur = conn.cursor()
    logger.info("Fetching historical observations and route schedules from PostgreSQL...")

    sql = """
        SELECT 
            hr.run_id, hr.train_number, hr.run_date,
            hso.observation_id, hso.station_code, hso.stop_order,
            hso.arrival_delay_min, hso.departure_delay_min,
            ts.arrival_time::text AS sched_arr_str, ts.departure_time::text AS sched_dep_str,
            t.distance_km AS total_train_dist
        FROM historical_runs hr
        JOIN historical_station_observations hso ON hr.run_id = hso.run_id
        LEFT JOIN train_stops ts ON hr.train_number = ts.train_number AND hso.station_code = ts.station_code
        LEFT JOIN trains t ON hr.train_number = t.train_number
        ORDER BY hr.run_date ASC, hr.train_number ASC, hso.stop_order ASC;
    """
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    logger.info(f"Loaded {len(rows)} raw observation records from PostgreSQL.")
    df = pd.DataFrame(rows)
    return df


def prepare_features(df: pd.DataFrame, train_dates=None):
    """Builds leakage-free 28-feature vectors and target next_station_delay_change."""
    logger.info("Building 28-feature vectors...")
    df["run_date"] = pd.to_datetime(df["run_date"])

    # Calculate target: delay change to next station within the same run
    df = df.sort_values(["run_id", "stop_order"]).reset_index(drop=True)
    df["next_arrival_delay_min"] = df.groupby("run_id")["arrival_delay_min"].shift(-1)
    df["next_station_code"] = df.groupby("run_id")["station_code"].shift(-1)
    df["target_delay_change"] = df["next_arrival_delay_min"] - df["arrival_delay_min"]

    # Filter valid pairs
    df_clean = df.dropna(subset=["next_station_code", "target_delay_change"]).copy()

    # Distance calculations
    df_clean["total_dist"] = df_clean["total_train_dist"].fillna(450.0).astype(float)
    df_clean["max_stops"] = df_clean.groupby("run_id")["stop_order"].transform("max")
    df_clean["progress_fraction"] = (df_clean["stop_order"] - 1) / np.maximum(df_clean["max_stops"] - 1, 1)
    df_clean["distance_from_source"] = df_clean["progress_fraction"] * df_clean["total_dist"]
    df_clean["distance_remaining"] = df_clean["total_dist"] - df_clean["distance_from_source"]
    df_clean["segment_distance"] = df_clean["total_dist"] / np.maximum(df_clean["max_stops"] - 1, 1)

    # Prev station delays & trends (strictly grouped by run_id: no cross-run lag contamination)
    df_clean["prev_station_delay"] = df_clean.groupby("run_id")["arrival_delay_min"].shift(1)
    df_clean["prev_2_station_delay"] = df_clean.groupby("run_id")["arrival_delay_min"].shift(2)
    df_clean["current_arrival_delay"] = df_clean["arrival_delay_min"].fillna(0.0)
    df_clean["current_departure_delay"] = df_clean["departure_delay_min"].fillna(df_clean["current_arrival_delay"])
    df_clean["delay_trend"] = df_clean["current_arrival_delay"] - df_clean["prev_station_delay"]
    df_clean["delay_trend2"] = df_clean["prev_station_delay"] - df_clean["prev_2_station_delay"]

    # Date / Time features
    df_clean["day_of_week"] = df_clean["run_date"].dt.dayofweek
    df_clean["month"] = df_clean["run_date"].dt.month
    df_clean["sched_arr_hour"] = 12.0
    df_clean["sched_dep_hour"] = 12.0
    df_clean["dwell_time"] = np.nan
    df_clean["sched_dwell_time"] = 2.0
    df_clean["sched_section_travel"] = 30.0

    # Train/Station/Section Historical Aggregates (strictly leakage-free from train partition)
    df_train_only = df_clean[df_clean["run_date"].isin(train_dates)] if train_dates is not None else df_clean

    train_stats = df_train_only.groupby("train_number")["current_arrival_delay"].agg(
        hist_avg_train_delay="mean",
        hist_median_train_delay="median",
        hist_std_train_delay="std"
    ).reset_index()
    train_stats["hist_on_time_rate_train"] = df_train_only.groupby("train_number")["current_arrival_delay"].apply(lambda s: (s <= 5).mean()).values

    stn_stats = df_train_only.groupby("station_code")["current_arrival_delay"].agg(
        hist_avg_station_delay="mean",
        hist_std_station_delay="std"
    ).reset_index()

    sec_stats = df_train_only.groupby(["station_code", "next_station_code"])["target_delay_change"].agg(
        hist_avg_section_delay_chg="mean",
        hist_std_section_delay_chg="std"
    ).reset_index()

    df_clean = df_clean.merge(train_stats, on="train_number", how="left")
    df_clean = df_clean.merge(stn_stats, on="station_code", how="left")
    df_clean = df_clean.merge(sec_stats, on=["station_code", "next_station_code"], how="left")

    # Categorical Encoding
    from sklearn.preprocessing import LabelEncoder
    le_train = LabelEncoder()
    le_stn = LabelEncoder()
    le_next = LabelEncoder()

    df_clean["train_number_enc"] = le_train.fit_transform(df_clean["train_number"].astype(str))
    df_clean["station_code_enc"] = le_stn.fit_transform(df_clean["station_code"].astype(str))
    df_clean["next_station_code_enc"] = le_next.fit_transform(df_clean["next_station_code"].astype(str))

    encoders = {"train": le_train, "station": le_stn, "next_station": le_next}

    logger.info(f"Feature dataset prepared: {len(df_clean)} samples.")
    return df_clean, encoders


def train_and_evaluate():
    df_raw = load_dataset_from_db()
    if len(df_raw) < 500:
        logger.error("Insufficient observation count in PostgreSQL database.")
        return

    # Chronological Train / Val / Test Split Dates (leakage-free)
    dates = sorted(pd.to_datetime(df_raw["run_date"]).unique())
    n_dates = len(dates)

    train_end_idx = int(n_dates * 0.70)
    val_end_idx = int(n_dates * 0.85)

    train_dates = set(dates[:train_end_idx])
    val_dates = set(dates[train_end_idx:val_end_idx])
    test_dates = set(dates[val_end_idx:])

    df, encoders = prepare_features(df_raw, train_dates=train_dates)

    df_train = df[df["run_date"].isin(train_dates)]
    df_val = df[df["run_date"].isin(val_dates)]
    df_test = df[df["run_date"].isin(test_dates)]

    logger.info(f"Chronological Dataset Split:")
    logger.info(f"  - Train Set : {len(df_train)} samples ({min(train_dates).strftime('%Y-%m-%d')} to {max(train_dates).strftime('%Y-%m-%d')})")
    logger.info(f"  - Val Set   : {len(df_val)} samples ({min(val_dates).strftime('%Y-%m-%d')} to {max(val_dates).strftime('%Y-%m-%d')})")
    logger.info(f"  - Test Set  : {len(df_test)} samples ({min(test_dates).strftime('%Y-%m-%d')} to {max(test_dates).strftime('%Y-%m-%d')})")

    X_train = df_train[FEATURE_COLS].fillna(0)
    y_train = df_train["target_delay_change"]

    X_val = df_val[FEATURE_COLS].fillna(0)
    y_val = df_val["target_delay_change"]

    X_test = df_test[FEATURE_COLS].fillna(0)
    y_test = df_test["target_delay_change"]

    # Baseline Model: Predict Constant Mean Delay Change
    y_pred_base = np.full_like(y_test, fill_value=y_train.mean())
    mae_base = float(np.mean(np.abs(y_test - y_pred_base)))
    rmse_base = float(np.sqrt(np.mean((y_test - y_pred_base)**2)))
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    r2_base = 0.0

    logger.info(f"\n--- Naive Mean Baseline Performance ---")
    logger.info(f"  MAE  : {mae_base:.3f} min")
    logger.info(f"  RMSE : {rmse_base:.3f} min")
    logger.info(f"  R2   : {r2_base:.3f}")

    # Train XGBoost Model
    if HAS_XGB:
        logger.info("\nTraining XGBoost Regressor model...")
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred_xgb = model.predict(X_test)

        mae_xgb = float(np.mean(np.abs(y_test - y_pred_xgb)))
        rmse_xgb = float(np.sqrt(np.mean((y_test - y_pred_xgb)**2)))
        r2_xgb = float(1.0 - (np.sum((y_test - y_pred_xgb)**2) / max(ss_tot, 1e-6)))

        logger.info(f"\n--- XGBoost Model Performance (Chronological Test Holdout) ---")
        logger.info(f"  MAE  : {mae_xgb:.3f} min (Improvement: {mae_base - mae_xgb:+.3f} min)")
        logger.info(f"  RMSE : {rmse_xgb:.3f} min (Improvement: {rmse_base - rmse_xgb:+.3f} min)")
        logger.info(f"  R2   : {r2_xgb:.3f}")

        # Save Model Artifacts
        model_path = os.path.join(MODEL_DIR, "xgb_model.ubj")
        encoders_path = os.path.join(MODEL_DIR, "encoders.joblib")
        meta_path = os.path.join(MODEL_DIR, "metadata.json")

        model.save_model(model_path)
        joblib.dump(encoders, encoders_path)

        metadata = {
            "trained_at": datetime.now().isoformat(),
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
            "feature_cols": FEATURE_COLS,
            "metrics": {
                "baseline_mae": round(mae_base, 3),
                "baseline_rmse": round(rmse_base, 3),
                "xgb_mae": round(mae_xgb, 3),
                "xgb_rmse": round(rmse_xgb, 3),
                "xgb_r2": round(r2_xgb, 3)
            }
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\n✓ Saved model artifacts to {MODEL_DIR}")
        return metadata

if __name__ == "__main__":
    train_and_evaluate()
