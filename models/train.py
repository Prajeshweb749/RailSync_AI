"""
RailSync AI — models/train.py
Steps 7, 8: Baseline models + XGBoost tuning + training pipeline
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not available — skipping.")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH  = os.path.join(ROOT_DIR, "data", "train.csv")
VAL_PATH    = os.path.join(ROOT_DIR, "data", "val.csv")
TEST_PATH   = os.path.join(ROOT_DIR, "data", "test.csv")
MODEL_DIR   = os.path.join(ROOT_DIR, "models", "saved")

FEATURE_COLS = [
    "sched_arr_hour", "sched_dep_hour", "day_of_week", "month",
    "distance_from_source", "distance_remaining", "progress_fraction",
    "segment_distance",
    "current_arrival_delay", "current_departure_delay",
    "prev_station_delay", "prev_2_station_delay",
    "delay_trend", "delay_trend2",
    "dwell_time", "sched_dwell_time", "sched_section_travel",
    "hist_avg_train_delay", "hist_median_train_delay",
    "hist_std_train_delay", "hist_on_time_rate_train",
    "hist_avg_station_delay", "hist_std_station_delay",
    "hist_avg_section_delay_chg", "hist_std_section_delay_chg",
    # Label-encoded categoricals
    "train_number_enc", "station_code_enc", "next_station_code_enc",
]
TARGET = "next_station_delay_change"


def load_splits():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    return train_df, val_df, test_df


def encode_categoricals(train_df, val_df, test_df):
    """Label-encode train_number, station_code, next_station_code."""
    le_train   = LabelEncoder()
    le_station = LabelEncoder()
    le_next    = LabelEncoder()

    all_trains   = pd.concat([train_df["train_number"], val_df["train_number"], test_df["train_number"]]).astype(str)
    all_stations = pd.concat([train_df["station_code"], val_df["station_code"], test_df["station_code"]]).astype(str)
    all_next     = pd.concat([train_df["next_station_code"], val_df["next_station_code"], test_df["next_station_code"]]).astype(str)

    le_train.fit(all_trains)
    le_station.fit(all_stations)
    le_next.fit(all_next)

    for df in [train_df, val_df, test_df]:
        df["train_number_enc"]    = le_train.transform(df["train_number"].astype(str))
        df["station_code_enc"]    = le_station.transform(df["station_code"].astype(str))
        df["next_station_code_enc"] = le_next.transform(df["next_station_code"].astype(str))

    encoders = {"train": le_train, "station": le_station, "next_station": le_next}
    return train_df, val_df, test_df, encoders


def prepare_xy(df, feature_cols=FEATURE_COLS, target=TARGET):
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    y = df[target].copy()
    # Impute remaining NaNs: use column median, fallback to 0 if median is NaN
    for col in X.columns:
        if X[col].isna().any():
            med = X[col].median()
            fill_val = med if pd.notna(med) else 0.0
            X[col] = X[col].fillna(fill_val)
    # Final safety: replace any remaining NaN/inf with 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y, available


def metrics(y_true, y_pred, prefix=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    tag  = f"[{prefix}]" if prefix else ""
    print(f"  {tag} MAE={mae:.4f} min  |  RMSE={rmse:.4f} min  |  R²={r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ─────────────────────────────────────────────
# STEP 7: BASELINES
# ─────────────────────────────────────────────
def run_baselines(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 70)
    print("STEP 7 — BASELINE MODELS")
    print("=" * 70)

    results = {}

    # 1. Naive: always predict 0 (no change)
    pred_naive = np.zeros(len(y_val))
    print("\nBaseline 1: Naive (predict 0 delay change)")
    results["naive_zero"] = metrics(y_val, pred_naive, "Val")

    # 2. Naive: predict mean of training targets
    mean_pred = np.full(len(y_val), y_train.mean())
    print("\nBaseline 2: Mean predictor (train set mean)")
    results["naive_mean"] = metrics(y_val, mean_pred, "Val")

    # 3. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_val)
    print("\nBaseline 3: Linear Regression")
    results["linear_regression"] = metrics(y_val, pred_lr, "Val")

    # 4. Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_val)
    print("\nBaseline 4: Random Forest (100 trees, max_depth=10)")
    results["random_forest"] = metrics(y_val, pred_rf, "Val")

    return results, lr, rf


# ─────────────────────────────────────────────
# STEP 8: XGBOOST MODEL
# ─────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 70)
    print("STEP 8 — XGBOOST MODEL")
    print("=" * 70)

    # Manual grid search over key hyperparameters
    param_grid = [
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.8,
         "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.7,
         "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03,
         "subsample": 0.9, "colsample_bytree": 0.8,
         "reg_alpha": 0.05, "reg_lambda": 1.5},
        {"n_estimators": 400, "max_depth": 7, "learning_rate": 0.05,
         "subsample": 0.75, "colsample_bytree": 0.75,
         "reg_alpha": 0.1, "reg_lambda": 2.0},
    ]

    best_mae = float("inf")
    best_model = None
    best_params = None

    for i, params in enumerate(param_grid):
        model = xgb.XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=30,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        print(f"  Config {i+1}: {params} → Val MAE={mae:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_params = params

    print(f"\n  Best config: {best_params}")
    print(f"  Best Val MAE: {best_mae:.4f} min")
    return best_model, best_params


def train_lightgbm(X_train, y_train, X_val, y_val):
    if not HAS_LGB:
        return None, None

    print("\n--- LightGBM ---")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    pred = lgb_model.predict(X_val)
    lgb_metrics = metrics(y_val, pred, "Val-LGB")
    return lgb_model, lgb_metrics


def run():
    print("Loading split data...")
    train_df, val_df, test_df = load_splits()

    print("Encoding categoricals...")
    train_df, val_df, test_df, encoders = encode_categoricals(train_df, val_df, test_df)

    print("Preparing X/y matrices...")
    X_train, y_train, feat_cols = prepare_xy(train_df)
    X_val,   y_val,   _         = prepare_xy(val_df)
    X_test,  y_test,  _         = prepare_xy(test_df)

    print(f"  Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")
    print(f"  Features used ({len(feat_cols)}): {feat_cols}")

    # Baselines
    baseline_results, lr_model, rf_model = run_baselines(X_train, y_train, X_val, y_val)

    # XGBoost
    xgb_model, best_params = train_xgboost(X_train, y_train, X_val, y_val)

    # LightGBM
    lgb_model, lgb_m = train_lightgbm(X_train, y_train, X_val, y_val)

    # Final comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Validation set)")
    print("=" * 70)
    all_results = {}
    all_results.update(baseline_results)

    for name, model in [("xgboost", xgb_model), ("linear_regression", lr_model),
                         ("random_forest", rf_model)]:
        if model is not None:
            pred = model.predict(X_val)
            print(f"\n{name}:")
            all_results[name] = metrics(y_val, pred, "Val")

    if lgb_model is not None:
        print("\nlightgbm:")
        all_results["lightgbm"] = lgb_m

    best_name = min(all_results, key=lambda k: all_results[k]["mae"])
    print(f"\n>>> Best model by Val MAE: {best_name.upper()} (MAE={all_results[best_name]['mae']:.4f} min)")

    # Test set evaluation of XGBoost
    print("\n" + "=" * 70)
    print("XGBOOST TEST SET EVALUATION")
    print("=" * 70)
    pred_test = xgb_model.predict(X_test)
    test_metrics = metrics(y_test, pred_test, "Test")

    # Save everything
    os.makedirs(MODEL_DIR, exist_ok=True)

    xgb_model.save_model(os.path.join(MODEL_DIR, "xgb_model.ubj"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.joblib"))

    metadata = {
        "model_version": "1.0.0",
        "feature_cols": feat_cols,
        "target": TARGET,
        "best_xgb_params": best_params,
        "val_metrics": all_results.get("xgboost", {}),
        "test_metrics": test_metrics,
        "baseline_results": {k: v for k, v in baseline_results.items()},
        "best_model": best_name,
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ XGBoost model saved to: {MODEL_DIR}/xgb_model.ubj")
    print(f"✓ Encoders saved to:        {MODEL_DIR}/encoders.joblib")
    print(f"✓ Metadata saved to:        {MODEL_DIR}/metadata.json")

    return xgb_model, encoders, feat_cols, X_train, y_train, X_val, y_val, X_test, y_test, all_results, pred_test


if __name__ == "__main__":
    run()
