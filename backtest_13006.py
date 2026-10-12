"""
backtest_13006.py
=================
Backtesting the production delay prediction pipeline on historical runs of Train 13006.

Evaluation methodology:
- For each historical run in railway_db, for each halt station S (up to RBL):
  Seed the predictor with (S, observed delay at S).
  Predict arrival delay at:
    1. Next halt (S + 1)
    2. 5 halts ahead (S + 5, if <= RBL)
    3. Destination halt RBL (stop_order 28)
- Compare against actual observed arrival delay.
- Baselines:
  (a) Persistence: predicted delay = seed delay
  (b) Historical Median Change: seed delay + median(delay_target - delay_S) for earlier runs
      with seed delay within +/- 10 min (leakage-free: only runs earlier than target run).
- Reports MAE, Median Absolute Error, and Bias (mean predicted - actual),
  stratified by Horizon and Seed Delay Bucket (<20, 20-50, 50-80, >80 min).
- Saves all per-prediction rows to backtest_results.csv.
"""

import os
import sys
from datetime import datetime, date
import numpy as np
import pandas as pd

# Set working directory to project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from db_adapter import get_db_connection
import prediction.predict as _pred_module

_pred_module._load_resources()
model = _pred_module._model
feat_cols = _pred_module._feat_cols

def run_backtest():
    print("=" * 80, flush=True)
    print("  RailSync AI — Historical Backtest for Train 13006", flush=True)
    print("=" * 80, flush=True)

    # 1. Load historical runs and observations from PostgreSQL
    print("[1/4] Fetching historical runs and observations from railway_db...", flush=True)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT hr.run_id, hr.run_date, hso.station_code, hso.stop_order, hso.arrival_delay_min
        FROM historical_runs hr
        JOIN historical_station_observations hso ON hr.run_id = hso.run_id
        WHERE hr.train_number = '13006' AND hso.arrival_delay_min IS NOT NULL
        ORDER BY hr.run_date ASC, hso.stop_order ASC;
    """)
    rows = cur.fetchall()

    # Load timetable route metadata for halts (scheduled arrival/departure, distances)
    cur.execute("""
        SELECT ts.stop_order, ts.station_code, 
               ts.arrival_time::text as sched_arr, ts.departure_time::text as sched_dep,
               t.distance_km as total_train_dist
        FROM train_stops ts
        JOIN trains t ON ts.train_number = t.train_number
        WHERE ts.train_number = '13006'
        ORDER BY ts.stop_order ASC;
    """)
    ts_rows = cur.fetchall()
    cur.close()
    conn.close()

    obs_df = pd.DataFrame(rows)
    obs_df['run_date'] = pd.to_datetime(obs_df['run_date'])

    # Halts up to RBL (RBL is stop_order 28)
    sample_run = obs_df[obs_df['run_id'] == obs_df['run_id'].iloc[0]]
    halts_df = sample_run[sample_run['station_code'] == 'RBL']
    rbl_order = halts_df['stop_order'].iloc[0] if not halts_df.empty else 28

    halt_order_df = sample_run[sample_run['stop_order'] <= rbl_order][['stop_order', 'station_code']].sort_values('stop_order').reset_index(drop=True)
    halt_codes = halt_order_df['station_code'].tolist()
    rbl_idx = halt_codes.index('RBL')

    print(f"Loaded {len(obs_df)} observations across {obs_df['run_id'].nunique()} runs.", flush=True)
    print(f"Halts up to RBL ({len(halt_codes)} halts): {' -> '.join(halt_codes)}", flush=True)

    # Pre-build timetable distance and schedule lookup for halt stations
    ts_dict = {r['station_code']: r for r in ts_rows}
    tot_dist = float(ts_rows[0]['total_train_dist'] or 1910.0) if ts_rows else 1910.0

    halt_info = []
    n_halts = len(halt_codes)
    for i, sc in enumerate(halt_codes):
        d_src = round((i / max(n_halts - 1, 1)) * 866.4, 1)  # distance to RBL is ~866 km
        d_rem = round(tot_dist - d_src, 1)
        ts_entry = ts_dict.get(sc, {})
        halt_info.append({
            'station_code': sc,
            'distance_from_source': d_src,
            'distance_to_destination': d_rem,
            'sched_arr_str': ts_entry.get('sched_arr', '10:00')[:5] if ts_entry.get('sched_arr') else '10:00',
            'sched_dep_str': ts_entry.get('sched_dep', '10:05')[:5] if ts_entry.get('sched_dep') else '10:05',
        })

    # Historical stats cache for 13006
    hist_stats = _pred_module._build_historical_stats_for_train('13006', date.today().isoformat())

    # 2. Pivot observed delays by run_id
    print("[2/4] Indexing run delays and preparing chronological history...", flush=True)
    run_groups = []
    for run_id, grp in obs_df.groupby('run_id', sort=False):
        r_date = grp['run_date'].iloc[0]
        delay_map = dict(zip(grp['station_code'], grp['arrival_delay_min']))
        run_groups.append({
            'run_id': run_id,
            'run_date': r_date,
            'delay_map': delay_map
        })

    run_groups.sort(key=lambda x: x['run_date'])

    # Pre-compute historical differences lookup for Baseline (b) without data leakage
    # For each run_idx, baseline (b) only uses runs[:run_idx]
    print("[3/4] Running vectorized production inference across all runs...", flush=True)

    # Initialize all simulation paths across (run, seed)
    # A path tracks: run_id, run_date, seed_idx, cur_step_idx, target_dict, prev_d, prev2_d, prev3_d, sim_delays
    active_paths = []
    path_meta = []

    for r_idx, run in enumerate(run_groups):
        delay_map = run['delay_map']
        j_date_str = run['run_date'].strftime('%Y-%m-%d')

        for s_idx in range(rbl_idx):
            s_code = halt_codes[s_idx]
            if s_code not in delay_map:
                continue

            seed_delay = float(delay_map[s_code])
            targets = {}
            if s_idx + 1 <= rbl_idx:
                targets['next_halt'] = s_idx + 1
            if s_idx + 5 <= rbl_idx:
                targets['5_halts_ahead'] = s_idx + 5
            targets['RBL'] = rbl_idx

            max_target = max(targets.values())

            active_paths.append({
                'path_id': len(path_meta),
                'r_idx': r_idx,
                'run_id': run['run_id'],
                'run_date': j_date_str,
                'seed_station': s_code,
                'seed_delay': seed_delay,
                'seed_idx': s_idx,
                'cur_step': s_idx,
                'max_target': max_target,
                'targets': targets,
                'prev_d': seed_delay,
                'prev2_d': np.nan,
                'prev3_d': np.nan,
                'sim_delays': {s_idx: seed_delay}
            })
            path_meta.append(active_paths[-1])

    print(f"Total evaluation paths initialized: {len(active_paths)}", flush=True)

    # Step through paths synchronously in rounds (max 27 rounds)
    for round_num in range(rbl_idx):
        stepping_paths = [p for p in active_paths if p['cur_step'] < p['max_target']]
        if not stepping_paths:
            break

        # Batch build feature vectors
        batch_fvecs = []
        for p in stepping_paths:
            cur_idx = p['cur_step']
            nxt_idx = cur_idx + 1
            cur_h = halt_info[cur_idx]
            nxt_h = halt_info[nxt_idx]
            j_date = p['run_date']

            sched_arr = pd.to_datetime(f"{j_date} {cur_h['sched_arr_str']}")
            sched_dep = pd.to_datetime(f"{j_date} {cur_h['sched_dep_str']}")
            sched_nxt = pd.to_datetime(f"{j_date} {nxt_h['sched_arr_str']}")

            fv = _pred_module._build_feature_vector(
                train_number              = '13006',
                station_code              = cur_h['station_code'],
                next_station_code         = nxt_h['station_code'],
                current_arrival_delay     = p['prev_d'],
                current_departure_delay   = p['prev_d'],
                prev_delay                = p['prev2_d'],
                prev2_delay               = p['prev3_d'],
                scheduled_arrival         = sched_arr,
                scheduled_departure       = sched_dep,
                scheduled_next_arrival    = sched_nxt,
                distance_from_source      = cur_h['distance_from_source'],
                distance_remaining        = cur_h['distance_to_destination'],
                next_distance_from_source = nxt_h['distance_from_source'],
                journey_date              = j_date,
                hist_stats                = hist_stats,
            )
            batch_fvecs.append([fv.get(c, np.nan) for c in feat_cols])

        # Batch predict with XGBoost
        X_mat = np.nan_to_num(np.array(batch_fvecs, dtype=np.float32), nan=0.0)
        deltas = model.predict(X_mat)

        # Update each path
        for i, p in enumerate(stepping_paths):
            pred_delta = float(deltas[i])
            next_d = max(p['prev_d'] + pred_delta, 0.0)

            nxt_idx = p['cur_step'] + 1
            p['sim_delays'][nxt_idx] = next_d
            p['prev3_d'] = p['prev2_d']
            p['prev2_d'] = p['prev_d']
            p['prev_d'] = next_d
            p['cur_step'] = nxt_idx

        if (round_num + 1) % 5 == 0 or round_num == rbl_idx - 1:
            print(f"  Completed stepping round {round_num + 1}/{rbl_idx}...", flush=True)

    # 4. Compute metrics and baselines
    print("\n[4/4] Calculating baselines and aggregating metrics...", flush=True)
    results = []

    # Map historical run delays for baseline (b)
    # Cache station-pair differences per run for instant median computation
    earlier_diff_cache = {}  # (s_code, tgt_code) -> list of (e_seed, diff)

    for r_idx, run in enumerate(run_groups):
        delay_map = run['delay_map']

        # Find paths belonging to this run
        paths_for_run = [p for p in path_meta if p['r_idx'] == r_idx]

        for p in paths_for_run:
            s_code = p['seed_station']
            seed_delay = p['seed_delay']

            for h_name, tgt_idx in p['targets'].items():
                tgt_code = halt_codes[tgt_idx]
                if tgt_code not in delay_map:
                    continue

                actual_delay = float(delay_map[tgt_code])
                pred_model = p['sim_delays'][tgt_idx]
                pred_pers = seed_delay

                # Baseline (b): Historical median change strictly from earlier runs
                pair_key = (s_code, tgt_code)
                earlier_history = earlier_diff_cache.get(pair_key, [])

                if earlier_history:
                    # Filter within +/- 10 min
                    matching = [diff for (e_seed, diff) in earlier_history if abs(e_seed - seed_delay) <= 10.0]
                    if matching:
                        pred_hist = max(seed_delay + float(np.median(matching)), 0.0)
                    else:
                        all_diffs = [diff for (e_seed, diff) in earlier_history]
                        pred_hist = max(seed_delay + float(np.median(all_diffs)), 0.0)
                else:
                    pred_hist = seed_delay

                # Seed delay bucket
                if seed_delay < 20:
                    bucket = "<20"
                elif seed_delay < 50:
                    bucket = "20-50"
                elif seed_delay < 80:
                    bucket = "50-80"
                else:
                    bucket = ">80"

                results.append({
                    'run_id': p['run_id'],
                    'run_date': p['run_date'],
                    'seed_station': s_code,
                    'seed_delay': seed_delay,
                    'target_station': tgt_code,
                    'horizon': h_name,
                    'bucket': bucket,
                    'actual_delay': actual_delay,
                    'pred_model': round(pred_model, 2),
                    'pred_persistence': round(pred_pers, 2),
                    'pred_hist_median': round(pred_hist, 2),
                    'err_model': round(pred_model - actual_delay, 2),
                    'err_pers': round(pred_pers - actual_delay, 2),
                    'err_hist': round(pred_hist - actual_delay, 2),
                    'abs_err_model': round(abs(pred_model - actual_delay), 2),
                    'abs_err_pers': round(abs(pred_pers - actual_delay), 2),
                    'abs_err_hist': round(abs(pred_hist - actual_delay), 2),
                })

        # Append current run's station observations into earlier_diff_cache
        for s_i in range(rbl_idx):
            s_c = halt_codes[s_i]
            if s_c in delay_map:
                s_d = delay_map[s_c]
                for t_i in range(s_i + 1, rbl_idx + 1):
                    t_c = halt_codes[t_i]
                    if t_c in delay_map:
                        t_d = delay_map[t_c]
                        pair = (s_c, t_c)
                        if pair not in earlier_diff_cache:
                            earlier_diff_cache[pair] = []
                        earlier_diff_cache[pair].append((s_d, t_d - s_d))

    res_df = pd.DataFrame(results)
    csv_path = os.path.join(ROOT_DIR, "backtest_results.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"Saved {len(res_df)} prediction records to {csv_path}.\n", flush=True)

    def summarize(df_sub):
        if df_sub.empty:
            return {'N': 0, 'Model_MAE': 0, 'Model_MedAE': 0, 'Model_Bias': 0,
                    'Pers_MAE': 0, 'Pers_MedAE': 0, 'Pers_Bias': 0,
                    'Hist_MAE': 0, 'Hist_MedAE': 0, 'Hist_Bias': 0}
        return {
            'N': len(df_sub),
            'Model_MAE': df_sub['abs_err_model'].mean(),
            'Model_MedAE': df_sub['abs_err_model'].median(),
            'Model_Bias': df_sub['err_model'].mean(),
            'Pers_MAE': df_sub['abs_err_pers'].mean(),
            'Pers_MedAE': df_sub['abs_err_pers'].median(),
            'Pers_Bias': df_sub['err_pers'].mean(),
            'Hist_MAE': df_sub['abs_err_hist'].mean(),
            'Hist_MedAE': df_sub['abs_err_hist'].median(),
            'Hist_Bias': df_sub['err_hist'].mean(),
        }

    print("=" * 96, flush=True)
    print(f"{'HORIZON SPLIT':<18} | {'N':>5} | {'MODEL (MAE / MedAE / Bias)':<23} | {'PERSISTENCE (MAE / MedAE / Bias)':<24} | {'HIST MEDIAN (MAE / MedAE / Bias)':<24}", flush=True)
    print("-" * 96, flush=True)
    for h in ['next_halt', '5_halts_ahead', 'RBL']:
        sub = res_df[res_df['horizon'] == h]
        m = summarize(sub)
        m_str = f"{m['Model_MAE']:5.2f} / {m['Model_MedAE']:5.2f} / {m['Model_Bias']:+5.2f}"
        p_str = f"{m['Pers_MAE']:5.2f} / {m['Pers_MedAE']:5.2f} / {m['Pers_Bias']:+5.2f}"
        h_str = f"{m['Hist_MAE']:5.2f} / {m['Hist_MedAE']:5.2f} / {m['Hist_Bias']:+5.2f}"
        print(f"{h:<18} | {m['N']:>5} | {m_str:<23} | {p_str:<24} | {h_str:<24}", flush=True)

    print("\n" + "=" * 96, flush=True)
    print(f"{'SEED DELAY BUCKET':<18} | {'N':>5} | {'MODEL (MAE / MedAE / Bias)':<23} | {'PERSISTENCE (MAE / MedAE / Bias)':<24} | {'HIST MEDIAN (MAE / MedAE / Bias)':<24}", flush=True)
    print("-" * 96, flush=True)
    for b in ['<20', '20-50', '50-80', '>80']:
        sub = res_df[res_df['bucket'] == b]
        m = summarize(sub)
        m_str = f"{m['Model_MAE']:5.2f} / {m['Model_MedAE']:5.2f} / {m['Model_Bias']:+5.2f}"
        p_str = f"{m['Pers_MAE']:5.2f} / {m['Pers_MedAE']:5.2f} / {m['Pers_Bias']:+5.2f}"
        h_str = f"{m['Hist_MAE']:5.2f} / {m['Hist_MedAE']:5.2f} / {m['Hist_Bias']:+5.2f}"
        print(f"{b:<18} | {m['N']:>5} | {m_str:<23} | {p_str:<24} | {h_str:<24}", flush=True)

    print("\n" + "=" * 96, flush=True)
    overall = summarize(res_df)
    m_str = f"{overall['Model_MAE']:5.2f} / {overall['Model_MedAE']:5.2f} / {overall['Model_Bias']:+5.2f}"
    p_str = f"{overall['Pers_MAE']:5.2f} / {overall['Pers_MedAE']:5.2f} / {overall['Pers_Bias']:+5.2f}"
    h_str = f"{overall['Hist_MAE']:5.2f} / {overall['Hist_MedAE']:5.2f} / {overall['Hist_Bias']:+5.2f}"
    print(f"{'OVERALL TOTAL':<18} | {overall['N']:>5} | {m_str:<23} | {p_str:<24} | {h_str:<24}", flush=True)
    print("=" * 96, flush=True)

if __name__ == '__main__':
    run_backtest()
