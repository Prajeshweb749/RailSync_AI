"""
delay_model.py
==============
ST-GNN Inference Engine for Railway Delay Distributions.
Replaces heuristic Monte Carlo loops with parametric neural predictions.
"""

import json
import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from stgnn_model import RailwaySTGNN

# Path Resolution
INDEX_PATH = os.path.join(BASE_DIR, "station_index.json")
ADJ_NPZ_PATH = os.path.join(BASE_DIR, "adj_matrix.npz")
ADJ_PT_PATH = os.path.join(BASE_DIR, "adj_matrix.pt")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "stgnn_weights.pt")

if not os.path.exists(INDEX_PATH):
    INDEX_PATH = os.path.join(os.path.dirname(BASE_DIR), "station_index.json")
if not os.path.exists(ADJ_NPZ_PATH):
    ADJ_NPZ_PATH = os.path.join(os.path.dirname(BASE_DIR), "adj_matrix.npz")

# Load station mapping
if os.path.exists(INDEX_PATH):
    with open(INDEX_PATH, "r") as f:
        STATION_TO_IDX = json.load(f)
else:
    STATION_TO_IDX = {}

NUM_NODES = len(STATION_TO_IDX) if STATION_TO_IDX else 4626

# Load graph topology
if os.path.exists(ADJ_NPZ_PATH):
    data = np.load(ADJ_NPZ_PATH)["adj_matrix"]
    ADJ_TENSOR = torch.tensor(data, dtype=torch.float32)
elif os.path.exists(ADJ_PT_PATH):
    ADJ_TENSOR = torch.load(ADJ_PT_PATH)
else:
    ADJ_TENSOR = torch.eye(NUM_NODES, dtype=torch.float32)

DEVICE = torch.device("cpu")
MODEL = RailwaySTGNN(num_nodes=NUM_NODES, in_features=4, hidden_dim=32).to(DEVICE)
ADJ = ADJ_TENSOR.to(DEVICE)

# Auto-detect trained weights
if os.path.exists(MODEL_WEIGHTS):
    MODEL.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    print("[+] Loaded trained ST-GNN model weights from stgnn_weights.pt.")
else:
    print("[*] Running ST-GNN with initial weights.")

MODEL.eval()


def get_current_feature_snapshot(time_bin: int = 24) -> torch.Tensor:
    seq_len = 6
    x = torch.zeros(1, seq_len, NUM_NODES, 4, dtype=torch.float32)
    for t in range(seq_len):
        cur_bin = (time_bin - (seq_len - 1 - t)) % 48
        angle = 2.0 * np.pi * cur_bin / 48.0
        x[0, t, :, 2] = np.sin(angle)
        x[0, t, :, 3] = np.cos(angle)
        x[0, t, :, 0] = 0.05
        x[0, t, :, 1] = 0.05
    return x


def predict_station_delays(time_bin: int = 24):
    x = get_current_feature_snapshot(time_bin).to(DEVICE)
    with torch.no_grad():
        mu, sigma = MODEL(x, ADJ)
    return mu[0].cpu().numpy(), sigma[0].cpu().numpy()


def simulate_leg_delay(leg_key, incoming_delay_samples=None, n_samples=3000, rng=None):
    """
    Parametric delay distribution sampler compatible with reliability.py.
    """
    rng = rng or np.random.default_rng()
    train_id, origin, dest = leg_key
    
    dest_idx = STATION_TO_IDX.get(dest)
    pred_mu_all, pred_sigma_all = predict_station_delays(time_bin=24)
    
    if dest_idx is not None:
        leg_mu = float(pred_mu_all[dest_idx])
        leg_sigma = float(pred_sigma_all[dest_idx])
    else:
        leg_mu = 10.0
        leg_sigma = 8.0

    leg_sigma = max(leg_sigma, 2.0)
    generated_delays = rng.normal(loc=leg_mu, scale=leg_sigma, size=n_samples)

    if incoming_delay_samples is not None and len(incoming_delay_samples) > 0:
        absorbed_incoming = 0.85 * incoming_delay_samples
        total_delay = absorbed_incoming + generated_delays
    else:
        total_delay = generated_delays

    return np.clip(total_delay, 0.0, None)


# Aliases for compatibility with ranker.py
predict_delay_distribution = simulate_leg_delay


def quantiles(arr):
    arr = np.array(arr)
    return {
        "P10": float(np.percentile(arr, 10)),
        "P50": float(np.percentile(arr, 50)),
        "P90": float(np.percentile(arr, 90)),
    }