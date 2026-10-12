"""
train_stgnn.py
==============
Trains RailwaySTGNN on real chronological sequences.
Uses Masked Gaussian NLL Loss on all stations active during target time bins.
"""

import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from stgnn_model import RailwaySTGNN

print("[*] Starting Chronological ST-GNN Training...", flush=True)

INDEX_PATH = os.path.join(BASE_DIR, "station_index.json")
ADJ_PATH = os.path.join(BASE_DIR, "adj_matrix.npz")
DATASET_PATH = os.path.join(BASE_DIR, "historical_dataset_split.npz")
WEIGHTS_PATH = os.path.join(BASE_DIR, "stgnn_weights.pt")

if not os.path.exists(DATASET_PATH):
    print(f"[!] Error: {DATASET_PATH} not found. Run build_dataset.py first!", flush=True)
    sys.exit(1)

with open(INDEX_PATH, "r") as f:
    station_to_idx = json.load(f)
num_nodes = len(station_to_idx)

adj_data = np.load(ADJ_PATH)["adj_matrix"]
adj_tensor = torch.tensor(adj_data, dtype=torch.float32)

raw = np.load(DATASET_PATH)
train_dataset = TensorDataset(
    torch.tensor(raw["X_train"]), torch.tensor(raw["Y_train"]), torch.tensor(raw["Mask_train"])
)
val_dataset = TensorDataset(
    torch.tensor(raw["X_val"]), torch.tensor(raw["Y_val"]), torch.tensor(raw["Mask_val"])
)
test_dataset = TensorDataset(
    torch.tensor(raw["X_test"]), torch.tensor(raw["Y_test"]), torch.tensor(raw["Mask_test"])
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print(f"[+] Partitions Loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}", flush=True)


def masked_gaussian_nll(mu, sigma, target, mask):
    variance = torch.clamp(sigma ** 2, min=1e-4)
    loss = 0.5 * torch.log(variance) + 0.5 * ((target - mu) ** 2) / variance
    masked_loss = loss * mask
    count = torch.clamp(mask.sum(), min=1.0)
    return masked_loss.sum() / count


def compute_metrics(loader, model, adj, device):
    model.eval()
    total_nll = 0.0
    total_mae = 0.0
    total_count = 0.0

    with torch.no_grad():
        for x_b, y_b, m_b in loader:
            x_b, y_b, m_b = x_b.to(device), y_b.to(device), m_b.to(device)
            mu, sigma = model(x_b, adj)
            
            nll = masked_gaussian_nll(mu, sigma, y_b, m_b)
            total_nll += nll.item() * len(x_b)

            mae = (torch.abs(mu - y_b) * m_b).sum().item()
            total_mae += mae
            total_count += m_b.sum().item()

    avg_nll = total_nll / max(1, len(loader.dataset))
    avg_mae = total_mae / max(1.0, total_count)
    return avg_nll, avg_mae


def train():
    device = torch.device("cpu")
    print(f"[*] Initializing RailwaySTGNN on {device}...", flush=True)
    model = RailwaySTGNN(num_nodes=num_nodes, in_features=4, hidden_dim=32).to(device)
    adj = adj_tensor.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)

    epochs = 12
    best_val_loss = float("inf")
    print(f"[*] Commencing training loop ({epochs} epochs)...", flush=True)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        train_nll = 0.0

        for x_b, y_b, m_b in train_loader:
            x_b, y_b, m_b = x_b.to(device), y_b.to(device), m_b.to(device)

            optimizer.zero_grad()
            mu, sigma = model(x_b, adj)
            loss = masked_gaussian_nll(mu, sigma, y_b, m_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

            train_nll += loss.item()

        avg_train_nll = train_nll / max(1, len(train_loader))
        val_nll, val_mae = compute_metrics(val_loader, model, adj, device)

        if val_nll < best_val_loss:
            best_val_loss = val_nll
            torch.save(model.state_dict(), WEIGHTS_PATH)

        duration = time.time() - ep_start
        print(f"--> Epoch {epoch:02d}/{epochs:02d} | Train NLL: {avg_train_nll:.4f} | Val NLL (Jul 2026): {val_nll:.4f} | Val MAE: {val_mae:.2f}m | Time: {duration:.2f}s", flush=True)

    print(f"\n[+] Training completed in {time.time() - t_start:.1f}s!")

    # Evaluate on held-out August 2026 test partition
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    test_nll, test_mae = compute_metrics(test_loader, model, adj, device)
    print(f"[*] Unseen Out-of-Sample Test Evaluation (August 2026):")
    print(f"    - Test NLL : {test_nll:.4f}")
    print(f"    - Test MAE : {test_mae:.2f} minutes")
    print(f"[+] Calibrated weights verified and saved to: {WEIGHTS_PATH}")


if __name__ == "__main__":
    train()
