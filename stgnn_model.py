"""
stgnn_model.py
==============
Railway Spatio-Temporal Graph Neural Network (ST-GNN).
Outputs parametric delay distribution (mu, sigma) across station nodes.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """
    Spatial Graph Convolution: H' = A_hat * H * W + b
    Propagates track/station congestion across physical rail neighbors.
    """
    def __init__(self, in_features: int, out_features: int):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x shape:   [Batch, T, Nodes, In_Features]
        # adj shape: [Nodes, Nodes]
        # returns:   [Batch, T, Nodes, Out_Features]
        b, t, n, f = x.shape
        x_reshaped = x.contiguous().reshape(b * t, n, f)
        
        # Batch matrix multiplication: (B*T, N, N) x (B*T, N, F) -> (B*T, N, F)
        support = torch.bmm(adj.unsqueeze(0).expand(b * t, -1, -1), x_reshaped)
        out = torch.matmul(support, self.weight) + self.bias
        return out.contiguous().reshape(b, t, n, -1)


class TemporalConv(nn.Module):
    """
    Temporal 1D Convolution with Gated Linear Unit (GLU).
    Uses left-sided causal padding to prevent future-step leakage.
    """
    def __init__(self, in_features: int, out_features: int, kernel_size: int = 3):
        super(TemporalConv, self).__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=in_features, 
            out_channels=out_features * 2, 
            kernel_size=kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, T, Nodes, Features] -> [Batch * Nodes, Features, T]
        b, t, n, f = x.shape
        x_trans = x.permute(0, 2, 3, 1).reshape(b * n, f, t)
        
        # Causal padding on left side only
        x_padded = F.pad(x_trans, (self.pad, 0))
        conv_out = self.conv(x_padded)
        
        # Split channels for Gated Linear Unit
        p, q = torch.chunk(conv_out, 2, dim=1)
        out = p * torch.sigmoid(q)
        
        # Reshape back to [Batch, T, Nodes, Out_Features]
        return out.contiguous().reshape(b, n, -1, t).permute(0, 3, 1, 2).contiguous()


class STGNNBlock(nn.Module):
    """
    Spatio-Temporal Pipeline: Temporal-Conv -> Spatial-GCN -> Temporal-Conv -> LayerNorm
    """
    def __init__(self, in_features: int, hidden_dim: int):
        super(STGNNBlock, self).__init__()
        self.tconv1 = TemporalConv(in_features, hidden_dim)
        self.gconv = GraphConv(hidden_dim, hidden_dim)
        self.tconv2 = TemporalConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.tconv1(x))
        h = F.relu(self.gconv(h, adj))
        h = F.relu(self.tconv2(h))
        return self.norm(h)


class RailwaySTGNN(nn.Module):
    """
    Full Spatio-Temporal Network.
    Outputs:
      - mu:    Predicted mean delay in minutes (>= 0 via ReLU)
      - sigma: Predicted uncertainty/dispersion in minutes (> 0 via Softplus)
    """
    def __init__(self, num_nodes: int = 4626, in_features: int = 4, hidden_dim: int = 32):
        super(RailwaySTGNN, self).__init__()
        self.block1 = STGNNBlock(in_features, hidden_dim)
        self.block2 = STGNNBlock(hidden_dim, hidden_dim)
        
        self.fc_mu = nn.Linear(hidden_dim, 1)
        self.fc_sigma = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # x:   [Batch, T, Nodes, Features]
        # adj: [Nodes, Nodes]
        h = self.block1(x, adj)
        h = self.block2(h, adj)
        
        last_step = h[:, -1, :, :]  # Last time step: [Batch, Nodes, Hidden]
        
        mu = F.relu(self.fc_mu(last_step)).squeeze(-1)            # Mean delay >= 0
        sigma = F.softplus(self.fc_sigma(last_step)).squeeze(-1) + 1e-3  # Uncertainty > 0
        return mu, sigma


if __name__ == "__main__":
    print("[*] Running RailwaySTGNN architecture verification...")

    # Locate adjacency matrix
    adj_paths = [
        "adj_matrix.pt",
        os.path.join(os.path.expanduser("~"), "Downloads", "adj_matrix.pt"),
        "adj_matrix.npz",
        os.path.join(os.path.expanduser("~"), "Downloads", "adj_matrix.npz")
    ]
    
    adj = None
    for p in adj_paths:
        if os.path.exists(p):
            print(f"[+] Found graph topology file at: {p}")
            if p.endswith(".pt"):
                adj = torch.load(p)
            elif p.endswith(".npz"):
                import numpy as np
                data = np.load(p)["adj_matrix"]
                adj = torch.tensor(data, dtype=torch.float32)
            break

    if adj is None:
        print("[!] No adjacency file found on disk. Initializing synthetic [4626, 4626] tensor...")
        adj = torch.eye(4626, dtype=torch.float32)

    num_nodes = adj.shape[0]
    batch_size = 1
    seq_len = 12   # Past 12 time slices (e.g. 6 hours of half-hour bins)
    features = 4   # (dep_count, arr_count, sin_time, cos_time)

    print(f"[*] Initializing model: {num_nodes} stations, {features} node features, hidden_dim=32...")
    model = RailwaySTGNN(num_nodes=num_nodes, in_features=features, hidden_dim=32)
    model.eval()

    # Create dummy input tensor: [Batch=1, Time=12, Stations=4626, Features=4]
    dummy_x = torch.randn(batch_size, seq_len, num_nodes, features)

    print("[*] Passing tensors through spatial convolutions and temporal GLU blocks...")
    with torch.no_grad():
        pred_mu, pred_sigma = model(dummy_x, adj)

    print("\n[+] FORWARD PASS SUCCESSFUL:")
    print(f"    - Input Tensor shape                   : {list(dummy_x.shape)}")
    print(f"    - Adjacency Matrix shape               : {list(adj.shape)}")
    print(f"    - Output Mean Delay (mu) shape         : {list(pred_mu.shape)}")
    print(f"    - Output Uncertainty (sigma) shape     : {list(pred_sigma.shape)}")
    print(f"    - Station 0 Sample Prediction          : {pred_mu[0, 0].item():.2f} min (± {pred_sigma[0, 0].item():.2f})")
    print(f"    - Station 3002 (NDLS) Sample Prediction: {pred_mu[0, min(3002, num_nodes - 1)].item():.2f} min (± {pred_sigma[0, min(3002, num_nodes - 1)].item():.2f})")
