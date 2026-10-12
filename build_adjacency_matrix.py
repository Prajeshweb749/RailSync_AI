"""
build_adjacency_matrix.py (Pure NumPy Version)
==============================================
Builds the static spatial graph topology using only NumPy and psycopg2:
1. Queries network_edges from PostgreSQL.
2. Creates station_index.json mapping (station_code -> int).
3. Normalizes using D^(-1/2) * A_tilde * D^(-1/2).
4. Saves adj_matrix.npz.
"""

import json
import os
import psycopg2
import numpy as np

DB_CONFIG = {
    "dbname": os.getenv("PGDATABASE", "SIH_Railway_DB"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "postgres"),
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", 5432))
}

SIGMA_KM = 50.0  # Gaussian threshold distance scaling


def build_graph():
    print("[*] Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        SELECT source_station, target_station, distance_km, train_frequency_count
        FROM network_edges
        WHERE distance_km > 0;
    """)
    edges = cur.fetchall()
    cur.close()
    conn.close()

    print(f"[+] Loaded {len(edges)} directed edges from network_edges.")

    # 1. Build contiguous station index mapping
    stations = set()
    for src, tgt, _, _ in edges:
        stations.add(src)
        stations.add(tgt)

    station_list = sorted(list(stations))
    station_to_idx = {st: i for i, st in enumerate(station_list)}
    num_nodes = len(station_list)
    print(f"[+] Graph contains {num_nodes} unique station nodes.")

    # Save mapping for lookups
    with open("station_index.json", "w") as f:
        json.dump(station_to_idx, f, indent=2)
    print("[+] Saved station_index.json")

    # 2. Build Gaussian distance-weighted adjacency matrix
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for src, tgt, dist_km, _ in edges:
        u = station_to_idx[src]
        v = station_to_idx[tgt]
        dist = float(dist_km)

        # Standard Gaussian radial basis weighting
        weight = np.exp(- (dist ** 2) / (SIGMA_KM ** 2))

        # Bidirectional track assumption
        A[u, v] = max(A[u, v], weight)
        A[v, u] = max(A[v, u], weight)

    # 3. Add Self-Loops (A_tilde = A + I)
    A_tilde = A + np.eye(num_nodes, dtype=np.float32)

    # 4. Degree Normalization: D^(-1/2) * A_tilde * D^(-1/2)
    d = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.power(d, -0.5, where=d > 0)
    d_inv_sqrt[d == 0.0] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)

    A_normalized = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    # 5. Persist normalized adjacency matrix as compressed NumPy archive
    np.savez_compressed("adj_matrix.npz", adj_matrix=A_normalized)
    print(f"[+] Successfully saved adj_matrix.npz with shape: {A_normalized.shape}")


if __name__ == "__main__":
    build_graph()
