import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from delay_model import simulate_leg_delay
import numpy as np


def evaluate_connection(leg1_key, leg2_key, scheduled_layover_min=45):
    """
    Simulates a 2-leg multi-hop itinerary using ST-GNN inference
    to assess the probability of a missed connection.
    """
    print(f"\n[*] Evaluating Itinerary:")
    print(f"    Leg 1: Train {leg1_key[0]} ({leg1_key[1]} -> {leg1_key[2]})")
    print(f"    Transfer Window: {scheduled_layover_min} minutes at {leg1_key[2]}")
    print(f"    Leg 2: Train {leg2_key[0]} ({leg2_key[1]} -> {leg2_key[2]})")

    n_samples = 5000

    # 1. Simulate Leg 1 arrival delay at interchange
    leg1_arrival_delays = simulate_leg_delay(leg1_key, n_samples=n_samples)

    # 2. Connection feasibility check
    # Missed if arrival delay exceeds available layover buffer
    missed_mask = leg1_arrival_delays > scheduled_layover_min
    miss_prob = np.mean(missed_mask)

    # 3. Simulate Leg 2 for passengers who successfully make the connection
    leg2_delays = simulate_leg_delay(
        leg2_key,
        incoming_delay_samples=leg1_arrival_delays,
        n_samples=n_samples
    )

    print(f"\n[+] Results:")
    print(f"    Leg 1 Mean Delay       : {np.mean(leg1_arrival_delays):.2f} min (90th Pct: {np.percentile(leg1_arrival_delays, 90):.2f} min)")
    print(f"    Missed Transfer Risk   : {miss_prob * 100:.2f}%")
    print(f"    Final Dest Mean Delay  : {np.mean(leg2_delays):.2f} min")


if __name__ == "__main__":
    # Example multi-hop itinerary via major interchange:
    # Leg 1: NDLS -> BPL (New Delhi to Bhopal)
    # Leg 2: BPL -> NGP  (Bhopal to Nagpur)
    evaluate_connection(
        leg1_key=("12626", "NDLS", "BPL"),
        leg2_key=("12722", "BPL", "NGP"),
        scheduled_layover_min=30
    )
