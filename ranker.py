"""
Rank candidate itineraries by a reliability-vs-speed trade-off, per the
PDF's "Rank Itineraries" step. Exposes the full pipeline entry point used
by app.py: predict -> evaluate -> rank.
"""
import numpy as np

from data import LEGS, ITINERARIES, MIN_TRANSFER_BUFFER_MIN
from delay_model import predict_delay_distribution, quantiles
from reliability import itinerary_reliability, leg_pairs_needing_transfer, _to_minutes


def scheduled_journey_minutes(leg_sequence):
    start = _to_minutes(LEGS[leg_sequence[0]]["sched_dep"])
    end = _to_minutes(LEGS[leg_sequence[-1]]["sched_arr"])
    total = end - start
    if total < 0:
        total += 24 * 60
    return total


def evaluate_itinerary(name, leg_sequence, initial_delay_min=0, n_samples=3000,
                        min_buffer=MIN_TRANSFER_BUFFER_MIN, seed=None):
    samples_by_leg = predict_delay_distribution(
        leg_sequence, initial_delay_min=initial_delay_min, n_samples=n_samples, seed=seed
    )
    final_leg = leg_sequence[-1]
    final_delay_q = quantiles(samples_by_leg[final_leg])

    overall_reliability, transfer_details = itinerary_reliability(
        leg_sequence, samples_by_leg, min_buffer=min_buffer
    )

    sched_min = scheduled_journey_minutes(leg_sequence)
    expected_total_min = sched_min + final_delay_q["P50"]
    n_transfers = len(leg_pairs_needing_transfer(leg_sequence))

    return {
        "name": name,
        "legs": [
            {"leg": lk, "train": LEGS[lk]["train_name"], "from": LEGS[lk]["from_station"],
             "to": LEGS[lk]["to_station"], "sched_dep": LEGS[lk]["sched_dep"], "sched_arr": LEGS[lk]["sched_arr"]}
            for lk in leg_sequence
        ],
        "scheduled_journey_min": sched_min,
        "expected_total_min": round(expected_total_min, 1),
        "final_arrival_delay_p10_p50_p90": final_delay_q,
        "num_transfers": n_transfers,
        "transfer_details": transfer_details,
        "reliability": round(overall_reliability, 3),
    }


def rank_itineraries(itineraries=None, initial_delay_min=0, risk_weight=0.5,
                      n_samples=3000, min_buffer=MIN_TRANSFER_BUFFER_MIN, seed=42):
    """
    risk_weight in [0,1]: 0 = purely fastest, 1 = purely most reliable.
    Score = risk_weight * reliability - (1 - risk_weight) * normalized_time.
    """
    itineraries = itineraries or ITINERARIES
    evaluated = [
        evaluate_itinerary(name, legs, initial_delay_min, n_samples, min_buffer, seed)
        for name, legs in itineraries.items()
    ]

    times = np.array([e["expected_total_min"] for e in evaluated])
    t_min, t_max = times.min(), times.max()
    for e, t in zip(evaluated, times):
        norm_time = 0.0 if t_max == t_min else (t - t_min) / (t_max - t_min)
        e["score"] = round(risk_weight * e["reliability"] - (1 - risk_weight) * norm_time, 4)

    evaluated.sort(key=lambda e: e["score"], reverse=True)

    if evaluated:
        best_reliability = max(evaluated, key=lambda e: e["reliability"])["name"]
        fastest = min(evaluated, key=lambda e: e["expected_total_min"])["name"]
        for e in evaluated:
            if e["name"] == evaluated[0]["name"]:
                e["label"] = "Recommended"
            elif e["name"] == best_reliability:
                e["label"] = "Most Reliable"
            elif e["name"] == fastest:
                e["label"] = "Fastest"
            else:
                e["label"] = "Balanced"
    return evaluated
