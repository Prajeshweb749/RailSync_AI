"""
Connection reliability, per the PDF's formula:

    R_c = P(S2 - A1 >= m_x)

where A1 is the actual arrival time of the incoming train, S2 is the
scheduled departure of the outgoing train, and m_x is the minimum required
transfer time. We evaluate this per-transfer using the Monte Carlo arrival
delay distribution from delay_model.py, then combine transfers into an
overall itinerary reliability.
"""
import numpy as np

from data import LEGS, MIN_TRANSFER_BUFFER_MIN


def _to_minutes(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def leg_pairs_needing_transfer(leg_sequence):
    """Consecutive leg pairs where the train_id changes -> a real transfer."""
    pairs = []
    for a, b in zip(leg_sequence, leg_sequence[1:]):
        if a[0] != b[0]:
            pairs.append((a, b))
    return pairs


def transfer_reliability(leg_a, leg_b, arrival_delay_samples_a, min_buffer=MIN_TRANSFER_BUFFER_MIN):
    """
    R_c for one interchange: fraction of Monte Carlo runs where the
    passenger still makes leg_b given leg_a's simulated arrival delay.
    """
    sched_arr_a = _to_minutes(LEGS[leg_a]["sched_arr"])
    sched_dep_b = _to_minutes(LEGS[leg_b]["sched_dep"])
    scheduled_buffer = sched_dep_b - sched_arr_a
    if scheduled_buffer < 0:
        scheduled_buffer += 24 * 60  # overnight wrap

    available = scheduled_buffer - arrival_delay_samples_a
    made_it = available >= min_buffer
    r_c = float(np.mean(made_it))
    return {
        "leg_a": leg_a, "leg_b": leg_b,
        "scheduled_buffer_min": scheduled_buffer,
        "min_required_buffer_min": min_buffer,
        "reliability": r_c,
    }


def itinerary_reliability(leg_sequence, samples_by_leg, min_buffer=MIN_TRANSFER_BUFFER_MIN):
    """
    Overall itinerary reliability = product of each transfer's reliability
    (transfers treated as independent — a simplification; a fuller model
    would carry a "missed connection" branch into a rebooking simulation).
    Itineraries with zero transfers (single train, or all same train_id)
    are reliability 1.0 by definition — nothing to miss.
    """
    pairs = leg_pairs_needing_transfer(leg_sequence)
    if not pairs:
        return 1.0, []
    details = []
    overall = 1.0
    for leg_a, leg_b in pairs:
        d = transfer_reliability(leg_a, leg_b, samples_by_leg[leg_a], min_buffer)
        details.append(d)
        overall *= d["reliability"]
    return overall, details
