"""
Synthetic Tamil Nadu railway network used for the RailSync prototype.

Real RailSync (per the project's other work) pulls this from the RailRadar
API into a PostgreSQL dataset. This prototype ships a small, self-contained
synthetic network so the pipeline is runnable and testable offline, with the
same shape (stations, scheduled legs, historical delay stats) that a real
data-loader would produce. Swap `LEGS` / `ITINERARIES` for real query
results and the rest of the pipeline (delay_model.py, reliability.py,
ranker.py) is unchanged.
"""

# Each leg = one scheduled train segment between two stations.
# historical_delay_mean_min / historical_delay_std_min are *added* delay
# (minutes) accrued over that specific segment, based on synthetic
# "historical" stats — stand-in for what the ST-GNN would learn per
# station/segment in the real system.
LEGS = {
    # (train_id, from, to): leg dict
    ("T1", "MAS", "VM"): dict(
        train_name="Chennai–Villupuram Exp", from_station="Chennai (MAS)", to_station="Villupuram",
        sched_dep="06:00", sched_arr="08:30", hist_delay_mean=6, hist_delay_std=8,
    ),
    ("T1", "VM", "TPJ"): dict(
        train_name="Chennai–Trichy Exp", from_station="Villupuram", to_station="Trichy",
        sched_dep="08:45", sched_arr="11:15", hist_delay_mean=8, hist_delay_std=10,
    ),
    ("T2", "TPJ", "DG"): dict(
        train_name="Trichy–Dindigul Pass", from_station="Trichy", to_station="Dindigul",
        sched_dep="11:40", sched_arr="12:40", hist_delay_mean=4, hist_delay_std=6,
    ),
    ("T3", "DG", "MDU"): dict(
        train_name="Dindigul–Madurai Pass", from_station="Dindigul", to_station="Madurai",
        sched_dep="13:00", sched_arr="13:50", hist_delay_mean=3, hist_delay_std=5,
    ),
    ("T4", "MAS", "TPJ"): dict(
        train_name="Chennai–Trichy Direct", from_station="Chennai (MAS)", to_station="Trichy",
        sched_dep="06:00", sched_arr="14:30", hist_delay_mean=18, hist_delay_std=20,
    ),
    ("T5", "TPJ", "MDU"): dict(
        train_name="Trichy–Madurai Exp", from_station="Trichy", to_station="Madurai",
        sched_dep="14:45", sched_arr="14:45", hist_delay_mean=7, hist_delay_std=9,
    ),
    ("T6", "MAS", "CBE"): dict(
        train_name="Chennai–Coimbatore SF", from_station="Chennai (MAS)", to_station="Coimbatore",
        sched_dep="06:15", sched_arr="14:00", hist_delay_mean=15, hist_delay_std=18,
    ),
    ("T7", "CBE", "DG"): dict(
        train_name="Coimbatore–Dindigul Pass", from_station="Coimbatore", to_station="Dindigul",
        sched_dep="14:30", sched_arr="16:45", hist_delay_mean=9, hist_delay_std=11,
    ),
    ("T8", "DG", "MDU2"): dict(
        train_name="Dindigul–Madurai Local", from_station="Dindigul", to_station="Madurai",
        sched_dep="17:10", sched_arr="18:00", hist_delay_mean=4, hist_delay_std=5,
    ),
    ("T9", "MAS", "DG"): dict(
        train_name="Chennai–Dindigul Exp", from_station="Chennai (MAS)", to_station="Dindigul",
        sched_dep="07:00", sched_arr="14:30", hist_delay_mean=20, hist_delay_std=25,
    ),
    ("T10", "DG", "MDU3"): dict(
        train_name="Dindigul–Madurai Fast", from_station="Dindigul", to_station="Madurai",
        sched_dep="14:50", sched_arr="15:40", hist_delay_mean=4, hist_delay_std=5,
    ),
}

# Candidate itineraries Chennai -> Madurai, each a list of leg keys taken in
# order. This mirrors the four options shown in the PDF (via Trichy,
# via Villupuram, via Dindigul, via Coimbatore).
ITINERARIES = {
    "Option A (via Trichy+Dindigul)": [("T1", "MAS", "VM"), ("T1", "VM", "TPJ"), ("T2", "TPJ", "DG"), ("T3", "DG", "MDU")],
    "Option B (via Villupuram+Trichy direct)": [("T1", "MAS", "VM"), ("T1", "VM", "TPJ"), ("T5", "TPJ", "MDU")],
    "Option C (via Dindigul, fastest)": [("T9", "MAS", "DG"), ("T10", "DG", "MDU3")],
    "Option D (via Coimbatore, conservative)": [("T6", "MAS", "CBE"), ("T7", "CBE", "DG"), ("T8", "DG", "MDU2")],
}

MIN_TRANSFER_BUFFER_MIN = 10  # minimum dwell time needed to physically change platforms/trains
