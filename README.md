# RailSync — Prototype

A working prototype of the "Predict → Evaluate → Rank" pipeline described in
the RailSync PDF (Spatio-Temporal GNN framework for railway delay prediction
and reliable itinerary ranking).

## What this implements

1. **Predict Delay Evolution** (`delay_model.py`) — Monte Carlo simulation of
   how a train's delay evolves segment-by-segment, producing a full
   distribution (not a point estimate) at each station, with P10/P50/P90
   reported.
2. **Evaluate Connection Reliability** (`reliability.py`) — implements the
   PDF's formula `R_c = P(S2 - A1 >= m_x)` per interchange, using the
   simulated arrival-delay distribution.
3. **Rank Itineraries** (`ranker.py`) — scores each candidate itinerary on a
   configurable fastest ⟷ most-reliable trade-off and labels the top picks
   (Recommended / Fastest / Most Reliable / Balanced), matching the PDF's
   ranked-recommendations output.

Served via a Flask app with a small web UI (`app.py`, `templates/index.html`)
and a JSON API (`POST /api/rank`).

## Run it

```bash
pip install flask numpy
python app.py
```

Then open **http://localhost:5000**. Set a current delay (minutes) and a
fastest↔reliable slider, click "Predict & Rank" — four candidate
Chennai→Madurai itineraries (mirroring the PDF's example) are simulated and
ranked live.

To test the API directly:
```bash
curl -X POST http://localhost:5000/api/rank \
  -H "Content-Type: application/json" \
  -d '{"initial_delay_min": 30, "risk_weight": 0.5}'
```

## Assumptions & scope decisions

- **No stated tech stack** in the PDF → chose Python/Flask (matches your
  existing Flask/React freelance stack) + a plain JS frontend, no build step
  needed.
- **No real dataset available here.** The PDF's actual pipeline (per your
  other RailSync work) pulls from the RailRadar API into PostgreSQL. This
  prototype ships a small synthetic Tamil Nadu network (`data.py`: Chennai,
  Villupuram, Trichy, Dindigul, Coimbatore, Madurai) with synthetic
  historical delay stats per segment, matching the PDF's Chennai→Madurai
  example and its 4 itinerary options. Swapping in real crawled data only
  requires replacing `LEGS`/`ITINERARIES` in `data.py`.
- **ST-GNN replaced with a documented stand-in.** Training an actual graph
  neural network needs a labeled historical dataset this prototype doesn't
  have. `delay_model.py` implements the same *interface* (delay in →
  predicted delay distribution out) using a stochastic propagation +
  Monte Carlo model, so a trained GNN can later be substituted without
  touching `reliability.py` or `ranker.py`.
- **Transfers assumed independent** when compounding multi-leg itinerary
  reliability — a simplification noted in `reliability.py`; a fuller model
  would simulate a "missed connection → rebook" branch.
- **Minimum transfer buffer** fixed at 10 minutes (`MIN_TRANSFER_BUFFER_MIN`
  in `data.py`) — adjust per station if real platform-change times differ.
- Only the **core pipeline** (predict/evaluate/rank) was built, since it's
  the paper's central contribution; the PDF's optional extras (weather/live
  data ingestion, conformal calibration, real-time replanning) are out of
  scope for this prototype and noted here rather than stubbed.

## Files

```
app.py              Flask routes + JSON API
data.py             Synthetic station/leg network + itinerary options
delay_model.py       Monte Carlo delay propagation (ST-GNN stand-in)
reliability.py       R_c connection-reliability calculation
ranker.py             Full predict→evaluate→rank pipeline + scoring
templates/index.html  Web UI
```
