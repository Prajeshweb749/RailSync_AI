/* static/js/indirect.js - RailSync AI Indirect Itinerary Search & Ranking Client */

document.addEventListener("DOMContentLoaded", function() {
  loadStations();
  
  const riskSlider = document.getElementById("riskSlider");
  const riskLabel = document.getElementById("riskLabel");
  if (riskSlider && riskLabel) {
    riskSlider.addEventListener("input", function() {
      const val = parseFloat(riskSlider.value);
      if (val <= 0.3) riskLabel.textContent = `Fastest Journey Priority (${val})`;
      else if (val >= 0.7) riskLabel.textContent = `Maximum Connection Reliability Priority (${val})`;
      else riskLabel.textContent = `Balanced Priority (${val})`;
    });
  }
});

async function loadStations() {
  try {
    const resp = await fetch("/api/stations");
    const data = await resp.json();
    if (data.stations) {
      populateStationDropdowns(data.stations);
    }
  } catch (err) {
    console.error("Failed to load stations:", err);
  }
}

function populateStationDropdowns(stations) {
  const origSel = document.getElementById("originSel");
  const destSel = document.getElementById("destSel");
  if (!origSel || !destSel) return;

  origSel.innerHTML = '<option value="">Select Origin Station...</option>';
  destSel.innerHTML = '<option value="">Select Destination Station...</option>';

  stations.forEach(s => {
    const opt1 = document.createElement("option");
    opt1.value = s.code;
    opt1.textContent = `${s.name} (${s.code})`;
    origSel.appendChild(opt1);

    const opt2 = document.createElement("option");
    opt2.value = s.code;
    opt2.textContent = `${s.name} (${s.code})`;
    destSel.appendChild(opt2);
  });

  // Defaults
  origSel.value = "HWH";
  destSel.value = "NDLS";
}

async function searchIndirectItineraries() {
  const origin = document.getElementById("originSel").value;
  const destination = document.getElementById("destSel").value;
  const delay = parseFloat(document.getElementById("delayInput").value || 0);
  const riskWeight = parseFloat(document.getElementById("riskSlider").value || 0.5);
  const journeyDate = document.getElementById("dateInput").value || "2026-09-10";

  const resultsBox = document.getElementById("resultsContainer");
  const errorBox = document.getElementById("errorBanner");
  errorBox.style.display = "none";

  if (!origin || !destination) {
    showError("Please select both origin and destination stations.");
    return;
  }
  if (origin === destination) {
    showError("Origin and destination stations cannot be identical.");
    return;
  }

  resultsBox.innerHTML = `
    <div style="text-align:center; padding: 48px; color: #64748B;">
      <div style="font-size: 2.5rem; margin-bottom: 12px;" class="spin">🚄</div>
      <h3 style="font-size: 1.15rem; color:#0F172A; margin:0 0 6px;">Querying candidate itineraries &amp; running ML predictions...</h3>
      <p style="font-size:0.9rem; margin:0;">Simulating multi-leg delay propagation and connection reliability (R_c)</p>
    </div>
  `;

  try {
    const resp = await fetch("/api/rank-indirect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        origin: origin,
        destination: destination,
        initial_delay_min: delay,
        risk_weight: riskWeight,
        journey_date: journeyDate,
      })
    });

    const data = await resp.json();
    if (!resp.ok || !data.success) {
      throw new Error(data.error || "Failed to calculate itinerary rankings.");
    }

    renderItineraries(data.results);
  } catch (err) {
    resultsBox.innerHTML = "";
    showError(err.message);
  }
}

function showError(msg) {
  const errorBox = document.getElementById("errorBanner");
  errorBox.textContent = "Error: " + msg;
  errorBox.style.display = "block";
}

function renderItineraries(results) {
  const box = document.getElementById("resultsContainer");
  box.innerHTML = "";

  if (!results || results.length === 0) {
    box.innerHTML = `
      <div style="text-align:center; padding: 48px; background:white; border-radius:16px; border:1px solid #E2E8F0;">
        <h3>No valid itineraries found</h3>
        <p style="color:#64748B;">Try selecting a different origin or destination station pair.</p>
      </div>
    `;
    return;
  }

  results.forEach((it, idx) => {
    const relPct = Math.round(it.reliability * 100);
    const card = document.createElement("div");
    card.className = "itinerary-card";

    let legsHtml = "";
    it.legs.forEach(leg => {
      legsHtml += `
        <div class="leg-card">
          <div class="leg-header">
            <span>🚆 ${leg.train_name} (${leg.train_number})</span>
            <span style="color:#2563EB;">${leg.from_name} → ${leg.to_name}</span>
          </div>
          <div class="leg-details">
            <div>Sched. Departure: <b>${leg.sched_dep}</b> | Sched. Arrival: <b>${leg.sched_arr}</b></div>
            <div>Predicted Arrival Delay: <b style="color:${leg.predicted_arr_delay_min > 15 ? '#EF4444' : '#10B981'};">+${leg.predicted_arr_delay_min} min</b></div>
          </div>
        </div>
      `;

      if (leg.transfer_info) {
        const tr = leg.transfer_info;
        const trRelPct = Math.round(tr.connection_reliability * 100);
        legsHtml += `
          <div class="transfer-banner">
            <div>
              🔄 Interchange Transfer at <b>${tr.interchange_name}</b>
              <span style="margin-left: 12px; opacity:0.85;">Buffer: ${tr.scheduled_buffer_min} mins</span>
            </div>
            <div>
              Connection Feasibility: <b style="color:${tr.is_feasible ? '#059669' : '#DC2626'};">${trRelPct}% (R_c)</b>
            </div>
          </div>
        `;
      }
    });

    card.innerHTML = `
      <div class="itinerary-head">
        <div class="itinerary-title-wrap">
          <h3>${it.name}</h3>
          <span class="rank-badge ${it.label.replace(/\s+/g, '')}">${it.label}</span>
        </div>
        <div style="font-size:0.88rem; color:#64748B;">
          Rank Score: <b>${it.score}</b>
        </div>
      </div>

      <div class="itinerary-metrics">
        <div class="metric-item">
          <span class="metric-label">Expected Total Journey</span>
          <span class="metric-val">${(it.expected_total_min / 60).toFixed(1)} hrs</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Transfers</span>
          <span class="metric-val">${it.num_transfers} transfer${it.num_transfers !== 1 ? 's' : ''}</span>
        </div>
        <div class="rel-bar-container">
          <div style="display:flex; justify-content:space-between; font-size:0.8rem; font-weight:700;">
            <span>Connection Reliability (R_c)</span>
            <span style="color:#059669">${relPct}%</span>
          </div>
          <div class="rel-bar-wrap">
            <div class="rel-bar-fill" style="width: ${relPct}%;"></div>
          </div>
        </div>
      </div>

      <div class="legs-timeline">
        <h4 style="font-size:0.9rem; font-weight:700; color:#334155; margin:0 0 12px; text-transform:uppercase;">Journey Legs &amp; Connections</h4>
        ${legsHtml}
      </div>

      <div class="p10-percentiles">
        <span>📊 Final Arrival Delay Distribution:</span>
        <span>P10: <b>${it.final_arrival_delay_p10_p50_p90.P10}m</b></span>
        <span>P50 (Median): <b>${it.final_arrival_delay_p10_p50_p90.P50}m</b></span>
        <span>P90: <b>${it.final_arrival_delay_p10_p50_p90.P90}m</b></span>
      </div>
    `;

    box.appendChild(card);
  });
}
