/**
 * predict.js — RailSync AI  |  Track & Predict Page
 * Handles: train selection → journey setup → live-predict API → render results
 */

/* ── State ─────────────────────────────────────────────────────── */
const State = {
  selectedTrain:  null,   // { train_number, name, color, emoji, ... }
  boardingCode:   null,
  destCode:       null,
  journeyDate:    null,   // Specific departure date for multi-day train instances (e.g. "2026-10-11")
  activeInstances: [],    // Array of active/candidate dated runs
  lastResult:     null,   // last API response
  fullRoute:      [],     // full_journey array
  userRoute:      [],     // user_journey array
  filteredRoute:  [],     // for table search
  currentPage:    1,
  rowsPerPage:    30,
  autoRefreshTimer: null,
  chartInstance:  null,
};

/* ── Train colour mapping ──────────────────────────────────────── */
const TRAIN_STYLES = {
  "12301": { emoji: "🔵", bg: "#EBF2FA", accent: "#1D3557" },
  "12627": { emoji: "🟢", bg: "#EBF7F0", accent: "#1A6640" },
  "12628": { emoji: "🟣", bg: "#F2EBF7", accent: "#4A1D7A" },
  "12951": { emoji: "🔴", bg: "#FAF0EB", accent: "#7A1D1D" },
  "16089": { emoji: "🟡", bg: "#FAF7EB", accent: "#7A6018" },
};

/* ── Init ───────────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  loadTrains();
  setupStationWatchers();
});

/* ══════════════════════════════════════════════════════════════════
   STEP 1 — Train Selection
══════════════════════════════════════════════════════════════════ */
async function loadTrains() {
  try {
    const res  = await fetch("/api/trains");
    const data = await res.json();
    renderTrainCards(data.trains || []);
  } catch (e) {
    document.getElementById("trainGrid").innerHTML =
      `<div style="color:#D03C3C;font-size:.85rem;padding:12px;">
         ⚠ Could not load trains. Is the server running?
       </div>`;
  }
}

let searchDebounceTimer = null;
function onTrainSearch(query) {
  clearTimeout(searchDebounceTimer);
  searchDebounceTimer = setTimeout(async () => {
    try {
      const res = await fetch(`/api/trains?search=${encodeURIComponent(query)}`);
      const data = await res.json();
      renderTrainCards(data.trains || []);
    } catch (e) {
      console.error("Search error:", e);
    }
  }, 250);
}

function renderTrainCards(trains) {
  const grid = document.getElementById("trainGrid");
  grid.innerHTML = "";

  if (!trains || trains.length === 0) {
    grid.innerHTML = `<div style="color:#64748B;font-size:.9rem;padding:16px;text-align:center;grid-column:1/-1;">No trains found matching your query.</div>`;
    return;
  }

  trains.forEach(t => {
    const st  = TRAIN_STYLES[t.train_number] || { emoji: "🚆", bg: "#EBF2FA", accent: "#1D3557" };
    const avg = t.avg_delay_min != null ? `~${t.avg_delay_min} min avg delay` : "";

    const card = document.createElement("div");
    card.className = "train-card";
    card.id = `tc-${t.train_number}`;
    card.setAttribute("role", "button");
    card.setAttribute("tabindex", "0");
    card.innerHTML = `
      <div class="tc-icon" style="background:${st.bg}">${st.emoji}</div>
      <div class="tc-info">
        <div class="tc-num">${t.train_number}</div>
        <div class="tc-name">${t.name}</div>
        <div class="tc-route">${t.source_name || t.source_code} → ${t.dest_name || t.dest_code}</div>
      </div>
      <div class="tc-right">
        <div class="tc-delay">${avg}</div>
        <div class="tc-check" id="chk-${t.train_number}"></div>
      </div>`;

    card.onclick = () => selectTrain(t, st);
    card.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") selectTrain(t, st); };
    grid.appendChild(card);
  });
}

function selectTrain(trainObj, style) {
  // Deselect all
  document.querySelectorAll(".train-card").forEach(c => c.classList.remove("selected"));

  // Select this one
  const card = document.getElementById(`tc-${trainObj.train_number}`);
  if (card) card.classList.add("selected");

  State.selectedTrain = { ...trainObj, ...style };
  document.getElementById("btnStep1Next").disabled = false;
  document.getElementById("btnStep1Next").onclick = () => goToStep(2);
}

/* ══════════════════════════════════════════════════════════════════
   STEP 2 — Journey Details
══════════════════════════════════════════════════════════════════ */
async function goToStep(n) {
  document.querySelectorAll(".wizard-step").forEach(s => s.classList.remove("active"));
  document.getElementById(`step${n}`).classList.add("active");

  // Breadcrumb
  updateBreadcrumb(n);

  if (n === 2 && State.selectedTrain) {
    populateSelectedTrainPill();
    await loadJourneyInstancesForTrain(State.selectedTrain.train_number);
    await loadStationsForTrain(State.selectedTrain.train_number, State.journeyDate);
  }
}

function updateBreadcrumb(step) {
  for (let i = 1; i <= 3; i++) {
    const el = document.getElementById(`bc${i}`);
    el.classList.remove("active", "done");
    if (i < step)  el.classList.add("done");
    if (i === step) el.classList.add("active");
  }
}

function populateSelectedTrainPill() {
  const t  = State.selectedTrain;
  document.getElementById("stpIcon").textContent = t.emoji || "🚄";
  document.getElementById("stpNum").textContent  = t.train_number;
  document.getElementById("stpName").textContent = t.name;
}

async function loadJourneyInstancesForTrain(trainNumber) {
  const grp = document.getElementById("journeyInstanceGrp");
  const list = document.getElementById("journeyInstanceList");
  if (!grp || !list) return;

  grp.style.display = "none";
  list.innerHTML = "";
  State.journeyDate = null;
  State.activeInstances = [];

  try {
    const res = await fetch(`/api/trains/${trainNumber}/instances`);
    const data = await res.json();

    if (data.success && data.instances && data.instances.length > 0) {
      State.activeInstances = data.instances;

      if (data.is_multiday) {
        grp.style.display = "block";
        list.innerHTML = "";

        // Default selection: if user hasn't selected yet, default to active instance
        if (!State.journeyDate) {
          const defaultInst = data.instances.find(i => i.is_active) || data.instances[0];
          State.journeyDate = defaultInst.journey_start_date;
        }

        data.instances.forEach(inst => {
          const isSelected = (inst.journey_start_date === State.journeyDate);
          const card = document.createElement("div");
          card.className = `journey-card ${isSelected ? 'selected' : ''}`;
          card.id = `jc-${inst.journey_instance_id}`;

          const badgeClass = inst.is_active ? 'active-badge' : '';
          const currentText = inst.current_station_name
            ? ` • Near ${inst.current_station_name}`
            : (inst.current_station ? ` • Near ${inst.current_station}` : '');

          card.innerHTML = `
            <div class="jc-info">
              <div class="jc-head">
                <span class="jc-title">${inst.relative_label}</span>
                <span class="jc-badge ${badgeClass}">${inst.is_active ? 'Active En-Route' : 'Scheduled'}</span>
              </div>
              <div class="jc-sub">Day ${inst.day_of_journey}${currentText} • Origin: ${inst.origin_station} (${inst.scheduled_origin_departure ? inst.scheduled_origin_departure.substring(11, 16) : 'Scheduled'}) → ${inst.destination_station}</div>
            </div>
            <div class="jc-radio"></div>
          `;

          card.onclick = async () => {
            document.querySelectorAll(".journey-card").forEach(c => c.classList.remove("selected"));
            card.classList.add("selected");
            State.journeyDate = inst.journey_start_date;
            await loadStationsForTrain(State.selectedTrain.train_number, State.journeyDate);
          };

          list.appendChild(card);
        });
      } else {
        State.journeyDate = data.instances[0].journey_start_date;
        grp.style.display = "none";
      }
    }
  } catch (err) {
    console.warn("Could not load journey instances:", err);
    grp.style.display = "none";
  }
}

async function loadStationsForTrain(trainNumber, journeyDate = null) {
  // Show skeletons while loading
  showStationLoading(true);

  try {
    // Fetch live route from API (gives us all station codes + names from live data)
    const url = `/api/live/${trainNumber}${journeyDate ? '?journey_date=' + encodeURIComponent(journeyDate) : ''}`;
    const res  = await fetch(url);
    const data = await res.json();

    if (!data.success || !data.data.route) {
      throw new Error("No route in API response");
    }

    const route = data.data.route;
    populateStationDropdowns(route);
    showStationLoading(false);
  } catch (e) {
    // Fallback: use historical route
    try {
      const res  = await fetch(`/api/route/${trainNumber}`);
      const data = await res.json();
      const route = (data.route || []).map(s => ({
        station_code: s.station_code,
        station_name: s.station_code,  // historical route may not have names
        sequence:     s.station_sequence,
      }));
      populateStationDropdowns(route);
      showStationLoading(false);
    } catch (e2) {
      showStationError();
    }
  }
}

function populateStationDropdowns(route) {
  const boarding = document.getElementById("boardingSel");
  const dest     = document.getElementById("destSel");

  const savedBoarding = State.boardingCode;
  const savedDest = State.destCode;

  boarding.innerHTML = `<option value="">Select where you board...</option>`;
  dest.innerHTML     = `<option value="">Select your destination...</option>`;

  const haltStops = route.filter(s => s.is_halt !== false);
  const dropdownStops = haltStops.length > 0 ? haltStops : route;

  dropdownStops.forEach(s => {
    const label = s.station_name && s.station_name !== s.station_code
      ? `${s.station_name} (${s.station_code})`
      : s.station_code;
    boarding.innerHTML += `<option value="${s.station_code}">${label}</option>`;
    dest.innerHTML     += `<option value="${s.station_code}">${label}</option>`;
  });

  if (savedBoarding && dropdownStops.some(s => s.station_code === savedBoarding)) {
    boarding.value = savedBoarding;
    State.boardingCode = savedBoarding;
    dest.disabled = false;
  }
  if (savedDest && dropdownStops.some(s => s.station_code === savedDest)) {
    dest.value = savedDest;
    State.destCode = savedDest;
  }
  checkPredictReady();
}

function showStationLoading(show) {
  const boardLoad = document.getElementById("boardingLoadWrap");
  const boardSel  = document.getElementById("boardingSelWrap");
  const destLoad  = document.getElementById("destLoadWrap");
  const destSel   = document.getElementById("destSelWrap");

  if (show) {
    boardLoad.classList.remove("hidden"); boardSel.classList.add("hidden");
    destLoad.classList.remove("hidden");  destSel.classList.add("hidden");
  } else {
    boardLoad.classList.add("hidden"); boardSel.classList.remove("hidden");
    destLoad.classList.add("hidden");  destSel.classList.remove("hidden");
  }
}

function showStationError() {
  document.getElementById("boardingLoadWrap").innerHTML =
    `<div style="color:#D03C3C;font-size:.8rem;padding:10px 0;">⚠ Could not load stations.</div>`;
  document.getElementById("destLoadWrap").innerHTML = "";
}

function setupStationWatchers() {
  document.addEventListener("change", (e) => {
    if (e.target.id === "boardingSel") {
      State.boardingCode = e.target.value || null;
      const destSel = document.getElementById("destSel");
      destSel.disabled = !State.boardingCode;
      checkPredictReady();
    }
    if (e.target.id === "destSel") {
      State.destCode = e.target.value || null;
      checkPredictReady();
    }
  });
}

function checkPredictReady() {
  document.getElementById("btnPredict").disabled = !(State.boardingCode && State.destCode);
}

/* ══════════════════════════════════════════════════════════════════
   PREDICTION — Creative loader + API call
══════════════════════════════════════════════════════════════════ */
async function runPrediction() {
  if (!State.selectedTrain || !State.boardingCode || !State.destCode) return;

  showState("loading");
  updateBreadcrumb(3);
  await animateLoader();

  try {
    // Call /api/live-predict with exact journey instance date
    const res = await fetch("/api/live-predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        train_number:        State.selectedTrain.train_number,
        boarding_station:    State.boardingCode,
        destination_station: State.destCode,
        journey_date:        State.journeyDate || undefined,
      }),
    });

    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Prediction failed");

    State.lastResult    = data.data;
    State.fullRoute     = data.data.full_journey  || [];
    State.userRoute     = data.data.user_journey  || [];
    State.filteredRoute = [...State.fullRoute];
    State.currentPage   = 1;

    finishLoader();
    renderResults(data.data);
    startAutoRefresh();

  } catch (err) {
    showState("empty");
    document.querySelector(".empty-title").textContent = "⚠ Prediction Error";
    document.querySelector(".empty-desc").textContent  = err.message || "Could not fetch live data. Please try again.";
  }
}

async function refreshPrediction() {
  if (!State.selectedTrain || !State.boardingCode || !State.destCode) return;
  document.getElementById("btnRefresh").disabled = true;
  document.getElementById("btnRefresh").innerHTML = "↻ Refreshing...";

  try {
    const res = await fetch("/api/live-predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        train_number:        State.selectedTrain.train_number,
        boarding_station:    State.boardingCode,
        destination_station: State.destCode,
        journey_date:        State.journeyDate || undefined,
      }),
    });
    const data = await res.json();
    if (data.success) {
      State.lastResult    = data.data;
      State.fullRoute     = data.data.full_journey || [];
      State.userRoute     = data.data.user_journey || [];
      State.filteredRoute = [...State.fullRoute];
      State.currentPage   = 1;
      renderResults(data.data);
    }
  } catch (e) { /* silent fail on refresh */ }

  document.getElementById("btnRefresh").disabled = false;
  document.getElementById("btnRefresh").innerHTML =
    `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> Refresh Live Data`;
}

function startAutoRefresh() {
  if (State.autoRefreshTimer) clearInterval(State.autoRefreshTimer);
  let countdown = 60;
  const tick = () => {
    countdown--;
    const el = document.getElementById("autoRefreshTxt");
    if (el) el.textContent = `Auto-refreshes in ${countdown}s`;
    if (countdown <= 0) {
      countdown = 60;
      refreshPrediction();
    }
  };
  State.autoRefreshTimer = setInterval(tick, 1000);
}

/* ── Loader animation ─────────────────────────────────────────── */
function animateLoader() {
  return new Promise(resolve => {
    const stages  = [document.getElementById("ls1"), document.getElementById("ls2"), document.getElementById("ls3")];
    const fills   = [document.getElementById("railFill"), document.getElementById("railFill2")];
    const train   = document.getElementById("loaderTrain");
    const pct     = document.getElementById("loaderPct");
    const sub     = document.getElementById("loaderSub");
    const stops   = [1,2,3,4,5].map(i => document.getElementById(`lstop${i}`));
    const subs    = ["Connecting to RailRadar network…", "Downloading live train data…", "Running AI delay model…"];

    let progress = 0;
    stages.forEach(s => s && s.classList.remove("active", "done"));

    const step = (target, stageIdx, afterCb) => {
      if (stageIdx < stages.length && stages[stageIdx]) {
        stages[stageIdx].classList.add("active");
        if (sub) sub.textContent = subs[stageIdx] || "";
      }
      const interval = setInterval(() => {
        progress++;
        const p = Math.min(progress, target);
        fills.forEach(f => { if (f) f.style.width = p + "%"; });
        if (train) train.style.left = p + "%";
        if (pct)   pct.textContent  = p + "%";
        stops.forEach((s, i) => {
          if (s && p >= (i + 1) * 20) s.classList.add("passed");
        });
        if (p >= target) {
          clearInterval(interval);
          if (stageIdx < stages.length && stages[stageIdx]) {
            stages[stageIdx].classList.remove("active");
            stages[stageIdx].classList.add("done");
          }
          afterCb && afterCb();
        }
      }, 18);
    };

    step(33, 0, () =>
      setTimeout(() => step(66, 1, () =>
        setTimeout(() => { resolve(); }, 200)
      ), 300)
    );
  });
}

function finishLoader() {
  const fills = [document.getElementById("railFill"), document.getElementById("railFill2")];
  const train = document.getElementById("loaderTrain");
  const pct   = document.getElementById("loaderPct");
  const sub   = document.getElementById("loaderSub");
  const ls3   = document.getElementById("ls3");

  if (ls3)  ls3.classList.add("active");
  fills.forEach(f => { if (f) f.style.width = "100%"; });
  if (train) train.style.left = "100%";
  if (pct)   pct.textContent  = "100%";
  if (sub)   sub.textContent  = "Predictions ready!";

  setTimeout(() => showState("results"), 500);
}

/* ── State switching ─────────────────────────────────────────── */
function showState(which) {
  document.getElementById("stateEmpty").classList.add("hidden");
  document.getElementById("stateLoading").classList.add("hidden");
  document.getElementById("stateResults").classList.add("hidden");

  if (which === "empty")   document.getElementById("stateEmpty").classList.remove("hidden");
  if (which === "loading") document.getElementById("stateLoading").classList.remove("hidden");
  if (which === "results") document.getElementById("stateResults").classList.remove("hidden");
}

/* ══════════════════════════════════════════════════════════════════
   RENDER RESULTS
══════════════════════════════════════════════════════════════════ */
function renderResults(d) {
  renderInstanceSwitcher();
  renderLiveBanner(d);
  renderETACards(d);
  renderTimeline(d.user_journey || []);
  renderTable(d.full_journey  || []);
  renderChart(d.user_journey  || []);
}

function renderInstanceSwitcher() {
  const bar = document.getElementById("resultsInstanceBar");
  const pills = document.getElementById("resultsInstancePills");
  if (!bar || !pills) return;

  if (!State.activeInstances || State.activeInstances.length <= 1) {
    bar.style.display = "none";
    return;
  }

  bar.style.display = "flex";
  pills.innerHTML = "";

  State.activeInstances.forEach(inst => {
    const isSelected = (inst.journey_start_date === State.journeyDate);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.style.cssText = `
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 700;
      cursor: pointer;
      border: 1.5px solid ${isSelected ? 'var(--navy)' : 'rgba(29,53,87,0.2)'};
      background: ${isSelected ? 'var(--navy)' : 'white'};
      color: ${isSelected ? 'white' : 'var(--navy)'};
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    `;
    const stnLabel = inst.current_station_name ? ` (near ${inst.current_station_name})` : (inst.current_station ? ` (near ${inst.current_station})` : '');
    btn.innerHTML = `${isSelected ? '✓ ' : ''}${inst.relative_label} • Day ${inst.day_of_journey}${stnLabel}`;
    btn.onclick = async () => {
      if (inst.journey_start_date === State.journeyDate) return;
      State.journeyDate = inst.journey_start_date;
      document.querySelectorAll(".journey-card").forEach(c => {
        c.classList.toggle("selected", c.id === `jc-${inst.journey_instance_id}`);
      });
      await runPrediction();
    };
    pills.appendChild(btn);
  });
}

/* ── Live Banner ────────────────────────────────────────────────── */
function renderLiveBanner(d) {
  const currStn = d.current_station_name ? `${d.current_station_name} (${d.current_station})` : (d.current_station || "—");
  set("lbStation", currStn);
  set("lbSpeed",   d.speed_kmh != null ? `${d.speed_kmh} km/h` : "— km/h");
  set("lbDelay",   d.current_delay_minutes != null ? `${d.current_delay_minutes} min late` : "—");
  set("lbUpdated", d.last_updated_at ? `Updated: ${fmtRelTime(d.last_updated_at)}` : "");

  // Previous / next halt
  const prev = d.previous_halt;
  const next = d.next_halt;
  const prevEl = document.getElementById("lbPrevHalt");
  const nextEl = document.getElementById("lbNextHalt");
  if (prevEl && prev) {
    const distStr = prev.distance_km != null ? ` (${prev.distance_km} km ago)` : '';
    prevEl.textContent = `Last stop: ${prev.station_name || prev.station_code}${distStr}`;
  }
  if (nextEl && next) {
    const distStr = next.distance_km != null ? ` (${next.distance_km} km ahead)` : '';
    nextEl.textContent = `Next stop: ${next.station_name || next.station_code}${distStr}`;
  }

  // Source departure
  const srcEl = document.getElementById("lbSourceDep");
  if (srcEl && d.train_source_name) {
    const depTime = fmtETA(d.source_departure);
    const actualTime = d.source_departure_actual ? ` (actual: ${fmtETA(d.source_departure_actual)})` : " (actual not recorded)";
    const dateLabel = State.journeyDate ? ` • Run: ${State.journeyDate}` : "";
    srcEl.textContent = `Departed ${d.train_source_name}${dateLabel}: ${depTime}${actualTime}`;
  }
}

/* ── ETA Cards ─────────────────────────────────────────────────── */
function renderETACards(d) {
  // Boarding card
  const bName  = d.boarding_station || "—";
  const bEntry = (d.user_journey || []).find(s => s.is_boarding)
               || (d.full_journey || []).find(s => s.station === d.boarding_station);
  set("etaBoardName",     bEntry ? (bEntry.station_name || bName) : bName);
  set("etaBoardCode",     bName);
  set("etaBoardTime",     fmtETA(d.boarding_eta));
  renderBadge("etaBoardBadge", d.boarding_delay_minutes, d.boarding_severity);
  set("etaBoardDelayTxt", d.boarding_delay_minutes != null ? delayText(d.boarding_delay_minutes) : "");

  // Destination card
  const dCode  = d.destination_station || "—";
  const dEntry = (d.user_journey || []).find(s => s.is_user_destination)
               || (d.full_journey || []).find(s => s.station === d.destination_station);
  set("etaDestName",     dEntry ? (dEntry.station_name || dCode) : (d.destination_name || dCode));
  set("etaDestCode",     dCode);
  set("etaDestTime",     fmtETA(d.destination_eta, d.boarding_eta));
  renderBadge("etaDestBadge", d.destination_delay_minutes, d.destination_severity);
  set("etaDestDelayTxt", d.destination_delay_minutes != null ? delayText(d.destination_delay_minutes) : "");
}

function renderBadge(id, delayMin, severity) {
  const el = document.getElementById(id);
  if (!el) return;
  const sev = severity || classifySeverity(delayMin);
  const labels = { "on-time": "On Time ✓", minor: "Minor", moderate: "Moderate", major: "Major", severe: "Severe ⚠", unknown: "—" };
  el.textContent = delayMin != null ? labels[sev] || sev : "—";
  el.className   = `delay-badge ${sev}`;
}

/* ── Journey Timeline ──────────────────────────────────────────── */
function renderTimeline(journey) {
  const wrap = document.getElementById("journeyTimeline");
  if (!journey.length) {
    wrap.innerHTML = `<div class="timeline-empty">No journey data available.</div>`;
    return;
  }

  const boardEta = journey.find(s => s.is_boarding)?.predicted_eta || journey[0]?.predicted_eta;

  wrap.innerHTML = journey.map((s, idx) => {
    const nodeClass = s.is_boarding         ? "board"
                    : s.is_user_destination ? "exit"
                    : s.is_current          ? "current"
                    : s.is_past             ? "past"
                    : "future";

    const tags = [];
    if (s.is_boarding)                   tags.push(`<span class="tl-tag tag-board">BOARD</span>`);
    if (s.is_user_destination)           tags.push(`<span class="tl-tag tag-exit">EXIT</span>`);
    if (s.is_current)                    tags.push(`<span class="tl-tag tag-current">NOW HERE</span>`);
    if (s.data_source === "live" && !s.is_boarding && !s.is_user_destination && !s.is_current)
                                         tags.push(`<span class="tl-tag tag-live">LIVE</span>`);
    if (s.data_source === "model")       tags.push(`<span class="tl-tag tag-ai">AI</span>`);

    const itemCls = s.is_boarding ? "tl-item is-board"
                  : s.is_user_destination ? "tl-item is-exit"
                  : "tl-item";

    const nameCls = s.is_past && !s.is_boarding && !s.is_user_destination ? "tl-stn-name muted" : "tl-stn-name";

    const etaStr  = fmtETA(s.predicted_eta, boardEta);
    const dlyStr  = s.predicted_delay != null ? `+${s.predicted_delay} min` : "";
    const dlyCol  = delayColor(s.predicted_delay);

    return `
      <div class="${itemCls}">
        <div class="tl-left">
          <div class="tl-node ${nodeClass}"></div>
          <div class="tl-line"></div>
        </div>
        <div class="tl-content">
          <div class="tl-row">
            <div>
              <div class="tl-stn-code">${s.station}</div>
              <div class="${nameCls}">${s.station_name || s.station}</div>
            </div>
            <div class="tl-right">
              ${etaStr ? `<div class="tl-eta">${etaStr}</div>` : ""}
              ${dlyStr ? `<div class="tl-delay" style="color:${dlyCol}">${dlyStr}</div>` : ""}
            </div>
          </div>
          ${tags.length ? `<div class="tl-tags">${tags.join("")}</div>` : ""}
        </div>
      </div>`;
  }).join("");
}

/* ── Full Route Table ──────────────────────────────────────────── */
function renderTable(route) {
  State.filteredRoute = [...route];
  drawTablePage();
}

function drawTablePage() {
  const route  = State.filteredRoute;
  const total  = route.length;
  const pages  = Math.ceil(total / State.rowsPerPage);
  const start  = (State.currentPage - 1) * State.rowsPerPage;
  const slice  = route.slice(start, start + State.rowsPerPage);

  set("pgInfo", `Page ${State.currentPage} / ${pages || 1}  (${total} stations)`);
  document.getElementById("pgPrev").disabled = State.currentPage <= 1;
  document.getElementById("pgNext").disabled = State.currentPage >= pages;

  const body = document.getElementById("routeTableBody");
  if (!slice.length) {
    body.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:20px;color:var(--navy-60);">No stations found.</td></tr>`;
    return;
  }

  body.innerHTML = slice.map((s, i) => {
    const rowCls = s.is_boarding  ? "tr-boarding"
                 : s.is_user_destination ? "tr-destination"
                 : s.is_current  ? "tr-current" : "";

    const srcBadge = s.data_source === "live"
      ? `<span class="src-badge src-live">LIVE</span>`
      : `<span class="src-badge src-ai">AI</span>`;

    const sched = s.scheduled_arrival ? fmtISO(s.scheduled_arrival) : "—";
    const eta   = s.predicted_eta     ? fmtISO(s.predicted_eta)     : "—";
    const delay = s.predicted_delay   != null ? `+${s.predicted_delay} min` : "—";
    const dlyCol= delayColor(s.predicted_delay);

    return `
      <tr class="${rowCls}">
        <td style="color:var(--navy-60);font-family:var(--font-mono,monospace);font-size:0.72rem;">${start + i + 1}</td>
        <td>
          <div class="stn-code-cell">${s.station}</div>
          <div style="font-size:0.75rem;color:var(--navy-60)">${s.station_name || ""}</div>
        </td>
        <td style="font-family:monospace;font-size:0.78rem;color:var(--navy-60);">${sched}</td>
        <td style="font-family:monospace;font-size:0.78rem;font-weight:700;color:var(--navy);">${eta}</td>
        <td style="font-weight:700;font-size:0.78rem;color:${dlyCol};">${delay}</td>
        <td>${srcBadge}</td>
      </tr>`;
  }).join("");
}

function changePage(dir) {
  const pages = Math.ceil(State.filteredRoute.length / State.rowsPerPage);
  State.currentPage = Math.max(1, Math.min(pages, State.currentPage + dir));
  drawTablePage();
}

function filterTable() {
  const q = (document.getElementById("tableSearch").value || "").toLowerCase().trim();
  State.filteredRoute = q
    ? State.fullRoute.filter(s =>
        (s.station || "").toLowerCase().includes(q) ||
        (s.station_name || "").toLowerCase().includes(q)
      )
    : [...State.fullRoute];
  State.currentPage = 1;
  drawTablePage();
}

/* ── Delay Chart ─────────────────────────────────────────────── */
function renderChart(journey) {
  const ctx = document.getElementById("delayChart");
  if (!ctx) return;
  if (State.chartInstance) { State.chartInstance.destroy(); State.chartInstance = null; }

  const labels = journey.map(s => s.station || "");
  const delays = journey.map(s => s.predicted_delay ?? null);
  const colors = journey.map(s => s.data_source === "live" ? "#4ADE80" : "#457B9D");

  State.chartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Predicted Delay (min)",
          data:  delays,
          borderColor: "#1D3557",
          backgroundColor: "rgba(168,218,220,0.18)",
          pointBackgroundColor: colors,
          pointBorderColor:     colors,
          pointRadius: 4,
          pointHoverRadius: 7,
          fill: true,
          tension: 0.35,
          borderWidth: 2.5,
        },
        {
          label: "On-Time (0 min)",
          data:  delays.map(() => 0),
          borderColor: "rgba(168,218,220,0.6)",
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
          tension: 0,
          borderWidth: 1.5,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1D3557",
          titleFont:   { size: 12, weight: "700" },
          bodyFont:    { size: 11 },
          padding:     10,
          callbacks: {
            title:  (items) => items[0]?.label || "",
            label:  (item) => item.datasetIndex === 0 ? ` Delay: ${item.raw ?? "—"} min` : "",
          },
        },
      },
      scales: {
        x: {
          ticks: {
            maxTicksLimit: 12,
            font: { size: 10, family: "monospace" },
            color: "#3D5A82",
            maxRotation: 45,
          },
          grid: { color: "rgba(168,218,220,0.25)" },
        },
        y: {
          beginAtZero: true,
          ticks: { font: { size: 10 }, color: "#3D5A82" },
          grid: { color: "rgba(168,218,220,0.25)" },
          title: { display: true, text: "Delay (minutes)", font: { size: 11 }, color: "#3D5A82" },
        },
      },
    },
  });
}

/* ── Tab switching ──────────────────────────────────────────────── */
function switchTab(which) {
  const panes = { journey: "paneJourney", full: "paneFull", chart: "paneChart" };
  const tabs  = { journey: "tabJrny",    full: "tabFull",   chart: "tabChart" };

  Object.values(panes).forEach(id => document.getElementById(id)?.classList.remove("active"));
  Object.values(tabs ).forEach(id => document.getElementById(id)?.classList.remove("active"));

  document.getElementById(panes[which])?.classList.add("active");
  document.getElementById(tabs[which])?.classList.add("active");
}

/* ── Helpers ────────────────────────────────────────────────────── */
function set(id, html) {
  const el = document.getElementById(id);
  if (el) el.textContent = html;
}

function fmtETA(isoStr, refIsoStr = null) {
  if (!isoStr) return "—";
  try {
    const d = new Date(isoStr);
    const timeStr = d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: true });
    if (!refIsoStr) return timeStr;
    const refDate = new Date(refIsoStr);
    const dDateOnly = new Date(d.getFullYear(), d.getMonth(), d.getDate());
    const refDateOnly = new Date(refDate.getFullYear(), refDate.getMonth(), refDate.getDate());
    const diffDays = Math.round((dDateOnly - refDateOnly) / (1000 * 60 * 60 * 24));
    if (diffDays === 1) return `${timeStr} (+1 Day)`;
    if (diffDays > 1) return `${timeStr} (+${diffDays} Days)`;
    return timeStr;
  } catch { return "—"; }
}

function fmtISO(isoStr) {
  if (!isoStr) return "—";
  try {
    const d = new Date(isoStr);
    return d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: true });
  } catch { return isoStr.slice(11, 16) || "—"; }
}

function fmtRelTime(isoStr) {
  try {
    const diff = Math.floor((Date.now() - new Date(isoStr).getTime()) / 1000);
    if (diff < 60)  return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
    return `${Math.floor(diff/3600)}h ago`;
  } catch { return ""; }
}

function classifySeverity(min) {
  if (min == null) return "unknown";
  if (min <= 5)    return "on-time";
  if (min <= 15)   return "minor";
  if (min <= 30)   return "moderate";
  if (min <= 60)   return "major";
  return "severe";
}

function delayText(min) {
  if (min == null) return "";
  if (min <= 0)    return "No delay expected";
  return `+${min} min behind schedule`;
}

function delayColor(min) {
  if (min == null || min <= 5)  return "#2DB67D";
  if (min <= 15)  return "#E8A020";
  if (min <= 30)  return "#D96B2A";
  if (min <= 60)  return "#D03C3C";
  return "#8B1A1A";
}
