document.addEventListener("DOMContentLoaded", () => {
  loadLocations();
  loadForecast();
  initIcons();
  initNavbar();
  bindForecastInputs();
  initGhiTrendChart();
});

let availableLocations = [];
let selectedLocationKey = null;

/* ----------------------------- */
/* Icons                         */
/* ----------------------------- */
function initIcons() {
  if (window.lucide && typeof window.lucide.createIcons === "function") {
    window.lucide.createIcons();
  }
}

/* ----------------------------- */
/* Navbar                        */
/* ----------------------------- */
function initNavbar() {
  const locationTrigger = document.getElementById("locationTrigger");
  const locationMenu = document.getElementById("locationMenu");
  const selectedLocation = document.getElementById("selectedLocation");
  const themeToggle = document.getElementById("themeToggle");

  if (!locationTrigger || !locationMenu || !selectedLocation || !themeToggle) {
    return;
  }

  function openLocationMenu() {
    locationMenu.classList.remove("hidden");
    locationTrigger.classList.add("is-open");
    locationTrigger.setAttribute("aria-expanded", "true");
  }

  function closeLocationMenu() {
    locationMenu.classList.add("hidden");
    locationTrigger.classList.remove("is-open");
    locationTrigger.setAttribute("aria-expanded", "false");
  }

  function toggleLocationMenu() {
    const isHidden = locationMenu.classList.contains("hidden");
    if (isHidden) {
      openLocationMenu();
    } else {
      closeLocationMenu();
    }
  }

  locationTrigger.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleLocationMenu();
  });

  document.addEventListener("click", (event) => {
    if (
      !locationMenu.contains(event.target) &&
      !locationTrigger.contains(event.target)
    ) {
      closeLocationMenu();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeLocationMenu();
    }
  });

  themeToggle.dataset.mode = "dark";

  themeToggle.addEventListener("click", () => {
    const currentMode = themeToggle.dataset.mode;
    const nextMode = currentMode === "dark" ? "light" : "dark";

    themeToggle.dataset.mode = nextMode;
    themeToggle.setAttribute(
      "aria-pressed",
      nextMode === "dark" ? "true" : "false"
    );
  });
}

/* ----------------------------- */
/* Comparison chart              */
/* ----------------------------- */
let ghiTrendChart = null;

function parseLocalDate(dateStr) {
  if (!dateStr || typeof dateStr !== "string") return null;

  const parts = dateStr.split("-");
  if (parts.length !== 3) return null;

  const year = Number(parts[0]);
  const month = Number(parts[1]) - 1;
  const day = Number(parts[2]);

  const date = new Date(year, month, day);
  return Number.isNaN(date.getTime()) ? null : date;
}

function initGhiTrendChart() {
  const ghiCanvas = document.getElementById("ghiTrendChart");
  if (!ghiCanvas || typeof Chart === "undefined") {
    return;
  }

  const context = ghiCanvas.getContext("2d");
  const gradient = context.createLinearGradient(0, 0, 0, 300);
  gradient.addColorStop(0, "rgba(223,234,243,0.25)");
  gradient.addColorStop(1, "rgba(223,234,243,0)");

  ghiTrendChart = new Chart(ghiCanvas, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Actual GHI",
          data: [],
          borderColor: "#7c8ea3",
          backgroundColor: "rgba(223, 234, 243, 0.08)",
          borderWidth: 2,
          borderDash: [6, 6],
          tension: 0.42,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: "#7c8ea3",
          pointHoverBorderColor: "#101010",
          pointHoverBorderWidth: 2,
          fill: false
        },
        {
          label: "Predicted GHI",
          data: [],
          borderColor: "#d6e4ee",
          backgroundColor: gradient,
          borderWidth: 3.5,
          tension: 0.42,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: "#d6e4ee",
          pointHoverBorderColor: "#101010",
          pointHoverBorderWidth: 2,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      animation: {
        duration: 900,
        easing: "easeOutQuart"
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          backgroundColor: "rgba(20, 20, 22, 0.96)",
          titleColor: "#ffffff",
          bodyColor: "rgba(255,255,255,0.82)",
          borderColor: "rgba(255,255,255,0.08)",
          borderWidth: 1,
          padding: 12,
          displayColors: true,
          usePointStyle: true,
          cornerRadius: 14
        }
      },
      scales: {
        x: {
          grid: {
            display: false,
            drawBorder: false
          },
          border: {
            display: false
          },
          ticks: {
            color: "rgba(255,255,255,0.5)",
            font: {
              size: 12,
              weight: "500"
            },
            padding: 10
          }
        },
        y: {
          beginAtZero: true,
          grid: {
            color: "rgba(255,255,255,0.08)",
            drawBorder: false
          },
          border: {
            display: false
          },
          ticks: {
            color: "rgba(255,255,255,0.42)",
            font: {
              size: 12,
              weight: "500"
            },
            padding: 10
          }
        }
      },
      elements: {
        line: {
          capBezierPoints: true
        }
      }
    }
  });
}

function renderGhiTrendChart(history) {
  if (!ghiTrendChart || !Array.isArray(history)) return;

  const sortedHistory = [...history].sort((a, b) =>
    (a.date || "").localeCompare(b.date || "")
  );

  const labels = sortedHistory.map((entry) => formatChartDate(entry.date));
  const predicted = sortedHistory.map((entry) =>
    entry.predicted_ghi_kwh_m2 ?? null
  );
  const actual = sortedHistory.map((entry) =>
    entry.actual_ghi_kwh_m2 ?? null
  );

  ghiTrendChart.data.labels = labels;
  ghiTrendChart.data.datasets[0].data = actual;
  ghiTrendChart.data.datasets[1].data = predicted;
  ghiTrendChart.update();
}

/* ----------------------------- */
/* API load / render             */
/* ----------------------------- */
async function loadForecast() {
  try {
    const arrayArea = getArrayArea();
    const efficiency = getEfficiency();

    const locationParam = selectedLocationKey
      ? `location_key=${encodeURIComponent(selectedLocationKey)}&`
      : "";

    const res = await fetch(
      `/api/forecast?${locationParam}array_area_m2=${arrayArea}&panel_efficiency=${efficiency}`
    );

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();
    console.log("forecast data:", data);

    renderAll(data);
  } catch (err) {
    console.error("Forecast load failed:", err);
  }
}

function renderAll(data) {
  if (!data) return;

  renderLocation(data.meta);

  const days = data.forecast_days || [];
  const today = days[0] || null;
  const future = days.slice(1);

  renderTodaySummary(today);
  renderPills(days);
  renderSolarEstimator(today);
  
  renderAccuracy(data.accuracy_bands_percent);
  renderGhiTrendChart(data.history);
}

/* ----------------------------- */
/* Location / today              */
/* ----------------------------- */
function renderLocation(meta) {
  const selectedLocation = document.getElementById("selectedLocation");
  if (!selectedLocation || !meta) return;

  selectedLocation.textContent = meta.location_name;
}

function renderTodaySummary(today) {
  if (!today) return;

  const todayActualEl = document.getElementById("todayActualGhi");
  const todayDeltaEl = document.getElementById("todayDeltaGhi");

  if (todayActualEl) {
    todayActualEl.textContent =
      today.actual_ghi_kwh_m2 == null
        ? "--"
        : today.actual_ghi_kwh_m2.toFixed(2);
  }

  if (todayDeltaEl) {
    todayDeltaEl.textContent =
      today.ghi_delta_kwh_m2 == null
        ? "--"
        : `${today.ghi_delta_kwh_m2 > 0 ? "+" : ""}${today.ghi_delta_kwh_m2.toFixed(2)}`;
  }
}

/* ----------------------------- */
/* Forecast pills                */
/* ----------------------------- */
function renderPills(days) {
  const pillRow = document.getElementById("pillRow");
  if (!pillRow || !Array.isArray(days)) return;

  pillRow.innerHTML = "";

  days.forEach((day, index) => {
    const isToday = index === 0;
    const pill = document.createElement("button");
    pill.type = "button";
    pill.className = `forecast-pill ${index === 0 ? "is-active" : ""}`;
    pill.dataset.ghi = day.predicted_ghi_kwh_m2.toFixed(2);

    const bands = day.ghi_bands_kwh_m2;

    pill.innerHTML = `
      <div class="pill-closed">
        <div class="pill-day">${isToday ? "Today" : getClosedDayLabel(day.label, day.date)}</div>
        <div class="temp-small">${day.predicted_ghi_kwh_m2.toFixed(2)}</div>
        <div class="unit-small">kWh/m²</div>
      </div>

      <div class="pill-open">
        <div class="pill-head">
          <span class="pill-day-open">${isToday ? "Today" : getOpenDayLabel(day.label, day.date)}</span>
          <span class="pill-time">${formatDate(day.date)}</span>
        </div>

        <div class="pill-main-open">
          <div class="temp-large">${day.predicted_ghi_kwh_m2.toFixed(2)}</div>
          <div class="unit-large">kWh/m²</div>
        </div>

        <div class="pill-details">
          <div class="pill-details-output">
            <p>${formatBand(bands.mae)}</p>
            <p>± MAE</p>
          </div>
          <div class="pill-details-output">
            <p>${formatBand(bands["1std"])}</p>
            <p>±1 STD</p>
          </div>
          <div class="pill-details-output">
            <p>${formatBand(bands["2std"])}</p>
            <p>±2 STD</p>
          </div>
          <div class="pill-details-output">
            <p>${formatBand(bands["3std"])}</p>
            <p>±3 STD</p>
          </div>
        </div>
      </div>
    `;

    pill.addEventListener("mouseenter", () => {
      setActivePill(pillRow, pill);
      renderSolarEstimator(day);
    });

    pill.addEventListener("focus", () => {
      setActivePill(pillRow, pill);
      renderSolarEstimator(day);
    });

    pill.addEventListener("click", () => {
      setActivePill(pillRow, pill);
      renderSolarEstimator(day);
    });

    pillRow.appendChild(pill);
  });
}

function setActivePill(container, activePill) {
  const pills = container.querySelectorAll(".forecast-pill");
  pills.forEach((pill) => pill.classList.remove("is-active"));
  activePill.classList.add("is-active");
}

/* ----------------------------- */
/* Estimator                     */
/* ----------------------------- */
function renderSolarEstimator(day) {
  if (!day) return;

  const estimatedOutput = document.getElementById("estimatedOutput");
  const currentGhiValue = document.getElementById("currentGhiValue");
  const actualGhiValue = document.getElementById("actualGhiValue");
  const ghiDeltaValue = document.getElementById("ghiDeltaValue");

  if (estimatedOutput) {
    estimatedOutput.textContent = day.estimated_output_kwh.toFixed(2);
  }

  if (currentGhiValue) {
    currentGhiValue.textContent = day.predicted_ghi_kwh_m2.toFixed(2);
  }

  if (actualGhiValue) {
    actualGhiValue.textContent =
      day.actual_ghi_kwh_m2 == null ? "--" : day.actual_ghi_kwh_m2.toFixed(2);
  }

  if (ghiDeltaValue) {
    ghiDeltaValue.textContent =
      day.ghi_delta_kwh_m2 == null
        ? "--"
        : `${day.ghi_delta_kwh_m2 > 0 ? "+" : ""}${day.ghi_delta_kwh_m2.toFixed(2)}`;
  }
}

/* ----------------------------- */
/* Accuracy cards                */
/* ----------------------------- */
function renderAccuracy(acc) {
  const metricCards = document.querySelectorAll(".metric-card-compact");
  if (!metricCards.length || !acc) return;

  const metricConfig = [
    {
      key: "MAE",
      label: "±MAE",
      description: (model, actual) =>
        `${model}% of predictions fall within the MAE band, while ${actual}% of actual values fall inside that same range.`
    },
    {
      key: "1std",
      label: "±1 STD",
      description: (model, actual) =>
        `${model}% of predictions fall within one standard deviation, compared with ${actual}% coverage from actual values.`
    },
    {
      key: "2std",
      label: "±2 STD",
      description: (model, actual) =>
        `${model}% of predictions fall within two standard deviations, while actual values land inside that interval ${actual}% of the time.`
    },
    {
      key: "3std",
      label: "±3 STD",
      description: (model, actual) =>
        `${model}% of predictions fall within three standard deviations, versus ${actual}% for actual observed values.`
    }
  ];

  metricCards.forEach((card, index) => {
    const cfg = metricConfig[index];
    if (!cfg) return;

    const payload = acc[cfg.key];
    if (!payload) return;

    const model = Number(payload.model);
    const actual = Number(payload.actual);
    const delta = Number(payload.delta);

    const bandEl = card.querySelector(".metric-band");
    const inlineValues = card.querySelectorAll(".metric-inline-value");
    const deltaEl = card.querySelector(".metric-delta-pill strong");
    const descEl = card.querySelector(".metric-compact-description");
    const deltaPill = card.querySelector(".metric-delta-pill");

    if (bandEl) {
      bandEl.textContent = cfg.label;
    }

    if (inlineValues[0]) {
      inlineValues[0].textContent = `${model.toFixed(2)}%`;
    }

    if (inlineValues[1]) {
      inlineValues[1].textContent = `${actual.toFixed(2)}%`;
    }

    if (deltaEl) {
      deltaEl.textContent = `${delta >= 0 ? "+" : ""}${delta.toFixed(2)}%`;
    }

    if (deltaPill) {
      deltaPill.classList.toggle("metric-delta-negative", delta < 0);
    }

    if (descEl) {
      descEl.textContent = cfg.description(model.toFixed(2), actual.toFixed(2));
    }
  });
}

/* ----------------------------- */
/* Inputs                        */
/* ----------------------------- */
function getArrayArea() {
  const input = document.getElementById("arraySize");
  if (!input) return 10;

  const value = parseFloat(input.value);
  return Number.isFinite(value) && value > 0 ? value : 10;
}

function getEfficiency() {
  const input = document.getElementById("panelEfficiency");
  if (!input) return 0.15;

  const raw = parseFloat(input.value);
  if (!Number.isFinite(raw) || raw <= 0) return 0.15;

  return raw > 1 ? raw / 100 : raw;
}

function bindForecastInputs() {
  const arrayInput = document.getElementById("arraySize");
  const efficiencyInput = document.getElementById("panelEfficiency");

  if (arrayInput) {
    arrayInput.addEventListener("change", loadForecast);
    arrayInput.addEventListener("input", debounce(loadForecast, 250));
  }

  if (efficiencyInput) {
    efficiencyInput.addEventListener("change", loadForecast);
    efficiencyInput.addEventListener("input", debounce(loadForecast, 250));
  }
}

function debounce(fn, delay = 250) {
  let timer;

  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

/* ----------------------------- */
/* Formatting                    */
/* ----------------------------- */
function formatBand(range) {
  if (!range) return "--";
  return `${Number(range.low).toFixed(2)} - ${Number(range.high).toFixed(2)} kWh/m²`;
}

function formatDate(dateStr) {
  const date = parseLocalDate(dateStr);
  if (!date) return dateStr;

  return date.toLocaleDateString("en-US", {
    month: "2-digit",
    day: "2-digit",
    year: "numeric"
  });
}

function formatChartDate(dateStr) {
  const date = parseLocalDate(dateStr);
  if (!date) return dateStr;

  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "2-digit"
  });
}

function getClosedDayLabel(label, dateStr) {
  const date = parseLocalDate(dateStr);
  if (!date) return label;

  return date.toLocaleDateString("en-US", { weekday: "short" });
}

function getOpenDayLabel(label, dateStr) {
  const date = parseLocalDate(dateStr);
  if (!date) return label;

  return date.toLocaleDateString("en-US", { weekday: "long" });
}

async function loadLocations() {
  try {
    const res = await fetch("api/locations");

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const locations = await res.json();
    availableLocations = Array.isArray(locations) ? locations : [];

    if (!availableLocations.length) {
      return;
    }

    selectedLocationKey = availableLocations[0].key;
    renderLocationMenuOptions();
    updateSelectedLocationLabel();
  } catch (err) {
    console.error("Failed to load locations:", err);
  }
}

function renderLocationMenuOptions() {
  const locationMenu = document.getElementById("locationMenu");
  const locationTrigger = document.getElementById("locationTrigger");

  if (!locationMenu) return;

  locationMenu.innerHTML = "";

  availableLocations.forEach((location) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "location-option";
    button.textContent = location.name;
    button.dataset.key = location.key;

    if (location.key === selectedLocationKey) {
      button.classList.add("is-selected");
    }

    button.addEventListener("click", () => {
      selectedLocationKey = location.key;
      updateSelectedLocationLabel();
      renderLocationMenuOptions();

      locationMenu.classList.add("hidden");
      if (locationTrigger) {
        locationTrigger.classList.remove("is-open");
        locationTrigger.setAttribute("aria-expanded", "false");
      }

      loadForecast();
    });

    locationMenu.appendChild(button);
  });
}

function updateSelectedLocationLabel() {
  const selectedLocation = document.getElementById("selectedLocation");
  if (!selectedLocation) return;

  const location = availableLocations.find(
    (item) => item.key === selectedLocationKey
  );

  selectedLocation.textContent = location ? location.name : "Select Location";
}