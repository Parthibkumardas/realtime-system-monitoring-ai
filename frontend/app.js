const API_BASE = "http://localhost:8000/api";

let forecastChart, historyChart;

async function fetchJSON(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

function setStatus(text, ok = true) {
  const el = document.getElementById("connectionStatus");
  el.textContent = text;
  el.style.color = ok ? "#9fe1cb" : "#f09595";
}

async function refreshMetrics() {
  try {
    const latest = await fetchJSON("/metrics/latest");
    document.getElementById("cpuValue").textContent = `${Math.round(latest.cpu_percent)}%`;
    document.getElementById("memoryValue").textContent = `${Math.round(latest.memory_percent)}%`;
    document.getElementById("diskValue").textContent = `${Math.round(latest.disk_percent)}%`;
    setStatus("live", true);
  } catch (e) {
    setStatus("disconnected", false);
  }
}

async function refreshAnomaly() {
  try {
    const status = await fetchJSON("/anomaly/status");
    const el = document.getElementById("anomalyValue");
    const card = document.getElementById("anomalyCard");
    if (status.is_anomaly) {
      el.textContent = "Anomaly";
      card.classList.add("danger");
    } else {
      el.textContent = "Normal";
      card.classList.remove("danger");
    }
  } catch (e) {
    document.getElementById("anomalyValue").textContent = "unknown";
  }
}

async function refreshHistory() {
  try {
    const { samples } = await fetchJSON("/metrics/history?n=30");
    const labels = samples.map((_, i) => i.toString());
    const cpu = samples.map((s) => s.cpu_percent);
    const memory = samples.map((s) => s.memory_percent);
    const disk = samples.map((s) => s.disk_percent);

    if (!historyChart) {
      historyChart = new Chart(document.getElementById("historyChart"), {
        type: "line",
        data: {
          labels,
          datasets: [
            { label: "CPU", data: cpu, borderColor: "#2a78d6", tension: 0.3, pointRadius: 0 },
            { label: "Memory", data: memory, borderColor: "#7f77dd", tension: 0.3, pointRadius: 0 },
            { label: "Disk", data: disk, borderColor: "#1baf7a", tension: 0.3, pointRadius: 0 },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { min: 0, max: 100, ticks: { color: "#7b7f92" }, grid: { color: "#181a22" } },
            x: { ticks: { color: "#7b7f92" }, grid: { display: false } },
          },
          plugins: { legend: { labels: { color: "#d7d6ea" } } },
        },
      });
    } else {
      historyChart.data.labels = labels;
      historyChart.data.datasets[0].data = cpu;
      historyChart.data.datasets[1].data = memory;
      historyChart.data.datasets[2].data = disk;
      historyChart.update();
    }
  } catch (e) {
    // backend not reachable yet -- ignore until next poll
  }
}

async function refreshForecast() {
  try {
    const { history, forecast } = await fetchJSON("/forecast/cpu");
    const historyLabels = history.map((_, i) => `-${(history.length - i) * 5}s`);
    const forecastLabels = forecast.map((_, i) => `+${(i + 1) * 5}s`);
    const labels = [...historyLabels, ...forecastLabels];

    const historyData = [...history, ...Array(forecast.length).fill(null)];
    const forecastData = [
      ...Array(history.length - 1).fill(null),
      history[history.length - 1],
      ...forecast,
    ];

    if (!forecastChart) {
      forecastChart = new Chart(document.getElementById("forecastChart"), {
        type: "line",
        data: {
          labels,
          datasets: [
            { label: "Actual", data: historyData, borderColor: "#2a78d6", backgroundColor: "rgba(42,120,214,0.12)", fill: true, tension: 0.3, pointRadius: 2 },
            { label: "Forecast", data: forecastData, borderColor: "#eb6834", borderDash: [5, 4], tension: 0.3, pointRadius: 2 },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { min: 0, max: 100, ticks: { color: "#7b7f92" }, grid: { color: "#181a22" } },
            x: { ticks: { color: "#7b7f92" }, grid: { display: false } },
          },
          plugins: { legend: { labels: { color: "#d7d6ea" } } },
        },
      });
    } else {
      forecastChart.data.labels = labels;
      forecastChart.data.datasets[0].data = historyData;
      forecastChart.data.datasets[1].data = forecastData;
      forecastChart.update();
    }
  } catch (e) {
    // not enough history yet to forecast -- ignore until next poll
  }
}

function appendChatMessage(role, text) {
  const log = document.getElementById("chatLog");
  const div = document.createElement("div");
  div.className = `chat-message ${role}`;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

async function askAgent() {
  const input = document.getElementById("chatInput");
  const question = input.value.trim();
  if (!question) return;

  appendChatMessage("user", question);
  input.value = "";
  appendChatMessage("assistant", "Thinking...");

  try {
    const result = await fetchJSON("/agent/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const log = document.getElementById("chatLog");
    log.lastChild.remove(); // remove "Thinking..." placeholder
    appendChatMessage("assistant", result.answer);
  } catch (e) {
    const log = document.getElementById("chatLog");
    log.lastChild.remove();
    appendChatMessage("assistant", "Couldn't reach the backend. Is it running on localhost:8000?");
  }
}

document.getElementById("chatSend").addEventListener("click", askAgent);
document.getElementById("chatInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") askAgent();
});

async function pollLoop() {
  await refreshMetrics();
  await refreshAnomaly();
  await refreshHistory();
  await refreshForecast();
}

pollLoop();
setInterval(pollLoop, 5000);
