/* ═══════════════════════════════════════════════════════════════
   ai2agi — Main Frontend Script
   Ai70000, Ltd. | POC v2.0
═══════════════════════════════════════════════════════════════ */

"use strict";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const chatWindow  = document.getElementById("chatWindow");
const taskInput   = document.getElementById("taskInput");
const submitBtn   = document.getElementById("submitBtn");
const resetBtn    = document.getElementById("resetBtn");
const statusDot   = document.getElementById("statusDot");
const devToggle   = document.getElementById("devToggle");
const devPanel    = document.querySelector(".dev-panel");

// Dev panel refs
const mQueries    = document.getElementById("mQueries");
const mTopics     = document.getElementById("mTopics");
const mDopamine   = document.getElementById("mDopamine");
const mCortisol   = document.getElementById("mCortisol");
const pipelineLog = document.getElementById("pipelineLog");

// ── State ─────────────────────────────────────────────────────────────────────
let queryCount = 0;
let isProcessing = false;

// ── Auto-resize textarea ──────────────────────────────────────────────────────
taskInput.addEventListener("input", () => {
  taskInput.style.height = "auto";
  taskInput.style.height = Math.min(taskInput.scrollHeight, 160) + "px";
});

// Enter = submit (Shift+Enter = newline)
taskInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSubmit();
  }
});

submitBtn.addEventListener("click", handleSubmit);
resetBtn.addEventListener("click", handleReset);

// Dev panel toggle
devToggle.addEventListener("click", () => {
  devPanel.classList.toggle("collapsed");
});

// ── Submit ────────────────────────────────────────────────────────────────────
async function handleSubmit() {
  const task = taskInput.value.trim();
  if (!task || isProcessing) return;

  setProcessing(true);
  queryCount++;

  // Append user message
  appendMsg("USER", task, "user-block");

  // Clear input
  taskInput.value = "";
  taskInput.style.height = "auto";

  // Show processing indicator
  const processingId = "proc-" + Date.now();
  appendProcessing(processingId);

  // Animate schematic
  animateSchematic();

  // Log start
  clearLog();
  logLine(`[${timestamp()}] Query #${queryCount}: "${task.substring(0, 60)}${task.length > 60 ? '…' : ''}"`, "log-step");
  logLine("Running 5-component AGI pipeline...", "log-meta");

  try {
    const res = await fetch("/api/infer", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ task }),
    });

    const data = await res.json();
    removeProcessing(processingId);

    if (!res.ok || data.error) {
      appendMsg("ERROR", data.error || "Pipeline error.", "error-block");
      logLine(`✗ Error: ${data.error}`, "log-err");
      setStatus("ERROR", "error");
    } else {
      renderAGIResponse(data);
      updateDevPanel(data);
      setStatus("READY", "ready");
    }

  } catch (err) {
    removeProcessing(processingId);
    appendMsg("ERROR", `Network error: ${err.message}`, "error-block");
    logLine(`✗ ${err.message}`, "log-err");
    setStatus("ERROR", "error");
  }

  setProcessing(false);
}

// ── Reset ─────────────────────────────────────────────────────────────────────
async function handleReset() {
  try {
    await fetch("/api/reset", { method: "POST" });
  } catch (_) {}

  queryCount = 0;
  isProcessing = false;
  setProcessing(false);

  // Clear chat (keep welcome)
  chatWindow.innerHTML = "";
  appendWelcome();

  // Clear dev panel
  resetSchematic();
  clearLog();
  logLine("Session reset.", "log-meta");
  mQueries.textContent = "0";
  mTopics.textContent  = "0";
  mDopamine.textContent = "—";
  mCortisol.textContent = "—";
  ["G","R","L","DTB","MQ"].forEach(k => {
    document.getElementById(`bar-${k}`).style.width = "0%";
    document.getElementById(`val-${k}`).textContent = "—";
  });

  setStatus("READY", "ready");
}

// ── Render AGI response ───────────────────────────────────────────────────────
function renderAGIResponse(d) {
  const response   = d.response || d.llm_answer || "[No response]";
  const domain     = d.domain || "—";
  const confidence = d.confidence ?? 0;
  const mq         = d.mq_score ?? 0;
  const epsilon    = d.epsilon ?? 0;
  const runtime    = d.runtime_s ?? "—";
  const tier       = d.confidence_tier || "";

  // Separate main answer from AGI trace line
  let mainText = response;
  let traceLine = "";
  const traceIdx = response.lastIndexOf("\n\n[AGI signal:");
  if (traceIdx !== -1) {
    mainText  = response.substring(0, traceIdx).trim();
    traceLine = response.substring(traceIdx + 2).trim();
  }

  // Confidence color class
  const confClass = confidence >= 0.70 ? "high" : confidence >= 0.45 ? "medium" : "low";
  const confPct   = (confidence * 100).toFixed(1) + "%";

  const block = document.createElement("div");
  block.className = "msg-block agi-block";
  block.innerHTML = `
    <div class="msg-label">AGI RESPONSE · ${domain} · Q#${queryCount}</div>
    <div class="msg-content">
      <div class="agi-response">${escapeHtml(mainText)}</div>
      ${traceLine ? `<div class="agi-trace">${escapeHtml(traceLine)}</div>` : ""}
      <div class="agi-meta">
        <div class="agi-meta-item">
          <span class="agi-meta-key">CONFIDENCE</span>
          <span class="agi-meta-val ${confClass}">${confPct}</span>
        </div>
        <div class="agi-meta-item">
          <span class="agi-meta-key">TIER</span>
          <span class="agi-meta-val">${tier.split(" ")[0]}</span>
        </div>
        <div class="agi-meta-item">
          <span class="agi-meta-key">MQ</span>
          <span class="agi-meta-val">${mq.toFixed(3)}</span>
        </div>
        <div class="agi-meta-item">
          <span class="agi-meta-key">ε</span>
          <span class="agi-meta-val">${epsilon.toFixed(4)}</span>
        </div>
        <div class="agi-meta-item">
          <span class="agi-meta-key">RUNTIME</span>
          <span class="agi-meta-val">${runtime}s</span>
        </div>
      </div>
    </div>
  `;
  chatWindow.appendChild(block);
  scrollChat();

  // Pipeline log
  const cs = d.component_scores || {};
  const rs = d.r_scores || {};
  const ds = d.dtb_state || {};

  logLine("", "");
  logLine("✓ [0] Foundation Initializer — 66 books, 15,914 passages", "log-ok");
  logLine(`✓ [1] G_enhanced  T_G(100,32) · ε=${(d.epsilon||0).toFixed(4)} · FSL=${(d.domain_signals?.gen_block_mean||0).toFixed(3)}`, "log-ok");
  logLine(`      NS=${(rs.NS||0).toFixed(3)} CR=${(rs.CR||0).toFixed(3)} CA=${(rs.CA||0).toFixed(3)}`, "log-meta");
  logLine(`✓ [2] R_enhanced  T_R(100,64) · ε_r=${(rs.epsilon_r||0).toFixed(3)}`, "log-ok");
  logLine(`✓ [3] ai_LLM      T_LLM(100,32) · Claude API · ${((d.llm_answer||"").split(" ").length)} words`, "log-ok");
  logLine(`✓ [4] DTB         T_DTB(100,24) · dopa=${(ds.dopamine||0).toFixed(3)} cortisol=${(ds.cortisol||0).toFixed(3)}`, "log-ok");
  logLine(`✓ [5] CV_Adapt    T_AGI(1,1024) — AGI-DNA`, "log-agi");
  logLine("", "");
  logLine(`   Domain     : ${domain}`, "log-meta");
  logLine(`   Confidence : ${confPct} [${tier.split("(")[0].trim()}]`, "log-meta");
  logLine(`   MQ Score   : ${mq.toFixed(3)}`, "log-meta");
  logLine(`   Runtime    : ${runtime}s`, "log-meta");
}

// ── Dev panel updates ─────────────────────────────────────────────────────────
function updateDevPanel(d) {
  // Session metrics
  mQueries.textContent = queryCount;
  mTopics.textContent  = d.session_n ?? "—";

  const ds = d.dtb_state || {};
  mDopamine.textContent = ds.dopamine != null ? ds.dopamine.toFixed(3) : "—";
  mCortisol.textContent = ds.cortisol != null ? ds.cortisol.toFixed(3) : "—";

  // Score bars
  const cs = d.component_scores || {};
  setBar("G",   cs.generalization ?? 0);
  setBar("R",   cs.reasoning ?? 0);
  setBar("L",   cs.linguistic ?? 0);
  setBar("DTB", cs.neuronal_dtb ?? 0);
  setBar("MQ",  d.mq_score ?? 0);

  // Schematic details
  document.getElementById("sch0-detail").textContent = "NIV · 66 books · 15,914 passages";
  document.getElementById("sch1-detail").textContent =
    `T_G(100,32) · ε=${(d.epsilon||0).toFixed(4)}`;
  document.getElementById("sch2-detail").textContent =
    `T_R(100,64) · ε_r=${(d.r_scores?.epsilon_r||0).toFixed(3)}`;
  document.getElementById("sch3-detail").textContent =
    `T_LLM(100,32) · ${((d.llm_answer||"").split(" ").length)}w`;
  document.getElementById("sch4-detail").textContent =
    `T_DTB(100,24) · dopa=${(ds.dopamine||0).toFixed(3)}`;
  document.getElementById("sch5-detail").textContent =
    `T_AGI(1,1024) · conf=${((d.confidence||0)*100).toFixed(1)}%`;

  // Activate all steps
  [0,1,2,3,4,5].forEach(i => {
    document.getElementById(`sch${i}`).classList.add("active");
  });
}

function setBar(key, val) {
  const pct = Math.round(Math.min(val, 1) * 100);
  document.getElementById(`bar-${key}`).style.width = pct + "%";
  document.getElementById(`val-${key}`).textContent = val.toFixed(3);
}

// ── Schematic animation ───────────────────────────────────────────────────────
function animateSchematic() {
  resetSchematic();
  [0,1,2,3,4,5].forEach((i, idx) => {
    setTimeout(() => {
      const el = document.getElementById(`sch${i}`);
      if (el) el.classList.add("active");
    }, idx * 180);
  });
}

function resetSchematic() {
  [0,1,2,3,4,5].forEach(i => {
    const el = document.getElementById(`sch${i}`);
    if (el) el.classList.remove("active");
  });
}

// ── Status ────────────────────────────────────────────────────────────────────
function setStatus(text, state) {
  const dot  = statusDot.querySelector(".dot");
  const span = statusDot.querySelector(".status-text");
  dot.className  = "dot" + (state !== "ready" ? ` ${state}` : "");
  span.textContent = text;
}

function setProcessing(active) {
  isProcessing = active;
  submitBtn.disabled = active;
  if (active) {
    setStatus("PROCESSING", "processing");
  }
}

// ── Message helpers ───────────────────────────────────────────────────────────
function appendMsg(label, text, cssClass) {
  const block = document.createElement("div");
  block.className = `msg-block ${cssClass}`;
  block.innerHTML = `
    <div class="msg-label">${label}</div>
    <div class="msg-content">${escapeHtml(text)}</div>
  `;
  chatWindow.appendChild(block);
  scrollChat();
}

function appendProcessing(id) {
  const block = document.createElement("div");
  block.className = "msg-block processing-block";
  block.id = id;
  block.innerHTML = `
    <div class="msg-label">AGI PIPELINE</div>
    <div class="msg-content">
      <div class="processing-dots">
        <span></span><span></span><span></span>
      </div>
      <div class="processing-text">Running 5-component inference...</div>
    </div>
  `;
  chatWindow.appendChild(block);
  scrollChat();
}

function removeProcessing(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendWelcome() {
  const block = document.createElement("div");
  block.className = "msg-block system-block";
  block.innerHTML = `
    <div class="msg-label">SYSTEM</div>
    <div class="msg-content">
      <p class="welcome-title">AGI Inference System — POC v2.0</p>
      <p>Five-component architecture running. Foundation initialized from NIV Scripture corpus — 66 books, 15,914 passages. Submit any query.</p>
      <div class="component-pills">
        <span class="pill">G_enhanced</span>
        <span class="pill">R_enhanced</span>
        <span class="pill pill-accent">ai_LLM ◈ Claude</span>
        <span class="pill">DTB</span>
        <span class="pill">CV_Adapt → T_AGI</span>
      </div>
    </div>
  `;
  chatWindow.appendChild(block);
}

// ── Log helpers ───────────────────────────────────────────────────────────────
function clearLog() {
  pipelineLog.innerHTML = "";
}

function logLine(text, cssClass) {
  if (!text) { pipelineLog.appendChild(document.createElement("br")); return; }
  const div = document.createElement("div");
  div.className = `log-line ${cssClass}`;
  div.textContent = text;
  pipelineLog.appendChild(div);
  pipelineLog.scrollTop = pipelineLog.scrollHeight;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function scrollChat() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function timestamp() {
  return new Date().toLocaleTimeString("en-US", { hour12: false });
}
