"use strict";

const state = {
  token: sessionStorage.getItem("laplace_operator_token") || "",
  csrf: "",
  role: "",
  userId: "",
  capabilityTier: "",
  agentSession: "",
  currentResearchJob: "",
  eventCursor: 0,
};

const byId = (id) => document.getElementById(id);
const text = (tag, value, className = "") => {
  const node = document.createElement(tag);
  node.textContent = String(value ?? "—");
  if (className) node.className = className;
  return node;
};

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Authorization", `Bearer ${state.token}`);
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  if (options.method && options.method !== "GET" && path !== "/api/v1/session") {
    headers.set("X-CSRF-Token", state.csrf);
  }
  const response = await fetch(path, { ...options, headers });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("json") ? await response.json() : await response.text();
  if (!response.ok) {
    const message = payload.detail || payload.failure_category || `HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload;
}

function announce(message) {
  byId("live-status").textContent = message;
}

function warn(message = "") {
  const banner = byId("warning-banner");
  banner.textContent = message;
  banner.hidden = !message;
}

async function establishSession() {
  const session = await api("/api/v1/session", { method: "POST" });
  state.csrf = session.csrf_token;
  state.role = session.role;
  state.userId = session.user_id;
  state.capabilityTier = session.capability_tier;
  byId("role-badge").textContent = `${session.capability_tier} · ${session.role}`;
  byId("connection-dot").classList.add("online");
  byId("connection-label").textContent = "Local API online";
  announce(`Authenticated as ${session.user_id} with ${session.capability_tier} capability`);
  return session;
}

function showLogin(message = "") {
  byId("login-error").textContent = message;
  const dialog = byId("login-dialog");
  if (!dialog.open) dialog.showModal();
  byId("token-input").focus();
}

async function signIn(token) {
  state.token = token;
  try {
    await establishSession();
    sessionStorage.setItem("laplace_operator_token", token);
    byId("login-dialog").close();
    configureShell();
    if (state.capabilityTier === "operator") {
      await loadDashboard();
      startEventStream();
    } else {
      await loadTierWorkspace();
    }
  } catch (error) {
    state.token = "";
    sessionStorage.removeItem("laplace_operator_token");
    showLogin(`Authentication failed: ${error.message}`);
  }
}

function signOut() {
  sessionStorage.removeItem("laplace_operator_token");
  state.token = "";
  state.csrf = "";
  state.role = "";
  state.userId = "";
  state.capabilityTier = "";
  state.agentSession = "";
  byId("role-badge").textContent = "signed out";
  byId("connection-dot").classList.remove("online");
  showLogin();
}

function activateView(name) {
  if (state.capabilityTier && state.capabilityTier !== "operator" && name !== "tier-workspace") {
    name = "tier-workspace";
  }
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === name));
  document.querySelectorAll("#primary-nav a").forEach((link) => {
    if (link.dataset.view === name) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  const view = byId(name);
  byId("view-title").textContent = view?.dataset.title || "Laplace";
  byId("workspace").focus({ preventScroll: true });
  if (name === "approvals") loadApprovals();
  if (name === "hardware") loadHardware();
  if (name === "corpora") loadCorpora();
  if (name === "diagnostics") loadDiagnostics();
}

function configureShell() {
  const operator = state.capabilityTier === "operator";
  document.querySelectorAll("#primary-nav a").forEach((link) => {
    link.hidden = operator ? link.dataset.view === "tier-workspace" : link.dataset.view !== "tier-workspace";
  });
  byId("refresh-button").hidden = !operator;
  if (operator) {
    activateView(location.hash.slice(1) && location.hash.slice(1) !== "tier-workspace" ? location.hash.slice(1) : "dashboard");
  } else {
    activateView("tier-workspace");
  }
}

async function loadTierWorkspace() {
  const capability = await api("/api/v1/tier/capabilities");
  byId("capability-label").textContent = capability.capability_tier;
  byId("tier-user").textContent = capability.user_id;
  byId("plus-agent-panel").hidden = !capability.agent_enabled;
  const standard = capability.routes?.standard || {};
  byId("tier-route-model").textContent = standard.model_id || "—";
  byId("tier-queue-status").textContent = `${(capability.queue?.waiting || []).length} waiting`;
  byId("tier-limits").textContent = `${capability.effective_limits?.standard_capacity || "—"} standard`;
  announce(`${capability.capability_tier} workspace ready`);
}

async function sendTierChat(form) {
  const values = new FormData(form);
  const lane = String(values.get("lane"));
  byId("lane-label").textContent = lane;
  const result = await api("/api/v1/chat", {
    method: "POST",
    body: JSON.stringify({
      lane,
      domain: String(values.get("domain")),
      messages: [{ role: "user", content: String(values.get("message")) }],
    }),
  });
  byId("tier-chat-response").textContent = JSON.stringify(result, null, 2);
  announce(`Chat completed on ${result.effective_lane} lane`);
}

async function bindAgentSession(form) {
  const values = new FormData(form);
  const sessionId = String(values.get("session_id"));
  const result = await api("/api/v1/agent/sessions", {
    method: "POST",
    body: JSON.stringify({
      repo_id: String(values.get("repo_id")),
      session_id: sessionId,
      allowed_tools: ["read_file", "apply_patch", "run_validation"],
    }),
  });
  state.agentSession = sessionId;
  byId("agent-response").textContent = JSON.stringify(result, null, 2);
  announce(`Agent bound to isolated session ${sessionId}`);
}

async function runAgent(form) {
  if (!state.agentSession) throw new Error("Bind an authorized repository session first");
  const values = new FormData(form);
  const result = await api(`/api/v1/agent/sessions/${encodeURIComponent(state.agentSession)}/run`, {
    method: "POST",
    body: JSON.stringify({
      lane: String(values.get("lane")),
      domain: String(values.get("domain")),
      instruction: String(values.get("instruction")),
    }),
  });
  byId("agent-response").textContent = JSON.stringify(result, null, 2);
  announce(`Agent request ${result.request_id} completed`);
}

async function cancelAgent() {
  if (!state.agentSession) return;
  const result = await api(`/api/v1/agent/sessions/${encodeURIComponent(state.agentSession)}/cancel`, { method: "POST" });
  byId("agent-response").textContent = JSON.stringify(result, null, 2);
  if (result.status === "RELEASED_CLEAN_WORKTREE") state.agentSession = "";
  announce(result.status);
}

function renderTable(container, rows, columns) {
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach(([label]) => headerRow.append(text("th", label)));
  head.append(headerRow);
  table.append(head);
  const body = document.createElement("tbody");
  rows.forEach((row) => {
    const line = document.createElement("tr");
    columns.forEach(([, key]) => line.append(text("td", row[key])));
    body.append(line);
  });
  table.append(body);
  container.replaceChildren(table);
}

async function loadDashboard() {
  try {
    const data = await api("/api/v1/dashboard");
    warn("");
    const counts = data.run_counts || {};
    const active = Number(counts.RUNNING || 0) + Number(counts.QUEUED || 0);
    byId("hero-active-runs").textContent = String(active);
    const cards = [
      ["Prepared", counts.PREPARED || 0],
      ["Complete", counts.COMPLETE || 0],
      ["Pending approvals", (data.approval_counts || {}).PENDING || 0],
      ["Research jobs", (data.research_jobs || []).length],
    ];
    byId("dashboard-cards").replaceChildren(...cards.map(([label, value]) => {
      const card = text("article", "", "metric");
      card.append(text("span", label), text("strong", value));
      return card;
    }));
    const recent = data.recent_runs || [];
    if (recent.length) renderTable(byId("recent-runs"), recent, [["Run", "run_id"], ["Arm", "arm_id"], ["State", "state"], ["Updated", "updated_at_utc"]]);
    else byId("recent-runs").textContent = "No prepared runs yet.";
    const jobs = data.research_jobs || [];
    byId("research-summary").replaceChildren(...jobs.map((job) => {
      const item = text("div", "", "stack-item");
      item.append(text("strong", job.question), text("small", `${job.status} · ${job.current_stage || "not started"}`));
      return item;
    }));
    if (!jobs.length) byId("research-summary").textContent = "No research jobs yet.";
  } catch (error) {
    warn(`Dashboard unavailable: ${error.message}`);
  }
}

async function startEventStream() {
  if (!state.token) return;
  try {
    const response = await fetch(`/api/v1/events?after_sequence=${state.eventCursor}&once=true`, {
      headers: { Authorization: `Bearer ${state.token}` },
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const body = await response.text();
    body.split("\n").forEach((line) => {
      if (!line.startsWith("data: ") || line === "data: {}") return;
      const event = JSON.parse(line.slice(6));
      state.eventCursor = Math.max(state.eventCursor, Number(event.sequence || 0));
      addTimelineEvent(event);
    });
  } catch (error) {
    warn(`Live event stream warning: ${error.message}`);
  }
  window.setTimeout(startEventStream, 1500);
}

function addTimelineEvent(event) {
  const item = text("li", `${event.action} · ${event.entity_id}`, "done");
  item.title = event.timestamp_utc || "";
  byId("event-timeline").prepend(item);
}

async function prepareRun(form) {
  const values = new FormData(form);
  const configuration = {
    task_id: values.get("task_id"),
    arm_id: values.get("arm_id"),
    model_route: values.get("model_route"),
    corpus_snapshot_sha256: values.get("corpus_snapshot_sha256"),
    skills_lock_sha256: values.get("skills_lock_sha256"),
    smoke_profile: values.get("smoke_profile"),
    request_sha256: values.get("request_sha256"),
    gpu_required: values.get("gpu_required") === "on",
  };
  const result = await api("/api/v1/runs", { method: "POST", body: JSON.stringify({ configuration }) });
  byId("run-preview").textContent = JSON.stringify(result, null, 2);
  byId("live-run-id").value = result.run_id;
  announce(`Prepared run ${result.run_id}`);
  await loadDashboard();
}

async function inspectRun(form) {
  const runId = new FormData(form).get("run_id");
  const result = await api(`/api/v1/runs/${encodeURIComponent(runId)}`);
  byId("run-record").textContent = JSON.stringify(result, null, 2);
  const terminal = result.terminal_result || {};
  const verification = terminal.verification || {};
  const matrix = verification.gate_matrix || [];
  byId("gate-matrix").replaceChildren(...matrix.map((gate) => {
    const item = text("article", "", `mini-card ${gate.status === "PASS" ? "pass" : "fail"}`);
    item.append(
      text("strong", gate.gate_id),
      text("p", `${gate.status} · ${gate.tool} · executed ${gate.executed === true ? "yes" : "no"}`)
    );
    return item;
  }));
  if (!matrix.length) byId("gate-matrix").textContent = "No verification gate matrix recorded.";
  announce(`Loaded immutable run evidence for ${runId}`);
}

async function inspectArtifact(form) {
  const path = new FormData(form).get("path");
  const result = await api(`/api/v1/artifacts?path=${encodeURIComponent(path)}`);
  byId("artifact-name").textContent = `${result.name} · ${result.sha256.slice(0, 12)}`;
  byId("artifact-preview").textContent = result.preview || "Binary artifact; use download.";
  const link = byId("artifact-download");
  link.href = "#";
  link.dataset.path = String(path);
  link.hidden = false;
}

async function createResearch(form) {
  const values = new FormData(form);
  const body = {
    job: {
      question: values.get("question"),
      scope: values.get("scope"),
      research_mode: values.get("research_mode"),
      search_backends: [values.get("backend")],
      source_policy: values.get("source_policy"),
      model_route: "deterministic",
    },
  };
  const result = await api("/api/v1/research/jobs", { method: "POST", body: JSON.stringify(body) });
  state.currentResearchJob = result.research_job_id;
  byId("run-research-button").disabled = false;
  announce(`Created research job ${state.currentResearchJob}`);
}

async function runResearch() {
  if (!state.currentResearchJob) return;
  const result = await api(`/api/v1/research/jobs/${encodeURIComponent(state.currentResearchJob)}/run`, { method: "POST" });
  const report = await api(`/api/v1/research/jobs/${encodeURIComponent(state.currentResearchJob)}/report`);
  byId("research-report").textContent = report.report_markdown;
  const stages = report.job.completed_stages || [];
  byId("research-stages").replaceChildren(...stages.map((stage) => text("li", stage, "done")));
  const claims = report.evidence_ledger.claims || [];
  byId("evidence-ledger").replaceChildren(...claims.map((claim) => {
    const item = text("div", "", "stack-item");
    item.append(text("strong", claim.normalized_claim), text("small", `${claim.status} · ${claim.supporting_source_ids.length} supporting · ${claim.contradicting_source_ids.length} contradicting`));
    return item;
  }));
  announce(`Research report ${result.status}`);
}

async function loadHardware() {
  const container = byId("model-servers");
  container.textContent = "Probing local endpoints…";
  try {
    const data = await api("/api/v1/model-servers/status");
    const gpu = data.gpu_observation?.gpu || {};
    byId("gpu-status").replaceChildren(...[
      ["GPU", gpu.name || data.gpu_observation?.status],
      ["VRAM free", gpu.memory_free_mib ? `${gpu.memory_free_mib} MiB` : "—"],
      ["Utilization", gpu.utilization_percent !== undefined ? `${gpu.utilization_percent}%` : "—"],
      ["Owned processes", (data.laplace_owned_processes || []).length],
    ].map(([label, value]) => {
      const card = text("article", "", "metric");
      card.append(text("span", label), text("strong", value));
      return card;
    }));
    container.replaceChildren(...(data.servers || []).map((server) => {
      const item = text("div", "", "stack-item");
      item.append(text("strong", `${server.profile} · ${server.expected_model_id}`), text("small", `${server.endpoint_observation?.status || "unknown"} · port ${server.port}`));
      return item;
    }));
    try {
      const profiles = await api("/api/v1/serving-profiles/status");
      const owned = profiles.owned_profile;
      byId("serving-profile-status").textContent = owned
        ? `${owned.profile_id} · PID ${owned.pid} · ${owned.alive_exact_pid ? "running" : "not running"}`
        : `No owned profile · available: ${(profiles.available_profiles || []).join(", ")}`;
    } catch (error) {
      byId("serving-profile-status").textContent = `Profile runtime unavailable: ${error.message}`;
    }
  } catch (error) {
    container.textContent = `Probe failed safely: ${error.message}`;
  }
}

async function servingProfileAction(form) {
  const values = new FormData(form);
  const action = String(values.get("action"));
  const result = await api("/api/v1/serving-profiles/action", {
    method: "POST",
    body: JSON.stringify({
      action,
      profile_id: action === "start" ? String(values.get("profile_id")) : null,
    }),
  });
  byId("serving-profile-status").textContent = JSON.stringify(result, null, 2);
  announce(`Serving profile action ${action}: ${result.status}`);
}

async function loadApprovals() {
  const data = await api("/api/v1/approvals");
  const container = byId("approval-list");
  const approvals = data.approvals || [];
  container.replaceChildren(...approvals.map((approval) => {
    const item = text("article", "", "stack-item");
    item.append(text("strong", approval.action), text("small", `${approval.entity_id} · ${approval.state}`));
    if (approval.state === "PENDING" && ["approve", "admin"].includes(state.role)) {
      const actions = text("div", "", "form-actions");
      ["Approve", "Reject"].forEach((label) => {
        const button = text("button", label, `button ${label === "Approve" ? "primary" : "secondary"}`);
        button.type = "button";
        button.addEventListener("click", async () => {
          await api(`/api/v1/approvals/${approval.approval_id}/decision`, { method: "POST", body: JSON.stringify({ approve: label === "Approve" }) });
          await loadApprovals();
        });
        actions.append(button);
      });
      item.append(actions);
    }
    return item;
  }));
  if (!approvals.length) container.textContent = "No approvals recorded.";
}

async function loadCorpora() {
  const data = await api("/api/v1/corpora");
  const snapshots = data.governed_snapshots || [];
  byId("corpus-list").replaceChildren(...snapshots.map((snapshot) => {
    const card = text("article", "", "mini-card");
    card.append(text("strong", snapshot.domain), text("p", snapshot.current.snapshot_sha256));
    return card;
  }));
  if (!snapshots.length) byId("corpus-list").textContent = "No promoted snapshots. Existing measured corpus locks are unaffected.";
}

async function compareRuns(form) {
  const values = new FormData(form);
  const data = await api(`/api/v1/runs/compare/${encodeURIComponent(values.get("left"))}/${encodeURIComponent(values.get("right"))}`);
  const rows = Object.entries(data.comparison).map(([field, value]) => ({ field, left: value.left, right: value.right, equal: value.equal }));
  renderTable(byId("comparison-table"), rows, [["Field", "field"], ["Left", "left"], ["Right", "right"], ["Equal", "equal"]]);
}

async function loadDiagnostics() {
  const data = await api("/api/v1/diagnostics");
  const entries = Object.entries(data).filter(([key]) => key !== "research_backends_supported");
  byId("diagnostic-cards").replaceChildren(...entries.map(([key, value]) => {
    const card = text("article", "", "mini-card");
    card.append(text("strong", key.replaceAll("_", " ")), text("p", Array.isArray(value) ? value.join(", ") : value));
    return card;
  }));
}

function bindEvents() {
  window.addEventListener("hashchange", () => activateView(location.hash.slice(1) || "dashboard"));
  byId("login-form").addEventListener("submit", (event) => { event.preventDefault(); signIn(byId("token-input").value); });
  byId("sign-out-button").addEventListener("click", signOut);
  byId("refresh-button").addEventListener("click", loadDashboard);
  byId("hardware-refresh").addEventListener("click", loadHardware);
  byId("approvals-refresh").addEventListener("click", loadApprovals);
  byId("run-form").addEventListener("submit", (event) => { event.preventDefault(); prepareRun(event.currentTarget).catch((error) => warn(error.message)); });
  byId("live-run-form").addEventListener("submit", (event) => { event.preventDefault(); inspectRun(event.currentTarget).catch((error) => warn(error.message)); });
  byId("artifact-form").addEventListener("submit", (event) => { event.preventDefault(); inspectArtifact(event.currentTarget).catch((error) => warn(error.message)); });
  byId("artifact-download").addEventListener("click", async (event) => {
    event.preventDefault();
    const path = event.currentTarget.dataset.path;
    if (!path) return;
    try {
      const response = await fetch(`/api/v1/artifacts/download?path=${encodeURIComponent(path)}`, {
        headers: { Authorization: `Bearer ${state.token}` },
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const objectUrl = URL.createObjectURL(await response.blob());
      const download = document.createElement("a");
      download.href = objectUrl;
      download.download = byId("artifact-name").textContent.split(" · ")[0];
      download.click();
      URL.revokeObjectURL(objectUrl);
    } catch (error) {
      warn(`Download failed: ${error.message}`);
    }
  });
  byId("research-form").addEventListener("submit", (event) => { event.preventDefault(); createResearch(event.currentTarget).catch((error) => warn(error.message)); });
  byId("run-research-button").addEventListener("click", () => runResearch().catch((error) => warn(error.message)));
  byId("compare-form").addEventListener("submit", (event) => { event.preventDefault(); compareRuns(event.currentTarget).catch((error) => warn(error.message)); });
  byId("tier-chat-form").addEventListener("submit", (event) => { event.preventDefault(); sendTierChat(event.currentTarget).catch((error) => warn(error.message)); });
  byId("agent-session-form").addEventListener("submit", (event) => { event.preventDefault(); bindAgentSession(event.currentTarget).catch((error) => warn(error.message)); });
  byId("agent-run-form").addEventListener("submit", (event) => { event.preventDefault(); runAgent(event.currentTarget).catch((error) => warn(error.message)); });
  byId("agent-cancel").addEventListener("click", () => cancelAgent().catch((error) => warn(error.message)));
  byId("serving-profile-form").addEventListener("submit", (event) => { event.preventDefault(); servingProfileAction(event.currentTarget).catch((error) => warn(error.message)); });
}

async function boot() {
  bindEvents();
  activateView(location.hash.slice(1) || "dashboard");
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
  if (!state.token) return showLogin();
  try {
    await establishSession();
    configureShell();
    if (state.capabilityTier === "operator") {
      await loadDashboard();
      startEventStream();
    } else {
      await loadTierWorkspace();
    }
  } catch {
    signOut();
  }
}

document.addEventListener("DOMContentLoaded", boot);
