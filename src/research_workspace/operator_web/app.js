"use strict";

const state = {
  csrf: null,
  account: null,
  capabilities: null,
  conversationId: null,
  conversations: [],
  messages: [],
  lastChatRequest: null,
  chatController: null,
  agentSessionId: null,
  researchJobId: null,
  draftTimer: null,
};

const $ = (selector, root = document) => root.querySelector(selector);

function element(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(options)) {
    if (key === "className") node.className = value;
    else if (key === "text") node.textContent = String(value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "attributes") {
      for (const [name, attributeValue] of Object.entries(value)) {
        node.setAttribute(name, String(attributeValue));
      }
    } else if (key in node) {
      node[key] = value;
    }
  }
  for (const child of children) {
    if (child !== null && child !== undefined) {
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
  }
  return node;
}

function setLive(message) {
  $("#live-status").textContent = message;
}

function showNotice(message, kind = "info") {
  const notice = $("#notice");
  notice.textContent = message;
  notice.className = kind === "error" ? "notice error" : "notice";
  notice.hidden = false;
  setLive(message);
}

function clearNotice() {
  $("#notice").hidden = true;
  $("#notice").textContent = "";
}

const friendlyErrors = {
  authentication_failed: "The email or credential was not accepted.",
  authentication_required: "Your session has ended. Sign in again to continue.",
  session_expired: "Your session expired. Your unsent draft is still here.",
  session_revoked: "Your access changed and this session was closed. Sign in again.",
  csrf_validation_failed: "The secure form token expired. Sign in again and retry.",
  origin_not_allowed: "This browser origin is not authorized.",
  host_not_allowed: "This host name is not authorized.",
  repository_not_authorized: "That repository is not authorized for this account.",
  capacity_guardrail: "This job is queued until research capacity is available.",
  model_endpoint_unavailable: "The selected local model is unavailable. Your draft was preserved.",
  response_validation_failed: "The model response did not pass deterministic validation.",
  agent_validation_failed: "The agent result did not pass verification.",
  conversation_not_found: "That conversation is unavailable to this account.",
  research_job_not_found: "That research job is unavailable to this account.",
  artifact_not_found: "That artifact is unavailable to this account.",
  artifact_integrity_failure: "The artifact hash no longer matches its registry record.",
  internal_error: "Laplace encountered an internal error. Use the trace ID when asking an operator for help.",
};

class ApiError extends Error {
  constructor(status, category, payload) {
    super(category);
    this.status = status;
    this.category = category;
    this.payload = payload;
  }
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (options.mutation && state.csrf) headers.set("X-CSRF-Token", state.csrf);
  const response = await fetch(path, {
    method: options.method || "GET",
    headers,
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    credentials: "same-origin",
    signal: options.signal,
    cache: "no-store",
  });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = payload && typeof payload === "object" ? payload.detail : null;
    const category = (
      typeof detail === "string" ? detail :
      payload && typeof payload === "object" && typeof payload.failure_category === "string" ?
        payload.failure_category : `http_${response.status}`
    );
    if (response.status === 401 && !path.startsWith("/api/v1/auth/")) {
      showAuthentication(category);
    }
    throw new ApiError(response.status, category, payload);
  }
  return payload;
}

function errorMessage(error) {
  if (error && error.name === "AbortError") return "Cancelled.";
  if (error instanceof ApiError) return friendlyErrors[error.category] || `Request failed: ${error.category}`;
  return "Laplace is unreachable. Your draft has been preserved.";
}

function switchAuthMode(mode) {
  const activation = mode === "activation";
  $("#login-form").hidden = activation;
  $("#activation-form").hidden = !activation;
  $("#login-tab").setAttribute("aria-selected", String(!activation));
  $("#activation-tab").setAttribute("aria-selected", String(activation));
  $("#auth-title").textContent = activation ? "Activate your account" : "Sign in";
  $("#auth-copy").textContent = activation ?
    "Use the one-time code printed locally by your administrator, then create your password." :
    "Use an email registered by your local administrator.";
  $("#auth-error").textContent = "";
  const first = activation ? $("#activation-form input") : $("#login-form input");
  first.focus();
}

function showAuthentication(reason = null) {
  state.csrf = null;
  state.account = null;
  $("#app-shell").setAttribute("aria-hidden", "true");
  $("#auth-error").textContent = reason ? (friendlyErrors[reason] || "Please sign in again.") : "";
  const dialog = $("#auth-dialog");
  if (!dialog.open) dialog.showModal();
  document.body.classList.add("dialog-open");
  $("#login-form input[name=email]").focus();
}

function hideAuthentication() {
  const dialog = $("#auth-dialog");
  if (dialog.open) dialog.close();
  document.body.classList.remove("dialog-open");
  $("#app-shell").setAttribute("aria-hidden", "false");
}

async function submitLogin(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  $("#auth-error").textContent = "";
  try {
    const result = await api("/api/v1/auth/login", {
      method: "POST",
      body: {email: form.get("email"), password: form.get("password")},
    });
    await acceptSession(result);
    event.currentTarget.reset();
  } catch (error) {
    $("#auth-error").textContent = errorMessage(error);
  }
}

async function submitActivation(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  if (form.get("new_password") !== form.get("confirm_password")) {
    $("#auth-error").textContent = "The new passwords do not match.";
    return;
  }
  $("#auth-error").textContent = "";
  try {
    const result = await api("/api/v1/auth/activate", {
      method: "POST",
      body: {
        email: form.get("email"),
        activation_code: form.get("activation_code"),
        new_password: form.get("new_password"),
      },
    });
    await acceptSession(result);
    event.currentTarget.reset();
  } catch (error) {
    $("#auth-error").textContent = errorMessage(error);
  }
}

async function acceptSession(session) {
  state.csrf = session.csrf_token;
  state.account = session.account;
  if (session.development_http && session.deployment_mode !== "local" && session.deployment_mode !== "ssh-tunnel") {
    $("#security-warning").hidden = false;
  }
  hideAuthentication();
  await initializeWorkspace();
}

async function restoreSession() {
  try {
    const session = await api("/api/v1/auth/session");
    if (session.status !== "AUTHENTICATED") throw new ApiError(401, "authentication_required", session);
    await acceptSession(session);
  } catch {
    try {
      const health = await api("/api/v1/health");
      $("#connection-label").textContent = health.status === "OK" ? "Local API online" : "Service degraded";
      $("#connection-dot").className = health.status === "OK" ? "status-dot online" : "status-dot degraded";
    } catch {
      $("#connection-label").textContent = "Service unreachable";
    }
    showAuthentication();
  }
}

function navDefinition() {
  const items = [
    ["chat", "Chat", "C"],
    ["help", "Help", "?"],
    ["about", "System", "i"],
  ];
  if (state.capabilities?.agent_enabled) items.splice(1, 0, ["agent", "Agent", "A"]);
  if (state.capabilities?.operator_enabled) {
    items.splice(1, 0, ["research", "Research", "R"]);
    items.splice(items.length - 2, 0, ["operations", "Operations", "O"], ["users", "Users", "U"], ["models", "Models & GPU", "M"]);
  }
  return items;
}

function buildNavigation() {
  const nav = $("#primary-nav");
  nav.replaceChildren();
  for (const [id, label, glyph] of navDefinition()) {
    const button = element("button", {
      className: "nav-button",
      type: "button",
      dataset: {view: id},
      attributes: {"aria-current": id === "chat" ? "page" : "false"},
    }, [
      element("span", {className: "nav-glyph", text: glyph, attributes: {"aria-hidden": "true"}}),
      element("span", {text: label}),
    ]);
    button.addEventListener("click", () => activateView(id));
    nav.append(button);
  }
}

function activateView(id) {
  const target = document.getElementById(id);
  if (!target) return;
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === id));
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.setAttribute("aria-current", button.dataset.view === id ? "page" : "false");
  });
  $("#view-title").textContent = target.dataset.title || "Laplace";
  $("#view-eyebrow").textContent = (
    id === "agent" ? "Repository-bound workspace" :
    id === "research" ? "Evidence-led workflow" :
    id === "operations" || id === "users" || id === "models" ? "Operator controls" :
    "Private local inference"
  );
  $("#workspace").focus({preventScroll: true});
  window.location.hash = id;
  if (id === "operations") loadOperations();
  if (id === "users") loadUsers();
  if (id === "models") loadModels();
}

async function initializeWorkspace() {
  clearNotice();
  const [capabilities, help, about] = await Promise.all([
    api("/api/v1/tier/capabilities"),
    api("/api/v1/help"),
    api("/api/v1/about"),
  ]);
  state.capabilities = capabilities;
  $("#account-name").textContent = state.account.display_name || state.account.email;
  $("#account-tier").textContent = `${capabilities.capability_tier} · ${capabilities.role}`;
  $("#chat-lane").value = capabilities.default_lane;
  $("#connection-label").textContent = "Local API online";
  $("#connection-dot").className = "status-dot online";
  buildNavigation();
  buildRoleWorkspace();
  renderHelp(help);
  renderAbout(about);
  renderAccount();
  await loadConversations();
  const requested = window.location.hash.slice(1);
  const allowed = navDefinition().some(([id]) => id === requested);
  activateView(allowed ? requested : "chat");
  setLive(`Signed in as ${state.account.display_name || state.account.email}`);
}

function buildRoleWorkspace() {
  const mount = $("#role-workspace");
  mount.replaceChildren();
  if (state.capabilities?.agent_enabled) mount.append(buildAgentView());
  if (state.capabilities?.operator_enabled) {
    mount.append(buildOperationsView(), buildUsersView(), buildModelsView());
  }
}

function buildAgentView() {
  const repoSelect = element("select", {id: "agent-repository", name: "repo_id", required: true});
  repoSelect.append(element("option", {value: "", text: "Choose an authorized repository"}));
  for (const repo of state.capabilities.authorized_repositories || []) {
    repoSelect.append(element("option", {value: repo.repo_id, text: repo.logical_name || repo.repo_id}));
  }
  const domainSelect = element("select", {id: "agent-domain", name: "domain"}, [
    element("option", {value: "python", text: "Python"}),
    element("option", {value: "systemverilog", text: "SystemVerilog"}),
  ]);
  const form = element("form", {id: "agent-form", className: "surface form-grid"}, [
    element("label", {text: "Authorized repository"}, [repoSelect]),
    element("label", {text: "Quality lane"}, [
      element("select", {name: "lane"}, [
        element("option", {value: "quality", text: "Quality"}),
        element("option", {value: "standard", text: "Standard", selected: true}),
        element("option", {value: "economy", text: "Economy"}),
      ]),
    ]),
    element("label", {text: "Engineering domain"}, [domainSelect]),
    element("label", {className: "wide", text: "Bounded task"}, [
      element("textarea", {name: "instruction", rows: 6, required: true, placeholder: "Describe the requested repository change…"}),
    ]),
    element("div", {className: "wide button-row"}, [
      element("button", {className: "button primary", type: "submit", text: "Start isolated agent"}),
      element("button", {id: "cancel-agent", className: "button danger", type: "button", text: "Cancel", disabled: true}),
    ]),
  ]);
  form.addEventListener("submit", runAgent);
  return element("section", {id: "agent", className: "view", dataset: {title: "Repository Agent"}}, [
    element("div", {className: "page-intro"}, [
      element("div", {}, [element("p", {className: "eyebrow", text: "Plus capability"}), element("h2", {text: "Repository-bound agent"})]),
      element("p", {text: "The server resolves this logical repository ID, creates an isolated worktree, denies network access, and verifies the resulting patch."}),
    ]),
    form,
    element("article", {className: "surface"}, [
      element("div", {className: "section-heading"}, [element("h3", {text: "Plan and status"}), element("span", {id: "agent-state", className: "state-pill", text: "Not started"})]),
      element("div", {id: "agent-plan", className: "stack-list"}, [element("p", {className: "subtle", text: "Select a repository and describe a bounded task."})]),
    ]),
    element("article", {className: "surface"}, [element("h3", {text: "File changes"}), element("div", {id: "agent-files", className: "stack-list"})]),
    element("article", {className: "surface"}, [element("h3", {text: "Unified diff"}), element("pre", {id: "agent-diff", className: "diff-view", text: "No diff yet."})]),
    element("article", {className: "surface"}, [element("h3", {text: "Tests and verification"}), element("ul", {id: "agent-tests", className: "verification-list"})]),
    element("details", {className: "help-card"}, [
      element("summary", {text: "Repository isolation and allowed tools"}),
      element("p", {text: "Allowed tools are read_file, apply_patch, and run_validation. The client cannot submit a filesystem path. Absolute paths, traversal, links, mounts, submodules, nested repositories, and sibling worktrees are rejected server-side."}),
    ]),
  ]);
}

function buildOperationsView() {
  const reload = element("button", {className: "button secondary", type: "button", text: "Reload registry"});
  reload.addEventListener("click", reloadRegistry);
  return element("section", {id: "operations", className: "view", dataset: {title: "Operations"}}, [
    element("div", {className: "page-intro"}, [
      element("div", {}, [element("p", {className: "eyebrow", text: "Non-secret state"}), element("h2", {text: "Operator dashboard"})]),
      reload,
    ]),
    element("div", {id: "operations-cards", className: "card-grid"}),
    element("div", {className: "two-column"}, [
      element("article", {className: "surface"}, [element("h3", {text: "Queues and guardrails"}), element("div", {id: "operations-queues", className: "stack-list"})]),
      element("article", {className: "surface"}, [element("h3", {text: "Readiness"}), element("div", {id: "operations-readiness", className: "stack-list"})]),
    ]),
    element("article", {className: "surface"}, [element("h3", {text: "Repositories and approvals"}), element("div", {id: "operations-repositories", className: "table-wrap"})]),
    element("details", {className: "help-card"}, [element("summary", {text: "About approvals, queues, and safe lifecycle controls"}), element("p", {text: "Risk-bearing server actions require the configured approval role. Queue reservations keep Quality available. Stop controls act only on Laplace-owned PIDs recorded by the lifecycle manager."})]),
  ]);
}

function buildUsersView() {
  return element("section", {id: "users", className: "view", dataset: {title: "User Management"}}, [
    element("div", {className: "page-intro"}, [
      element("div", {}, [element("p", {className: "eyebrow", text: "Registered accounts"}), element("h2", {text: "Users and access"})]),
      element("p", {text: "Passwords, hashes, activation codes, and session identifiers are never returned here."}),
    ]),
    element("article", {className: "surface"}, [element("div", {id: "user-table", className: "table-wrap"})]),
    element("details", {className: "help-card"}, [element("summary", {text: "Managing roles, tiers, repositories, and sessions"}), element("p", {text: "Use the local user_admin command for durable access changes. Any role, tier, password, enabled-state, default-lane, or repository change revokes affected sessions."})]),
  ]);
}

function buildModelsView() {
  const refresh = element("button", {className: "button secondary", type: "button", text: "Refresh status"});
  refresh.addEventListener("click", loadModels);
  return element("section", {id: "models", className: "view", dataset: {title: "Models & GPU"}}, [
    element("div", {className: "page-intro"}, [
      element("div", {}, [element("p", {className: "eyebrow", text: "Local serving"}), element("h2", {text: "Models and hardware"})]),
      refresh,
    ]),
    element("div", {id: "model-cards", className: "card-grid"}),
    element("article", {className: "surface"}, [element("h3", {text: "Sanitized model endpoints"}), element("div", {id: "model-endpoints", className: "stack-list"})]),
    element("article", {className: "surface"}, [element("h3", {text: "Serving profile"}), element("div", {id: "profile-status", className: "stack-list"})]),
    element("details", {className: "help-card"}, [element("summary", {text: "GPU memory and ownership"}), element("p", {text: "GPU VRAM, system RAM, iGPU memory, and NPU memory are separate resources. Laplace keeps one main generative model resident by default and never stops an unrelated process."})]),
  ]);
}

function renderHelp(payload) {
  const mount = $("#help-functions");
  mount.replaceChildren();
  for (const item of payload.functions || []) {
    mount.append(element("article", {className: "info-card"}, [
      element("h3", {text: item.name || "Function"}),
      element("p", {text: item.description || "Available to this account."}),
    ]));
  }
}

function renderAbout(payload) {
  const cards = $("#about-cards");
  cards.replaceChildren(
    infoCard("Application", payload.application_version || "unknown"),
    infoCard("Git revision", String(payload.git_revision || "unavailable").slice(0, 12)),
    infoCard("API", payload.api_version || "v1"),
    infoCard("Access", `${payload.capability_tier} · ${payload.role}`),
    infoCard("Remote mode", payload.remote_access_mode || "local"),
    infoCard("Health", payload.health || "unknown"),
  );
  const lanes = Object.entries(payload.model_lanes || {}).map(([lane, route]) => [
    lane, route.display_name, route.context_limit, route.output_limit,
  ]);
  $("#about-lanes").replaceChildren(makeTable(["Lane", "Model", "Context", "Output"], lanes));
  const links = $("#documentation-links");
  links.replaceChildren();
  for (const name of payload.documentation || []) {
    links.append(element("span", {className: "state-pill", text: name}));
  }
}

function renderAccount() {
  const details = $("#account-details");
  details.replaceChildren();
  const rows = [
    ["Email", state.account.email],
    ["Display name", state.account.display_name],
    ["Role", state.account.role],
    ["Capability", state.account.capability_tier],
    ["Default lane", state.account.default_lane],
    ["Authorized repositories", (state.capabilities.authorized_repositories || []).map((repo) => repo.logical_name || repo.repo_id).join(", ") || "None"],
  ];
  for (const [label, value] of rows) details.append(element("dt", {text: label}), element("dd", {text: value || "—"}));
}

function infoCard(label, value, description = "") {
  return element("article", {className: "info-card"}, [
    element("h3", {text: label}),
    element("strong", {text: value ?? "—"}),
    description ? element("p", {text: description}) : null,
  ]);
}

function makeTable(headers, rows) {
  const table = element("table");
  table.append(element("thead", {}, [element("tr", {}, headers.map((name) => element("th", {text: name})))]));
  const body = element("tbody");
  for (const row of rows) body.append(element("tr", {}, row.map((value) => element("td", {text: value ?? "—"}))));
  table.append(body);
  return table;
}

async function loadConversations() {
  const payload = await api("/api/v1/conversations");
  state.conversations = payload.conversations || [];
  renderConversationList();
  if (state.conversationId && !state.conversations.some((item) => item.conversation_id === state.conversationId)) {
    state.conversationId = null;
    state.messages = [];
    renderMessages();
  }
}

function renderConversationList() {
  const list = $("#conversation-list");
  list.replaceChildren();
  if (!state.conversations.length) {
    list.append(element("p", {className: "subtle", text: "No conversations yet."}));
    return;
  }
  for (const conversation of state.conversations) {
    const open = element("button", {className: "conversation-open", type: "button", text: conversation.title});
    open.addEventListener("click", () => openConversation(conversation.conversation_id));
    const menu = element("details", {className: "conversation-menu"}, [
      element("summary", {attributes: {"aria-label": `Actions for ${conversation.title}`}, text: "…"}),
    ]);
    const actions = element("div", {className: "stack-list"});
    for (const [label, action] of [["Rename", "rename"], [conversation.archived ? "Reopen" : "Archive", "archive"], ["Delete", "delete"]]) {
      const button = element("button", {className: "text-button", type: "button", text: label});
      button.addEventListener("click", () => conversationAction(conversation, action));
      actions.append(button);
    }
    menu.append(actions);
    list.append(element("div", {className: `conversation-item${conversation.conversation_id === state.conversationId ? " active" : ""}`}, [open, menu]));
  }
}

async function createConversation() {
  try {
    const result = await api("/api/v1/conversations", {method: "POST", mutation: true, body: {title: "New conversation"}});
    state.conversationId = result.conversation.conversation_id;
    state.messages = [];
    $("#conversation-title").textContent = result.conversation.title;
    $("#chat-message").value = result.conversation.draft || "";
    renderMessages();
    await loadConversations();
    activateView("chat");
    $("#chat-message").focus();
  } catch (error) {
    showNotice(errorMessage(error), "error");
  }
}

async function openConversation(id) {
  try {
    const result = await api(`/api/v1/conversations/${encodeURIComponent(id)}`);
    state.conversationId = id;
    state.messages = result.conversation.messages || [];
    $("#conversation-title").textContent = result.conversation.title;
    $("#chat-message").value = result.conversation.draft || "";
    autoSizeComposer();
    renderConversationList();
    renderMessages();
    activateView("chat");
  } catch (error) {
    showNotice(errorMessage(error), "error");
  }
}

async function conversationAction(conversation, action) {
  try {
    if (action === "delete") {
      if (!window.confirm(`Delete “${conversation.title}”?`)) return;
      await api(`/api/v1/conversations/${encodeURIComponent(conversation.conversation_id)}`, {method: "DELETE", mutation: true});
      if (state.conversationId === conversation.conversation_id) {
        state.conversationId = null;
        state.messages = [];
        $("#conversation-title").textContent = "New conversation";
        renderMessages();
      }
    } else {
      let body;
      if (action === "rename") {
        const title = window.prompt("Conversation title", conversation.title);
        if (!title) return;
        body = {title};
      } else {
        body = {archived: !conversation.archived};
      }
      await api(`/api/v1/conversations/${encodeURIComponent(conversation.conversation_id)}`, {method: "PATCH", mutation: true, body});
    }
    await loadConversations();
  } catch (error) {
    showNotice(errorMessage(error), "error");
  }
}

function renderMessages() {
  const mount = $("#message-list");
  mount.replaceChildren();
  if (!state.messages.length) {
    mount.append(element("div", {className: "empty-state"}, [
      element("span", {className: "empty-mark", text: "λ", attributes: {"aria-hidden": "true"}}),
      element("h2", {text: "Start with a question"}),
      element("p", {text: "Laplace keeps this conversation in your private server-side workspace."}),
    ]));
    return;
  }
  for (const message of state.messages) mount.append(messageCard(message));
}

function messageCard(message) {
  const isUser = message.role === "user";
  const content = isUser ? element("p", {text: message.content}) : renderMarkdown(message.content);
  const card = element("article", {className: `message-card ${isUser ? "user" : "assistant"}`}, [
    element("span", {className: "message-label", text: isUser ? "You" : "Laplace"}),
    content,
  ]);
  if (!isUser) {
    const copy = element("button", {className: "text-button", type: "button", text: "Copy response"});
    copy.addEventListener("click", () => copyText(message.content, copy));
    card.append(element("div", {className: "message-actions"}, [copy]));
    if (message.metadata && Object.keys(message.metadata).length) card.append(responseDetails(message.metadata));
  }
  return card;
}

function responseDetails(metadata, raw = null) {
  const rows = [
    ["Request ID", metadata.request_id],
    ["Trace ID", metadata.trace_id],
    ["Requested lane", metadata.requested_lane],
    ["Effective lane", metadata.effective_lane],
    ["Model", metadata.model_id],
    ["Queue wait", metadata.queue_wait_seconds !== undefined ? `${metadata.queue_wait_seconds}s` : null],
    ["Queue position", metadata.queue_position],
    ["Context limit", metadata.context_limit],
    ["Output limit", metadata.output_limit],
    ["Finish reason", metadata.finish_reason],
    ["Token usage", metadata.token_usage ? JSON.stringify(metadata.token_usage) : null],
    ["Escalation", metadata.escalation ? JSON.stringify(metadata.escalation) : "None"],
  ].filter(([, value]) => value !== undefined && value !== null);
  const dl = element("dl", {className: "metadata-grid"});
  for (const [label, value] of rows) dl.append(element("dt", {text: label}), element("dd", {text: value}));
  const details = element("details", {className: "response-details"}, [element("summary", {text: "Response details"}), dl]);
  if (raw && state.capabilities?.operator_enabled) {
    const rawDetails = element("details", {}, [element("summary", {text: "Operator debug JSON"}), element("pre", {className: "diff-view", text: JSON.stringify(raw, null, 2)})]);
    details.append(rawDetails);
  }
  return details;
}

function renderMarkdown(source) {
  const root = element("div", {className: "markdown"});
  const lines = String(source || "").replaceAll("\r\n", "\n").split("\n");
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (line.startsWith("```")) {
      const language = line.slice(3).trim().slice(0, 40) || "text";
      const code = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith("```")) code.push(lines[index++]);
      if (index < lines.length) index += 1;
      const copy = element("button", {className: "text-button", type: "button", text: "Copy code"});
      copy.addEventListener("click", () => copyText(code.join("\n"), copy));
      root.append(element("div", {className: "code-block"}, [
        element("div", {className: "code-header"}, [element("span", {text: language}), copy]),
        element("pre", {}, [element("code", {text: code.join("\n")})]),
      ]));
      continue;
    }
    const heading = /^(#{1,3})\s+(.+)$/.exec(line);
    if (heading) {
      root.append(element(`h${heading[1].length}`, {}, inlineMarkdown(heading[2])));
      index += 1;
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const list = element("ul");
      while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) {
        list.append(element("li", {}, inlineMarkdown(lines[index].replace(/^\s*[-*]\s+/, ""))));
        index += 1;
      }
      root.append(list);
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const list = element("ol");
      while (index < lines.length && /^\s*\d+\.\s+/.test(lines[index])) {
        list.append(element("li", {}, inlineMarkdown(lines[index].replace(/^\s*\d+\.\s+/, ""))));
        index += 1;
      }
      root.append(list);
      continue;
    }
    if (line.startsWith("> ")) {
      root.append(element("blockquote", {}, inlineMarkdown(line.slice(2))));
      index += 1;
      continue;
    }
    if (!line.trim()) {
      index += 1;
      continue;
    }
    const paragraph = [line];
    index += 1;
    while (index < lines.length && lines[index].trim() && !/^(#{1,3})\s|^```|^\s*[-*]\s+|^\s*\d+\.\s+|^> /.test(lines[index])) {
      paragraph.push(lines[index++]);
    }
    root.append(element("p", {}, inlineMarkdown(paragraph.join(" "))));
  }
  return root;
}

function inlineMarkdown(text) {
  const nodes = [];
  const pattern = /(`[^`\n]+`)|(\[([^\]\n]{1,300})\]\(([^)\s]{1,2048})\))|(\*\*([^*\n]+)\*\*)/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > cursor) nodes.push(document.createTextNode(text.slice(cursor, match.index)));
    if (match[1]) {
      nodes.push(element("code", {className: "inline-code", text: match[1].slice(1, -1)}));
    } else if (match[2]) {
      const safe = safeLink(match[4]);
      if (safe) {
        const link = element("a", {href: safe.href, text: match[3]});
        if (safe.external) {
          link.className = "external";
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          link.setAttribute("aria-label", `${match[3]} (external link)`);
        }
        nodes.push(link);
      } else {
        nodes.push(document.createTextNode(match[3]));
      }
    } else {
      nodes.push(element("strong", {text: match[6]}));
    }
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) nodes.push(document.createTextNode(text.slice(cursor)));
  return nodes;
}

function safeLink(raw) {
  try {
    const parsed = new URL(raw, window.location.origin);
    if (!["http:", "https:"].includes(parsed.protocol)) return null;
    return {href: parsed.href, external: parsed.origin !== window.location.origin};
  } catch {
    return null;
  }
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    const prior = button.textContent;
    button.textContent = "Copied";
    window.setTimeout(() => { button.textContent = prior; }, 1200);
  } catch {
    showNotice("Clipboard access is unavailable in this browser.", "error");
  }
}

async function submitChat(event) {
  event.preventDefault();
  clearNotice();
  const textarea = $("#chat-message");
  const content = textarea.value.trim();
  if (!content) return;
  const requestMessages = [...state.messages.map(({role, content: messageContent}) => ({role, content: messageContent})), {role: "user", content}];
  state.lastChatRequest = {content, messages: requestMessages};
  state.messages.push({role: "user", content, metadata: {}});
  renderMessages();
  $("#chat-state").textContent = "Queued · waiting for the selected local lane";
  $("#stop-chat").hidden = false;
  $("#retry-chat").hidden = true;
  const started = performance.now();
  const controller = new AbortController();
  state.chatController = controller;
  try {
    const result = await api("/api/v1/chat", {
      method: "POST",
      mutation: true,
      signal: controller.signal,
      body: {
        lane: $("#chat-lane").value,
        domain: $("#chat-domain").value,
        conversation_id: state.conversationId,
        messages: requestMessages,
      },
    });
    state.conversationId = result.conversation_id;
    const response = result.response || {};
    const assistant = {
      role: "assistant",
      content: response.content || "The local model returned no readable content.",
      metadata: {
        request_id: result.request_id,
        trace_id: result.trace_id,
        requested_lane: result.requested_lane,
        effective_lane: result.effective_lane,
        model_id: result.model_id,
        queue_wait_seconds: result.queue_wait_seconds,
        queue_position: result.queue_position,
        context_limit: result.context_limit,
        output_limit: result.output_limit,
        finish_reason: response.finish_reason,
        token_usage: response.usage,
        escalation: result.escalation,
        raw: result,
      },
    };
    state.messages.push(assistant);
    textarea.value = "";
    autoSizeComposer();
    renderMessages();
    const lastCard = $("#message-list .message-card:last-child");
    const existingDetails = lastCard?.querySelector(".response-details");
    if (existingDetails && state.capabilities.operator_enabled) {
      existingDetails.append(element("details", {}, [
        element("summary", {text: "Operator debug JSON"}),
        element("pre", {className: "diff-view", text: JSON.stringify(result, null, 2)}),
      ]));
    }
    $("#chat-state").textContent = `Complete · ${result.effective_lane} · ${result.model_id} · ${((performance.now() - started) / 1000).toFixed(1)}s`;
    await loadConversations();
  } catch (error) {
    state.messages.pop();
    renderMessages();
    $("#retry-chat").hidden = false;
    $("#chat-state").textContent = error?.name === "AbortError" ? "Cancelled" : "Failed · draft preserved";
    showNotice(errorMessage(error), error?.name === "AbortError" ? "info" : "error");
  } finally {
    state.chatController = null;
    $("#stop-chat").hidden = true;
  }
}

async function retryChat() {
  if (!state.lastChatRequest) return;
  $("#chat-message").value = state.lastChatRequest.content;
  autoSizeComposer();
  $("#chat-form").requestSubmit();
}

function autoSizeComposer() {
  const textarea = $("#chat-message");
  const visualLines = textarea.value.split("\n").reduce(
    (total, line) => total + Math.max(1, Math.ceil(line.length / 88)),
    0,
  );
  textarea.rows = Math.min(10, Math.max(2, visualLines));
}

function saveDraftSoon() {
  window.clearTimeout(state.draftTimer);
  if (!state.conversationId) return;
  state.draftTimer = window.setTimeout(async () => {
    try {
      await api(`/api/v1/conversations/${encodeURIComponent(state.conversationId)}`, {
        method: "PATCH", mutation: true, body: {draft: $("#chat-message").value},
      });
    } catch {
      // The textarea remains intact; a later authenticated edit retries persistence.
    }
  }, 450);
}

async function runAgent(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const repoId = String(form.get("repo_id") || "");
  if (!repoId) {
    showNotice("Choose a server-authorized repository.", "error");
    return;
  }
  const sessionId = `agent-${crypto.randomUUID().replaceAll("-", "")}`;
  state.agentSessionId = sessionId;
  $("#agent-state").textContent = "Binding isolated worktree";
  $("#agent-plan").replaceChildren(stackItem("1 · Bind", "Resolving the server-side grant and creating an isolated worktree."));
  try {
    const bound = await api("/api/v1/agent/sessions", {
      method: "POST", mutation: true,
      body: {repo_id: repoId, session_id: sessionId, allowed_tools: ["read_file", "apply_patch", "run_validation"], max_commands: 100, max_wall_seconds: 1800},
    });
    $("#cancel-agent").disabled = false;
    $("#agent-plan").append(stackItem("2 · Execute", `${bound.binding.logical_repository_name} · ${bound.binding.worktree_status}`));
    $("#agent-state").textContent = "Running bounded task";
    const result = await api(`/api/v1/agent/sessions/${encodeURIComponent(sessionId)}/messages`, {
      method: "POST", mutation: true,
      body: {lane: form.get("lane"), instruction: form.get("instruction"), domain: form.get("domain")},
    });
    renderAgentResult(result);
  } catch (error) {
    $("#agent-state").textContent = "Failed";
    $("#agent-state").className = "state-pill error";
    showNotice(errorMessage(error), "error");
  }
}

function renderAgentResult(result) {
  const response = result.response || {};
  $("#agent-state").textContent = "Complete";
  $("#agent-state").className = "state-pill";
  $("#agent-plan").append(stackItem("3 · Verify", `${response.verification_status || "UNKNOWN"} · ${result.effective_lane || "—"} · ${result.model_id || "—"}`));
  const files = $("#agent-files");
  files.replaceChildren();
  for (const path of response.modified_paths || result.modified_paths || []) files.append(stackItem(path, "Modified inside the isolated worktree"));
  if (!files.children.length) files.append(element("p", {className: "subtle", text: "No modified paths were reported."}));
  $("#agent-diff").textContent = response.diff || result.diff || "The backend reported validated file changes without an inline diff.";
  const tests = $("#agent-tests");
  tests.replaceChildren();
  const results = response.tests || result.tests || [{name: "Deterministic patch validation", status: response.verification_status || "PASSED"}];
  for (const test of results) {
    const passed = String(test.status).toUpperCase().includes("PASS");
    tests.append(element("li", {className: passed ? "verification-pass" : "verification-fail", text: `${test.name || "Verification"} · ${test.status || "UNKNOWN"}`}));
  }
}

async function cancelAgent() {
  if (!state.agentSessionId) return;
  try {
    await api(`/api/v1/agent/sessions/${encodeURIComponent(state.agentSessionId)}/cancel`, {method: "POST", mutation: true});
    $("#agent-state").textContent = "Cancelled";
    $("#cancel-agent").disabled = true;
  } catch (error) {
    showNotice(errorMessage(error), "error");
  }
}

const researchStageNames = [
  "Request validation", "Question decomposition", "Source discovery", "Deduplication",
  "Source fetch", "Extraction", "Claim normalization", "Conflict analysis",
  "Evidence scoring", "Synthesis", "Citation check", "Report assembly", "Final verification",
];

function renderResearchStages(completed = []) {
  const mount = $("#research-stages");
  mount.replaceChildren();
  for (const stage of researchStageNames) {
    const normalized = stage.toLowerCase().replaceAll(" ", "_");
    mount.append(element("li", {className: completed.includes(normalized) || completed.includes(stage) ? "done" : "", text: stage}));
  }
}

async function createResearch(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const jobId = `research-${crypto.randomUUID().replaceAll("-", "")}`;
  try {
    const result = await api("/api/v1/research/jobs", {
      method: "POST", mutation: true,
      body: {
        research_job_id: jobId,
        job: {
          question: form.get("question"),
          scope: form.get("scope"),
          research_mode: form.get("research_mode"),
          search_backends: [form.get("backend")],
          source_policy: form.get("source_policy"),
          model_route: "deterministic",
        },
      },
    });
    state.researchJobId = result.research_job_id;
    const admission = result.admission || {};
    $("#research-queue").textContent = admission.state === "QUEUED" ? `Queued · position ${admission.queue_position}` : "Admitted";
    $("#research-queue").className = admission.state === "QUEUED" ? "state-pill warning" : "state-pill";
    $("#run-research").disabled = admission.state === "QUEUED";
    $("#cancel-research").disabled = false;
    renderResearchStages([]);
  } catch (error) {
    showNotice(errorMessage(error), "error");
  }
}

async function runResearch() {
  if (!state.researchJobId) return;
  $("#research-queue").textContent = "Running";
  $("#run-research").disabled = true;
  try {
    await api(`/api/v1/research/jobs/${encodeURIComponent(state.researchJobId)}/run`, {method: "POST", mutation: true});
    const report = await api(`/api/v1/research/jobs/${encodeURIComponent(state.researchJobId)}/report`);
    renderResearchReport(report);
    $("#research-queue").textContent = "Complete";
    $("#cancel-research").disabled = true;
    $("#export-research").disabled = false;
  } catch (error) {
    $("#research-queue").textContent = "Failed";
    $("#research-queue").className = "state-pill error";
    showNotice(errorMessage(error), "error");
  }
}

function renderResearchReport(payload) {
  renderResearchStages(payload.job.completed_stages || []);
  const sources = $("#research-sources");
  sources.replaceChildren();
  for (const source of payload.job.source_records || []) {
    let domain = "local source";
    try { domain = new URL(source.canonical_url).hostname || "local source"; } catch { /* display logical local label */ }
    sources.append(stackItem(source.title, `${domain} · retrieved ${source.retrieved_at || "unknown"} · ${source.source_type}`));
  }
  if (!sources.children.length) sources.append(element("p", {className: "subtle", text: "No source records were returned."}));
  $("#research-report").replaceChildren(renderMarkdown(payload.report_markdown || ""));
  const claims = payload.evidence_ledger?.claims || payload.job.claims || [];
  const contested = claims.filter((claim) => claim.status === "contested").length;
  $("#research-confidence").textContent = contested ? `${contested} contested claim(s)` : `${claims.length} claim(s) · conflicts visible`;
  $("#research-confidence").className = contested ? "state-pill warning" : "state-pill";
}

async function cancelResearch() {
  if (!state.researchJobId) return;
  try {
    await api(`/api/v1/research/jobs/${encodeURIComponent(state.researchJobId)}/cancel`, {method: "POST", mutation: true});
    $("#research-queue").textContent = "Cancelled";
    $("#run-research").disabled = true;
    $("#cancel-research").disabled = true;
  } catch (error) {
    showNotice(errorMessage(error), "error");
  }
}

async function exportResearch() {
  if (!state.researchJobId) return;
  try {
    const result = await api(`/api/v1/research/jobs/${encodeURIComponent(state.researchJobId)}/export`, {method: "POST", mutation: true});
    const link = element("a", {className: "button secondary", href: result.download_url, text: `Download ${result.artifact.name}`});
    link.setAttribute("download", result.artifact.name);
    $("#research-report").prepend(link);
  } catch (error) {
    showNotice(errorMessage(error), "error");
  }
}

function stackItem(title, detail) {
  return element("div", {className: "stack-item"}, [element("strong", {text: title}), element("small", {text: detail})]);
}

async function loadOperations() {
  const cards = $("#operations-cards");
  if (!cards) return;
  try {
    const [dashboard, readiness, version] = await Promise.all([
      api("/api/v1/dashboard"), api("/api/v1/readiness"), api("/api/v1/version"),
    ]);
    cards.replaceChildren(
      infoCard("Application", version.application_version),
      infoCard("Active sessions", dashboard.active_browser_sessions ?? 0),
      infoCard("Registered users", (dashboard.users || []).length),
      infoCard("Artifacts", dashboard.provenance?.registered_artifacts ?? 0),
      infoCard("Deployment", dashboard.deployment?.mode || "local"),
      infoCard("Configuration", String(dashboard.registry_revision || "unavailable").slice(0, 12)),
    );
    const queues = $("#operations-queues");
    queues.replaceChildren();
    for (const [name, value] of Object.entries(dashboard.queue_guardrails || {})) queues.append(stackItem(name.replaceAll("_", " "), typeof value === "object" ? JSON.stringify(value) : value));
    const ready = $("#operations-readiness");
    ready.replaceChildren(stackItem(readiness.status, readiness.reasons?.length ? readiness.reasons.join(", ") : "Registry, sessions, state, and lanes are ready."));
    $("#operations-repositories").replaceChildren(makeTable(
      ["Repository", "Grants", "Registered"],
      (dashboard.repositories || []).map((repo) => [repo.logical_name || repo.repo_id, repo.active_grants, repo.registered_at_utc]),
    ));
  } catch (error) {
    showNotice(errorMessage(error), "error");
  }
}

async function reloadRegistry() {
  try {
    const result = await api("/api/v1/admin/registry/reload", {method: "POST", mutation: true});
    showNotice(result.status === "RELOADED" ? `Registry reloaded; ${result.revoked_user_count} account(s) changed.` : "Invalid registry rejected; the last valid configuration remains active.");
    if (result.status === "RELOADED") await loadOperations();
  } catch (error) {
    showNotice(errorMessage(error), "error");
  }
}

async function loadUsers() {
  const mount = $("#user-table");
  if (!mount) return;
  try {
    const result = await api("/api/v1/admin/users");
    const table = element("table");
    table.append(element("thead", {}, [element("tr", {}, ["Email", "Name", "Enabled", "Capability", "Role", "Default lane", "Repositories", "Sessions"].map((name) => element("th", {text: name})))]));
    const body = element("tbody");
    for (const user of result.users || []) {
      const revoke = element("button", {className: "text-button", type: "button", text: "Revoke"});
      revoke.addEventListener("click", () => revokeUserSessions(user.user_id));
      body.append(element("tr", {}, [
        element("td", {text: user.email}), element("td", {text: user.display_name}),
        element("td", {text: user.enabled ? "Enabled" : "Disabled"}),
        element("td", {text: user.capability_tier}), element("td", {text: user.role}),
        element("td", {text: user.default_lane}),
        element("td", {text: (user.authorized_repo_ids || []).join(", ") || "None"}),
        element("td", {}, [revoke]),
      ]));
    }
    table.append(body);
    mount.replaceChildren(table);
  } catch (error) {
    showNotice(errorMessage(error), "error");
  }
}

async function revokeUserSessions(userId) {
  try {
    const result = await api(`/api/v1/admin/users/${encodeURIComponent(userId)}/sessions/revoke`, {method: "POST", mutation: true});
    showNotice(`${result.count} active session(s) revoked.`);
  } catch (error) {
    showNotice(errorMessage(error), "error");
  }
}

async function loadModels() {
  const cards = $("#model-cards");
  if (!cards) return;
  try {
    const [dashboard, profiles] = await Promise.all([
      api("/api/v1/dashboard"),
      api("/api/v1/serving-profiles/status").catch(() => ({status: "UNAVAILABLE"})),
    ]);
    const gpu = dashboard.model_servers?.gpu_observation?.gpu || {};
    cards.replaceChildren(
      infoCard("GPU", gpu.name || "Not observed"),
      infoCard("GPU memory free", gpu.memory_free_mib !== undefined ? `${gpu.memory_free_mib} MiB` : "Unavailable"),
      infoCard("GPU utilization", gpu.utilization_percent !== undefined ? `${gpu.utilization_percent}%` : "Unavailable"),
      infoCard("Laplace-owned processes", (dashboard.model_servers?.laplace_owned_processes || []).length),
      infoCard("Profile", profiles.profile_id || profiles.status || "Unavailable"),
      infoCard("Profile state", profiles.status || "Unavailable"),
    );
    const endpoints = $("#model-endpoints");
    endpoints.replaceChildren();
    for (const server of dashboard.model_servers?.servers || []) {
      endpoints.append(stackItem(server.expected_model_id || server.profile || "Local model", `${server.endpoint_observation?.status || server.status || "unknown"} · loopback port ${server.port || "—"}`));
    }
    if (!endpoints.children.length) endpoints.append(element("p", {className: "subtle", text: "No model endpoint is currently healthy."}));
    $("#profile-status").replaceChildren(stackItem(profiles.profile_id || "No active profile", profiles.status || "UNAVAILABLE"));
  } catch (error) {
    showNotice(errorMessage(error), "error");
  }
}

async function signOut(all = false) {
  try {
    await api(all ? "/api/v1/auth/logout-all" : "/api/v1/auth/logout", {method: "POST", mutation: true});
  } catch {
    // Clear the local authenticated view even when the server has already expired it.
  }
  state.csrf = null;
  state.account = null;
  state.capabilities = null;
  state.messages = [];
  if ($("#account-dialog").open) $("#account-dialog").close();
  showAuthentication();
}

async function changePassword(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const current = form.get("current_password");
  const replacement = form.get("new_password");
  if (!current || !replacement) return;
  try {
    const result = await api("/api/v1/auth/change-password", {
      method: "POST", mutation: true,
      body: {current_password: current, new_password: replacement},
    });
    state.csrf = result.csrf_token;
    state.account = result.account;
    event.currentTarget.reset();
    $("#account-dialog").close();
    showNotice("Password changed and prior sessions revoked.");
  } catch (error) {
    showNotice(errorMessage(error), "error");
  }
}

function bindEvents() {
  $("#login-tab").addEventListener("click", () => switchAuthMode("login"));
  $("#activation-tab").addEventListener("click", () => switchAuthMode("activation"));
  $("#login-form").addEventListener("submit", submitLogin);
  $("#activation-form").addEventListener("submit", submitActivation);
  document.querySelectorAll(".show-password").forEach((button) => button.addEventListener("click", () => {
    const input = document.getElementById(button.dataset.target);
    const reveal = input.type === "password";
    input.type = reveal ? "text" : "password";
    button.textContent = reveal ? "Hide" : "Show";
    button.setAttribute("aria-pressed", String(reveal));
  }));
  $("#chat-form").addEventListener("submit", submitChat);
  $("#chat-message").addEventListener("input", () => { autoSizeComposer(); saveDraftSoon(); });
  $("#chat-message").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("#chat-form").requestSubmit();
    }
  });
  $("#stop-chat").addEventListener("click", () => state.chatController?.abort());
  $("#retry-chat").addEventListener("click", retryChat);
  $("#new-conversation").addEventListener("click", createConversation);
  $("#research-form").addEventListener("submit", createResearch);
  $("#run-research").addEventListener("click", runResearch);
  $("#cancel-research").addEventListener("click", cancelResearch);
  $("#export-research").addEventListener("click", exportResearch);
  $("#account-button").addEventListener("click", () => $("#account-dialog").showModal());
  $("#close-account").addEventListener("click", () => $("#account-dialog").close());
  $("#account-form").addEventListener("submit", changePassword);
  $("#logout-all").addEventListener("click", () => signOut(true));
  $("#sign-out-button").addEventListener("click", () => signOut(false));
}

bindEvents();
renderResearchStages([]);
restoreSession();
