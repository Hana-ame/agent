/**
 * Vertex-Edge Agent Framework (VEA) V4 Dashboard Application Logic
 */

let currentSession = "default_session";
let eventSource = null;
let cachedVertices = {};
let cachedEdges = {};
let cachedGraph = { vertices: {}, edges: {}, tiers: {} };
let chatMessages = [];
let availableModels = ["default"];

// Helper: Escape HTML string
function esc(s) {
  if (s === null || s === undefined) return "";
  const d = document.createElement("div");
  d.textContent = String(s);
  return d.innerHTML;
}


// API-key support: when the server enforces VEA_API_KEY, the browser must send
// it. Set it once via localStorage.vea_api_key (the dashboard prompts on 401).
function getApiKey() {
  try { return localStorage.getItem("vea_api_key") || ""; } catch (e) { return ""; }
}

async function apiFetch(url, options = {}) {
  const opts = Object.assign({}, options);
  const key = getApiKey();
  if (key) {
    opts.headers = Object.assign({}, opts.headers || {}, { "X-API-Key": key });
  }
  const res = await fetch(url, opts);
  if (res.status === 401) {
    const entered = window.prompt("This server requires an API key (VEA_API_KEY). Enter it to continue:");
    if (entered) {
      try { localStorage.setItem("vea_api_key", entered.trim()); } catch (e) {}
      return apiFetch(url, options);
    }
  }
  return res;
}

// Helper: Get state CSS badge class
function getBadgeClass(state) {
  const s = (state || "").toLowerCase().replace(/ /g, "-");
  return "badge badge-" + s;
}

// Switch Active Tab
function switchTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
  
  const targetBtn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
  const targetPane = document.getElementById(`tab-${tabId}`);
  if (targetBtn) targetBtn.classList.add("active");
  if (targetPane) targetPane.classList.add("active");

  if (tabId === "topology") renderDagGraph();
  if (tabId === "metrics") refreshMetrics();
  if (tabId === "chat") scrollChatToBottom();
}

// ---------------------------------------------------------------------------
// Session Management
// ---------------------------------------------------------------------------

async function loadSessions() {
  try {
    const res = await apiFetch("/api/db/sessions");
    const sessions = await res.json();
    const select = document.getElementById("sessionSelect");
    select.innerHTML = "";

    if (!sessions || sessions.length === 0) {
      sessions.push(currentSession);
    }
    sessions.forEach(s => {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s;
      select.appendChild(opt);
    });

    if (sessions.includes(currentSession)) {
      select.value = currentSession;
    } else {
      select.value = sessions[0];
      currentSession = sessions[0];
    }
  } catch (err) {
    console.error("Failed loading sessions:", err);
  }
}

async function onSessionChange() {
  const select = document.getElementById("sessionSelect");
  currentSession = select.value;
  chatMessages = [];
  renderChatHistory();
  await refreshAll();
  connectEventStream();
}

async function createNewSession() {
  const name = prompt("Enter new session ID:");
  if (!name || !name.trim()) return;
  const clean = name.trim();
  currentSession = clean;
  await loadSessions();
  const select = document.getElementById("sessionSelect");
  select.value = clean;
  await onSessionChange();
}

async function clearCurrentSession() {
  if (!confirm(`Permanently clear all graph data, staging, and metrics for session '${currentSession}'?`)) {
    return;
  }
  try {
    const res = await apiFetch(`/api/db/sessions/${currentSession}/clear`, { method: "POST" });
    const data = await res.json();
    alert(`Session cleared. Purged ${data.purged_vertices || 0} vertices.`);
    await refreshAll();
  } catch (err) {
    alert("Error clearing session: " + err);
  }
}

// ---------------------------------------------------------------------------
// Interactive SVG DAG Topology Renderer
// ---------------------------------------------------------------------------

function renderDagGraph() {
  const svg = document.getElementById("dagSvg");
  if (!svg) return;
  svg.innerHTML = "";

  const vertices = Object.values(cachedVertices);
  const edges = Object.values(cachedEdges);

  if (vertices.length === 0) {
    svg.innerHTML = `<text x="50%" y="50%" text-anchor="middle" fill="#64748b" font-size="14">No vertices in session '${esc(currentSession)}'. Add vertices or run a pipeline to view DAG.</text>`;
    return;
  }

  // Define Arrow Marker
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  defs.innerHTML = `
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b" />
    </marker>
    <marker id="arrow-active" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8" />
    </marker>
  `;
  svg.appendChild(defs);

  // Compute Topological Levels (Tiers)
  const nodeLevels = {};
  const inDegree = {};
  const adj = {};

  vertices.forEach(v => {
    nodeLevels[v.name] = 0;
    inDegree[v.name] = 0;
    adj[v.name] = [];
  });

  edges.forEach(e => {
    if (e.input_vertex !== e.output_vertex && adj[e.input_vertex]) {
      adj[e.input_vertex].push(e.output_vertex);
      inDegree[e.output_vertex] = (inDegree[e.output_vertex] || 0) + 1;
    }
  });

  // Kahn's level assignment
  const queue = vertices.filter(v => inDegree[v.name] === 0).map(v => v.name);
  while (queue.length > 0) {
    const u = queue.shift();
    const nextLevel = nodeLevels[u] + 1;
    (adj[u] || []).forEach(v => {
      if (nextLevel > nodeLevels[v]) {
        nodeLevels[v] = nextLevel;
      }
      inDegree[v]--;
      if (inDegree[v] === 0) {
        queue.push(v);
      }
    });
  }

  // Group vertices by level
  const levels = {};
  vertices.forEach(v => {
    const lvl = nodeLevels[v.name] || 0;
    if (!levels[lvl]) levels[lvl] = [];
    levels[lvl].push(v);
  });

  // Position nodes
  const nodeCoords = {};
  const colWidth = 240;
  const rowHeight = 110;
  const startX = 60;
  const startY = 60;

  const sortedLevels = Object.keys(levels).map(Number).sort((a, b) => a - b);
  sortedLevels.forEach(lvl => {
    const group = levels[lvl];
    const totalH = group.length * rowHeight;
    group.forEach((v, idx) => {
      const x = startX + lvl * colWidth;
      const y = startY + idx * rowHeight;
      nodeCoords[v.name] = { x, y, width: 170, height: 60 };
    });
  });

  // Draw Edges (Layer 1)
  const gEdges = document.createElementNS("http://www.w3.org/2000/svg", "g");
  edges.forEach(e => {
    const src = nodeCoords[e.input_vertex];
    const dst = nodeCoords[e.output_vertex];
    if (!src || !dst) return;

    if (e.input_vertex === e.output_vertex || e.type === "reflexive") {
      // Reflexive Loop Curve over top
      const x = src.x + 85;
      const y = src.y;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", `M ${x - 20} ${y} C ${x - 30} ${y - 45}, ${x + 30} ${y - 45}, ${x + 20} ${y}`);
      path.setAttribute("class", "edge-path");
      path.setAttribute("marker-end", "url(#arrow)");
      path.setAttribute("stroke-dasharray", "4 2");
      gEdges.appendChild(path);

      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", x);
      label.setAttribute("y", y - 50);
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("fill", "#f59e0b");
      label.setAttribute("font-size", "10");
      label.textContent = `↺ ${e.id} (${e.trigger_state || "reject"})`;
      gEdges.appendChild(label);
    } else {
      // Directed Bezier Curve
      const startX = src.x + src.width;
      const startY = src.y + src.height / 2;
      const endX = dst.x;
      const endY = dst.y + dst.height / 2;
      const c1X = startX + (endX - startX) / 2;
      const c1Y = startY;
      const c2X = startX + (endX - startX) / 2;
      const c2Y = endY;

      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", `M ${startX} ${startY} C ${c1X} ${c1Y}, ${c2X} ${c2Y}, ${endX} ${endY}`);
      path.setAttribute("class", "edge-path");
      path.setAttribute("id", `edge-${e.id}`);
      path.setAttribute("marker-end", "url(#arrow)");
      gEdges.appendChild(path);

      // Edge Type Tag
      const midX = (startX + endX) / 2;
      const midY = (startY + endY) / 2;
      const tag = document.createElementNS("http://www.w3.org/2000/svg", "text");
      tag.setAttribute("x", midX);
      tag.setAttribute("y", midY - 6);
      tag.setAttribute("text-anchor", "middle");
      tag.setAttribute("fill", "#94a3b8");
      tag.setAttribute("font-size", "10");
      tag.textContent = `${e.id} [${e.type}]`;
      gEdges.appendChild(tag);
    }
  });
  svg.appendChild(gEdges);

  // Draw Nodes (Layer 2)
  const gNodes = document.createElementNS("http://www.w3.org/2000/svg", "g");
  vertices.forEach(v => {
    const pos = nodeCoords[v.name];
    if (!pos) return;

    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "node-card");
    g.setAttribute("id", `node-${v.name}`);
    g.onclick = () => selectVertex(v.name);

    // State Color Palette
    let strokeColor = "#334155";
    let statusBg = "#1e293b";
    let stateColor = "#94a3b8";

    const st = (v.state || "").toLowerCase();
    if (st === "data ready") {
      strokeColor = "#10b981";
      statusBg = "#064e3b";
      stateColor = "#34d399";
    } else if (st === "todo") {
      strokeColor = "#3b82f6";
      statusBg = "#1e3a8a";
      stateColor = "#60a5fa";
    } else if (st === "todo urgent") {
      strokeColor = "#f59e0b";
      statusBg = "#78350f";
      stateColor = "#fbbf24";
    } else if (st === "reject") {
      strokeColor = "#ef4444";
      statusBg = "#7f1d1d";
      stateColor = "#f87171";
    }

    g.innerHTML = `
      <rect x="${pos.x}" y="${pos.y}" width="${pos.width}" height="${pos.height}" rx="8" fill="#111827" stroke="${strokeColor}" stroke-width="1.5" />
      <text x="${pos.x + 12}" y="${pos.y + 22}" fill="#f8fafc" font-size="12" font-weight="600">${esc(v.name)}</text>
      <rect x="${pos.x + 12}" y="${pos.y + 32}" width="80" height="18" rx="9" fill="${statusBg}" />
      <text x="${pos.x + 52}" y="${pos.y + 44}" fill="${stateColor}" font-size="10" font-weight="600" text-anchor="middle">${esc(v.state)}</text>
      <text x="${pos.x + pos.width - 12}" y="${pos.y + 44}" fill="#64748b" font-size="10" text-anchor="end">#${v.processed_count || 0}</text>
    `;
    gNodes.appendChild(g);
  });
  svg.appendChild(gNodes);
}

// ---------------------------------------------------------------------------
// Data Refresh & Inspection
// ---------------------------------------------------------------------------

async function refreshAll() {
  await Promise.all([
    refreshVertices(),
    refreshEdges(),
    refreshStaging(),
    refreshMetrics(),
  ]);
  renderDagGraph();
}

async function refreshVertices() {
  try {
    const res = await apiFetch(`/api/db/sessions/${currentSession}/vertices`);
    const vertices = await res.json();
    cachedVertices = {};
    const tbody = document.getElementById("verticesTableBody");
    if (tbody) tbody.innerHTML = "";

    vertices.forEach(v => {
      cachedVertices[v.name] = v;
      if (!tbody) return;
      const row = document.createElement("tr");
      row.style.cursor = "pointer";
      row.onclick = () => selectVertex(v.name);
      row.innerHTML = `
        <td><strong>${esc(v.name)}</strong></td>
        <td><span class="${getBadgeClass(v.state)}">${esc(v.state)}</span></td>
        <td>${v.processed_count}</td>
        <td><code>${esc((v.content || "").substring(0, 45))}</code></td>
        <td>
          <button class="btn-secondary" style="padding: 2px 8px; font-size: 0.75rem;" onclick="event.stopPropagation(); reenterVertex('${esc(v.name)}')">↺ Replay</button>
        </td>
      `;
      tbody.appendChild(row);
    });
  } catch (err) {
    console.error("Error refreshing vertices:", err);
  }
}

async function refreshEdges() {
  try {
    const res = await apiFetch(`/api/sessions/${currentSession}/graph`);
    const graph = await res.json();
    cachedGraph = graph;
    cachedEdges = graph.edges || {};

    const tbody = document.getElementById("edgesTableBody");
    if (tbody) tbody.innerHTML = "";

    const badge = document.getElementById("graphValidationBadge");
    if (badge) {
      if (graph.valid) {
        badge.className = "badge badge-data-ready";
        badge.textContent = "DAG Valid";
      } else {
        badge.className = "badge badge-reject";
        badge.textContent = "Cycle / Invalid";
      }
    }

    const tiers = graph.tiers || {};
    for (const [eId, e] of Object.entries(cachedEdges)) {
      if (!tbody) continue;
      const tier = tiers[eId] !== undefined ? tiers[eId] : "-";
      const row = document.createElement("tr");
      row.style.cursor = "pointer";
      row.onclick = () => selectEdge(eId);
      row.innerHTML = `
        <td><strong>${esc(eId)}</strong></td>
        <td><span class="badge badge-idle">${esc(e.type)}</span></td>
        <td><code>${esc(e.input_vertex)}</code> &rarr; <code>${esc(e.output_vertex)}</code></td>
        <td>Tier ${esc(String(tier))}</td>
      `;
      tbody.appendChild(row);
    }
  } catch (err) {
    console.error("Error refreshing edges:", err);
  }
}

async function refreshStaging() {
  try {
    const res = await apiFetch(`/api/db/sessions/${currentSession}/staging`);
    const staging = await res.json();
    const tbody = document.getElementById("stagingTableBody");
    if (!tbody) return;
    tbody.innerHTML = "";
    staging.forEach(s => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${esc(String(s.id))}</td>
        <td><code>${esc(s.edge_id)}</code></td>
        <td>${esc(s.vertex_name || "-")}</td>
        <td><strong>${esc(s.key)}</strong></td>
        <td><code>${esc((s.value || "").substring(0, 80))}</code></td>
        <td>${esc(s.created_at)}</td>
      `;
      tbody.appendChild(row);
    });
  } catch (err) {
    console.error("Error refreshing staging:", err);
  }
}

// ---------------------------------------------------------------------------
// Edge Metrics & Performance Benchmarks
// ---------------------------------------------------------------------------

async function refreshMetrics() {
  try {
    const res = await apiFetch(`/api/sessions/${currentSession}/metrics`);
    if (!res.ok) return;
    const summary = await res.json();

    // Update Top Stat Cards
    const elExec = document.getElementById("statTotalExec");
    const elLatency = document.getElementById("statAvgLatency");
    const elSuccess = document.getElementById("statSuccessRate");
    const elTokens = document.getElementById("statTotalTokens");

    if (elExec) elExec.textContent = summary.total_executions || 0;
    if (elLatency) elLatency.textContent = `${(summary.avg_execution_time_ms || 0).toFixed(1)} ms`;
    if (elSuccess) {
      const rate = ((1.0 - (summary.error_rate || 0)) * 100).toFixed(1);
      elSuccess.textContent = `${rate}%`;
    }
    if (elTokens) elTokens.textContent = (summary.total_tokens || 0).toLocaleString();

    // Render Metrics Table
    const tbody = document.getElementById("metricsTableBody");
    if (!tbody) return;
    tbody.innerHTML = "";

    const byEdge = summary.by_edge || {};
    const maxLat = Math.max(...Object.values(byEdge).map(e => e.avg_execution_time_ms || 0), 1);

    for (const [edgeId, info] of Object.entries(byEdge)) {
      const row = document.createElement("tr");
      const pct = Math.min(100, Math.round(((info.avg_execution_time_ms || 0) / maxLat) * 100));
      row.innerHTML = `
        <td><strong>${esc(edgeId)}</strong></td>
        <td><span class="badge badge-idle">${esc(info.edge_type)}</span></td>
        <td>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span>${(info.total_execution_time_ms || 0).toFixed(1)} ms</span>
            <div style="flex: 1; height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; min-width: 60px;">
              <div style="width: ${pct}%; height: 100%; background: var(--accent);"></div>
            </div>
          </div>
        </td>
        <td>${info.executions}</td>
        <td><span style="color: ${info.failed_executions > 0 ? 'var(--rose)' : 'var(--emerald)'}">${info.failed_executions}</span></td>
        <td>${(info.total_tokens || 0).toLocaleString()}</td>
        <td>$${(info.total_cost_usd || 0).toFixed(4)}</td>
      `;
      tbody.appendChild(row);
    }
  } catch (err) {
    console.error("Error refreshing metrics:", err);
  }
}

// ---------------------------------------------------------------------------
// Online Graph Mutation Handlers
// ---------------------------------------------------------------------------

// Set a <select> to `value`, appending an option when the value is not listed.
// Assigning an unknown value silently leaves the select empty, which used to
// rewrite an edge/vertex to a default type or state on save.
function setSelectValue(id, value, fallback) {
  const sel = document.getElementById(id);
  if (!sel) return;
  const v = value || fallback || "";
  if (!Array.from(sel.options).some((o) => o.value === v)) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  }
  sel.value = v;
}

function selectVertex(name) {
  const v = cachedVertices[name];
  if (!v) return;
  switchTab("editor");
  document.getElementById("vName").value = v.name || "";
  setSelectValue("vState", v.state, "todo");
  document.getElementById("vAttrs").value = (v.attributes || []).join(", ");
  document.getElementById("vContent").value = v.content || "";
  document.getElementById("vProcessed").value = v.processed_count || 0;
}

function clearVertexForm() {
  document.getElementById("vName").value = "";
  setSelectValue("vState", "todo", "todo");
  document.getElementById("vAttrs").value = "";
  document.getElementById("vContent").value = "";
  document.getElementById("vProcessed").value = "";
}

async function saveVertex() {
  const name = document.getElementById("vName").value.trim();
  const state = document.getElementById("vState").value;
  const content = document.getElementById("vContent").value;
  const rawAttrs = document.getElementById("vAttrs").value;
  const attributes = rawAttrs ? rawAttrs.split(",").map(a => a.trim()).filter(Boolean) : [];
  if (!name) return alert("Vertex name required");

  await apiFetch(`/api/sessions/${currentSession}/graph/vertices`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, state, content, attributes, processed_count: cachedVertices[name] ? cachedVertices[name].processed_count : 0 })
  });
  await refreshAll();
}

async function deleteSelectedVertex() {
  const name = document.getElementById("vName").value.trim();
  if (!name) return alert("Select a vertex first");
  if (!confirm(`Delete vertex '${name}'?`)) return;
  await apiFetch(`/api/sessions/${currentSession}/graph/vertices/${name}`, { method: "DELETE" });
  clearVertexForm();
  await refreshAll();
}

async function reenterVertex(name) {
  const targetName = name || document.getElementById("vName").value.trim();
  if (!targetName) return alert("Specify vertex name to replay");

  const newContent = prompt(`Reenter vertex '${targetName}'. Optionally modify input content:`, (cachedVertices[targetName] || {}).content || "");
  if (newContent === null) return;

  try {
    const res = await apiFetch(`/api/sessions/${currentSession}/graph/vertices/${targetName}/reenter`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: newContent })
    });
    const data = await res.json();
    alert(`Reentered '${targetName}'. Invalidated downstream nodes: ${(data.downstream_invalidated || []).join(", ") || "none"}`);
    await refreshAll();
  } catch (err) {
    alert("Reenter failed: " + err);
  }
}

function selectEdge(eId) {
  const e = cachedEdges[eId];
  if (!e) return;
  switchTab("editor");
  document.getElementById("eId").value = e.id || "";
  document.getElementById("eType").value = e.type || "code";
  document.getElementById("eIn").value = e.input_vertex || "";
  document.getElementById("eOut").value = e.output_vertex || "";
  document.getElementById("eScript").value = e.script || "";
  document.getElementById("eTriggerState").value = e.trigger_state || "";
  document.getElementById("eTargetState").value = e.target_state || "";
  document.getElementById("eMaxRetries").value = e.max_retries !== undefined ? e.max_retries : 3;
  document.getElementById("eSettings").value = JSON.stringify(e.settings || {}, null, 2);
}

function clearEdgeForm() {
  document.getElementById("eId").value = "";
  document.getElementById("eType").value = "code";
  document.getElementById("eIn").value = "";
  document.getElementById("eOut").value = "";
  document.getElementById("eScript").value = "";
  document.getElementById("eTriggerState").value = "";
  document.getElementById("eTargetState").value = "";
  document.getElementById("eMaxRetries").value = "3";
  document.getElementById("eSettings").value = "";
}

async function saveEdge() {
  const id = document.getElementById("eId").value.trim();
  const type = document.getElementById("eType").value;
  const inV = document.getElementById("eIn").value.trim();
  const outV = document.getElementById("eOut").value.trim();
  const script = document.getElementById("eScript").value.trim() || null;
  const trigger_state = document.getElementById("eTriggerState").value.trim() || null;
  const target_state = document.getElementById("eTargetState").value.trim() || null;
  const rawRetries = document.getElementById("eMaxRetries").value.trim();
  const max_retries = rawRetries !== "" ? parseInt(rawRetries) : 3;

  let settings = {};
  try {
    const rawSettings = document.getElementById("eSettings").value.trim();
    if (rawSettings) settings = JSON.parse(rawSettings);
  } catch (err) {
    return alert("Invalid JSON in settings: " + err);
  }

  if (!id || !inV || !outV) return alert("Edge ID, Input and Output vertices required");

  await apiFetch(`/api/sessions/${currentSession}/graph/edges`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, type, input_vertex: inV, output_vertex: outV, script, trigger_state, target_state, max_retries, settings })
  });
  await refreshAll();
}

async function deleteSelectedEdge() {
  const id = document.getElementById("eId").value.trim();
  if (!id) return alert("Select an edge first");
  if (!confirm(`Delete edge '${id}'?`)) return;
  await apiFetch(`/api/sessions/${currentSession}/graph/edges/${id}`, { method: "DELETE" });
  clearEdgeForm();
  await refreshAll();
}

async function runPipeline() {
  try {
    const res = await apiFetch(`/api/sessions/${currentSession}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_concurrency: 4 })
    });
    const data = await res.json();
    await refreshAll();
  } catch (err) {
    alert("Pipeline run failed: " + err);
  }
}

// ---------------------------------------------------------------------------
// OpenAI Chat Playground
// ---------------------------------------------------------------------------

async function loadModels() {
  try {
    const res = await apiFetch("/v1/models");
    const data = await res.json();
    const select = document.getElementById("chatModelSelect");
    if (!select || !data.data) return;
    select.innerHTML = "";
    data.data.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.id;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error("Failed loading models:", err);
  }
}

function renderChatHistory() {
  const container = document.getElementById("chatHistory");
  if (!container) return;
  container.innerHTML = "";

  chatMessages.forEach((msg, idx) => {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${msg.role}`;
    
    let contentHtml = esc(msg.content || "");
    
    // Render Tool Calls
    if (msg.tool_calls && msg.tool_calls.length > 0) {
      msg.tool_calls.forEach(tc => {
        contentHtml += `
          <div class="tool-call-card">
            <div class="tool-call-header">
              <span>🔧 Tool Call: <code>${esc(tc.function.name)}</code></span>
              <span style="font-size: 0.7rem; color: #94a3b8;">${esc(tc.id)}</span>
            </div>
            <pre style="color: #cbd5e1; white-space: pre-wrap; margin-bottom: 6px;">${esc(tc.function.arguments)}</pre>
            <button class="btn-success" style="font-size: 0.75rem; padding: 3px 8px;" onclick="simulateToolOutput('${esc(tc.id)}', '${esc(tc.function.name)}')">
              ▶ Send Simulated Sandbox Output
            </button>
          </div>
        `;
      });
    }

    bubble.innerHTML = contentHtml;
    container.appendChild(bubble);
  });

  scrollChatToBottom();
}

function scrollChatToBottom() {
  const container = document.getElementById("chatHistory");
  if (container) container.scrollTop = container.scrollHeight;
}

async function sendChatMessage() {
  const input = document.getElementById("chatInput");
  const text = input.value.trim();
  if (!text) return;

  const model = document.getElementById("chatModelSelect").value || "default";
  const stream = document.getElementById("chatStreamToggle").checked;

  chatMessages.push({ role: "user", content: text });
  input.value = "";
  renderChatHistory();

  try {
    const payload = {
      model: model,
      messages: chatMessages,
      stream: stream,
    };

    const res = await apiFetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (stream) {
      // SSE Stream reader
      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let assistantMsg = { role: "assistant", content: "" };
      chatMessages.push(assistantMsg);
      renderChatHistory();

      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ") && line.trim() !== "data: [DONE]") {
            try {
              const chunk = JSON.parse(line.substring(6));
              const delta = chunk.choices[0].delta || {};
              if (delta.content) {
                assistantMsg.content += delta.content;
                renderChatHistory();
              }
              if (delta.tool_calls) {
                assistantMsg.tool_calls = delta.tool_calls;
                renderChatHistory();
              }
            } catch (e) {}
          }
        }
      }
    } else {
      const data = await res.json();
      const msg = data.choices[0].message;
      chatMessages.push(msg);
      renderChatHistory();
    }

    await refreshAll();
  } catch (err) {
    chatMessages.push({ role: "system", content: `Error: ${err}` });
    renderChatHistory();
  }
}

async function simulateToolOutput(toolCallId, toolName) {
  const output = prompt(`Enter sandbox execution output for tool '${toolName}':`, "Exit code: 0\nOutput: Verified OK");
  if (output === null) return;

  chatMessages.push({
    role: "tool",
    tool_call_id: toolCallId,
    content: output,
  });
  renderChatHistory();

  // Pulse next harness step
  await sendChatMessageRaw();
}

async function sendChatMessageRaw() {
  const model = document.getElementById("chatModelSelect").value || "default";
  try {
    const res = await apiFetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages: chatMessages, stream: false })
    });
    const data = await res.json();
    const msg = data.choices[0].message;
    chatMessages.push(msg);
    renderChatHistory();
    await refreshAll();
  } catch (err) {
    chatMessages.push({ role: "system", content: `Error: ${err}` });
    renderChatHistory();
  }
}

// ---------------------------------------------------------------------------
// Real-Time Event Streaming
// ---------------------------------------------------------------------------

function connectEventStream() {
  if (eventSource) {
    eventSource.close();
  }
  const streamDiv = document.getElementById("eventStream");
  if (!streamDiv) return;
  streamDiv.innerHTML = "";

  eventSource = new EventSource(`/api/sessions/${currentSession}/events`);
  eventSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      const line = document.createElement("div");
      line.className = "event-line";
      line.innerHTML = `
        <span class="event-time">[${esc(new Date().toLocaleTimeString())}]</span>
        <span class="event-type">${esc(data.event_type)}</span>
        <span class="event-body">edge=${esc(data.edge_id || '-')} node=${esc(data.vertex_name || '-')}</span>
      `;
      streamDiv.prepend(line);

      // Animate active edge in DAG
      if (data.edge_id) {
        const p = document.getElementById(`edge-${data.edge_id}`);
        if (p) {
          p.classList.add("active");
          setTimeout(() => p.classList.remove("active"), 1200);
        }
      }

      refreshVertices();
      refreshStaging();
      refreshMetrics();
    } catch (err) {}
  };
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

// Populate the edge-type suggestion list from the server registry so custom
// edges (@register_edge_type) appear without editing the dashboard.
async function loadEdgeTypes() {
  const datalist = document.getElementById("edgeTypeOptions");
  if (!datalist) return;
  try {
    const res = await apiFetch("/api/edge-types");
    if (!res.ok) return;
    const data = await res.json();
    const types = Array.isArray(data.types) ? data.types : [];
    if (!types.length) return;
    datalist.innerHTML = types
      .map((t) => `<option value="${esc(t)}"></option>`)
      .join("");
    if (data.script_spec_supported) {
      datalist.innerHTML += `<option value="${esc(data.script_spec_example || "my_edge.py:MyEdge")}"></option>`;
    }
  } catch (err) {}
}

window.onload = async () => {
  await loadSessions();
  await loadModels();
  await loadEdgeTypes();
  await refreshAll();
  connectEventStream();

  // Chat Enter key handler
  const chatInput = document.getElementById("chatInput");
  if (chatInput) {
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });
  }
};
