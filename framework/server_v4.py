"""V4 Standalone Server and Database Viewer.

Provides:
1. Online Graph Mutation API: Modify vertices, edges, and topology per session in real-time.
2. Isolated Session Graphs: Each session maintains an independent GraphV4 instance with live event streaming.
3. Standalone Database Viewer & Dashboard: Inspect SQLite vertices and session_staging scratchpad tables.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from framework.edge_v4 import CodeEdgeV4, EdgeResultV4, EdgeV4, LLMEdgeV4, ReflexiveEdgeV4
from framework.executor_v4 import ExecutionResultV4, ExecutorV4, GraphEventV4
from framework.graph_v4 import GraphTopologyError, GraphV4
from framework.vertex_v4 import (
    StagingRecordV4,
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)

logger = logging.getLogger("vertex_edge_agent.server_v4")


# ---------------------------------------------------------------------------
# Pydantic Schemas for Online Graph API
# ---------------------------------------------------------------------------

class VertexCreateOrUpdateRequest(BaseModel):
    name: str = Field(..., description="Unique vertex name in session")
    content: str = Field(default="", description="Payload content")
    attributes: List[str] = Field(default_factory=list, description="Vertex attributes")
    state: str = Field(default=VertexStateV4.IDLE.value, description="Lifecycle state")
    processed_count: int = Field(default=0, description="Processing counter")


class EdgeCreateOrUpdateRequest(BaseModel):
    id: str = Field(..., description="Unique edge identifier")
    type: str = Field(default="code", description="Edge type: code, llm, or reflexive")
    input_vertex: str = Field(..., description="Input vertex name")
    output_vertex: str = Field(..., description="Output vertex name")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Edge settings")
    script: Optional[str] = Field(default=None, description="Script path or callable specification")
    trigger_state: Optional[str] = Field(default=None, description="Trigger state for reflexive edge")
    target_state: Optional[str] = Field(default=None, description="Target reset state for reflexive edge")
    max_retries: Optional[int] = Field(default=None, description="Max retries for reflexive edge")


class WorkflowRunRequest(BaseModel):
    max_concurrency: int = Field(default=4, ge=1, le=64, description="Max concurrent edges")
    timeout: float = Field(default=120.0, gt=0.0, description="Execution timeout in seconds")


# ---------------------------------------------------------------------------
# Session Graph Manager
# ---------------------------------------------------------------------------

class SessionGraphManagerV4:
    """Maintains isolated GraphV4 instances per session synchronized with VertexStoreV4."""

    def __init__(self, store: VertexStoreV4):
        self.store = store
        self._graphs: Dict[str, GraphV4] = {}
        self._event_broadcasters: Dict[str, List[asyncio.Queue]] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}

    def get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Fetch or create an asyncio.Lock for the session."""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    def get_or_create_graph(self, session_id: str) -> GraphV4:
        """Fetch or instantiate an isolated GraphV4 for a session, hydrating from store if available."""
        if session_id not in self._graphs:
            # Attempt to hydrate from database if vertices or edges exist
            hydrated = GraphV4.load_from_store(self.store, session_id, name=f"graph_{session_id}")
            if hydrated.vertices or hydrated.edges:
                self._graphs[session_id] = hydrated
            else:
                self._graphs[session_id] = GraphV4(session_id=session_id, name=f"graph_{session_id}")
        return self._graphs[session_id]

    def list_active_sessions(self) -> List[str]:
        """List distinct session IDs from in-memory graphs and database."""
        session_set: Set[str] = set(self._graphs.keys())
        session_set.update(self.store.list_sessions())
        return sorted(list(session_set))

    def add_or_update_vertex(
        self,
        session_id: str,
        name: str,
        content: str = "",
        attributes: Optional[List[str]] = None,
        state: str = VertexStateV4.IDLE.value,
        processed_count: int = 0,
    ) -> VertexRecordV4:
        """Online vertex addition or update. Syncs to SQLite and updates graph."""
        graph = self.get_or_create_graph(session_id)
        # Persist to database
        db_record = self.store.save_vertex(
            session_id=session_id,
            name=name,
            content=content,
            attributes=attributes,
            state=state,
            processed_count=processed_count,
        )
        graph.add_vertex(db_record)
        return db_record

    def delete_vertex(self, session_id: str, name: str) -> bool:
        """Delete vertex from graph and database, cleaning up connected edges in memory and SQLite."""
        graph = self.get_or_create_graph(session_id)
        # Find incident edges to delete from SQLite as well
        incident_edges = [
            eid for eid, e in graph.edges.items()
            if e.input_vertex == name or e.output_vertex == name
        ]
        for eid in incident_edges:
            self.store.delete_edge(session_id, eid)

        graph.delete_vertex(name)
        try:
            graph.compute_dag_tiers()
        except Exception:
            pass
        return self.store.delete_vertex(session_id, name)

    def add_or_update_edge(
        self,
        session_id: str,
        edge_id: str,
        edge_type: str,
        input_vertex: str,
        output_vertex: str,
        settings: Optional[Dict[str, Any]] = None,
        script: Optional[str] = None,
        trigger_state: Optional[str] = None,
        target_state: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> EdgeV4:
        """Online edge addition or update. Persists to SQLite and recalculates DAG tiers."""
        graph = self.get_or_create_graph(session_id)
        edge_settings = dict(settings or {})

        edge: EdgeV4
        if edge_type == "code":
            edge = CodeEdgeV4(
                edge_id=edge_id,
                input_vertex=input_vertex,
                output_vertex=output_vertex,
                script=script,
                settings=edge_settings,
            )
        elif edge_type == "llm":
            edge = LLMEdgeV4(
                edge_id=edge_id,
                input_vertex=input_vertex,
                output_vertex=output_vertex,
                model=edge_settings.get("model", "sensenova-6.8-flash-lite"),
                prompt_template=edge_settings.get("prompt"),
                settings=edge_settings,
            )
        elif edge_type == "reflexive" or input_vertex == output_vertex:
            edge = ReflexiveEdgeV4(
                edge_id=edge_id,
                vertex_name=input_vertex,
                trigger_state=trigger_state or VertexStateV4.REJECT.value,
                target_state=target_state or VertexStateV4.TODO_URGENT.value,
                max_retries=max_retries if max_retries is not None else int(edge_settings.get("max_retries", 3)),
                script=script,
                settings=edge_settings,
            )
        else:
            raise ValueError(f"Unsupported edge type: {edge_type}")

        graph.add_edge(edge)
        # Persist edge to SQLite database
        self.store.save_edge(
            session_id=session_id,
            edge_id=edge_id,
            edge_type=edge.type,
            input_vertex=input_vertex,
            output_vertex=output_vertex,
            script=script,
            trigger_state=getattr(edge, "trigger_state", None),
            target_state=getattr(edge, "target_state", None),
            max_retries=getattr(edge, "max_retries", 3),
            settings=edge_settings,
        )

        # Validate and recalculate DAG tiers if all endpoints are registered
        if input_vertex in graph.vertices and output_vertex in graph.vertices:
            try:
                graph.validate()
            except Exception as exc:
                logger.warning("[SessionGraphManagerV4] Graph validation note: %s", exc)

        return edge

    def delete_edge(self, session_id: str, edge_id: str) -> bool:
        """Delete edge from session graph and SQLite database."""
        graph = self.get_or_create_graph(session_id)
        deleted = graph.delete_edge(edge_id)
        self.store.delete_edge(session_id, edge_id)
        if deleted:
            try:
                graph.compute_dag_tiers()
            except Exception:
                pass
        return deleted

    def validate_graph(self, session_id: str) -> Dict[str, Any]:
        """Validate DAG constraints and return edge tiers."""
        graph = self.get_or_create_graph(session_id)
        try:
            graph.validate()
            return {
                "valid": True,
                "tiers": graph.edge_tiers,
                "vertex_count": len(graph.vertices),
                "edge_count": len(graph.edges),
            }
        except GraphTopologyError as err:
            return {
                "valid": False,
                "error": str(err),
                "vertex_count": len(graph.vertices),
                "edge_count": len(graph.edges),
            }

    def register_event_queue(self, session_id: str, q: asyncio.Queue) -> None:
        """Register an SSE broadcast subscriber queue for a session."""
        if session_id not in self._event_broadcasters:
            self._event_broadcasters[session_id] = []
        self._event_broadcasters[session_id].append(q)

    def unregister_event_queue(self, session_id: str, q: asyncio.Queue) -> None:
        """Remove an SSE broadcast subscriber queue and clean up empty session registry."""
        if session_id in self._event_broadcasters:
            try:
                self._event_broadcasters[session_id].remove(q)
            except ValueError:
                pass
            if not self._event_broadcasters[session_id]:
                self._event_broadcasters.pop(session_id, None)

    def broadcast_event(self, session_id: str, event: GraphEventV4) -> None:
        """Broadcast an execution event to all active SSE subscribers with backpressure protection."""
        queues = list(self._event_broadcasters.get(session_id, []))
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# HTML Dashboard Source
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VEA v4 Graph & Database Live Dashboard</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --text: #f8fafc;
      --muted: #94a3b8;
      --green: #22c55e;
      --blue: #3b82f6;
      --yellow: #eab308;
      --red: #ef4444;
      --purple: #a855f7;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding: 24px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }
    h1 { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
    .session-bar {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    select, input, button {
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.875rem;
    }
    button {
      background: var(--accent);
      color: #0f172a;
      font-weight: 600;
      cursor: pointer;
      border: none;
      transition: background 0.15s;
    }
    button:hover { background: var(--accent-hover); }
    .btn-danger { background: var(--red); color: white; }
    .btn-secondary { background: var(--border); color: var(--text); }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }
    .full-width { grid-column: span 2; }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }
    .card h2 {
      font-size: 1.1rem;
      margin-bottom: 12px;
      color: var(--accent);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
    }
    .badge-data-ready { background: #14532d; color: #4ade80; }
    .badge-todo { background: #1e3a8a; color: #60a5fa; }
    .badge-todo-urgent { background: #713f12; color: #facc15; }
    .badge-reject { background: #7f1d1d; color: #f87171; }
    .badge-idle { background: #334155; color: #cbd5e1; }
    .badge-forbidden { background: #451a03; color: #fb923c; }
    .badge-pruning { background: #3b0764; color: #c084fc; }

    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 0.85rem;
    }
    th, td {
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
    }
    th { color: var(--muted); font-weight: 600; }
    .live-events {
      height: 180px;
      overflow-y: auto;
      background: #090d16;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      font-family: monospace;
      font-size: 0.8rem;
    }
    .event-line { margin-bottom: 4px; border-bottom: 1px dashed #1e293b; padding-bottom: 2px; }
    .form-row { display: flex; gap: 8px; margin-bottom: 8px; }
    .form-row input, .form-row select { flex: 1; }
  </style>
</head>
<body>
  <header>
    <h1>VEA v4 Dynamic Graph & Database Viewer</h1>
    <div class="session-bar">
      <label for="sessionSelect">Session:</label>
      <select id="sessionSelect" onchange="onSessionChange()"></select>
      <input type="text" id="newSessionInput" placeholder="New session id" style="width: 140px;">
      <button onclick="createSession()">Create</button>
      <button onclick="runWorkflow()">Run Workflow</button>
      <button class="btn-secondary" onclick="refreshAll()">Refresh</button>
    </div>
  </header>

  <div class="grid">
    <!-- Live Graph & Vertices -->
    <div class="card">
      <h2>
        <span>Live Vertices (Click row to inspect/edit)</span>
        <span id="graphValidationBadge" class="badge badge-idle">Validating...</span>
      </h2>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>State</th>
            <th>Processed</th>
            <th>Content Preview</th>
          </tr>
        </thead>
        <tbody id="verticesTableBody"></tbody>
      </table>
    </div>

    <!-- Active Edges -->
    <div class="card">
      <h2>Session Edges (Click row to inspect/edit)</h2>
      <table>
        <thead>
          <tr>
            <th>Edge ID</th>
            <th>Type</th>
            <th>Connection</th>
            <th>Tier</th>
          </tr>
        </thead>
        <tbody id="edgesTableBody"></tbody>
      </table>
    </div>

    <!-- Selected Vertex Inspector / Editor -->
    <div class="card">
      <h2>Vertex Inspector & Online Editor</h2>
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div class="form-row">
          <input type="text" id="vName" placeholder="Vertex Name (e.g. v_start)">
          <select id="vState">
            <option value="data ready">data ready</option>
            <option value="todo" selected>todo</option>
            <option value="todo urgent">todo urgent</option>
            <option value="idle">idle</option>
            <option value="reject">reject</option>
            <option value="pruning">pruning</option>
            <option value="forbidden">forbidden</option>
          </select>
          <input type="text" id="vAttrs" placeholder="Attributes (e.g. start, json, subgraph)">
        </div>
        <div>
          <textarea id="vContent" rows="4" style="width: 100%; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 8px; font-family: monospace; font-size: 0.85rem;" placeholder="Vertex Content / Subgraph config"></textarea>
        </div>
        <div style="display: flex; gap: 8px;">
          <button onclick="saveVertex()">Save Vertex</button>
          <button class="btn-danger" onclick="deleteSelectedVertex()">Delete</button>
          <button class="btn-secondary" onclick="clearVertexForm()">Clear</button>
        </div>
      </div>
    </div>

    <!-- Selected Edge Inspector / Editor -->
    <div class="card">
      <h2>Edge Inspector & Online Editor</h2>
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div class="form-row">
          <input type="text" id="eId" placeholder="Edge ID (e.g. e_transform)">
          <select id="eType">
            <option value="code" selected>code</option>
            <option value="llm">llm</option>
            <option value="reflexive">reflexive</option>
          </select>
          <input type="text" id="eIn" placeholder="Input Node">
          <input type="text" id="eOut" placeholder="Output Node">
        </div>
        <div class="form-row">
          <input type="text" id="eScript" placeholder="Script path or callable (e.g. scripts/run.py:execute)">
        </div>
        <div class="form-row">
          <input type="text" id="eTriggerState" placeholder="Trigger State (e.g. reject)">
          <input type="text" id="eTargetState" placeholder="Target State (e.g. todo urgent)">
          <input type="number" id="eMaxRetries" placeholder="Max Retries" value="3">
        </div>
        <div>
          <textarea id="eSettings" rows="3" style="width: 100%; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 8px; font-family: monospace; font-size: 0.85rem;" placeholder="Edge Settings (JSON)"></textarea>
        </div>
        <div style="display: flex; gap: 8px;">
          <button onclick="saveEdge()">Save Edge</button>
          <button class="btn-danger" onclick="deleteSelectedEdge()">Delete</button>
          <button class="btn-secondary" onclick="clearEdgeForm()">Clear</button>
        </div>
      </div>
    </div>

    <!-- Live Event Stream -->
    <div class="card full-width">
      <h2>Real-Time Event Stream (SSE)</h2>
      <div id="eventStream" class="live-events"></div>
    </div>

    <!-- Staging Scratchpad Table -->
    <div class="card full-width">
      <h2>Session Staging Scratchpad Table (`session_staging`)</h2>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Edge Attribution</th>
            <th>Vertex</th>
            <th>Key</th>
            <th>Value</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody id="stagingTableBody"></tbody>
      </table>
    </div>
  </div>

  <script>
    let currentSession = "default_session";
    let eventSource = null;
    let cachedVertices = {};
    let cachedEdges = {};

    function esc(s) { const d = document.createElement('div'); d.textContent = String(s); return d.innerHTML; }

    function getBadgeClass(state) {
      const s = (state || "").toLowerCase().replace(" ", "-");
      return "badge badge-" + s;
    }

    async function loadSessions() {
      const res = await fetch("/api/db/sessions");
      const sessions = await res.json();
      const select = document.getElementById("sessionSelect");
      select.innerHTML = "";
      if (sessions.length === 0) {
        sessions.push(currentSession);
      }
      sessions.forEach(s => {
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = s;
        select.appendChild(opt);
      });
      select.value = sessions.includes(currentSession) ? currentSession : sessions[0];
      currentSession = select.value;
    }

    async function onSessionChange() {
      currentSession = document.getElementById("sessionSelect").value;
      refreshAll();
      connectEventStream();
    }

    function createSession() {
      const newSess = document.getElementById("newSessionInput").value.trim();
      if (!newSess) return;
      currentSession = newSess;
      document.getElementById("newSessionInput").value = "";
      const select = document.getElementById("sessionSelect");
      const opt = document.createElement("option");
      opt.value = newSess;
      opt.textContent = newSess;
      select.appendChild(opt);
      select.value = newSess;
      refreshAll();
      connectEventStream();
    }

    async function refreshVertices() {
      const res = await fetch(`/api/db/sessions/${currentSession}/vertices`);
      const vertices = await res.json();
      cachedVertices = {};
      const tbody = document.getElementById("verticesTableBody");
      tbody.innerHTML = "";
      vertices.forEach(v => {
        cachedVertices[v.name] = v;
        const row = document.createElement("tr");
        row.style.cursor = "pointer";
        row.onclick = () => selectVertex(v.name);
        row.innerHTML = `
          <td><strong>${esc(v.name)}</strong></td>
          <td><span class="${getBadgeClass(v.state)}">${esc(v.state)}</span></td>
          <td>${v.processed_count}</td>
          <td><code>${esc((v.content || "").substring(0, 45))}</code></td>
        `;
        tbody.appendChild(row);
      });
    }

    function selectVertex(name) {
      const v = cachedVertices[name];
      if (!v) return;
      document.getElementById("vName").value = v.name || "";
      document.getElementById("vState").value = v.state || "todo";
      document.getElementById("vAttrs").value = (v.attributes || []).join(", ");
      document.getElementById("vContent").value = v.content || "";
    }

    function clearVertexForm() {
      document.getElementById("vName").value = "";
      document.getElementById("vState").value = "todo";
      document.getElementById("vAttrs").value = "";
      document.getElementById("vContent").value = "";
    }

    async function refreshEdges() {
      const res = await fetch(`/api/sessions/${currentSession}/graph`);
      const graph = await res.json();
      cachedEdges = graph.edges || {};
      const tbody = document.getElementById("edgesTableBody");
      tbody.innerHTML = "";
      const validationBadge = document.getElementById("graphValidationBadge");
      if (graph.valid) {
        validationBadge.className = "badge badge-data-ready";
        validationBadge.textContent = "DAG Valid";
      } else {
        validationBadge.className = "badge badge-reject";
        validationBadge.textContent = "Cycle / Invalid";
      }

      const tiers = graph.tiers || {};
      for (const [eId, e] of Object.entries(cachedEdges)) {
        const tier = tiers[eId] !== undefined ? tiers[eId] : "-";
        const row = document.createElement("tr");
        row.style.cursor = "pointer";
        row.onclick = () => selectEdge(eId);
        row.innerHTML = `
          <td><strong>${esc(eId)}</strong></td>
          <td>${esc(e.type)}</td>
          <td>${esc(e.input_vertex)} &rarr; ${esc(e.output_vertex)}</td>
          <td><strong>Tier ${esc(String(tier))}</strong></td>
        `;
        tbody.appendChild(row);
      }
    }

    function selectEdge(eId) {
      const e = cachedEdges[eId];
      if (!e) return;
      document.getElementById("eId").value = e.id || "";
      document.getElementById("eType").value = e.type || "code";
      document.getElementById("eIn").value = e.input_vertex || "";
      document.getElementById("eOut").value = e.output_vertex || "";
      document.getElementById("eScript").value = e.script || "";
      document.getElementById("eTriggerState").value = e.trigger_state || "";
      document.getElementById("eTargetState").value = e.target_state || "";
      document.getElementById("eMaxRetries").value = e.max_retries !== undefined && e.max_retries !== null ? e.max_retries : 3;
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

    async function refreshStaging() {
      const res = await fetch(`/api/db/sessions/${currentSession}/staging`);
      const staging = await res.json();
      const tbody = document.getElementById("stagingTableBody");
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
    }

    async function saveVertex() {
      const name = document.getElementById("vName").value.trim();
      const state = document.getElementById("vState").value;
      const content = document.getElementById("vContent").value;
      const rawAttrs = document.getElementById("vAttrs").value;
      const attributes = rawAttrs ? rawAttrs.split(",").map(a => a.trim()).filter(Boolean) : [];
      if (!name) return alert("Vertex name required");

      await fetch(`/api/sessions/${currentSession}/graph/vertices`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, state, content, attributes, processed_count: cachedVertices[name] ? cachedVertices[name].processed_count : 0 })
      });
      refreshAll();
    }

    async function deleteSelectedVertex() {
      const name = document.getElementById("vName").value.trim();
      if (!name) return alert("Select a vertex first");
      if (!confirm(`Delete vertex '${name}'?`)) return;
      await fetch(`/api/sessions/${currentSession}/graph/vertices/${name}`, { method: "DELETE" });
      clearVertexForm();
      refreshAll();
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

      if (!id || !inV || !outV) return alert("Edge ID, Input and Output required");

      await fetch(`/api/sessions/${currentSession}/graph/edges`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id,
          type,
          input_vertex: inV,
          output_vertex: outV,
          script,
          trigger_state,
          target_state,
          max_retries,
          settings
        })
      });
      refreshAll();
    }

    async function deleteSelectedEdge() {
      const id = document.getElementById("eId").value.trim();
      if (!id) return alert("Select an edge first");
      if (!confirm(`Delete edge '${id}'?`)) return;
      await fetch(`/api/sessions/${currentSession}/graph/edges/${id}`, { method: "DELETE" });
      clearEdgeForm();
      refreshAll();
    }

    async function runWorkflow() {
      const res = await fetch(`/api/sessions/${currentSession}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ max_concurrency: 4 })
      });
      const data = await res.json();
      refreshAll();
    }

    function connectEventStream() {
      if (eventSource) {
        eventSource.close();
      }
      const streamDiv = document.getElementById("eventStream");
      streamDiv.innerHTML = "";
      eventSource = new EventSource(`/api/sessions/${currentSession}/events`);
      eventSource.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          const line = document.createElement("div");
          line.className = "event-line";
          line.innerHTML = `[${esc(new Date().toLocaleTimeString())}] <strong>${esc(data.event_type)}</strong> edge=${esc(data.edge_id || '-')} node=${esc(data.vertex_name || '-')}`;
          streamDiv.prepend(line);
          refreshVertices();
          refreshStaging();
        } catch (err) {}
      };
    }

    function refreshAll() {
      refreshVertices();
      refreshEdges();
      refreshStaging();
    }

    window.onload = async () => {
      await loadSessions();
      refreshAll();
      connectEventStream();
    };
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# FastAPI Application Factory
# ---------------------------------------------------------------------------

def create_v4_server(
    store_or_db: Union[VertexStoreV4, str, Path] = ":memory:",
    manager: Optional[SessionGraphManagerV4] = None,
    agent: Optional[Any] = None,
) -> FastAPI:
    """Create a FastAPI application powering online graph APIs and database dashboard."""
    if isinstance(store_or_db, VertexStoreV4):
        store = store_or_db
    else:
        store = VertexStoreV4(str(store_or_db))

    if manager is None:
        manager = SessionGraphManagerV4(store=store)
    else:
        store = manager.store

    app = FastAPI(
        title="VEA v4 Online Graph & Database Server",
        version="4.0.0",
        description="Online graph modification API and standalone database inspector.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach shared instances to app state
    app.state.store = store
    app.state.manager = manager
    app.state.agent = agent

    # -----------------------------------------------------------------------
    # Dashboard Endpoint
    # -----------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    async def get_dashboard() -> HTMLResponse:
        """Serve live single-page dashboard."""
        return HTMLResponse(content=DASHBOARD_HTML)

    # -----------------------------------------------------------------------
    # Database Inspection Endpoints
    # -----------------------------------------------------------------------

    @app.get("/api/db/stats")
    async def get_db_stats() -> Dict[str, Any]:
        """Return overall database statistics across all sessions."""
        return store.get_db_stats()

    @app.get("/api/db/sessions")
    async def list_sessions() -> List[str]:
        """List distinct session IDs."""
        return manager.list_active_sessions()

    @app.get("/api/db/sessions/{session_id}/vertices")
    async def get_db_vertices(
        session_id: str,
        state: Optional[str] = Query(None, description="Optional state filter"),
    ) -> List[Dict[str, Any]]:
        """Fetch raw vertices for a session directly from SQLite."""
        records = store.list_vertices(session_id=session_id, state=state)
        return [r.to_dict() for r in records]

    @app.get("/api/db/sessions/{session_id}/staging")
    async def get_db_staging(
        session_id: str,
        key: Optional[str] = Query(None),
        edge_id: Optional[str] = Query(None),
        vertex_name: Optional[str] = Query(None),
    ) -> List[Dict[str, Any]]:
        """Query raw records from session_staging table."""
        records = store.get_staged(
            session_id=session_id,
            key=key,
            edge_id=edge_id,
            vertex_name=vertex_name,
        )
        return [r.to_dict() for r in records]

    @app.post("/api/db/sessions/{session_id}/clear")
    async def clear_session_db(session_id: str) -> Dict[str, str]:
        """Purge all records for a session from SQLite and in-memory state."""
        store.clear_session(session_id)
        manager._graphs.pop(session_id, None)
        manager._event_broadcasters.pop(session_id, None)
        return {"status": "cleared", "session_id": session_id}

    # -----------------------------------------------------------------------
    # Online Graph Mutation API
    # -----------------------------------------------------------------------

    @app.get("/api/sessions/{session_id}/graph")
    async def get_session_graph(session_id: str) -> Dict[str, Any]:
        """Retrieve the current graph structure, endpoints, and DAG validation."""
        graph = manager.get_or_create_graph(session_id)
        validation = manager.validate_graph(session_id)
        edges_data = {}
        for eid, e in graph.edges.items():
            edges_data[eid] = {
                "id": e.id,
                "type": e.type,
                "input_vertex": e.input_vertex,
                "output_vertex": e.output_vertex,
                "is_reflexive": e.is_reflexive,
                "settings": e.settings,
                "script": getattr(e, 'script', None),
                "trigger_state": getattr(e, 'trigger_state', None),
                "target_state": getattr(e, 'target_state', None),
                "max_retries": getattr(e, 'max_retries', None),
            }

        # Query database store to always reflect latest real-time states and contents
        db_vertices = {v.name: v for v in store.list_vertices(session_id)}
        vertices_data = {}
        for vname, v in graph.vertices.items():
            if vname in db_vertices:
                vertices_data[vname] = db_vertices[vname].to_dict()
            else:
                vertices_data[vname] = v.to_dict()
        for vname, db_v in db_vertices.items():
            if vname not in vertices_data:
                vertices_data[vname] = db_v.to_dict()

        return {
            "session_id": session_id,
            "valid": validation.get("valid", False),
            "tiers": validation.get("tiers", {}),
            "error": validation.get("error"),
            "vertices": vertices_data,
            "edges": edges_data,
        }

    @app.post("/api/sessions/{session_id}/graph/vertices")
    async def create_or_update_vertex(
        session_id: str,
        req: VertexCreateOrUpdateRequest,
    ) -> Dict[str, Any]:
        """Add or update a vertex online."""
        v = manager.add_or_update_vertex(
            session_id=session_id,
            name=req.name,
            content=req.content,
            attributes=req.attributes,
            state=req.state,
            processed_count=req.processed_count,
        )
        return {"status": "saved", "vertex": v.to_dict()}

    @app.delete("/api/sessions/{session_id}/graph/vertices/{name}")
    async def delete_vertex(session_id: str, name: str) -> Dict[str, Any]:
        """Delete a vertex online."""
        deleted = manager.delete_vertex(session_id, name)
        return {"deleted": deleted, "name": name}

    @app.post("/api/sessions/{session_id}/graph/edges")
    async def create_or_update_edge(
        session_id: str,
        req: EdgeCreateOrUpdateRequest,
    ) -> Dict[str, Any]:
        """Add or update an edge online."""
        edge = manager.add_or_update_edge(
            session_id=session_id,
            edge_id=req.id,
            edge_type=req.type,
            input_vertex=req.input_vertex,
            output_vertex=req.output_vertex,
            settings=req.settings,
            script=req.script,
            trigger_state=req.trigger_state,
            target_state=req.target_state,
            max_retries=req.max_retries,
        )
        val = manager.validate_graph(session_id)
        return {
            "status": "saved",
            "edge": {
                "id": edge.id,
                "type": edge.type,
                "input": edge.input_vertex,
                "output": edge.output_vertex,
            },
            "validation": val,
        }

    @app.delete("/api/sessions/{session_id}/graph/edges/{edge_id}")
    async def delete_edge(session_id: str, edge_id: str) -> Dict[str, Any]:
        """Delete an edge online."""
        deleted = manager.delete_edge(session_id, edge_id)
        return {"deleted": deleted, "edge_id": edge_id}

    @app.post("/api/sessions/{session_id}/graph/validate")
    async def validate_graph(session_id: str) -> Dict[str, Any]:
        """Validate DAG topology and tiers for session graph."""
        return manager.validate_graph(session_id)

    # -----------------------------------------------------------------------
    # Dynamic Workflow Execution & Live SSE Events
    # -----------------------------------------------------------------------

    @app.post("/api/sessions/{session_id}/run")
    async def run_session_workflow(
        session_id: str,
        req: WorkflowRunRequest = Body(default_factory=WorkflowRunRequest),
    ) -> Dict[str, Any]:
        """Execute session graph with specified concurrency limit and broadcast events."""
        async with manager.get_session_lock(session_id):
            graph = manager.get_or_create_graph(session_id)
            executor = ExecutorV4(
                graph=graph,
                store=store,
                agent=app.state.agent,
                max_concurrency=req.max_concurrency,
                timeout=req.timeout,
            )

            # Broadcast events in real-time
            async def stream_and_broadcast():
                async for ev in executor.stream():
                    manager.broadcast_event(session_id, ev)

            await stream_and_broadcast()
            return executor._result.to_dict()

    @app.get("/api/sessions/{session_id}/events")
    async def stream_session_events(session_id: str) -> StreamingResponse:
        """SSE stream broadcasting execution events in real time."""
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        manager.register_event_queue(session_id, q)

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                while True:
                    ev: GraphEventV4 = await q.get()
                    data = {
                        "event_type": ev.event_type,
                        "edge_id": ev.edge_id,
                        "vertex_name": ev.vertex_name,
                        "payload": ev.payload,
                        "timestamp": ev.timestamp,
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                manager.unregister_event_queue(session_id, q)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # -----------------------------------------------------------------------
    # SSE Executor with Session Routing & Harness Tool Call Echo
    # -----------------------------------------------------------------------

    @app.post("/api/sse/execute")
    async def execute_via_sse(
        payload: Dict[str, Any] = Body(default_factory=dict),
    ) -> Any:
        """Execute workflow via SSEExecutor with session routing and harness echo output."""
        from framework.sse_executor_v4 import SSEExecutorV4

        session_id = payload.get("session_id")
        input_payload = payload.get("input_payload")
        manifest_path = payload.get("manifest_path")
        max_concurrency = int(payload.get("max_concurrency", 4))
        timeout = float(payload.get("timeout", 120.0))
        is_stream = bool(payload.get("stream", True))

        sse_exec = SSEExecutorV4(manager=manager, store=store)

        if is_stream:
            gen = sse_exec.execute_and_stream(
                session_id=session_id,
                input_payload=input_payload,
                manifest_path=manifest_path,
                max_concurrency=max_concurrency,
                timeout=timeout,
            )
            return StreamingResponse(
                gen,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            return await sse_exec.execute_harness_call(
                session_id=session_id,
                input_payload=input_payload,
                manifest_path=manifest_path,
                max_concurrency=max_concurrency,
                timeout=timeout,
            )

    return app


# ---------------------------------------------------------------------------
# CLI Entrypoint for Standalone Server
# ---------------------------------------------------------------------------

def parse_server_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VEA v4 Online Graph & Database Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--db", default=":memory:", help="SQLite database path")
    return parser.parse_args()


def main() -> None:
    """Run server directly from CLI."""
    import uvicorn
    args = parse_server_args()
    app = create_v4_server(store_or_db=args.db)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
