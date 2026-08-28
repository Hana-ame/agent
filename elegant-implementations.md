# More Elegant Implementations

After reading every line of the framework, here are the structural changes that would make the codebase significantly more elegant — not just cleaner, but fundamentally better abstractions.

---

## 1. Replace Static Fan-Out with a `map` Operator

The biggest inelegance in the entire project. The `hn_ai_report` config manually defines 10 vertices and 10 edges to process 5 items in parallel:

```
v_sel1, v_sel2, v_sel3, v_sel4, v_sel5
v_c1, v_c2, v_c3, v_c4, v_c5
e_sel1..e_sel5, e_fetch_c1..e_fetch_c5, e_sum1..e_sum5
```

This is **14 vertices and 17 edges** to express what should be one concept: *"for each AI story, fetch comments and summarize."*

### The Elegant Way: A `MapEdge`

```python
class MapEdge(Edge):
    """Fan-out a list through a sub-pipeline, fan-in the results.
    
    Config:
        {
            "id": "e_process_stories",
            "source": "v_router",
            "destination": "v_report", 
            "type": "map",
            "settings": {
                "pipeline": [
                    {"type": "fetch", "script": "hn_edges.py:FetchCommentsEdge"},
                    {"type": "llm",   "script": "hn_edges.py:SummarizeEdge",
                     "prompt": "Summarize this discussion..."}
                ],
                "max_concurrency": 3
            }
        }
    """
    
    async def execute(self, source_vertex, dest_vertex, agents, **kw):
        items = source_vertex.fetch_data(self.channel)
        if not isinstance(items, list):
            items = [items]
        
        sem = asyncio.Semaphore(self.settings.get("max_concurrency", 5))
        
        async def process_one(item):
            async with sem:
                result = item
                for step in self.settings["pipeline"]:
                    edge = self._build_step(step)
                    result = await edge.compute(result, agents, step)
                return result
        
        results = await asyncio.gather(
            *[process_one(item) for item in items],
            return_exceptions=True
        )
        
        # Fan-in: deliver all results
        for r in results:
            if not isinstance(r, Exception):
                await dest_vertex.receive_signal(
                    self.id, EdgeSignal.COMPLETED, r, self.channel
                )
```

The entire `hn_ai_report/config.json` collapses from **120 lines to ~40 lines**:

```json
{
    "vertices": [
        {"id": "v_start", "initial_data": [{"channel": "default", "value": "trigger"}]},
        {"id": "v_stories"},
        {"id": "v_router"},
        {"id": "v_report", "script": "vertex/report_hook.py"}
    ],
    "edges": [
        {"id": "e_fetch",  "source": "v_start",   "destination": "v_stories",
         "type": "fetch",  "script": "hn_edges.py:FetchTopStoriesEdge"},
        {"id": "e_filter", "source": "v_stories",  "destination": "v_router",
         "type": "llm",    "script": "hn_edges.py:FilterEdge",
         "settings": {"prompt": "...", "model": "hy3-free"}},
        {"id": "e_process","source": "v_router",   "destination": "v_report",
         "type": "map",
         "settings": {
             "pipeline": [
                 {"type": "fetch", "script": "hn_edges.py:FetchCommentsEdge"},
                 {"type": "llm",   "script": "hn_edges.py:SummarizeEdge",
                  "prompt": "Summarize the following Hacker News discussion..."}
             ],
             "max_concurrency": 3
         }}
    ]
}
```

> [!IMPORTANT]
> This is the single highest-impact change. It transforms the framework from "you wire every parallel path manually" to "you declare what to do with each item." Every real-world pipeline (process N documents, call N APIs, summarize N threads) benefits from this.

---

## 2. Composable Edge Stages via Protocols (Not Inheritance)

Currently, every custom behavior requires a new `Edge` subclass. `hn_edges.py` has 5 classes for what are really just 5 functions. The edge class hierarchy forces you to think in terms of "what kind of edge is this" rather than "what transforms does this data need."

### Current: One Class Per Operation

```python
class FetchCommentsEdge(Edge):
    def condition(self, data, settings):
        return isinstance(data, dict) and "id" in data

    async def pre_process(self, data, settings):
        comments_md = await fetch_hn_comments_md(data["id"])
        return f"# {data.get('title')}\n\n{comments_md}"

class SummarizeEdge(Edge):
    def pre_process(self, data, settings):
        if isinstance(data, dict):
            return f"Title: {data.get('title')}\n{data.get('content')}"
        return str(data)
```

### Elegant: Compose Stages as Functions

```python
# Instead of subclassing, register functions directly:

@edge_hook("pre_process")
async def fetch_comments(data: dict, settings: dict) -> str:
    """Fetch HN comments for a story dict."""
    comments = await fetch_hn_comments_md(data["id"])
    return f"# {data.get('title')}\n\n{comments}"

@edge_hook("condition") 
def has_story_id(data, settings) -> bool:
    return isinstance(data, dict) and "id" in data

# In config: reference hooks by name, not by class
{
    "id": "e_fetch_comments",
    "hooks": {
        "condition": "has_story_id",
        "pre_process": "fetch_comments"
    }
}
```

The implementation is straightforward:

```python
# framework/hooks.py

_HOOK_REGISTRY: Dict[str, Callable] = {}

def edge_hook(stage: str):
    """Decorator to register a named hook function."""
    def decorator(fn):
        _HOOK_REGISTRY[f"{stage}:{fn.__name__}"] = fn
        return fn
    return decorator

class Edge:
    def _resolve_hook(self, stage: str) -> Optional[Callable]:
        """Look up hook from config, falling back to subclass method."""
        hook_name = self.settings.get("hooks", {}).get(stage)
        if hook_name:
            return _HOOK_REGISTRY.get(f"{stage}:{hook_name}")
        return None  # fall through to subclass override
```

> [!TIP]
> This doesn't replace subclassing — it supplements it. Simple hooks use functions, complex behaviors use classes. The framework supports both without forcing a choice.

---

## 3. Vertex State Machine as a Descriptor

The vertex state transitions are scattered across [vertex.py](file:///home/gekkasayu/vertex_edge_agent/framework/vertex.py) — `receive_signal()` does settlement, `_fire_vertex()` in the executor sets `DONE`, `reset()` handles loops, `approve()` handles HITL. The valid transitions are implicit.

### Current: Scattered State Logic

```python
# In vertex.py
self.state = VertexState.READY  # just assigned anywhere
# In executor/base.py  
vertex.state = VertexState.DONE  # executor reaches in
# In checkpoint.py
v._state = VertexState(state_str)  # bypasses property setter
```

### Elegant: Declarative State Machine

```python
class StateMachine:
    """Declarative state machine with validated transitions."""
    
    TRANSITIONS = {
        VertexState.IDLE:           {VertexState.READY, VertexState.AWAITING_EDGES, VertexState.ABORTED},
        VertexState.AWAITING_EDGES: {VertexState.READY, VertexState.ABORTED, VertexState.ERROR},
        VertexState.READY:          {VertexState.DONE, VertexState.PAUSED, VertexState.ERROR},
        VertexState.PAUSED:         {VertexState.READY, VertexState.ERROR},
        VertexState.DONE:           {VertexState.IDLE},  # loop reset
        VertexState.ABORTED:        {VertexState.IDLE},  # loop reset
        VertexState.ERROR:          set(),                # terminal
    }
    
    def __set_name__(self, owner, name):
        self._name = f"_{name}"
    
    def __get__(self, obj, objtype=None):
        return getattr(obj, self._name, VertexState.IDLE)
    
    def __set__(self, obj, new_state: VertexState):
        current = self.__get__(obj)
        if new_state not in self.TRANSITIONS.get(current, set()):
            raise InvalidTransition(
                f"Vertex[{obj.id}]: {current.value} → {new_state.value} is not allowed"
            )
        logger.debug("Vertex[%s] %s → %s", obj.id, current.value, new_state.value)
        setattr(obj, self._name, new_state)


class Vertex:
    state = StateMachine()  # descriptor handles all validation
```

This makes invalid states **impossible** rather than relying on every caller to know the rules. The checkpoint restore uses a dedicated `force_state()` method that bypasses validation for recovery.

---

## 4. Immutable Execution Context with Typed Slots

The context is a `dict` that every vertex mutates in place. There's no way to know what data is available at any point in the graph, and key collisions between vertices are silent.

### Current: Shared Mutable Dict

```python
# vertex.py
self._data_store: Dict[str, Any] = {}  # anything goes
v.set_data("default", some_value)      # hope the key doesn't collide
```

### Elegant: Typed Dataflow Declarations

```python
from dataclasses import dataclass
from typing import TypeVar, Generic

T = TypeVar("T")

@dataclass(frozen=True)
class Slot(Generic[T]):
    """Typed, named data slot that edges produce and consume."""
    name: str
    type: type
    description: str = ""

# Declare slots at the edge level:
STORY_LIST = Slot("story_list", list, "List of HN story dicts")
AI_STORIES = Slot("ai_stories", list, "Filtered AI-related stories")  
REPORT_MD  = Slot("report_md",  str,  "Final markdown report")

# Edge declares what it produces:
class FilterEdge(Edge):
    produces = AI_STORIES
    consumes = STORY_LIST
```

The graph validator can now check at **build time**:
- Does every edge's `consumes` slot get produced by an upstream edge?
- Are there type mismatches between connected slots?
- Are there orphan slots that nothing reads?

This is a compile-time dataflow analysis — far more powerful than the current runtime `KeyError`.

---

## 5. Edge as Data, Not Just as Object

Many edges in `hn_ai_report` are pure data transforms with no real "identity." `SelectEdge` just does `data[index]`. `SummarizeEdge` just formats a string. These don't need classes.

### Elegant: Functional Edge Transforms

```python
# Instead of a class:
class SelectEdge(Edge):
    def condition(self, data, settings):
        return isinstance(data, list) and settings.get("index", 0) < len(data)
    def pre_process(self, data, settings):
        return data[settings.get("index", 0)]

# Write a one-liner:
select = edge_transform(
    pre=lambda data, s: data[s.get("index", 0)],
    guard=lambda data, s: isinstance(data, list) and s.get("index", 0) < len(data),
)

# Or even more concisely in config:
{
    "id": "e_sel",
    "source": "v_router",
    "destination": "v_sel",
    "pre_process": "data[settings['index']]",
    "condition": "isinstance(data, list) and settings.get('index', 0) < len(data)"
}
```

The implementation behind `edge_transform`:

```python
def edge_transform(
    pre: Optional[Callable] = None,
    post: Optional[Callable] = None,
    guard: Optional[Callable] = None,
) -> type:
    """Create an Edge subclass from plain functions."""
    
    class FunctionalEdge(Edge):
        if guard:
            condition = staticmethod(guard)
        if pre:
            pre_process = staticmethod(pre)
        if post:
            post_process = staticmethod(post)
    
    return FunctionalEdge
```

---

## 6. Declarative Guard DSL

The current guard evaluation in [edge.py](file:///home/gekkasayu/vertex_edge_agent/framework/edge.py) is a manual operator dispatch table:

```python
# Current: 15 lines of if/elif for what's really a pattern match
ops = {
    ">=": field_val >= self._threshold, 
    ">": field_val > self._threshold,
    "<=": field_val <= self._threshold,
    "<": field_val < self._threshold,
    "==": field_val == self._threshold,
    "contains": self._threshold in field_val,
}
return ops.get(op, False)
```

### Elegant: Guard as a Mini-Language

```python
import operator

OPERATORS = {
    ">=": operator.ge, ">": operator.gt,
    "<=": operator.le, "<": operator.lt,
    "==": operator.eq, "!=": operator.ne,
    "in": lambda a, b: a in b,
    "contains": lambda a, b: b in a,
    "matches": lambda a, b: re.match(b, str(a)) is not None,
}

@dataclass(frozen=True)
class Guard:
    """Declarative, composable guard condition."""
    field: Optional[str] = None
    op: str = ">="
    value: Any = None
    match: Optional[dict] = None
    
    def evaluate(self, data: Any) -> bool:
        if self.match is not None:
            return isinstance(data, dict) and all(
                data.get(k) == v for k, v in self.match.items()
            )
        
        target = data
        if self.field and isinstance(data, dict):
            target = data.get(self.field)
        
        fn = OPERATORS.get(self.op)
        if fn is None:
            raise ValueError(f"Unknown guard operator: {self.op}")
        return fn(target, self.value)
    
    # Composition:
    def __and__(self, other: "Guard") -> "CompositeGuard":
        return CompositeGuard([self, other], mode="all")
    
    def __or__(self, other: "Guard") -> "CompositeGuard":
        return CompositeGuard([self, other], mode="any")


# Usage in config becomes natural:
{"guard": {"field": "score", "op": ">=", "value": 10}}
{"guard": [  # implicit AND
    {"field": "type", "op": "==", "value": "story"},
    {"field": "score", "op": ">=", "value": 10}
]}
```

This is both safer (no `eval()`) and more expressive (composable, serializable, testable).

---

## 7. Unified Resource Lifecycle

Currently, resources (`HttpLLMAgent`, `MemoryStore`, `TelemetryTracker`) are created and managed ad-hoc. The demo does `try/finally` for agent cleanup. The executor doesn't own any resource lifecycle.

### Elegant: `ExecutionContext` as an Async Context Manager

```python
@dataclass
class ExecutionContext:
    """Owns all resources needed for a graph run."""
    agents: BaseAgent
    memory: MemoryStore = field(default_factory=MemoryStore)
    telemetry: TelemetryTracker = field(default_factory=TelemetryTracker) 
    schema_registry: SchemaRegistry = field(default_factory=SchemaRegistry)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *exc):
        if hasattr(self.agents, 'close'):
            await self.agents.close()

# Usage becomes:
async with ExecutionContext(
    agents=HttpLLMAgent(api_key=key),
) as ctx:
    executor = Executor(graph, context=ctx)
    result = await executor.run()
    print(result.summary())
    print(ctx.telemetry.summary())
# Agent, memory, telemetry all cleaned up automatically
```

---

## Summary: Impact × Effort Matrix

| Change | Elegance Impact | Effort | Risk |
|--------|:-:|:-:|:-:|
| **1. `MapEdge` operator** | 🔥🔥🔥 | Medium | Low — additive, doesn't break existing |
| **2. Hook composition** | 🔥🔥 | Medium | Low — supplements, doesn't replace |
| **3. State machine descriptor** | 🔥🔥 | Small | Medium — touches core Vertex |
| **4. Typed slots** | 🔥🔥🔥 | Large | Medium — pervasive change |
| **5. Functional edge transforms** | 🔥 | Small | Low — sugar on top |
| **6. Guard DSL** | 🔥🔥 | Small | Low — replaces isolated function |
| **7. Resource lifecycle** | 🔥 | Small | Low — additive |

> [!TIP]
> **Start with #1 (MapEdge) and #6 (Guard DSL).** Together they transform the framework's expressiveness with minimal disruption. The `hn_ai_report` example becomes a compelling 40-line config instead of a 120-line manual wiring exercise, and guards become safe and composable instead of relying on `eval()`.
