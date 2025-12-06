# Frontend Integration Guide

## OptimizationRunner

```python
from src.optimization import OptimizationRunner, OptimizationResult

runner = OptimizationRunner(
    data_path="data/raw/schema_guided_dialogue",
    strategy="mipro",        # or "bootstrap"
    num_train=50,
    num_eval=10,
    num_candidates=5,        # MIPRO prompt variants
    model="gemini-2.5-flash",
    callback=on_event,       # Progress callback
    output_path="checkpoints/my_agent.json",
)

result = runner.run()
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategy` | "bootstrap" | "bootstrap" or "mipro" |
| `num_train` | 50 | Training dialogues |
| `num_eval` | 10 | Evaluation dialogues |
| `num_candidates` | 5 | MIPRO prompt variants |
| `callback` | None | `Callable[[OptimizationEvent], None]` |
| `output_path` | None | Auto-saves agent + result |

---

## Event System

```python
def on_event(event: OptimizationEvent):
    print(f"[{event.type.value}] {event.data}")
```

| Event Type | Data Fields |
|------------|-------------|
| `log` | `message` |
| `start` | `total_steps`, `strategy` |
| `progress` | `step`, `total`, `metric` |
| `evaluation` | `step`, `total`, `current_score` |
| `complete` | `baseline_score`, `optimized_score`, `improvement_pct` |
| `error` | `message` |

---

## OptimizationResult

```python
@dataclass
class OptimizationResult:
    success: bool
    baseline_score: float
    optimized_score: float
    improvement_pct: float
    training_time: float
    
    initial_prompt: str
    optimized_prompt: str
    few_shot_demos: List[dict]
    events: List[OptimizationEvent]
    
    optimized_agent: SGDAgentModule  # In-memory only
```

### Saving/Loading

```python
# Auto-saved when output_path set:
# - checkpoints/my_agent.json        (DSPy module)
# - checkpoints/my_agent_result.json (metrics + prompts + events)

# Manual load
result = OptimizationResult.load("checkpoints/my_agent_result.json")
```

---

## Data Loading

```python
from src.data import SGDLoader

loader = SGDLoader("data/raw/schema_guided_dialogue")
dialogues = loader.load_dialogues("train", limit=100)
schemas = loader.load_schemas("train")
stats = loader.get_stats("train")
```

---

## Agent

```python
from src.agent import SGDAgentModule

# Load optimized agent
agent = SGDAgentModule(tool_catalog)
agent.load("checkpoints/my_agent.json")

result = agent(user_message="...", conversation_history="...")
# result.response, result.tool_call, result.reasoning
```

---

## Environment

```python
from src.environment import create_sgd_environment

env = create_sgd_environment(
    data_path="...",
    reward_mode="llm_judge",  # or "ground_truth"
    limit=50,
)

obs = env.reset(dialogue_idx=0)
obs, reward, done, info = env.step(action)
```

---

## REST API Example

```python
@app.post("/api/optimize")
async def start(strategy: str, num_train: int, bg: BackgroundTasks):
    job_id = uuid.uuid4()
    bg.add_task(run_optimization, job_id, strategy, num_train)
    return {"job_id": job_id}

@app.get("/api/optimize/{job_id}/status")
async def status(job_id: str):
    return {"status": jobs[job_id]["status"]}
```

---

## WebSocket Example

```python
@app.websocket("/ws/optimize")
async def ws_optimize(ws: WebSocket):
    await ws.accept()
    config = await ws.receive_json()
    
    def callback(event):
        asyncio.create_task(ws.send_json(event.to_dict()))
    
    runner = OptimizationRunner(..., callback=callback)
    result = runner.run()
    await ws.send_json({"type": "result", "data": result.to_dict()})
```

```javascript
// Client
ws.onmessage = (e) => {
  const data = JSON.parse(e.data);
  if (data.type === 'progress') {
    setProgress(data.step / data.total * 100);
  }
};
```
