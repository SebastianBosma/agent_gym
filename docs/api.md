# API Reference

## Main Entry Point

### `create_environment(traces)`

```python
from src import create_environment

query_agent, tool_mocker, reward_fn = create_environment(traces)
```

**Args:**
- `traces`: `List[Dict]` - Conversation traces in normalized format

**Returns:**
- `query_agent`: `Callable[[str, Optional[Dict]], str]`
- `tool_mocker`: `Callable[[str, Dict], Any]`
- `reward_fn`: `Callable[[str, Optional[str], Optional[Dict]], float]`

---

## Components

### QueryAgent.query()
```python
response = query_agent(user_message, context=None)
```

### ToolMocker.mock()
```python
result = tool_mocker(tool_name, tool_args)
```

### RewardFunction.score()
```python
score = reward_fn(agent_response, ground_truth=None, context=None)
```

---

## Environment (Optional)

For RL training loops:

```python
from src.environment import TraceEnvironment

env = TraceEnvironment(parsed_traces)
obs = env.reset()
obs, reward, done, info = env.step(action)
```

