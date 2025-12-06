# Architecture

## System Overview

```
SGD Dataset ──▶ OptimizationRunner ──▶ Saved Outputs
                     │                      │
                     │                      ├── agent.json (DSPy module)
                     │                      └── agent_result.json (metrics)
                     │
                     ├── Load data
                     ├── Build tool catalog
                     ├── Create baseline agent
                     ├── Run optimizer
                     └── Evaluate & save
```

---

## Components

```
src/
├── optimization/    OptimizationRunner, OptimizationResult
├── agent/           SGDAgentModule, build_tool_catalog
├── data/            SGDLoader, SGDDialogue, SGDSchema
├── environment/     SGDEnvironment, Observation
├── reward/          LLMJudge, ground truth comparison
└── tool_mocker/     SGDToolMocker
```

---

## Key Classes

### OptimizationRunner
```python
runner = OptimizationRunner(data_path, strategy, num_train, callback, output_path)
result = runner.run()  # Returns OptimizationResult
```

### SGDAgentModule
```python
agent = SGDAgentModule(tool_catalog)
result = agent(user_message, conversation_history)
# result.response, result.tool_call, result.reasoning
```

### SGDEnvironment
```python
env = create_sgd_environment(data_path, reward_mode, limit)
obs = env.reset(dialogue_idx)
obs, reward, done, info = env.step(action)
```

---

## Reward Modes

| Mode | Description |
|------|-------------|
| `ground_truth` | Compare to expected response from dataset |
| `llm_judge` | Gemini evaluates response quality |
| `hybrid` | 50/50 blend |

---

## Output Files

**agent.json** (DSPy format)
```json
{"respond.predict": {"demos": [...], "signature": {...}}}
```

**agent_result.json**
```json
{
  "baseline_score": 0.53,
  "optimized_score": 0.55,
  "improvement_pct": 3.8,
  "initial_prompt": "...",
  "optimized_prompt": "...",
  "few_shot_demos": [...],
  "events": [...]
}
```
