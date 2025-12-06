# Agent Gym

Convert conversation traces into RL environments for training AI agents.

## Overview

Agent Gym takes historical conversation logs (e.g., customer service transcripts) and transforms them into a reinforcement learning environment. This enables training agents that can:

- Respond to user queries naturally
- Use tools appropriately based on context
- Optimize for customer satisfaction and task completion

## User Journey

```
1. COLLECT        2. CONVERT         3. CREATE ENVIRONMENT        4. DEPLOY
   Raw Logs  ───▶  JSON Traces  ───▶  (parse + optimize)    ───▶  Agent
   (CSV, DB)       (normalized)       query_agent                  (production)
                                      tool_mocker
                                      reward_fn
```

| Step | What You Do | Output |
|------|-------------|--------|
| 1. Collect | Export customer service logs | Raw data files |
| 2. Convert | Normalize to trace schema | `data/processed/*.json` |
| 3. Create | `create_environment(traces)` | Optimized query_agent, tool_mocker, reward_fn |
| 4. Deploy | Use the agent | Production-ready responses |

### Phase 2: Policy Distillation (Coming Soon)

```
RL ENVIRONMENT                    POLICY TRAINING                 INFERENCE
┌─────────────────┐              ┌─────────────────┐             ┌──────────┐
│ query_agent     │   on-policy  │  Small LM       │   deploy    │ Fast,    │
│ tool_mocker     │─────────────▶│  (fine-tuned)   │────────────▶│ cheap    │
│ reward_fn       │   rollouts   │                 │             │ model    │
└─────────────────┘              └─────────────────┘             └──────────┘
```

Use the RL environment to train a smaller, domain-specific model:

- **Environment**: Gemini-powered simulation provides realistic interactions
- **Policy**: Train via PPO/DPO on collected trajectories  
- **Reward Model**: `reward_fn` scores completions for RLHF
- **Distillation**: Transfer capabilities from large LM to small, deployable model

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key
export GOOGLE_API_KEY="your-api-key-here"

# Run the demo
python examples/customer_service.py
```

## Main API

```python
from src import create_environment

# Load your traces (list of conversation dicts)
traces = [...]

# Create the RL environment
query_agent, tool_mocker, reward_fn = create_environment(traces)

# Use the components
response = query_agent("I want to cancel my subscription")
tool_result = tool_mocker("lookup_account", {"user_id": "123"})
score = reward_fn(response, ground_truth="Expected response...")
```

## Trace Format

Traces should be normalized to this JSON schema:

```json
{
  "conversation_id": "conv_001",
  "messages": [
    {"role": "user", "content": "User message here"},
    {"role": "assistant", "content": "Agent response", "tool_calls": [...]},
    {"role": "tool", "name": "tool_name", "result": {...}}
  ],
  "outcome": "resolved",
  "satisfaction_score": 4.5
}
```

## Project Structure

```
agent_gym/
├── src/
│   ├── __init__.py          # create_environment() factory
│   ├── trace_parser/         # Convert traces to unified format
│   ├── environment/          # RL environment wrapper
│   ├── tool_mocker/          # Mock tool responses from data
│   ├── reward/               # Reward function
│   └── agent/                # Gemini-powered query agent
├── data/
│   ├── raw/                  # Raw trace files
│   └── processed/            # Normalized JSON traces
├── examples/
│   └── customer_service.py   # Demo script
└── scripts/
    └── run_example.py        # Quick-run utility
```

## Components

| Component | Purpose |
|-----------|---------|
| `trace_parser` | Parse and validate traces into consistent schema |
| `tool_mocker` | Return realistic tool responses based on trace patterns |
| `reward` | Score agent responses (semantic similarity + quality) |
| `agent` | Gemini-powered agent that learns from trace context |

## Dependencies

- `google-generativeai` - Gemini SDK for LLM capabilities
- `dspy-ai` - Prompt optimization framework
- `pydantic` - Data validation and schemas
- `numpy` - Numerical utilities
- `rich` - Pretty console output

## License

MIT
