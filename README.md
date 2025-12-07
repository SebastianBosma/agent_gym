# Agent Gym

Create RL environments from dialogue traces to train customer service agents.

## Overview

Agent Gym transforms conversation datasets (like Schema-Guided Dialogue) into RL environments with:

- **Simulated User** - DSPy-optimized agent that generates realistic user messages
- **Tool Mocker** - Mocks API calls using recorded responses
- **Reward Function** - LLM-as-judge (Gemini) that scores agent responses

Use this environment to train a small model (e.g., Gemma via GRPO) to act as a customer service agent.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set your Gemini API key
export GOOGLE_API_KEY=your-key-here
```

### Create an Environment

```python
from agent_gym import create_environment

# Create environment components
simulated_user, tool_mocker, reward_fn = create_environment(
    "data/raw/schema_guided_dialogue"
)

# Get a user goal and generate initial message
goal = simulated_user.get_random_goal()
user_message = simulated_user.generate_initial_message(goal)
# "I'm looking for a restaurant in Oakland"

# Your agent responds...
agent_response = "What type of cuisine would you like?"

# Get next user message (dynamic, based on what agent said)
next_message = simulated_user.generate_response(
    goal=goal,
    history=f"USER: {user_message}\nASSISTANT: {agent_response}",
    assistant_message=agent_response,
)
# "Italian please"

# Score the agent's response
score = reward_fn.evaluate(
    history=[],
    user_message=user_message,
    response=agent_response,
    available_tools=[...],
)
print(score.quality_score)  # 0.75
```

### Optimize the Simulated User

```bash
# Train simulated user to generate realistic messages
python scripts/optimize_simulated_user.py \
    --strategy bootstrap \
    --num-train 100 \
    --output checkpoints/simulated_user.json
```

Then use the optimized checkpoint:

```python
simulated_user, tool_mocker, reward_fn = create_environment(
    "data/raw/schema_guided_dialogue",
    simulated_user_checkpoint="checkpoints/simulated_user.json",
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      RL ENVIRONMENT                             │
│                                                                 │
│   SimulatedUserAgent ──► generates user messages (DSPy)         │
│          │                                                      │
│          ▼                                                      │
│   Your Agent (Gemma) ──► generates responses + tool calls       │
│          │                                                      │
│          ▼                                                      │
│   ToolMocker ──► returns mocked API results                     │
│          │                                                      │
│          ▼                                                      │
│   LLMJudge ──► scores response (reward signal)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    GRPO updates agent weights
```

## Features

| Feature | Description |
|---------|-------------|
| **Simulated User** | DSPy-optimized to generate realistic user messages |
| **Dynamic Conversations** | Not fixed traces - evolves based on agent responses |
| **Tool Mocker** | Mocks 45+ SGD services with realistic responses |
| **LLM-as-Judge** | Gemini evaluates response quality (no ground truth needed) |
| **Multi-Turn** | Supports up to 10 turns per episode |
| **Checkpoint Saving** | Save/load optimized simulated user modules |

## Project Structure

```
agent_gym/
├── src/
│   ├── agent/
│   │   ├── simulated_user.py   # SimulatedUserAgent, UserGoal
│   │   └── sgd_agent.py        # SGDAgentModule (prompt optimization)
│   ├── data/
│   │   └── sgd_loader.py       # SGDLoader, dialogue parsing
│   ├── environment/
│   │   ├── multi_turn_env.py   # MultiTurnEnvironment
│   │   └── sgd_env.py          # SGDEnvironment (trace replay)
│   ├── reward/
│   │   └── llm_judge.py        # LLMJudge, CachedLLMJudge
│   └── tool_mocker/
│       └── sgd_mocker.py       # SGDToolMocker
├── scripts/
│   ├── optimize_simulated_user.py  # DSPy optimization for user sim
│   └── optimize_agent.py           # DSPy optimization for agent
├── checkpoints/                # Saved models and results
├── data/raw/schema_guided_dialogue/  # SGD dataset
└── docs/                       # Documentation
```

## Using MultiTurnEnvironment

For a more standard RL interface:

```python
from agent_gym import create_multi_turn_environment, AgentAction

env = create_multi_turn_environment(
    "data/raw/schema_guided_dialogue",
    simulated_user_checkpoint="checkpoints/simulated_user.json",
    max_turns=10,
)

obs = env.reset()
done = False

while not done:
    # Your agent produces an action
    action = AgentAction(
        type="response",
        response="What city are you looking for?",
        tool_call=None,  # or {"name": "FindRestaurants", "args": {...}}
    )
    
    obs, reward, done, info = env.step(action)
    print(f"User: {obs.user_message}")
    print(f"Reward: {reward}")
```

## Training Pipeline

### Phase 1: Optimize Simulated User

```bash
python scripts/optimize_simulated_user.py --strategy bootstrap
```

This teaches the simulated user to generate messages that match real users in the SGD dataset.

### Phase 2: Train Agent with GRPO

Use the optimized environment to train your agent (Gemma) via GRPO:

```bash
# See src/post-training/ for GRPO training scripts
cd src/post-training
uv run python scripts/train_grpo.py \
    --model_name google/gemma-2-2b-it \
    --use_lora
```

## Documentation

- [Architecture](docs/architecture.md) - System design and components
- [Post-Training](docs/post_training.md) - GRPO fine-tuning guide
- [Optimization](docs/optimization.md) - DSPy optimization details

## Requirements

- Python 3.10+
- `google-generativeai` - Gemini SDK
- `dspy-ai>=2.5` - Prompt optimization
- `litellm` - Model routing

## License

MIT
