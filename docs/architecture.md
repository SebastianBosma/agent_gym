# Architecture

## System Overview

Agent Gym creates RL environments from dialogue traces for training customer service agents.

```
SGD Traces ──► create_environment() ──► RL Environment Components
                                              │
                                              ├── SimulatedUserAgent (DSPy-optimized)
                                              ├── ToolMocker (API simulation)
                                              └── LLMJudge (reward function)
                                              
                                                    │
                                                    ▼
                                              GRPO/PPO trains
                                              small model (Gemma)
```

---

## Main API

```python
from agent_gym import create_environment

# Create environment components
simulated_user, tool_mocker, reward_fn = create_environment(
    "data/raw/schema_guided_dialogue",
    simulated_user_checkpoint="checkpoints/simulated_user.json",  # optional
)
```

---

## Components

```
src/
├── agent/
│   ├── simulated_user.py    SimulatedUserAgent, UserGoal, DSPy optimization
│   └── sgd_agent.py         SGDAgentModule (for prompt optimization)
├── data/
│   └── sgd_loader.py        SGDLoader, SGDDialogue, SGDSchema
├── environment/
│   ├── multi_turn_env.py    MultiTurnEnvironment (dynamic conversations)
│   └── sgd_env.py           SGDEnvironment (trace replay - legacy)
├── reward/
│   └── llm_judge.py         LLMJudge, CachedLLMJudge (Gemini-as-judge)
├── tool_mocker/
│   └── sgd_mocker.py        SGDToolMocker (API response simulation)
└── optimization/
    └── runner.py            OptimizationRunner (DSPy optimization)
```

---

## Key Classes

### SimulatedUserAgent

Generates realistic user messages based on goals extracted from SGD dialogues.

```python
from agent_gym import SimulatedUserAgent

agent = SimulatedUserAgent("data/raw/schema_guided_dialogue")

# Get a user goal
goal = agent.get_random_goal()
# UserGoal(intents=['FindRestaurant'], slot_values={'city': 'Oakland', ...})

# Generate initial message
message = agent.generate_initial_message(goal)
# "I'm looking for a restaurant"

# Generate response to assistant
response = agent.generate_response(
    goal=goal,
    history="USER: I'm looking for a restaurant\nASSISTANT: What city?",
    assistant_message="What city are you looking for?",
)
# "Oakland please"
```

### SGDToolMocker

Mocks API calls using recorded responses from SGD dialogues.

```python
from src.tool_mocker import SGDToolMocker

mocker = SGDToolMocker.from_loader(loader, "train")

# Mock a service call
results = mocker.call("FindRestaurants", {"city": "Oakland", "cuisine": "Italian"})
# [{"restaurant_name": "Olive Garden", "address": "123 Main St", ...}]
```

### LLMJudge (Reward Function)

Evaluates agent responses using Gemini.

```python
from agent_gym import CachedLLMJudge

judge = CachedLLMJudge()

result = judge.evaluate(
    history=[{"role": "user", "content": "Find me a restaurant"}],
    user_message="I want Italian food",
    response="What city are you looking for?",
    available_tools=[{"name": "FindRestaurants", ...}],
    tool_call=None,
)

print(result.quality_score)  # 0.75
print(result.tool_correct)   # None (no tool was called)
```

### MultiTurnEnvironment

Full RL environment with standard `reset()`/`step()` interface.

```python
from agent_gym import create_multi_turn_environment

env = create_multi_turn_environment(
    "data/raw/schema_guided_dialogue",
    simulated_user_checkpoint="checkpoints/simulated_user.json",
    max_turns=10,
)

obs = env.reset()
while not done:
    action = agent.act(obs)  # Your agent being trained
    obs, reward, done, info = env.step(action)
```

---

## Data Flow

### Training the Simulated User (DSPy Optimization)

```
SGD Traces ──► extract_user_training_examples() ──► DSPy Examples
                                                         │
                                                         ▼
                                                  BootstrapFewShot
                                                         │
                                                         ▼
                                          checkpoints/simulated_user.json
```

### Training the Customer Service Agent (GRPO)

```
┌─────────────────────────────────────────────────────────────────┐
│                      RL ENVIRONMENT                             │
│                                                                 │
│   SimulatedUserAgent ──► generates user messages                │
│          │                                                      │
│          ▼                                                      │
│   Agent (Gemma) ──► generates responses + tool calls            │
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
                    GRPO updates Gemma weights
```

---

## Saved Outputs

### simulated_user.json (DSPy format)

```json
{
  "respond.predict": {
    "demos": [
      {
        "user_goal": "Intent: FindRestaurant | Info: city=Oakland",
        "assistant_response": "What type of cuisine?",
        "user_message": "Italian please"
      }
    ],
    "signature": {...}
  }
}
```

### Optimization metrics

```json
{
  "baseline_score": 0.53,
  "optimized_score": 0.68,
  "improvement_pct": 28.3,
  "strategy": "bootstrap",
  "timestamp": "2025-01-15T10:30:00"
}
```

---

## Reward Components

| Component | Range | Description |
|-----------|-------|-------------|
| `response_quality` | 0-1 | LLM judge rating of helpfulness |
| `tool_accuracy` | 0-1 | Correct tool + argument accuracy |

Total reward per step: `response_quality + tool_accuracy` (0-2 range)
