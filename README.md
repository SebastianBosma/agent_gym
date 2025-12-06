# Agent Gym

Train and optimize task-oriented dialogue agents using conversation traces as training data.

## Overview

Agent Gym transforms conversation datasets (like Schema-Guided Dialogue) into RL environments for:

- **Training** agents to respond naturally and use tools appropriately
- **Optimizing** prompts with DSPy (BootstrapFewShot, MIPROv2)
- **Evaluating** agents with ground truth or LLM-as-judge

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set your Gemini API key
export GOOGLE_API_KEY=your-key-here

# Run optimization
python -c "
from src.optimization import OptimizationRunner

runner = OptimizationRunner(
    data_path='data/raw/schema_guided_dialogue',
    strategy='bootstrap',  # or 'mipro'
    num_train=50,
    output_path='checkpoints/my_agent.json',
)
result = runner.run()
print(f'Improvement: {result.improvement_pct:+.1f}%')
"
```

## Features

| Feature | Description |
|---------|-------------|
| **SGD Dataset** | Pre-loaded Schema-Guided Dialogue with 45 services |
| **Dynamic Tool Catalog** | Auto-extracted from dataset schemas |
| **BootstrapFewShot** | Fast optimization via example selection |
| **MIPROv2** | Gemini-powered prompt rewriting |
| **LLM-as-Judge** | Reward without ground truth dependency |
| **Result Saving** | Full metrics, prompts, and event logs |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     OptimizationRunner                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Load SGD data (dialogues + schemas)                         │
│  2. Build tool catalog from schemas                             │
│  3. Create baseline agent with prompt                           │
│  4. Run optimizer (BootstrapFewShot or MIPROv2)                 │
│  5. Evaluate and save results                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Saved Outputs                                │
├─────────────────────────────────────────────────────────────────┤
│  checkpoints/agent.json        - DSPy module (demos + prompt)   │
│  checkpoints/agent_result.json - Metrics, prompts, events       │
└─────────────────────────────────────────────────────────────────┘
```

## Usage

```python
from src.optimization import OptimizationRunner, OptimizationResult

# Run optimization
runner = OptimizationRunner(
    data_path="data/raw/schema_guided_dialogue",
    strategy="mipro",
    num_train=50,
    output_path="checkpoints/agent.json",
)
result = runner.run()

# Load saved results
result = OptimizationResult.load("checkpoints/agent_result.json")
print(result.initial_prompt)
print(result.few_shot_demos)
```

## Scoring Metric

Optimization uses a combined metric:

| Component | Weight | Description |
|-----------|--------|-------------|
| Response Similarity | 50% | Word overlap (Jaccard) with ground truth |
| Tool Accuracy | 50% | Correct tool name + arguments |

```python
final_score = (response_similarity + tool_accuracy) / 2
```

## Project Structure

```
agent_gym/
├── src/
│   ├── agent/           # SGDAgentModule, tool catalog builder
│   ├── data/            # SGDLoader, dialogue/schema parsing
│   ├── environment/     # SGDEnvironment, reward modes
│   ├── optimization/    # OptimizationRunner, result saving
│   ├── reward/          # LLMJudge, ground truth comparison
│   └── tool_mocker/     # SGDToolMocker for API simulation
├── data/
│   └── raw/schema_guided_dialogue/  # SGD dataset
├── checkpoints/         # Saved agents and results
├── docs/                # Documentation
└── scripts/             # Utility scripts
```

## Roadmap

### Phase 1: Prompt Optimization ✅
- SGD dataset integration
- BootstrapFewShot + MIPROv2
- LLM-as-judge reward
- Result saving/loading

### Phase 2: Policy Distillation (Next)
```
RL Environment              Policy Training              Inference
┌─────────────────┐        ┌─────────────────┐         ┌──────────┐
│ Gemini agent    │  PPO/  │  Small LM       │ deploy  │ Fast,    │
│ + tool mocker   │───────▶│  (fine-tuned)   │────────▶│ cheap    │
│ + reward_fn     │ DPO    │                 │         │ model    │
└─────────────────┘        └─────────────────┘         └──────────┘
```

Use the RL environment to distill a smaller, deployable model:
- Collect on-policy rollouts from Gemini agent
- Train via PPO/DPO on trajectories
- Deploy fast, domain-specific model

## Model

All components use `gemini-3-pro-preview` by default.

## Documentation

- [Frontend Integration](docs/frontend_integration.md) - API for building UIs
- [Optimization Guide](docs/optimization.md) - DSPy optimization details
- [Architecture](docs/architecture.md) - System design
- [Trace Schema](docs/trace_schema.md) - Data format

## Requirements

- Python 3.10+
- `google-generativeai` - Gemini SDK
- `dspy-ai>=2.5` - Prompt optimization
- `litellm` - Model routing

## License

MIT
