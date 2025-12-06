# DSPy Optimization

## Overview

Both `QueryAgent` and `ToolMocker` are built on DSPy modules, making them optimizable with traces as training data.

## Quick Start

```python
from dspy.teleprompt import BootstrapFewShot
from src import create_environment

# Create environment
query_agent, tool_mocker, reward_fn = create_environment(traces)

# Get the underlying DSPy module and training examples
agent_module = query_agent.__self__.get_module()
training_examples = query_agent.__self__.get_training_examples()

# Define metric using reward function
def metric(example, pred):
    return reward_fn(pred.response, example.response)

# Optimize
optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized_module = optimizer.compile(agent_module, trainset=training_examples)

# Use optimized module
query_agent.__self__.set_module(optimized_module)
```

## Available Optimizers

| Optimizer | Use Case |
|-----------|----------|
| `BootstrapFewShot` | Quick, finds good few-shot examples |
| `BootstrapFewShotWithRandomSearch` | More thorough search |
| `MIPRO` | Full prompt optimization |

## Training Data

Both components provide `get_training_examples()`:

```python
# Agent examples: (user_message, history) -> response
agent_examples = query_agent.__self__.get_training_examples()

# Tool examples: (tool_name, args) -> response
tool_examples = tool_mocker.__self__.get_training_examples()
```

## Saving/Loading

```python
# Save optimized module
optimized_module.save("checkpoints/agent_v1.json")

# Load later
agent_module.load("checkpoints/agent_v1.json")
```

