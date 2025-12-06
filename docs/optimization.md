# Prompt Optimization

## Quick Start

```python
from src.optimization import OptimizationRunner

runner = OptimizationRunner(
    data_path="data/raw/schema_guided_dialogue",
    strategy="mipro",  # or "bootstrap"
    num_train=50,
    output_path="checkpoints/my_agent.json",
)
result = runner.run()

print(f"Improvement: {result.improvement_pct:+.1f}%")
```

---

## Strategies

| Strategy | Speed | What it does |
|----------|-------|--------------|
| `bootstrap` | ~10s | Finds good few-shot examples |
| `mipro` | ~3min | Tests prompt rewrites + few-shot |

MIPROv2 may keep original instructions if they perform best—improvement often comes from better few-shot selection.

---

## Scoring Metric

```python
score = (response_similarity + tool_accuracy) / 2
```

| Component | How |
|-----------|-----|
| Response similarity | Jaccard word overlap |
| Tool accuracy | 1.0 if name+args match, 0.5 if name only, 0.0 otherwise |

---

## Results

```python
result.success           # bool
result.baseline_score    # Before optimization
result.optimized_score   # After optimization
result.improvement_pct   # Percentage change
result.initial_prompt    # Starting prompt
result.optimized_prompt  # Final prompt
result.few_shot_demos    # Added examples
result.events            # Event log
```

### Saving/Loading

```python
# Auto-saves when output_path set
result.save("checkpoints/result.json")
result = OptimizationResult.load("checkpoints/result.json")
```

---

## Tips

- Use `num_train=50+` for reliable results
- Use `num_candidates=5-10` for MIPRO
- Check `result.initial_prompt != result.optimized_prompt` to see if instructions changed
- Review `result.few_shot_demos` to understand what examples were selected
