# Trace Schema

## Required Format

```json
{
  "conversation_id": "string",
  "messages": [Message],
  "outcome": "resolved" | "escalated" | "abandoned" | null,
  "satisfaction_score": float | null,
  "metadata": {}
}
```

## Message Types

### User Message
```json
{"role": "user", "content": "I need help with..."}
```

### Assistant Message
```json
{
  "role": "assistant",
  "content": "Response text",
  "tool_calls": [{"name": "tool_name", "args": {...}}]  // optional
}
```

### Tool Response
```json
{
  "role": "tool",
  "name": "tool_name",
  "content": "",
  "result": {...}
}
```

## Validation

Traces are validated via Pydantic on ingestion. Invalid traces raise `ValueError` with the index and error details.

## Converting Raw Data

Common patterns:

```python
# From CSV logs
import csv
traces = [{"conversation_id": row["id"], "messages": json.loads(row["messages"])} for row in csv_reader]

# From JSONL
traces = [json.loads(line) for line in open("logs.jsonl")]
```

