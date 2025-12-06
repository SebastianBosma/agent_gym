# Architecture

## System Overview

```
┌─────────────┐     ┌──────────────────────────────────────────┐
│   Traces    │────▶│         create_environment()             │
│   (JSON)    │     └──────────────────────────────────────────┘
└─────────────┘                        │
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │              Returns 3 callables          │
                    ├──────────────┬─────────────┬─────────────┤
                    │ query_agent  │ tool_mocker │  reward_fn  │
                    └──────────────┴─────────────┴─────────────┘
```

## Data Flow

1. **Ingestion**: Raw traces → `trace_parser` → normalized `TraceSchema`
2. **Environment Creation**: Parsed traces initialize all three components
3. **Runtime**: Agent queries → tool mocks → reward scoring

## Component Contracts

### QueryAgent
```python
def query(user_message: str, context: Optional[Dict]) -> str
```
- Stateless per call (context passed explicitly)
- Uses Gemini for generation
- System prompt derived from trace patterns

### ToolMocker
```python
def mock(tool_name: str, tool_args: Dict) -> Any
```
- Exact match first, then fuzzy match
- Falls back to Gemini generation for unseen tools
- Returns JSON-serializable responses

### RewardFunction
```python
def score(response: str, ground_truth: Optional[str], context: Optional[Dict]) -> float
```
- Returns float in [-1.0, 1.0]
- Combines semantic similarity + quality assessment
- Handles missing ground truth gracefully

