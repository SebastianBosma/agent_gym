"""
Trace Parser - Ingest raw logs and output normalized JSON with consistent schema.

This module handles:
- Loading traces from various formats (JSON, JSONL, CSV)
- Validating trace structure
- Normalizing to a consistent schema for downstream use
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""
    role: str  # "user", "assistant", or "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None  # For tool messages
    result: Optional[Any] = None  # For tool messages


class TraceSchema(BaseModel):
    """Normalized schema for a conversation trace."""
    conversation_id: str
    messages: List[Message]
    outcome: Optional[str] = None  # e.g., "resolved", "escalated", "abandoned"
    satisfaction_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def parse_traces(traces: List[Dict[str, Any]]) -> List[TraceSchema]:
    """
    Parse and validate a list of raw traces into normalized TraceSchema objects.
    
    Args:
        traces: List of raw trace dictionaries
        
    Returns:
        List of validated TraceSchema objects
        
    Raises:
        ValueError: If traces don't match expected schema
    """
    parsed = []
    for i, trace in enumerate(traces):
        try:
            parsed.append(TraceSchema(**trace))
        except Exception as e:
            raise ValueError(f"Failed to parse trace {i}: {e}")
    return parsed


def load_traces_from_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Load traces from a JSON or JSONL file.
    
    Args:
        filepath: Path to the trace file
        
    Returns:
        List of raw trace dictionaries
    """
    import json
    
    with open(filepath, 'r') as f:
        if filepath.endswith('.jsonl'):
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)

