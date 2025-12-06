"""Agent module - Gemini-powered query agent."""

from .agent import QueryAgent
from .sgd_agent import (
    SGDAgent,
    SGDAgentModule,
    build_tool_catalog,
    build_tool_catalog_from_definitions,
    create_sgd_signature,
    extract_training_examples,
    create_metric,
)

__all__ = [
    "QueryAgent",
    "SGDAgent",
    "SGDAgentModule",
    "build_tool_catalog",
    "build_tool_catalog_from_definitions",
    "create_sgd_signature",
    "extract_training_examples",
    "create_metric",
]

