"""Agent module - Gemini-powered query agent and simulated user."""

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
from .simulated_user import (
    SimulatedUserAgent,
    SimulatedUserModule,
    UserGoal,
    extract_user_goal,
    extract_all_user_goals,
    extract_user_training_examples,
    create_user_simulation_metric,
)

__all__ = [
    # Query agent
    "QueryAgent",
    # SGD agent
    "SGDAgent",
    "SGDAgentModule",
    "build_tool_catalog",
    "build_tool_catalog_from_definitions",
    "create_sgd_signature",
    "extract_training_examples",
    "create_metric",
    # Simulated user
    "SimulatedUserAgent",
    "SimulatedUserModule",
    "UserGoal",
    "extract_user_goal",
    "extract_all_user_goals",
    "extract_user_training_examples",
    "create_user_simulation_metric",
]

