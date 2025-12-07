"""Environment module - RL environments for dialogue training."""

from .env import TraceEnvironment
from .sgd_env import SGDEnvironment, AgentAction, Observation, create_sgd_environment
from .multi_turn_env import (
    MultiTurnEnvironment,
    MultiTurnObservation,
    create_multi_turn_environment,
)

__all__ = [
    # Trace replay environments
    "TraceEnvironment",
    "SGDEnvironment",
    "AgentAction",
    "Observation",
    "create_sgd_environment",
    # Multi-turn environment with simulated user
    "MultiTurnEnvironment",
    "MultiTurnObservation",
    "create_multi_turn_environment",
]

