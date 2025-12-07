"""Optimization module - Create optimized RL environments for dialogue training."""

from .runner import (
    EnvironmentRunner,
    EnvironmentResult,
    OptimizationEvent,
    run_optimization,
    # Backward compatibility aliases
    OptimizationRunner,
    OptimizationResult,
)

__all__ = [
    "EnvironmentRunner",
    "EnvironmentResult",
    "OptimizationEvent",
    "run_optimization",
    # Backward compatibility
    "OptimizationRunner",
    "OptimizationResult",
]

