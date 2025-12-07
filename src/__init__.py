"""
Agent Gym - Convert traces into RL environments for training dialogue agents.

Main API:
    from agent_gym import create_environment
    
    # Create RL environment with simulated user, tool mocker, and reward function
    simulated_user, tool_mocker, reward_fn = create_environment(
        "data/raw/schema_guided_dialogue"
    )
    
    # Or with an optimized simulated user checkpoint
    simulated_user, tool_mocker, reward_fn = create_environment(
        "data/raw/schema_guided_dialogue",
        simulated_user_checkpoint="checkpoints/simulated_user.json",
    )

The environment consists of:
    - simulated_user: DSPy-optimized agent that generates realistic user messages
    - tool_mocker: Mocks API/service calls using recorded responses
    - reward_fn: LLM-as-judge that scores agent responses

Use this environment to train a model (e.g., via GRPO) to act as a customer service agent.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

from .trace_parser import parse_traces
from .agent import QueryAgent
from .tool_mocker import ToolMocker, SGDToolMocker, LLMToolMocker
from .reward import RewardFunction

# SGD-specific imports
from .data import SGDLoader
from .environment import SGDEnvironment, create_sgd_environment, AgentAction

# Multi-turn environment with simulated user
from .environment import MultiTurnEnvironment, create_multi_turn_environment

# Simulated user
from .agent import SimulatedUserAgent, UserGoal

# Reward / LLM Judge
from .reward import LLMJudge, CachedLLMJudge

# Optimization imports
from .optimization import (
    EnvironmentRunner,
    EnvironmentResult,
    OptimizationEvent,
    run_optimization,
    # Backward compatibility
    OptimizationRunner,
    OptimizationResult,
)


def create_environment(
    data_path: str,
    split: str = "train",
    simulated_user_checkpoint: Optional[str] = None,
    limit: Optional[int] = None,
    use_llm_tool_mocker: bool = True,
) -> Tuple[SimulatedUserAgent, SGDToolMocker, CachedLLMJudge]:
    """
    Factory function that creates an RL environment from SGD traces.
    
    The environment has three components:
    1. SimulatedUserAgent - generates realistic user messages based on goals
    2. ToolMocker - mocks API responses (uses LLM for unseen queries by default)
    3. LLMJudge - scores agent responses (reward function)
    
    Args:
        data_path: Path to SGD dataset (e.g., "data/raw/schema_guided_dialogue")
        split: Data split to use ("train", "dev", "test")
        simulated_user_checkpoint: Path to optimized simulated user module
        limit: Limit number of dialogues/goals to load
        use_llm_tool_mocker: If True, use LLM to generate plausible results
            for unseen queries (default: True). Set to False for faster
            execution without API calls for tool mocking.
    
    Returns:
        Tuple of (simulated_user, tool_mocker, reward_fn)
    
    Example:
        >>> simulated_user, tool_mocker, reward_fn = create_environment(
        ...     "data/raw/schema_guided_dialogue",
        ...     simulated_user_checkpoint="checkpoints/simulated_user.json",
        ... )
        >>> 
        >>> # Get a user goal and generate initial message
        >>> goal = simulated_user.get_random_goal()
        >>> user_message = simulated_user.generate_initial_message(goal)
        >>> 
        >>> # Agent responds...
        >>> agent_response = "What city are you looking for?"
        >>> 
        >>> # Mock tool calls (LLM generates plausible results for unseen queries)
        >>> results = tool_mocker.call("FindRestaurants", {"city": "Tokyo"})
        >>> # Returns plausible Tokyo restaurants even if not in training data
        >>> 
        >>> # Score the agent's response
        >>> score = reward_fn.evaluate(
        ...     history=[],
        ...     user_message=user_message,
        ...     response=agent_response,
        ...     available_tools=[...],
        ... )
    """
    import os
    
    # Load SGD data
    loader = SGDLoader(data_path)
    
    # Create simulated user with goals from dialogues
    simulated_user = SimulatedUserAgent(data_path=data_path, split=split)
    
    # Load optimized checkpoint if provided
    if simulated_user_checkpoint and os.path.exists(simulated_user_checkpoint):
        simulated_user.load_optimized(simulated_user_checkpoint)
    
    # Create tool mocker (with LLM fallback for unseen queries)
    if use_llm_tool_mocker:
        tool_mocker = LLMToolMocker.from_loader(loader, split, use_llm_fallback=True)
    else:
        tool_mocker = SGDToolMocker.from_loader(loader, split)
    
    # Create LLM judge (reward function)
    reward_fn = CachedLLMJudge()
    
    return simulated_user, tool_mocker, reward_fn


__all__ = [
    # Main API
    "create_environment",
    # Environment components
    "SimulatedUserAgent",
    "UserGoal",
    "SGDToolMocker",
    "LLMToolMocker",
    "LLMJudge",
    "CachedLLMJudge",
    # SGD environment (for backward compatibility / trace replay)
    "SGDLoader",
    "SGDEnvironment",
    "create_sgd_environment",
    "AgentAction",
    # Multi-turn environment wrapper
    "MultiTurnEnvironment",
    "create_multi_turn_environment",
    # Environment Runner (creates optimized environments)
    "EnvironmentRunner",
    "EnvironmentResult",
    "OptimizationEvent",
    "run_optimization",
    # Backward compatibility
    "OptimizationRunner",
    "OptimizationResult",
]

