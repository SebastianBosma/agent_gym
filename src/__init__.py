"""
Agent Gym - Convert traces into RL environments.

Main API:
    # For generic traces:
    from agent_gym import create_environment
    query_agent, tool_mocker, reward_fn = create_environment(traces)
    
    # For SGD dataset:
    from agent_gym import create_sgd_environment
    env = create_sgd_environment("data/raw/schema_guided_dialogue")
    
    # For optimization:
    from agent_gym import OptimizationRunner
    runner = OptimizationRunner(data_path, strategy="mipro", callback=my_callback)
    result = runner.run()
"""

from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Callable

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

from .trace_parser import parse_traces
from .agent import QueryAgent
from .tool_mocker import ToolMocker
from .reward import RewardFunction

# SGD-specific imports
from .data import SGDLoader
from .environment import SGDEnvironment, create_sgd_environment, AgentAction

# Optimization imports
from .optimization import OptimizationRunner, OptimizationResult, OptimizationEvent


def create_environment(
    traces: List[Dict[str, Any]]
) -> Tuple[Callable, Callable, Callable]:
    """
    Factory function that creates an RL environment from conversation traces.
    
    Args:
        traces: List of conversation traces in normalized JSON format.
                Each trace should contain conversation messages, tool calls,
                and outcome/satisfaction data.
    
    Returns:
        Tuple of three callables:
        - query_agent: Function to query the Gemini-powered agent
        - tool_mocker: Function to mock tool responses based on trace data
        - reward_fn: Function to score agent responses
    
    Example:
        >>> traces = load_traces("data/processed/customer_service.json")
        >>> query_agent, tool_mocker, reward_fn = create_environment(traces)
        >>> response = query_agent("I want to cancel my subscription")
        >>> score = reward_fn(response, ground_truth="I can help with that...")
    """
    # Parse and validate traces
    parsed_traces = parse_traces(traces)
    
    # Initialize components with parsed trace data
    agent = QueryAgent(parsed_traces)
    mocker = ToolMocker(parsed_traces)
    reward = RewardFunction(parsed_traces)
    
    return agent.query, mocker.mock, reward.score


__all__ = [
    # Generic environment
    "create_environment",
    # SGD environment
    "SGDLoader",
    "SGDEnvironment",
    "create_sgd_environment",
    "AgentAction",
    # Optimization
    "OptimizationRunner",
    "OptimizationResult",
    "OptimizationEvent",
]

