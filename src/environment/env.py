"""
Trace Environment - RL environment that replays conversations.

This module provides:
- State management for conversation context
- Action space for agent responses
- Step function for environment transitions
- Episode management (reset, done conditions)
"""

from typing import List, Dict, Any, Optional, Tuple
from ..trace_parser import TraceSchema


class TraceEnvironment:
    """
    RL Environment that replays conversations from traces.
    
    The environment simulates customer service interactions by:
    1. Presenting user messages as observations
    2. Accepting agent responses as actions
    3. Using tool_mocker for any tool calls
    4. Computing rewards based on similarity to ground truth
    
    Attributes:
        traces: List of parsed conversation traces
        current_trace_idx: Index of current trace being replayed
        current_step: Current position in the conversation
    """
    
    def __init__(self, traces: List[TraceSchema]):
        """
        Initialize environment with parsed traces.
        
        Args:
            traces: List of TraceSchema objects to replay
        """
        self.traces = traces
        self.current_trace_idx = 0
        self.current_step = 0
        self._done = False
    
    def reset(self, trace_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset environment to start of a trace.
        
        Args:
            trace_idx: Specific trace to start. If None, moves to next trace.
            
        Returns:
            Initial observation (first user message and context)
        """
        if trace_idx is not None:
            self.current_trace_idx = trace_idx
        else:
            self.current_trace_idx = (self.current_trace_idx + 1) % len(self.traces)
        
        self.current_step = 0
        self._done = False
        
        return self._get_observation()
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Agent's response to current observation
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        trace = self.traces[self.current_trace_idx]
        
        # Get ground truth response for reward calculation
        ground_truth = self._get_ground_truth()
        
        # Calculate reward (placeholder - actual implementation in reward module)
        reward = 0.0
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        self._done = self.current_step >= len(trace.messages)
        
        observation = self._get_observation() if not self._done else {}
        
        info = {
            "ground_truth": ground_truth,
            "trace_id": trace.conversation_id,
            "outcome": trace.outcome,
        }
        
        return observation, reward, self._done, info
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation (user message and context)."""
        trace = self.traces[self.current_trace_idx]
        
        # Get conversation history up to current point
        history = trace.messages[:self.current_step]
        
        # Find next user message
        for i, msg in enumerate(trace.messages[self.current_step:]):
            if msg.role == "user":
                return {
                    "user_message": msg.content,
                    "history": [m.model_dump() for m in history],
                    "step": self.current_step,
                }
        
        return {"user_message": "", "history": [], "step": self.current_step}
    
    def _get_ground_truth(self) -> Optional[str]:
        """Get ground truth assistant response for current step."""
        trace = self.traces[self.current_trace_idx]
        
        for msg in trace.messages[self.current_step:]:
            if msg.role == "assistant":
                return msg.content
        
        return None

