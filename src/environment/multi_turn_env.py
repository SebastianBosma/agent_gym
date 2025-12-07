"""
Multi-Turn Environment - RL environment with simulated user for dynamic conversations.

Unlike SGDEnvironment which replays fixed traces, this environment uses a
DSPy-optimized SimulatedUserAgent to generate realistic user responses
based on the agent's actual actions.

This enables true multi-turn RL training where the agent learns from
dynamic conversations rather than fixed trajectories.
"""

from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass, field
import os

from ..agent.simulated_user import (
    SimulatedUserAgent,
    SimulatedUserModule,
    UserGoal,
    extract_all_user_goals,
)
from ..data.sgd_loader import SGDLoader
from ..tool_mocker.sgd_mocker import SGDToolMocker, SGDToolMockerWithHistory
from ..reward.llm_judge import LLMJudge, CachedLLMJudge, JudgeResult

# Re-use types from sgd_env
from .sgd_env import ActionType, AgentAction, Observation, RewardComponents


@dataclass
class MultiTurnObservation:
    """Observation for multi-turn environment."""
    user_message: str
    history: List[Dict[str, Any]]
    available_tools: List[Dict[str, Any]]
    tool_result: Optional[Any] = None
    step: int = 0
    user_goal: Optional[str] = None  # The user's goal (for debugging/logging)


class MultiTurnEnvironment:
    """
    RL Environment with simulated user for dynamic multi-turn conversations.
    
    Key differences from SGDEnvironment:
    - Uses SimulatedUserAgent to generate user messages dynamically
    - Conversations are not fixed - they evolve based on agent's responses
    - Episodes limited to max_turns (default 10)
    - Reward computed via LLM judge only (no ground truth comparison)
    
    Example:
        env = MultiTurnEnvironment.from_sgd(
            data_path="data/raw/schema_guided_dialogue",
            simulated_user_checkpoint="checkpoints/simulated_user.json",
        )
        
        obs = env.reset()
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
    """
    
    def __init__(
        self,
        simulated_user: SimulatedUserAgent,
        tool_mocker: SGDToolMocker,
        tool_definitions: List[Dict[str, Any]],
        user_goals: List[UserGoal],
        *,
        llm_judge: Optional[LLMJudge] = None,
        max_turns: int = 10,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize multi-turn environment.
        
        Args:
            simulated_user: SimulatedUserAgent for generating user messages
            tool_mocker: Mocker for service/API calls
            tool_definitions: Available tool schemas
            user_goals: List of UserGoal objects to sample from
            llm_judge: LLM judge for reward computation (created if None)
            max_turns: Maximum turns per episode (default 10)
            reward_weights: Custom weights for reward components
        """
        self.simulated_user = simulated_user
        self.tool_mocker = tool_mocker
        self.tool_definitions = tool_definitions
        self.user_goals = user_goals
        self.max_turns = max_turns
        
        # LLM Judge for evaluation
        self._llm_judge = llm_judge or CachedLLMJudge()
        
        # Reward weights
        self.reward_weights = reward_weights or {
            "response_quality": 1.0,
            "tool_accuracy": 1.0,
        }
        
        # Episode state
        self._current_goal_idx = 0
        self._current_goal: Optional[UserGoal] = None
        self._current_turn = 0
        self._history: List[Dict[str, Any]] = []
        self._done = False
        self._last_tool_result: Optional[Any] = None
        self._last_user_message: str = ""
    
    @classmethod
    def from_sgd(
        cls,
        data_path: str,
        split: str = "train",
        simulated_user_checkpoint: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        max_turns: int = 10,
        limit: Optional[int] = None,
    ) -> "MultiTurnEnvironment":
        """
        Create environment from SGD dataset.
        
        Args:
            data_path: Path to SGD dataset
            split: Data split to use
            simulated_user_checkpoint: Path to optimized simulated user (optional)
            model: Gemini model for simulated user and judge
            max_turns: Maximum turns per episode
            limit: Limit number of user goals
            
        Returns:
            Configured MultiTurnEnvironment
        """
        # Load SGD data
        loader = SGDLoader(data_path)
        dialogues = loader.load_dialogues(split, limit=limit)
        
        # Extract user goals
        user_goals = extract_all_user_goals(dialogues)
        
        # Create tool mocker
        tool_mocker = SGDToolMockerWithHistory.from_loader(loader, split)
        
        # Get tool definitions
        tool_definitions = loader.get_tool_definitions(split)
        
        # Create simulated user
        simulated_user = SimulatedUserAgent(model=model)
        simulated_user.user_goals = user_goals
        
        # Load optimized module if checkpoint provided
        if simulated_user_checkpoint and os.path.exists(simulated_user_checkpoint):
            simulated_user.load_optimized(simulated_user_checkpoint)
        
        return cls(
            simulated_user=simulated_user,
            tool_mocker=tool_mocker,
            tool_definitions=tool_definitions,
            user_goals=user_goals,
            max_turns=max_turns,
        )
    
    @property
    def current_goal(self) -> Optional[UserGoal]:
        """Get current user goal."""
        return self._current_goal
    
    def reset(
        self,
        goal_idx: Optional[int] = None,
        *,
        random: bool = False,
    ) -> MultiTurnObservation:
        """
        Reset environment to start a new episode.
        
        Args:
            goal_idx: Specific goal index. If None, moves to next.
            random: If True, select random goal
            
        Returns:
            Initial observation with first user message
        """
        # Select goal
        if random:
            import random as rand
            self._current_goal_idx = rand.randint(0, len(self.user_goals) - 1)
        elif goal_idx is not None:
            self._current_goal_idx = goal_idx
        else:
            self._current_goal_idx = (self._current_goal_idx + 1) % len(self.user_goals)
        
        self._current_goal = self.user_goals[self._current_goal_idx]
        
        # Reset state
        self._current_turn = 0
        self._history = []
        self._done = False
        self._last_tool_result = None
        
        # Reset tool mocker if it tracks history
        if isinstance(self.tool_mocker, SGDToolMockerWithHistory):
            self.tool_mocker.reset()
        
        # Generate initial user message
        self._last_user_message = self.simulated_user.generate_initial_message(
            self._current_goal
        )
        
        return self._get_observation()
    
    def step(self, action: AgentAction) -> Tuple[MultiTurnObservation, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Agent's action (response + optional tool call)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        # Execute tool call if present
        tool_result = None
        if action.tool_call:
            tool_result = self.tool_mocker.call(
                action.tool_call["name"],
                action.tool_call.get("args", {}),
            )
            self._last_tool_result = tool_result
        
        # Compute reward using LLM judge
        reward_components, judge_result = self._compute_reward(action)
        reward = reward_components.total
        
        # Add to history
        self._history.append({
            "role": "user",
            "content": self._last_user_message,
        })
        self._history.append({
            "role": "assistant",
            "content": action.response,
            "tool_call": action.tool_call,
        })
        if tool_result is not None:
            self._history.append({
                "role": "tool",
                "name": action.tool_call["name"] if action.tool_call else "unknown",
                "result": tool_result,
            })
        
        # Increment turn counter
        self._current_turn += 1
        
        # Check if done
        self._done = self._current_turn >= self.max_turns
        
        # Generate next user message if not done
        if not self._done:
            self._last_user_message = self._generate_next_user_message(action.response)
            
            # Check if conversation naturally ended (user satisfied)
            if self._is_conversation_complete(self._last_user_message):
                self._done = True
        
        # Build info
        info = {
            "goal_id": self._current_goal.dialogue_id if self._current_goal else None,
            "goal": self._current_goal.to_prompt_string() if self._current_goal else None,
            "reward_components": reward_components,
            "judge_result": judge_result,
            "turn": self._current_turn,
        }
        
        if self._done:
            info["total_turns"] = self._current_turn
            info["outcome"] = "completed" if self._current_turn < self.max_turns else "max_turns"
        
        observation = self._get_observation() if not self._done else MultiTurnObservation(
            user_message="",
            history=self._history,
            available_tools=self._get_available_tools(),
            step=self._current_turn,
            user_goal=self._current_goal.to_prompt_string() if self._current_goal else None,
        )
        
        return observation, reward, self._done, info
    
    def _get_observation(self) -> MultiTurnObservation:
        """Build current observation."""
        return MultiTurnObservation(
            user_message=self._last_user_message,
            history=self._history.copy(),
            available_tools=self._get_available_tools(),
            tool_result=self._last_tool_result,
            step=self._current_turn,
            user_goal=self._current_goal.to_prompt_string() if self._current_goal else None,
        )
    
    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get tools available for current goal's services."""
        if not self._current_goal:
            return self.tool_definitions
        
        services = set(self._current_goal.services)
        return [
            tool for tool in self.tool_definitions
            if tool.get("service") in services
        ]
    
    def _generate_next_user_message(self, assistant_response: str) -> str:
        """Generate the next user message based on assistant's response."""
        # Format history for simulated user
        history_str = self._format_history()
        
        return self.simulated_user.generate_response(
            goal=self._current_goal,
            history=history_str,
            assistant_message=assistant_response,
        )
    
    def _format_history(self) -> str:
        """Format conversation history as string."""
        lines = []
        for msg in self._history[-10:]:  # Last 10 messages
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if role == "TOOL":
                continue  # Skip tool results in history string
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    def _is_conversation_complete(self, user_message: str) -> bool:
        """
        Check if the user's message indicates the conversation is complete.
        
        Simple heuristic: look for thank you / goodbye patterns.
        """
        lower = user_message.lower()
        completion_signals = [
            "thank you",
            "thanks",
            "that's all",
            "that is all",
            "goodbye",
            "bye",
            "perfect",
            "great, thanks",
        ]
        return any(signal in lower for signal in completion_signals)
    
    def _compute_reward(
        self,
        action: AgentAction,
    ) -> Tuple[RewardComponents, JudgeResult]:
        """
        Compute reward using LLM judge.
        
        Args:
            action: Agent's action
            
        Returns:
            Tuple of (RewardComponents, JudgeResult)
        """
        components = RewardComponents()
        
        if self._llm_judge is None:
            return components, JudgeResult(
                quality=None,
                quality_score=0.5,
            )
        
        # Get available tools for this goal
        available_tools = self._get_available_tools()
        
        # Call LLM judge
        judge_result = self._llm_judge.evaluate(
            history=self._history,
            user_message=self._last_user_message,
            response=action.response,
            available_tools=available_tools,
            tool_call=action.tool_call,
        )
        
        # Response quality from LLM rating
        components.response_quality = (
            judge_result.quality_score * self.reward_weights["response_quality"]
        )
        
        # Tool accuracy from LLM assessment
        if action.tool_call is None:
            # No tool call - neutral (judge determines appropriateness via quality)
            components.tool_accuracy = 1.0 * self.reward_weights["tool_accuracy"]
        elif judge_result.tool_correct is not None:
            if judge_result.tool_correct:
                # Right tool - combine with arg score
                arg_score = judge_result.tool_args_score if judge_result.tool_args_score else 0.5
                components.tool_accuracy = (0.5 + 0.5 * arg_score) * self.reward_weights["tool_accuracy"]
            else:
                # Wrong tool
                components.tool_accuracy = 0.0
        else:
            # Judge didn't evaluate tool - assume neutral
            components.tool_accuracy = 0.5 * self.reward_weights["tool_accuracy"]
        
        return components, judge_result
    
    def get_num_goals(self) -> int:
        """Get total number of user goals."""
        return len(self.user_goals)
    
    def get_goal_info(self, idx: int) -> Dict[str, Any]:
        """Get info about a specific goal."""
        goal = self.user_goals[idx]
        return {
            "dialogue_id": goal.dialogue_id,
            "services": goal.services,
            "intents": goal.intents,
            "slots": goal.slot_values,
            "prompt": goal.to_prompt_string(),
        }
    
    def get_tool_catalog(self) -> str:
        """Get formatted tool catalog for agent prompts."""
        from ..agent.sgd_agent import build_tool_catalog_from_definitions
        return build_tool_catalog_from_definitions(self.tool_definitions)
    
    def run_episode(
        self,
        agent_fn,
        goal_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete episode using an agent function.
        
        Args:
            agent_fn: Function that takes observation and returns AgentAction
            goal_idx: Specific goal to run (None = random)
            
        Returns:
            Dict with episode metrics
        """
        obs = self.reset(goal_idx=goal_idx, random=(goal_idx is None))
        
        total_reward = 0.0
        step_rewards = []
        actions_taken = []
        
        done = False
        while not done:
            action = agent_fn(obs)
            obs, reward, done, info = self.step(action)
            
            total_reward += reward
            step_rewards.append(reward)
            actions_taken.append({
                "user_message": obs.history[-2]["content"] if len(obs.history) >= 2 else "",
                "response": action.response[:100],
                "tool_call": action.tool_call,
                "reward": reward,
            })
        
        return {
            "goal_id": self._current_goal.dialogue_id if self._current_goal else None,
            "goal": self._current_goal.to_prompt_string() if self._current_goal else None,
            "total_reward": total_reward,
            "avg_reward": total_reward / len(step_rewards) if step_rewards else 0,
            "num_turns": len(step_rewards),
            "outcome": info.get("outcome", "unknown"),
            "actions": actions_taken,
        }


def create_multi_turn_environment(
    sgd_path: str,
    split: str = "train",
    simulated_user_checkpoint: Optional[str] = None,
    max_turns: int = 10,
    limit: Optional[int] = None,
) -> MultiTurnEnvironment:
    """
    Convenience function to create a multi-turn environment.
    
    Args:
        sgd_path: Path to SGD dataset
        split: Data split
        simulated_user_checkpoint: Path to optimized simulated user module
        max_turns: Maximum turns per episode (default 10)
        limit: Limit number of user goals
        
    Returns:
        Configured MultiTurnEnvironment
    """
    return MultiTurnEnvironment.from_sgd(
        data_path=sgd_path,
        split=split,
        simulated_user_checkpoint=simulated_user_checkpoint,
        max_turns=max_turns,
        limit=limit,
    )

