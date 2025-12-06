"""
SGD Environment - RL environment for Schema-Guided Dialogue training.

Provides:
- State management with tool availability
- Action space supporting tool calls
- Rich reward signals from SGD annotations (or LLM-as-judge)
- Episode management for dialogue replay
"""

from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum

from ..data.sgd_loader import SGDLoader, SGDDialogue, SGDTurn
from ..tool_mocker.sgd_mocker import SGDToolMocker, SGDToolMockerWithHistory
from ..reward.llm_judge import LLMJudge, CachedLLMJudge, JudgeResult, ResponseQuality


RewardMode = Literal["ground_truth", "llm_judge", "hybrid"]


class ActionType(str, Enum):
    RESPONSE = "response"
    TOOL_CALL = "tool_call"


@dataclass
class AgentAction:
    """An action taken by the agent."""
    type: ActionType
    response: str  # Natural language response
    tool_call: Optional[Dict[str, Any]] = None  # {"name": str, "args": dict}


@dataclass
class Observation:
    """What the agent observes at each step."""
    user_message: str
    history: List[Dict[str, Any]]
    available_tools: List[Dict[str, Any]]
    tool_result: Optional[Any] = None
    step: int = 0
    
    # Ground truth hints (can be hidden during evaluation)
    ground_truth_intent: Optional[str] = None
    ground_truth_slots: Optional[Dict[str, List[str]]] = None


@dataclass
class StepResult:
    """Result of taking a step in the environment."""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


@dataclass
class RewardComponents:
    """
    Simplified reward breakdown - same structure for GT and LLM modes.
    
    Components:
    - response_quality: Is the response helpful/appropriate? (0-1)
    - tool_accuracy: Did agent use the right tool with right args? (0-1)
    
    Both modes compute these the same way, just with different methods.
    """
    response_quality: float = 0.0  # 0-1: How good is the response?
    tool_accuracy: float = 0.0     # 0-1: Right tool + right args combined
    
    @property
    def total(self) -> float:
        """Total reward, scaled to roughly 0-2 range per step."""
        return self.response_quality + self.tool_accuracy


class SGDEnvironment:
    """
    RL Environment for training agents on SGD dialogues.
    
    The environment replays dialogues, presenting user messages as observations
    and comparing agent actions to ground truth for reward computation.
    
    IMPORTANT: Ground truth comparison only works for the first turn!
    After that, use reward_mode="llm_judge" for meaningful evaluation.
    
    Example:
        loader = SGDLoader("data/raw/schema_guided_dialogue")
        env = SGDEnvironment.from_loader(loader, split="train", reward_mode="llm_judge")
        
        obs = env.reset()
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
    """
    
    def __init__(
        self,
        dialogues: List[SGDDialogue],
        tool_mocker: SGDToolMocker,
        tool_definitions: List[Dict[str, Any]],
        *,
        include_ground_truth_hints: bool = False,
        reward_weights: Optional[Dict[str, float]] = None,
        reward_mode: RewardMode = "ground_truth",
        llm_judge: Optional[LLMJudge] = None,
    ):
        """
        Initialize environment.
        
        Args:
            dialogues: List of SGD dialogues to replay
            tool_mocker: Mocker for service calls
            tool_definitions: Available tool schemas
            include_ground_truth_hints: Include GT intent/slots in observations
            reward_weights: Custom weights for reward components
            reward_mode: How to compute rewards:
                - "ground_truth": Compare to recorded responses (only valid for turn 1)
                - "llm_judge": Use LLM to evaluate response quality
                - "hybrid": Ground truth for turn 1, LLM judge after
            llm_judge: Custom LLM judge instance (created automatically if needed)
        """
        self.dialogues = dialogues
        self.tool_mocker = tool_mocker
        self.tool_definitions = tool_definitions
        self.include_ground_truth_hints = include_ground_truth_hints
        self.reward_mode = reward_mode
        
        # LLM Judge for evaluation
        self._llm_judge = llm_judge
        if reward_mode in ("llm_judge", "hybrid") and llm_judge is None:
            self._llm_judge = CachedLLMJudge()
        
        # Reward weights (simple: response quality and tool accuracy, both 0-1)
        self.reward_weights = reward_weights or {
            "response_quality": 1.0,  # Max 1.0 per step
            "tool_accuracy": 1.0,     # Max 1.0 per step
        }
        
        # State
        self._current_dialogue_idx = 0
        self._current_turn_idx = 0
        self._step_in_episode = 0  # Track which step we're on
        self._history: List[Dict[str, Any]] = []
        self._done = False
        self._last_tool_result: Optional[Any] = None
    
    @classmethod
    def from_loader(
        cls,
        loader: SGDLoader,
        split: str = "train",
        reward_mode: RewardMode = "ground_truth",
        **kwargs,
    ) -> "SGDEnvironment":
        """
        Create environment from an SGDLoader.
        
        Args:
            loader: SGDLoader instance
            split: Data split to use
            reward_mode: How to compute rewards ("ground_truth", "llm_judge", "hybrid")
            **kwargs: Additional arguments to __init__
            
        Returns:
            Configured SGDEnvironment
        """
        dialogues = loader.load_dialogues(split)
        tool_mocker = SGDToolMocker.from_loader(loader, split)
        tool_definitions = loader.get_tool_definitions(split)
        
        return cls(
            dialogues=dialogues,
            tool_mocker=tool_mocker,
            tool_definitions=tool_definitions,
            reward_mode=reward_mode,
            **kwargs,
        )
    
    @property
    def current_dialogue(self) -> SGDDialogue:
        """Get current dialogue being replayed."""
        return self.dialogues[self._current_dialogue_idx]
    
    @property
    def current_turn(self) -> Optional[SGDTurn]:
        """Get current turn (user message)."""
        dialogue = self.current_dialogue
        if self._current_turn_idx < len(dialogue.turns):
            return dialogue.turns[self._current_turn_idx]
        return None
    
    @property
    def ground_truth_turn(self) -> Optional[SGDTurn]:
        """Get ground truth system turn for current position."""
        dialogue = self.current_dialogue
        # System turn follows user turn
        gt_idx = self._current_turn_idx + 1
        if gt_idx < len(dialogue.turns):
            turn = dialogue.turns[gt_idx]
            if turn.speaker == "SYSTEM":
                return turn
        return None
    
    def reset(
        self,
        dialogue_idx: Optional[int] = None,
        *,
        random: bool = False,
    ) -> Observation:
        """
        Reset environment to start of a dialogue.
        
        Args:
            dialogue_idx: Specific dialogue index. If None, moves to next.
            random: If True, select random dialogue
            
        Returns:
            Initial observation
        """
        if random:
            import random as rand
            self._current_dialogue_idx = rand.randint(0, len(self.dialogues) - 1)
        elif dialogue_idx is not None:
            self._current_dialogue_idx = dialogue_idx
        else:
            self._current_dialogue_idx = (self._current_dialogue_idx + 1) % len(self.dialogues)
        
        self._current_turn_idx = 0
        self._step_in_episode = 0
        self._history = []
        self._done = False
        self._last_tool_result = None
        
        # Reset tool mocker if it tracks history
        if isinstance(self.tool_mocker, SGDToolMockerWithHistory):
            self.tool_mocker.reset()
        
        return self._get_observation()
    
    def step(self, action: AgentAction) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Agent's action (response + optional tool call)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        gt_turn = self.ground_truth_turn
        
        # Compute reward based on mode
        use_llm_judge = (
            self.reward_mode == "llm_judge" or
            (self.reward_mode == "hybrid" and self._step_in_episode > 0)
        )
        
        if use_llm_judge and self._llm_judge is not None:
            reward_components, judge_result = self._compute_reward_llm(action)
        else:
            reward_components = self._compute_reward(action, gt_turn)
            judge_result = None
        
        reward = reward_components.total
        self._step_in_episode += 1
        
        # Execute tool call if present
        tool_result = None
        if action.tool_call:
            tool_result = self.tool_mocker.call(
                action.tool_call["name"],
                action.tool_call.get("args", {}),
            )
            self._last_tool_result = tool_result
        
        # Add to history
        self._history.append({
            "role": "user",
            "content": self.current_turn.utterance if self.current_turn else "",
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
        
        # Advance turn (skip to next user turn)
        self._advance_to_next_user_turn()
        
        # Check if done
        self._done = self._current_turn_idx >= len(self.current_dialogue.turns)
        
        # Build info
        info = {
            "dialogue_id": self.current_dialogue.dialogue_id,
            "reward_components": reward_components,
            "ground_truth_response": gt_turn.utterance if gt_turn else None,
            "ground_truth_tool_call": gt_turn.service_call if gt_turn else None,
            "reward_mode": "llm_judge" if use_llm_judge else "ground_truth",
            "step_in_episode": self._step_in_episode,
        }
        if judge_result is not None:
            info["judge_result"] = judge_result
        
        # Note outcome at end (no bonus - keep rewards consistent per step)
        if self._done:
            info["outcome"] = self.current_dialogue._determine_outcome()
        
        observation = self._get_observation() if not self._done else Observation(
            user_message="",
            history=self._history,
            available_tools=self._get_available_tools(),
            step=self._current_turn_idx,
        )
        
        return observation, reward, self._done, info
    
    def _get_observation(self) -> Observation:
        """Build current observation."""
        current = self.current_turn
        
        obs = Observation(
            user_message=current.utterance if current else "",
            history=self._history.copy(),
            available_tools=self._get_available_tools(),
            tool_result=self._last_tool_result,
            step=self._current_turn_idx,
        )
        
        # Include ground truth hints if enabled
        if self.include_ground_truth_hints and current:
            state = current.state
            if state:
                obs.ground_truth_intent = state.get("active_intent")
                obs.ground_truth_slots = state.get("slot_values")
        
        self._last_tool_result = None  # Clear after observation
        return obs
    
    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get tools available for current dialogue's services."""
        services = set(self.current_dialogue.services)
        return [
            tool for tool in self.tool_definitions
            if tool.get("service") in services
        ]
    
    def _advance_to_next_user_turn(self) -> None:
        """Advance to the next user turn."""
        self._current_turn_idx += 1  # Move past current user turn
        
        # Skip system turn (we just responded)
        while self._current_turn_idx < len(self.current_dialogue.turns):
            if self.current_dialogue.turns[self._current_turn_idx].speaker == "USER":
                break
            self._current_turn_idx += 1
    
    def _compute_reward(
        self,
        action: AgentAction,
        gt_turn: Optional[SGDTurn],
    ) -> RewardComponents:
        """
        Compute reward using ground truth comparison.
        
        Components (both 0-1 scale):
        - response_quality: Word overlap similarity with GT response
        - tool_accuracy: Combined score for tool selection + arguments
        
        Args:
            action: Agent's action
            gt_turn: Ground truth system turn
            
        Returns:
            RewardComponents with scores
        """
        components = RewardComponents()
        
        if gt_turn is None:
            return components
        
        # 1. Response quality: similarity to ground truth (0-1)
        similarity = self._compute_response_similarity(action.response, gt_turn.utterance)
        components.response_quality = similarity * self.reward_weights["response_quality"]
        
        # 2. Tool accuracy: combined tool selection + argument accuracy (0-1)
        gt_call = gt_turn.service_call
        
        if gt_call is None and action.tool_call is None:
            # No tool expected, none called - perfect
            components.tool_accuracy = 1.0 * self.reward_weights["tool_accuracy"]
        elif gt_call is None and action.tool_call is not None:
            # Unnecessary tool call
            components.tool_accuracy = 0.0
        elif gt_call is not None and action.tool_call is None:
            # Missed tool call
            components.tool_accuracy = 0.0
        elif gt_call is not None and action.tool_call is not None:
            # Both have tool calls - check correctness
            if action.tool_call["name"] == gt_call["method"]:
                # Right tool - now check arguments (50% for tool, 50% for args)
                arg_score = self._compute_argument_accuracy(
                    action.tool_call.get("args", {}),
                    gt_call.get("parameters", {}),
                )
                components.tool_accuracy = (0.5 + 0.5 * arg_score) * self.reward_weights["tool_accuracy"]
            else:
                # Wrong tool
                components.tool_accuracy = 0.0
        
        return components
    
    def _compute_reward_llm(
        self,
        action: AgentAction,
    ) -> Tuple[RewardComponents, JudgeResult]:
        """
        Compute reward using LLM-as-judge.
        
        Same structure as GT mode, but uses LLM evaluation instead of comparison.
        
        Components (both 0-1 scale):
        - response_quality: LLM's rating of response helpfulness
        - tool_accuracy: LLM's assessment of tool usage correctness
        
        Args:
            action: Agent's action
            
        Returns:
            Tuple of (RewardComponents, JudgeResult)
        """
        components = RewardComponents()
        
        current = self.current_turn
        if current is None or self._llm_judge is None:
            return components, JudgeResult(
                quality=ResponseQuality.ACCEPTABLE,
                quality_score=0.5,
            )
        
        # Get available tools for this dialogue
        available_tools = self._get_available_tools()
        
        # Call LLM judge
        judge_result = self._llm_judge.evaluate(
            history=self._history,
            user_message=current.utterance,
            response=action.response,
            available_tools=available_tools,
            tool_call=action.tool_call,
        )
        
        # 1. Response quality from LLM rating (0-1)
        components.response_quality = (
            judge_result.quality_score * self.reward_weights["response_quality"]
        )
        
        # 2. Tool accuracy from LLM assessment (0-1)
        if action.tool_call is None:
            # No tool call - judge determines if that was appropriate via quality score
            # If tool was needed, quality score would be lower
            components.tool_accuracy = 1.0 * self.reward_weights["tool_accuracy"]
        elif judge_result.tool_correct is not None:
            if judge_result.tool_correct:
                # Right tool - combine with arg score (50% tool, 50% args)
                arg_score = judge_result.tool_args_score if judge_result.tool_args_score else 0.5
                components.tool_accuracy = (0.5 + 0.5 * arg_score) * self.reward_weights["tool_accuracy"]
            else:
                # Wrong tool
                components.tool_accuracy = 0.0
        else:
            # Judge didn't evaluate tool - assume neutral
            components.tool_accuracy = 0.5 * self.reward_weights["tool_accuracy"]
        
        return components, judge_result
    
    def _compute_response_similarity(self, response: str, ground_truth: str) -> float:
        """Compute simple word overlap similarity."""
        if not response or not ground_truth:
            return 0.0
        
        response_words = set(response.lower().split())
        gt_words = set(ground_truth.lower().split())
        
        if not gt_words:
            return 0.0
        
        intersection = response_words & gt_words
        union = response_words | gt_words
        
        # Jaccard similarity
        return len(intersection) / len(union) if union else 0.0
    
    def _compute_argument_accuracy(
        self,
        predicted: Dict[str, Any],
        ground_truth: Dict[str, Any],
    ) -> float:
        """Compute F1-like score for argument accuracy."""
        if not ground_truth:
            return 1.0 if not predicted else 0.0
        
        pred_items = set((k, str(v).lower()) for k, v in predicted.items())
        gt_items = set((k, str(v).lower()) for k, v in ground_truth.items())
        
        if not gt_items:
            return 1.0
        
        correct = len(pred_items & gt_items)
        precision = correct / len(pred_items) if pred_items else 0.0
        recall = correct / len(gt_items) if gt_items else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def get_num_dialogues(self) -> int:
        """Get total number of dialogues."""
        return len(self.dialogues)
    
    def get_dialogue_info(self, idx: int) -> Dict[str, Any]:
        """Get info about a specific dialogue."""
        d = self.dialogues[idx]
        return {
            "dialogue_id": d.dialogue_id,
            "services": d.services,
            "num_turns": len(d.turns),
            "num_service_calls": d.num_service_calls,
            "intents": d.intents,
        }
    
    def get_tool_catalog(self) -> str:
        """
        Get formatted tool catalog for agent prompts.
        
        Returns:
            Formatted string with all available tools
        """
        from ..agent.sgd_agent import build_tool_catalog_from_definitions
        return build_tool_catalog_from_definitions(self.tool_definitions)
    
    def get_agent_action_from_module(self, agent_module, obs: Observation) -> AgentAction:
        """
        Use an SGDAgentModule to generate an action for the current observation.
        
        This integrates the DSPy agent with the environment.
        
        Args:
            agent_module: An SGDAgentModule instance
            obs: Current observation
            
        Returns:
            AgentAction based on agent's prediction
        """
        import json
        
        # Format history for the agent
        history_str = ""
        if obs.history:
            lines = []
            for msg in obs.history[-6:]:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                lines.append(f"{role}: {content}")
            history_str = "\n".join(lines)
        
        # Get prediction from agent
        try:
            result = agent_module(
                user_message=obs.user_message,
                conversation_history=history_str,
            )
            
            # Parse tool call
            tool_call = None
            tool_call_str = getattr(result, "tool_call", "none")
            if tool_call_str and tool_call_str.strip().lower() != "none":
                try:
                    tool_call = json.loads(tool_call_str)
                except json.JSONDecodeError:
                    pass
            
            return AgentAction(
                type=ActionType.TOOL_CALL if tool_call else ActionType.RESPONSE,
                response=getattr(result, "response", ""),
                tool_call=tool_call,
            )
        except Exception as e:
            # Fallback on error
            return AgentAction(
                type=ActionType.RESPONSE,
                response=f"I apologize, I encountered an error: {str(e)[:50]}",
            )
    
    def run_episode_with_agent(
        self,
        agent_module,
        dialogue_idx: Optional[int] = None,
        max_steps: int = 20,
    ) -> Dict[str, Any]:
        """
        Run a complete episode using an agent module.
        
        Args:
            agent_module: An SGDAgentModule instance
            dialogue_idx: Specific dialogue to run (None = random)
            max_steps: Maximum steps per episode
            
        Returns:
            Dict with episode metrics
        """
        obs = self.reset(dialogue_idx=dialogue_idx, random=(dialogue_idx is None))
        
        total_reward = 0.0
        step_rewards = []
        actions_taken = []
        
        for step in range(max_steps):
            action = self.get_agent_action_from_module(agent_module, obs)
            obs, reward, done, info = self.step(action)
            
            total_reward += reward
            step_rewards.append(reward)
            actions_taken.append({
                "response": action.response[:100],
                "tool_call": action.tool_call,
                "reward": reward,
            })
            
            if done:
                break
        
        return {
            "dialogue_id": self.current_dialogue.dialogue_id,
            "total_reward": total_reward,
            "avg_reward": total_reward / len(step_rewards) if step_rewards else 0,
            "steps": len(step_rewards),
            "outcome": info.get("outcome", "unknown"),
            "actions": actions_taken,
        }


def create_sgd_environment(
    data_path: str,
    split: str = "train",
    reward_mode: RewardMode = "ground_truth",
    **filter_kwargs,
) -> SGDEnvironment:
    """
    Convenience function to create an SGD environment.
    
    Args:
        data_path: Path to SGD dataset
        split: Data split
        reward_mode: How to compute rewards:
            - "ground_truth": Compare to recorded responses (only valid for turn 1)
            - "llm_judge": Use LLM to evaluate response quality (recommended)
            - "hybrid": Ground truth for turn 1, LLM judge after
        **filter_kwargs: Filtering options (min_domains, max_turns, etc.)
        
    Returns:
        Configured SGDEnvironment
    """
    loader = SGDLoader(data_path)
    dialogues = loader.load_dialogues(split, **filter_kwargs)
    tool_mocker = SGDToolMocker.from_loader(loader, split)
    tool_definitions = loader.get_tool_definitions(split)
    
    return SGDEnvironment(
        dialogues=dialogues,
        tool_mocker=tool_mocker,
        tool_definitions=tool_definitions,
        reward_mode=reward_mode,
    )

