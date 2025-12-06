"""
LLM-as-Judge - Evaluate agent responses using an LLM.

This is necessary because ground truth comparisons only work for the first turn.
After that, the conversation diverges from the recorded trajectory.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

import dspy


class ResponseQuality(str, Enum):
    EXCELLENT = "excellent"  # 1.0
    GOOD = "good"            # 0.75
    ACCEPTABLE = "acceptable"  # 0.5
    POOR = "poor"            # 0.25
    INAPPROPRIATE = "inappropriate"  # 0.0


QUALITY_SCORES = {
    ResponseQuality.EXCELLENT: 1.0,
    ResponseQuality.GOOD: 0.75,
    ResponseQuality.ACCEPTABLE: 0.5,
    ResponseQuality.POOR: 0.25,
    ResponseQuality.INAPPROPRIATE: 0.0,
}


class JudgeSignature(dspy.Signature):
    """Evaluate an assistant response in a task-oriented dialogue."""
    
    conversation_history: str = dspy.InputField(
        desc="The conversation history leading up to this response"
    )
    user_message: str = dspy.InputField(
        desc="The user's most recent message that the assistant is responding to"
    )
    assistant_response: str = dspy.InputField(
        desc="The assistant's response to evaluate"
    )
    available_tools: str = dspy.InputField(
        desc="Tools/APIs available to the assistant"
    )
    tool_call_made: str = dspy.InputField(
        desc="Tool call made by assistant, if any (JSON or 'none')"
    )
    
    quality: str = dspy.OutputField(
        desc="Quality rating: excellent, good, acceptable, poor, or inappropriate"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of the rating (1-2 sentences)"
    )


class ToolCallJudgeSignature(dspy.Signature):
    """Evaluate whether a tool call was appropriate."""
    
    conversation_history: str = dspy.InputField(
        desc="The conversation so far"
    )
    user_message: str = dspy.InputField(
        desc="The user's request"
    )
    tool_call: str = dspy.InputField(
        desc="The tool call made (name and arguments as JSON)"
    )
    available_tools: str = dspy.InputField(
        desc="Available tools with their descriptions"
    )
    
    correct_tool: str = dspy.OutputField(
        desc="yes or no - was this the right tool to call?"
    )
    correct_args: str = dspy.OutputField(
        desc="Score 0-100 for argument correctness"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation"
    )


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    quality: ResponseQuality
    quality_score: float
    tool_correct: Optional[bool] = None
    tool_args_score: Optional[float] = None
    reasoning: str = ""
    
    @property
    def total_score(self) -> float:
        """Combined score from response quality and tool usage."""
        score = self.quality_score
        
        if self.tool_correct is not None:
            if self.tool_correct:
                score += 0.5  # Bonus for correct tool
                if self.tool_args_score is not None:
                    score += 0.5 * self.tool_args_score  # Up to 0.5 more for args
            else:
                score -= 0.25  # Penalty for wrong tool
        
        return max(0.0, min(2.0, score))  # Clamp to [0, 2]


class LLMJudge:
    """
    LLM-based judge for evaluating agent responses.
    
    Uses Gemini to assess whether responses are appropriate given
    the conversation context, rather than comparing to ground truth.
    
    Example:
        judge = LLMJudge()
        
        result = judge.evaluate(
            history=[{"role": "user", "content": "Find me a restaurant"}],
            user_message="I want Italian food",
            response="I'll search for Italian restaurants. What city?",
            available_tools=[{"name": "FindRestaurants", ...}],
            tool_call={"name": "FindRestaurants", "args": {"cuisine": "Italian"}}
        )
        
        print(result.quality_score)  # 0.75
        print(result.tool_correct)   # True
    """
    
    def __init__(self, model: str = "gemini-3-pro-preview"):
        """
        Initialize the judge.
        
        Args:
            model: Gemini model to use
        """
        self.model = model
        self._configured = False
        self._response_judge = dspy.Predict(JudgeSignature)
        self._tool_judge = dspy.Predict(ToolCallJudgeSignature)
    
    def _ensure_configured(self):
        """Configure DSPy with Gemini on first use."""
        if not self._configured:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            
            lm = dspy.LM(f"gemini/{self.model}", api_key=api_key)
            dspy.configure(lm=lm)
            self._configured = True
    
    def evaluate(
        self,
        history: List[Dict[str, Any]],
        user_message: str,
        response: str,
        available_tools: List[Dict[str, Any]],
        tool_call: Optional[Dict[str, Any]] = None,
    ) -> JudgeResult:
        """
        Evaluate an agent response.
        
        Args:
            history: Conversation history
            user_message: Current user message being responded to
            response: Agent's response text
            available_tools: Tools available to the agent
            tool_call: Tool call made, if any
            
        Returns:
            JudgeResult with scores and reasoning
        """
        self._ensure_configured()
        
        # Format inputs
        history_str = self._format_history(history)
        tools_str = self._format_tools(available_tools)
        tool_call_str = self._format_tool_call(tool_call) if tool_call else "none"
        
        # Evaluate response quality
        try:
            quality_result = self._response_judge(
                conversation_history=history_str,
                user_message=user_message,
                assistant_response=response,
                available_tools=tools_str,
                tool_call_made=tool_call_str,
            )
            
            quality = self._parse_quality(quality_result.quality)
            reasoning = quality_result.reasoning
        except Exception as e:
            # Fallback on error
            quality = ResponseQuality.ACCEPTABLE
            reasoning = f"Evaluation error: {e}"
        
        result = JudgeResult(
            quality=quality,
            quality_score=QUALITY_SCORES[quality],
            reasoning=reasoning,
        )
        
        # Evaluate tool call if present
        if tool_call:
            tool_result = self._evaluate_tool_call(
                history_str, user_message, tool_call, tools_str
            )
            result.tool_correct = tool_result.get("correct", None)
            result.tool_args_score = tool_result.get("args_score", None)
            if tool_result.get("reasoning"):
                result.reasoning += f" Tool: {tool_result['reasoning']}"
        
        return result
    
    def _evaluate_tool_call(
        self,
        history_str: str,
        user_message: str,
        tool_call: Dict[str, Any],
        tools_str: str,
    ) -> Dict[str, Any]:
        """Evaluate a tool call."""
        try:
            result = self._tool_judge(
                conversation_history=history_str,
                user_message=user_message,
                tool_call=self._format_tool_call(tool_call),
                available_tools=tools_str,
            )
            
            correct = result.correct_tool.lower().strip() in ("yes", "true", "1")
            
            try:
                args_score = float(result.correct_args) / 100.0
            except (ValueError, TypeError):
                args_score = 0.5
            
            return {
                "correct": correct,
                "args_score": min(1.0, max(0.0, args_score)),
                "reasoning": result.reasoning,
            }
        except Exception:
            return {"correct": None, "args_score": None, "reasoning": ""}
    
    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history as string."""
        if not history:
            return "(no prior conversation)"
        
        lines = []
        for msg in history[-6:]:  # Last 6 messages for context
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")[:200]  # Truncate long messages
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def _format_tools(self, tools: List[Dict[str, Any]]) -> str:
        """Format available tools as string."""
        if not tools:
            return "(no tools available)"
        
        lines = []
        for tool in tools[:10]:  # Limit to 10 tools
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")[:100]
            lines.append(f"- {name}: {desc}")
        
        return "\n".join(lines)
    
    def _format_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """Format tool call as string."""
        import json
        name = tool_call.get("name", "unknown")
        args = tool_call.get("args", {})
        return f"{name}({json.dumps(args)})"
    
    def _parse_quality(self, quality_str: str) -> ResponseQuality:
        """Parse quality string to enum."""
        quality_str = quality_str.lower().strip()
        
        for q in ResponseQuality:
            if q.value in quality_str:
                return q
        
        # Default fallback
        return ResponseQuality.ACCEPTABLE


class CachedLLMJudge(LLMJudge):
    """
    LLM Judge with caching to reduce API calls.
    
    Caches results based on a hash of the inputs.
    """
    
    def __init__(self, *args, cache_size: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, JudgeResult] = {}
        self._cache_size = cache_size
    
    def _cache_key(
        self,
        history: List[Dict[str, Any]],
        user_message: str,
        response: str,
        tool_call: Optional[Dict[str, Any]],
    ) -> str:
        """Generate cache key from inputs."""
        import hashlib
        import json
        
        key_data = {
            "history": history[-3:],  # Last 3 messages
            "user": user_message,
            "response": response[:100],
            "tool": tool_call,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def evaluate(self, *args, **kwargs) -> JudgeResult:
        """Evaluate with caching."""
        key = self._cache_key(
            kwargs.get("history", args[0] if args else []),
            kwargs.get("user_message", args[1] if len(args) > 1 else ""),
            kwargs.get("response", args[2] if len(args) > 2 else ""),
            kwargs.get("tool_call"),
        )
        
        if key in self._cache:
            return self._cache[key]
        
        result = super().evaluate(*args, **kwargs)
        
        # Manage cache size
        if len(self._cache) >= self._cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._cache.keys())[:100]
            for k in keys_to_remove:
                del self._cache[k]
        
        self._cache[key] = result
        return result
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()

