"""
SGD Agent - Task-oriented dialogue agent with dynamic tool catalog.

This agent:
1. Extracts all tool definitions from the SGD schema
2. Builds a comprehensive prompt with the full tool catalog
3. Supports DSPy optimization (BootstrapFewShot, MIPRO)
"""

import os
import json
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass

import dspy

from ..data.sgd_loader import SGDLoader, SGDDialogue, SGDTurn, SGDSchema


# =============================================================================
# 1. Tool Catalog Builder
# =============================================================================

def build_tool_catalog(loader: SGDLoader, split: str = "train") -> str:
    """
    Extract ALL tools from SGD schema and format for the prompt.
    
    Args:
        loader: SGDLoader instance
        split: Data split to load schemas from
        
    Returns:
        Formatted string with all services and their tools
    """
    schemas = loader.load_schemas(split)
    
    lines = ["AVAILABLE TOOLS AND SERVICES:\n"]
    
    for schema in schemas:
        lines.append(f"## {schema.service_name}")
        lines.append(f"   {schema.description}")
        
        for intent in schema.intents:
            required = ", ".join(intent.required_slots) if intent.required_slots else "none"
            optional_keys = list(intent.optional_slots.keys()) if intent.optional_slots else []
            optional = ", ".join(optional_keys) if optional_keys else "none"
            
            lines.append(f"   - {intent.name}")
            lines.append(f"     Required: {required}")
            if optional != "none":
                lines.append(f"     Optional: {optional}")
        
        lines.append("")  # Blank line between services
    
    return "\n".join(lines)


def build_tool_catalog_from_definitions(tool_definitions: List[Dict[str, Any]]) -> str:
    """
    Build tool catalog from tool definition dicts (for use with environment).
    
    Args:
        tool_definitions: List of tool definition dictionaries
        
    Returns:
        Formatted string with all tools
    """
    # Group by service
    services: Dict[str, List[Dict]] = {}
    for tool in tool_definitions:
        service = tool.get("service", "Unknown")
        if service not in services:
            services[service] = []
        services[service].append(tool)
    
    lines = ["AVAILABLE TOOLS AND SERVICES:\n"]
    
    for service_name, tools in services.items():
        lines.append(f"## {service_name}")
        
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")[:100]
            params = tool.get("parameters", {})
            required = params.get("required", [])
            
            lines.append(f"   - {name}: {desc}")
            lines.append(f"     Required: {', '.join(required) if required else 'none'}")
        
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# 2. Dynamic Signature with Tool Catalog
# =============================================================================

def create_sgd_signature(tool_catalog: str) -> Type[dspy.Signature]:
    """
    Create a DSPy signature with the full tool catalog embedded in the docstring.
    
    The docstring becomes the system prompt, so embedding the tool catalog here
    ensures the model always knows what tools are available.
    
    Args:
        tool_catalog: Formatted string of all available tools
        
    Returns:
        A DSPy Signature class with the tool catalog in its docstring
    """
    docstring = f'''You are a task-oriented dialogue assistant helping users with various services.

{tool_catalog}

INSTRUCTIONS:
1. Identify the user's intent from their message
2. Check if a tool is needed and which one matches the intent
3. If a tool is needed, verify you have all REQUIRED parameters from the conversation
4. If missing required parameters, ask the user for them (one question at a time)
5. When you have all required parameters, call the tool with correct JSON format
6. After receiving tool results, present them clearly to the user

TOOL CALL FORMAT:
When calling a tool, output valid JSON: {{"name": "ToolName", "args": {{"param": "value"}}}}
If no tool is needed (e.g., answering a question or greeting), output: none

IMPORTANT:
- Only call a tool when you have ALL required parameters
- Be concise but helpful in your responses
- If the user's request is ambiguous, ask for clarification
'''
    
    # Create a new signature class with the dynamic docstring
    class SGDAgentSignature(dspy.Signature):
        __doc__ = docstring
        
        user_message: str = dspy.InputField(
            desc="The user's current message or request"
        )
        conversation_history: str = dspy.InputField(
            desc="Previous turns in the conversation (USER: ... ASSISTANT: ...)"
        )
        
        reasoning: str = dspy.OutputField(
            desc="Your step-by-step reasoning about what the user wants and what to do"
        )
        response: str = dspy.OutputField(
            desc="Your response to the user"
        )
        tool_call: str = dspy.OutputField(
            desc='Tool call as JSON {"name": "...", "args": {...}}, or "none" if not needed'
        )
    
    return SGDAgentSignature


# =============================================================================
# 3. SGD Agent Module
# =============================================================================

class SGDAgentModule(dspy.Module):
    """
    DSPy module for the SGD agent with ChainOfThought reasoning.
    
    This module can be optimized using DSPy's teleprompt optimizers
    (BootstrapFewShot, MIPRO, etc.).
    """
    
    def __init__(self, tool_catalog: str):
        """
        Initialize the agent module.
        
        Args:
            tool_catalog: Formatted string of all available tools
        """
        super().__init__()
        self.tool_catalog = tool_catalog
        self.signature = create_sgd_signature(tool_catalog)
        self.respond = dspy.ChainOfThought(self.signature)
    
    def forward(
        self,
        user_message: str,
        conversation_history: str = "",
    ) -> dspy.Prediction:
        """
        Generate a response to the user message.
        
        Args:
            user_message: The user's current message
            conversation_history: Formatted conversation history
            
        Returns:
            DSPy Prediction with reasoning, response, and tool_call fields
        """
        result = self.respond(
            user_message=user_message,
            conversation_history=conversation_history,
        )
        return result


class SGDAgent:
    """
    Full SGD agent that loads schema and creates the module.
    
    This is the main interface for using the agent.
    
    Example:
        agent = SGDAgent("data/raw/schema_guided_dialogue")
        result = agent.act(user_message="Find me a restaurant in Oakland")
        print(result.response)
        print(result.tool_call)
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        model: str = "gemini-3-pro-preview",
    ):
        """
        Initialize the agent.
        
        Args:
            data_path: Path to the SGD dataset
            split: Data split to load schemas from
            model: Gemini model to use
        """
        self.data_path = data_path
        self.split = split
        self.model = model
        
        # Load schema and build tool catalog
        self.loader = SGDLoader(data_path)
        self.tool_catalog = build_tool_catalog(self.loader, split)
        
        # Create module
        self.module = SGDAgentModule(self.tool_catalog)
        
        # Configure LM
        self._lm_configured = False
    
    def _ensure_lm_configured(self):
        """Configure DSPy with Gemini on first use."""
        if not self._lm_configured:
            api_key = os.getenv("GOOGLE_API_KEY")
            lm = dspy.LM(f"gemini/{self.model}", api_key=api_key)
            dspy.configure(lm=lm)
            self._lm_configured = True
    
    def act(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> dspy.Prediction:
        """
        Generate a response to the user message.
        
        Args:
            user_message: The user's current message
            conversation_history: List of previous messages
            
        Returns:
            DSPy Prediction with response and tool_call
        """
        self._ensure_lm_configured()
        
        # Format history
        history_str = ""
        if conversation_history:
            lines = []
            for msg in conversation_history[-6:]:  # Last 6 messages
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                lines.append(f"{role}: {content}")
            history_str = "\n".join(lines)
        
        return self.module(
            user_message=user_message,
            conversation_history=history_str,
        )
    
    def get_module(self) -> SGDAgentModule:
        """Return the DSPy module for optimization."""
        return self.module
    
    def set_module(self, module: SGDAgentModule) -> None:
        """Set an optimized module."""
        self.module = module


# =============================================================================
# 4. Training Data Extractor
# =============================================================================

def format_conversation_history(dialogue: SGDDialogue, up_to_turn: int) -> str:
    """
    Format conversation history up to a specific turn.
    
    Args:
        dialogue: The SGD dialogue
        up_to_turn: Index of the turn to stop at (exclusive)
        
    Returns:
        Formatted conversation history string
    """
    lines = []
    for i, turn in enumerate(dialogue.turns[:up_to_turn]):
        role = "USER" if turn.speaker == "USER" else "ASSISTANT"
        lines.append(f"{role}: {turn.utterance}")
    return "\n".join(lines)


def get_next_system_turn(dialogue: SGDDialogue, user_turn_idx: int) -> Optional[SGDTurn]:
    """
    Get the system turn following a user turn.
    
    Args:
        dialogue: The SGD dialogue
        user_turn_idx: Index of the user turn
        
    Returns:
        The next system turn, or None if not found
    """
    for i in range(user_turn_idx + 1, len(dialogue.turns)):
        if dialogue.turns[i].speaker == "SYSTEM":
            return dialogue.turns[i]
    return None


def format_tool_call(service_call: Optional[Dict[str, Any]]) -> str:
    """
    Format a service call as JSON string.
    
    Args:
        service_call: The service call dict from SGD
        
    Returns:
        JSON string or "none"
    """
    if service_call is None:
        return "none"
    
    return json.dumps({
        "name": service_call.get("method", "unknown"),
        "args": service_call.get("parameters", {}),
    })


def extract_training_examples(
    dialogues: List[SGDDialogue],
    max_history_turns: int = 6,
) -> List[dspy.Example]:
    """
    Convert SGD dialogues to DSPy training examples.
    
    Each example contains:
    - user_message: The user's message
    - conversation_history: Previous turns
    - response: Ground truth assistant response
    - tool_call: Ground truth tool call (or "none")
    
    Args:
        dialogues: List of SGD dialogues
        max_history_turns: Maximum number of history turns to include
        
    Returns:
        List of DSPy Example objects
    """
    examples = []
    
    for dialogue in dialogues:
        for i, turn in enumerate(dialogue.turns):
            if turn.speaker != "USER":
                continue
            
            # Get the next system turn as ground truth
            gt_turn = get_next_system_turn(dialogue, i)
            if gt_turn is None:
                continue
            
            # Format conversation history (limited to last N turns)
            history_start = max(0, i - max_history_turns)
            history = format_conversation_history(dialogue, i)
            if history_start > 0:
                # Truncate to last max_history_turns
                history_lines = history.split("\n")
                history = "\n".join(history_lines[-(max_history_turns):])
            
            # Format tool call
            tool_call_str = format_tool_call(gt_turn.service_call)
            
            # Create example
            example = dspy.Example(
                user_message=turn.utterance,
                conversation_history=history,
                response=gt_turn.utterance,
                tool_call=tool_call_str,
            ).with_inputs("user_message", "conversation_history")
            
            examples.append(example)
    
    return examples


# =============================================================================
# 5. Utility Functions for Evaluation
# =============================================================================

def compute_response_similarity(pred_response: str, gt_response: str) -> float:
    """
    Compute word overlap similarity between predicted and ground truth responses.
    
    Args:
        pred_response: Predicted response
        gt_response: Ground truth response
        
    Returns:
        Similarity score between 0 and 1
    """
    if not pred_response or not gt_response:
        return 0.0
    
    pred_words = set(pred_response.lower().split())
    gt_words = set(gt_response.lower().split())
    
    if not gt_words:
        return 0.0
    
    intersection = pred_words & gt_words
    union = pred_words | gt_words
    
    return len(intersection) / len(union) if union else 0.0


def compute_tool_accuracy(pred_tool_call: str, gt_tool_call: str) -> float:
    """
    Compute accuracy of tool call prediction.
    
    Args:
        pred_tool_call: Predicted tool call (JSON or "none")
        gt_tool_call: Ground truth tool call (JSON or "none")
        
    Returns:
        Accuracy score between 0 and 1
    """
    # Normalize
    pred = pred_tool_call.strip().lower()
    gt = gt_tool_call.strip().lower()
    
    # Both none
    if pred == "none" and gt == "none":
        return 1.0
    
    # One is none, other isn't
    if pred == "none" or gt == "none":
        return 0.0
    
    # Parse JSON
    try:
        pred_parsed = json.loads(pred_tool_call)
        gt_parsed = json.loads(gt_tool_call)
    except json.JSONDecodeError:
        return 0.0
    
    # Check tool name
    pred_name = pred_parsed.get("name", "").lower()
    gt_name = gt_parsed.get("name", "").lower()
    
    if pred_name != gt_name:
        return 0.0  # Wrong tool
    
    # Check arguments (F1-style)
    pred_args = pred_parsed.get("args", {})
    gt_args = gt_parsed.get("args", {})
    
    if not gt_args:
        return 1.0 if not pred_args else 0.5
    
    pred_items = set((k, str(v).lower()) for k, v in pred_args.items())
    gt_items = set((k, str(v).lower()) for k, v in gt_args.items())
    
    correct = len(pred_items & gt_items)
    precision = correct / len(pred_items) if pred_items else 0.0
    recall = correct / len(gt_items) if gt_items else 0.0
    
    if precision + recall == 0:
        return 0.5  # Right tool, wrong args
    
    f1 = 2 * precision * recall / (precision + recall)
    return 0.5 + 0.5 * f1  # 0.5 for correct tool + up to 0.5 for args


def create_metric():
    """
    Create the metric function for DSPy optimization.
    
    Returns:
        Metric function that takes (example, pred) and returns a score
    """
    def metric(example, pred, trace=None) -> float:
        """Score a prediction against ground truth."""
        # Get predicted values
        pred_response = getattr(pred, "response", "")
        pred_tool = getattr(pred, "tool_call", "none")
        
        # Get ground truth
        gt_response = example.response
        gt_tool = example.tool_call
        
        # Compute scores
        response_score = compute_response_similarity(pred_response, gt_response)
        tool_score = compute_tool_accuracy(pred_tool, gt_tool)
        
        # Combined score (equal weight)
        return (response_score + tool_score) / 2
    
    return metric

