"""
Simulated User Agent - DSPy module that simulates a user with a specific goal.

This module generates realistic user responses based on:
- The user's goal (intents + slots from SGD metadata)
- The conversation history
- What the assistant just said

The module can be optimized using DSPy to generate more realistic user behavior.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import dspy

from ..data.sgd_loader import SGDLoader, SGDDialogue, SGDTurn


# =============================================================================
# 1. DSPy Signature for User Simulation
# =============================================================================

class SimulatedUserSignature(dspy.Signature):
    """Simulate a user with a specific goal in a customer service conversation.
    
    You are role-playing as a customer who wants to accomplish a specific goal.
    Generate a realistic, natural response to the assistant's message.
    
    Guidelines:
    - Stay in character as the user with the given goal
    - Provide information when asked, but don't volunteer everything at once
    - If the assistant asks about something you don't have a preference for, say so naturally
    - Express satisfaction when your goal is being met
    - Ask clarifying questions if the assistant's response is unclear
    - Keep responses concise (1-2 sentences typically)
    """
    
    user_goal: str = dspy.InputField(
        desc="The user's goal: intents they want to accomplish and information they have (e.g., 'Intent: FindRestaurant, ReserveRestaurant | Info: city=Oakland, cuisine=Italian, party_size=2')"
    )
    conversation_history: str = dspy.InputField(
        desc="The conversation so far (USER: ... ASSISTANT: ...)"
    )
    assistant_response: str = dspy.InputField(
        desc="The assistant's most recent response that the user is replying to"
    )
    
    user_message: str = dspy.OutputField(
        desc="The user's natural response (1-2 sentences, staying in character)"
    )


class InitialUserMessageSignature(dspy.Signature):
    """Generate the user's opening message to start a conversation.
    
    You are role-playing as a customer who wants to accomplish a specific goal.
    Generate a natural opening message to start the conversation with an assistant.
    
    Guidelines:
    - Introduce your need naturally without providing all details upfront
    - Be conversational and realistic
    - Don't reveal all information at once - let the conversation unfold
    """
    
    user_goal: str = dspy.InputField(
        desc="The user's goal: intents they want to accomplish and information they have"
    )
    
    user_message: str = dspy.OutputField(
        desc="The user's opening message to start the conversation (1-2 sentences)"
    )


# =============================================================================
# 2. Simulated User Module
# =============================================================================

class SimulatedUserModule(dspy.Module):
    """
    DSPy module that simulates a user with a specific goal.
    
    Can generate both initial messages and responses to assistant messages.
    Optimizable with DSPy's teleprompt optimizers.
    """
    
    def __init__(self):
        super().__init__()
        self.respond = dspy.Predict(SimulatedUserSignature)
        self.start = dspy.Predict(InitialUserMessageSignature)
    
    def forward(
        self,
        user_goal: str,
        conversation_history: str = "",
        assistant_response: str = "",
    ) -> dspy.Prediction:
        """
        Generate a user message.
        
        Args:
            user_goal: The user's goal (intents + slots)
            conversation_history: Previous turns in the conversation
            assistant_response: The assistant's most recent message (empty for initial message)
            
        Returns:
            DSPy Prediction with user_message field
        """
        if not assistant_response:
            # Generate initial message
            return self.start(user_goal=user_goal)
        else:
            # Generate response to assistant
            return self.respond(
                user_goal=user_goal,
                conversation_history=conversation_history,
                assistant_response=assistant_response,
            )
    
    def generate_initial_message(self, user_goal: str) -> str:
        """Generate the opening user message."""
        result = self.start(user_goal=user_goal)
        return result.user_message
    
    def generate_response(
        self,
        user_goal: str,
        conversation_history: str,
        assistant_response: str,
    ) -> str:
        """Generate a response to the assistant."""
        result = self.respond(
            user_goal=user_goal,
            conversation_history=conversation_history,
            assistant_response=assistant_response,
        )
        return result.user_message


# =============================================================================
# 3. Goal Extraction from SGD Dialogues
# =============================================================================

@dataclass
class UserGoal:
    """Represents a user's goal extracted from an SGD dialogue."""
    intents: List[str]
    slot_values: Dict[str, str]
    services: List[str]
    dialogue_id: str
    
    def to_prompt_string(self) -> str:
        """Format goal for the simulated user prompt."""
        intent_str = ", ".join(self.intents) if self.intents else "General inquiry"
        
        slot_parts = []
        for key, value in self.slot_values.items():
            slot_parts.append(f"{key}={value}")
        slots_str = ", ".join(slot_parts) if slot_parts else "No specific preferences"
        
        return f"Intent: {intent_str} | Info: {slots_str}"


def extract_user_goal(dialogue: SGDDialogue) -> UserGoal:
    """
    Extract the user's goal from an SGD dialogue.
    
    Collects all intents and slot values mentioned throughout the dialogue.
    
    Args:
        dialogue: An SGDDialogue object
        
    Returns:
        UserGoal with intents and slot values
    """
    intents = []
    slot_values = {}
    
    for turn in dialogue.turns:
        if turn.speaker != "USER":
            continue
        
        state = turn.state
        if not state:
            continue
        
        # Extract intent
        active_intent = state.get("active_intent")
        if active_intent and active_intent != "NONE" and active_intent not in intents:
            intents.append(active_intent)
        
        # Extract slot values
        turn_slots = state.get("slot_values", {})
        for slot_name, values in turn_slots.items():
            if values:  # Values is a list
                # Take the first value (most common case)
                slot_values[slot_name] = values[0] if isinstance(values, list) else values
    
    return UserGoal(
        intents=intents,
        slot_values=slot_values,
        services=dialogue.services,
        dialogue_id=dialogue.dialogue_id,
    )


def extract_all_user_goals(
    dialogues: List[SGDDialogue],
) -> List[UserGoal]:
    """Extract user goals from a list of dialogues."""
    return [extract_user_goal(d) for d in dialogues]


# =============================================================================
# 4. Training Data Extraction for DSPy Optimization
# =============================================================================

def extract_user_training_examples(
    dialogues: List[SGDDialogue],
    max_history_turns: int = 10,
) -> List[dspy.Example]:
    """
    Convert SGD dialogues to DSPy training examples for user simulation.
    
    Creates examples where:
    - Input: goal, conversation history, assistant response
    - Output: actual user message from the trace
    
    Args:
        dialogues: List of SGD dialogues
        max_history_turns: Maximum conversation turns to include in history
        
    Returns:
        List of DSPy Example objects
    """
    examples = []
    
    for dialogue in dialogues:
        goal = extract_user_goal(dialogue)
        goal_str = goal.to_prompt_string()
        
        history_lines = []
        
        for i, turn in enumerate(dialogue.turns):
            if turn.speaker == "USER":
                # For user turns after the first, create training examples
                if i > 0:
                    # Find the preceding assistant response
                    assistant_response = ""
                    for j in range(i - 1, -1, -1):
                        if dialogue.turns[j].speaker == "SYSTEM":
                            assistant_response = dialogue.turns[j].utterance
                            break
                    
                    if assistant_response:
                        # Build conversation history (up to this point)
                        history_str = "\n".join(history_lines[-max_history_turns:])
                        
                        example = dspy.Example(
                            user_goal=goal_str,
                            conversation_history=history_str,
                            assistant_response=assistant_response,
                            user_message=turn.utterance,
                        ).with_inputs("user_goal", "conversation_history", "assistant_response")
                        
                        examples.append(example)
                else:
                    # First user turn - example for initial message generation
                    example = dspy.Example(
                        user_goal=goal_str,
                        user_message=turn.utterance,
                    ).with_inputs("user_goal")
                    
                    examples.append(example)
                
                # Add to history
                history_lines.append(f"USER: {turn.utterance}")
            
            else:  # SYSTEM turn
                history_lines.append(f"ASSISTANT: {turn.utterance}")
    
    return examples


# =============================================================================
# 5. Simulated User Agent (Full Interface)
# =============================================================================

class SimulatedUserAgent:
    """
    Full simulated user agent that can be used in the RL environment.
    
    Wraps the DSPy module with LLM configuration and provides a clean interface.
    
    Example:
        agent = SimulatedUserAgent("data/raw/schema_guided_dialogue")
        
        # Generate initial message
        goal = agent.get_random_goal()
        message = agent.generate_initial_message(goal)
        
        # Generate response to assistant
        response = agent.generate_response(
            goal=goal,
            history="USER: Find me a restaurant\nASSISTANT: What city?",
            assistant_message="What city are you looking for?"
        )
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        split: str = "train",
        model: str = "gemini-2.0-flash",
    ):
        """
        Initialize the simulated user agent.
        
        Args:
            data_path: Path to SGD dataset (for extracting goals)
            split: Data split to use
            model: Gemini model to use
        """
        self.model = model
        self.module = SimulatedUserModule()
        self._lm_configured = False
        
        # Load user goals if data path provided
        self.user_goals: List[UserGoal] = []
        if data_path:
            loader = SGDLoader(data_path)
            dialogues = loader.load_dialogues(split)
            self.user_goals = extract_all_user_goals(dialogues)
    
    def _ensure_lm_configured(self):
        """Configure DSPy with Gemini on first use."""
        if not self._lm_configured:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            
            lm = dspy.LM(f"gemini/{self.model}", api_key=api_key)
            dspy.configure(lm=lm)
            self._lm_configured = True
    
    def generate_initial_message(self, goal: UserGoal) -> str:
        """
        Generate the opening user message for a goal.
        
        Args:
            goal: UserGoal object
            
        Returns:
            Initial user message string
        """
        self._ensure_lm_configured()
        return self.module.generate_initial_message(goal.to_prompt_string())
    
    def generate_response(
        self,
        goal: UserGoal,
        history: str,
        assistant_message: str,
    ) -> str:
        """
        Generate a user response to the assistant.
        
        Args:
            goal: UserGoal object
            history: Conversation history string
            assistant_message: The assistant's most recent message
            
        Returns:
            User response string
        """
        self._ensure_lm_configured()
        return self.module.generate_response(
            user_goal=goal.to_prompt_string(),
            conversation_history=history,
            assistant_response=assistant_message,
        )
    
    def get_random_goal(self) -> UserGoal:
        """Get a random user goal from the loaded goals."""
        import random
        if not self.user_goals:
            raise ValueError("No user goals loaded. Provide data_path in constructor.")
        return random.choice(self.user_goals)
    
    def get_goal(self, idx: int) -> UserGoal:
        """Get a specific user goal by index."""
        return self.user_goals[idx]
    
    def get_num_goals(self) -> int:
        """Get the number of available user goals."""
        return len(self.user_goals)
    
    def get_module(self) -> SimulatedUserModule:
        """Return the DSPy module for optimization."""
        return self.module
    
    def set_module(self, module: SimulatedUserModule) -> None:
        """Set an optimized DSPy module."""
        self.module = module
    
    def load_optimized(self, checkpoint_path: str) -> None:
        """Load an optimized module from a checkpoint."""
        self.module.load(checkpoint_path)


# =============================================================================
# 6. Metric for User Simulation Quality
# =============================================================================

def compute_user_simulation_similarity(
    predicted: str,
    ground_truth: str,
) -> float:
    """
    Compute similarity between predicted and ground truth user messages.
    
    Uses word overlap (Jaccard similarity).
    
    Args:
        predicted: Predicted user message
        ground_truth: Actual user message from trace
        
    Returns:
        Similarity score between 0 and 1
    """
    if not predicted or not ground_truth:
        return 0.0
    
    pred_words = set(predicted.lower().split())
    gt_words = set(ground_truth.lower().split())
    
    if not gt_words:
        return 0.0
    
    intersection = pred_words & gt_words
    union = pred_words | gt_words
    
    return len(intersection) / len(union) if union else 0.0


def compute_slot_mention_accuracy(
    predicted: str,
    goal: UserGoal,
    assistant_response: str,
) -> float:
    """
    Check if the user mentioned expected slot values when asked.
    
    If the assistant asked for information and the user should provide it,
    check if the predicted response contains the relevant slot value.
    
    Args:
        predicted: Predicted user message
        goal: The user's goal with slot values
        assistant_response: What the assistant asked
        
    Returns:
        Accuracy score between 0 and 1
    """
    if not goal.slot_values:
        return 1.0  # No slots to check
    
    # Simple heuristic: check if any slot values appear in the response
    # when they might be relevant
    predicted_lower = predicted.lower()
    
    mentioned = 0
    relevant = 0
    
    for slot_name, slot_value in goal.slot_values.items():
        # Check if assistant might be asking about this slot
        slot_keywords = slot_name.lower().replace("_", " ").split()
        assistant_lower = assistant_response.lower()
        
        if any(kw in assistant_lower for kw in slot_keywords) or "what" in assistant_lower:
            relevant += 1
            if slot_value.lower() in predicted_lower:
                mentioned += 1
    
    if relevant == 0:
        return 1.0  # No relevant slots were asked about
    
    return mentioned / relevant


def create_user_simulation_metric():
    """
    Create the metric function for DSPy optimization of user simulation.
    
    Returns:
        Metric function that takes (example, pred) and returns a score
    """
    def metric(example, pred, trace=None) -> float:
        """Score a predicted user message against ground truth."""
        predicted = getattr(pred, "user_message", "")
        ground_truth = example.user_message
        
        # Word overlap similarity
        similarity = compute_user_simulation_similarity(predicted, ground_truth)
        
        return similarity
    
    return metric

