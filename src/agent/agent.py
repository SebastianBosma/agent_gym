"""
Query Agent - DSPy-powered agent for customer service interactions.

Uses DSPy with Gemini backend for:
- Optimizable prompt generation
- Few-shot learning from traces
- Chain-of-thought reasoning
"""

from typing import List, Dict, Any, Optional, Callable
import dspy

from ..trace_parser import TraceSchema


class CustomerServiceResponse(dspy.Signature):
    """Generate a helpful customer service response."""
    
    user_message: str = dspy.InputField(desc="The customer's message")
    conversation_history: str = dspy.InputField(desc="Previous messages in the conversation")
    response: str = dspy.OutputField(desc="Professional, helpful response to the customer")


class QueryAgentModule(dspy.Module):
    """DSPy module for generating customer service responses."""
    
    def __init__(self):
        super().__init__()
        self.respond = dspy.ChainOfThought(CustomerServiceResponse)
    
    def forward(self, user_message: str, conversation_history: str = "") -> str:
        result = self.respond(
            user_message=user_message,
            conversation_history=conversation_history
        )
        return result.response


class QueryAgent:
    """
    DSPy-powered agent that responds to customer queries.
    
    The agent can be optimized using traces via DSPy's teleprompt optimizers.
    
    Attributes:
        traces: Training traces for context and optimization
        module: The underlying DSPy module
        conversation_history: Current conversation state
    """
    
    def __init__(
        self,
        traces: List[TraceSchema],
        tool_executor: Optional[Callable] = None,
    ):
        """
        Initialize agent with traces.
        
        Args:
            traces: List of TraceSchema objects for learning
            tool_executor: Optional function to execute tool calls
        """
        self.traces = traces
        self.tool_executor = tool_executor
        self.module = QueryAgentModule()
        self.conversation_history: List[Dict[str, str]] = []
        self._lm_configured = False
    
    def _ensure_lm_configured(self):
        """Configure DSPy with Gemini on first use."""
        if not self._lm_configured:
            lm = dspy.Google("models/gemini-1.5-pro-latest")
            dspy.settings.configure(lm=lm)
            self._lm_configured = True
    
    def query(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a response to a user query.
        
        Args:
            user_message: The user's message
            context: Optional context (history, metadata)
            
        Returns:
            Agent's response string
        """
        self._ensure_lm_configured()
        
        # Build conversation history string
        history_str = ""
        if context and "history" in context:
            history_str = "\n".join(
                f"{msg['role']}: {msg['content']}"
                for msg in context["history"][-5:]
            )
        elif self.conversation_history:
            history_str = "\n".join(
                f"{msg['role']}: {msg['content']}"
                for msg in self.conversation_history[-5:]
            )
        
        try:
            response = self.module(
                user_message=user_message,
                conversation_history=history_str
            )
            
            # Store in conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            return f"I apologize, but I'm having trouble processing your request. Error: {str(e)}"
    
    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_module(self) -> QueryAgentModule:
        """Return the DSPy module for optimization."""
        return self.module
    
    def set_module(self, module: QueryAgentModule) -> None:
        """Set an optimized DSPy module."""
        self.module = module
    
    def get_training_examples(self) -> List[dspy.Example]:
        """
        Convert traces to DSPy Examples for optimization.
        
        Returns:
            List of dspy.Example objects
        """
        examples = []
        for trace in self.traces:
            history = []
            for msg in trace.messages:
                if msg.role == "user":
                    # Create example from user message and next assistant response
                    next_assistant = self._find_next_assistant_response(trace.messages, msg)
                    if next_assistant:
                        examples.append(dspy.Example(
                            user_message=msg.content,
                            conversation_history="\n".join(history),
                            response=next_assistant
                        ).with_inputs("user_message", "conversation_history"))
                
                history.append(f"{msg.role}: {msg.content}")
        
        return examples
    
    def _find_next_assistant_response(self, messages, user_msg) -> Optional[str]:
        """Find the assistant response following a user message."""
        found_user = False
        for msg in messages:
            if msg == user_msg:
                found_user = True
            elif found_user and msg.role == "assistant":
                return msg.content
        return None
