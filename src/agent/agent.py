"""
Query Agent - Gemini-powered agent for customer service interactions.

This module provides:
- Gemini-based response generation
- Context management for conversations
- Tool calling integration
- System prompt customization based on traces
"""

from typing import List, Dict, Any, Optional, Callable
import google.generativeai as genai

from ..trace_parser import TraceSchema


class QueryAgent:
    """
    Gemini-powered agent that responds to customer queries.
    
    The agent learns from trace data to understand:
    - Common query patterns
    - Appropriate response styles
    - When and how to use tools
    
    Attributes:
        traces: Training traces for context
        system_prompt: Generated system prompt based on traces
        tools: Available tools and their schemas
    """
    
    def __init__(
        self,
        traces: List[TraceSchema],
        tool_executor: Optional[Callable] = None,
    ):
        """
        Initialize agent with traces and optional tool executor.
        
        Args:
            traces: List of TraceSchema objects for learning
            tool_executor: Optional function to execute tool calls
        """
        self.traces = traces
        self.tool_executor = tool_executor
        self.system_prompt = self._generate_system_prompt()
        self.conversation_history: List[Dict[str, str]] = []
        self._model = None
    
    @property
    def model(self):
        """Lazy-load Gemini model."""
        if self._model is None:
            self._model = genai.GenerativeModel(
                'gemini-pro',
                system_instruction=self.system_prompt,
            )
        return self._model
    
    def _generate_system_prompt(self) -> str:
        """
        Generate a system prompt based on trace patterns.
        
        Analyzes traces to extract:
        - Common greeting patterns
        - Response style guidelines
        - Domain-specific knowledge
        """
        # Extract example responses from traces
        examples = []
        for trace in self.traces[:5]:  # Sample from first 5 traces
            for msg in trace.messages:
                if msg.role == "assistant" and len(examples) < 3:
                    examples.append(msg.content)
        
        examples_str = "\n".join(f"- {ex[:100]}..." for ex in examples)
        
        return f"""You are a helpful customer service agent. Your role is to assist customers with their inquiries professionally and efficiently.

Based on historical interactions, here are example response styles:
{examples_str}

Guidelines:
1. Be polite and professional at all times
2. Ask clarifying questions when needed
3. Use available tools to look up information
4. Provide clear, actionable responses
5. Acknowledge customer concerns empathetically"""
    
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
        # Build conversation context
        messages = []
        
        # Add history from context if available
        if context and "history" in context:
            for msg in context["history"]:
                role = "user" if msg["role"] == "user" else "model"
                messages.append({"role": role, "parts": [msg["content"]]})
        
        # Add current message
        messages.append({"role": "user", "parts": [user_message]})
        
        try:
            # Start chat with history
            chat = self.model.start_chat(history=messages[:-1])
            response = chat.send_message(user_message)
            
            # Store in conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response.text})
            
            return response.text
        except Exception as e:
            return f"I apologize, but I'm having trouble processing your request. Error: {str(e)}"
    
    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def set_tool_executor(self, executor: Callable) -> None:
        """
        Set the tool executor for handling tool calls.
        
        Args:
            executor: Function that takes (tool_name, tool_args) and returns result
        """
        self.tool_executor = executor
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return current conversation history."""
        return self.conversation_history.copy()

