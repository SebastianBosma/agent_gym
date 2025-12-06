"""
Tool Mocker - Return realistic tool responses based on trace data.

Uses DSPy for generating synthetic responses when no exact match exists.
"""

from typing import List, Dict, Any, Optional
import json
import dspy

from ..trace_parser import TraceSchema


class ToolResponseSignature(dspy.Signature):
    """Generate a realistic tool response."""
    
    tool_name: str = dspy.InputField(desc="Name of the tool being called")
    tool_args: str = dspy.InputField(desc="JSON string of tool arguments")
    examples: str = dspy.InputField(desc="Example responses from similar tool calls")
    response: str = dspy.OutputField(desc="JSON response from the tool")


class ToolResponseModule(dspy.Module):
    """DSPy module for generating tool responses."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(ToolResponseSignature)
    
    def forward(self, tool_name: str, tool_args: str, examples: str = "") -> str:
        result = self.generate(
            tool_name=tool_name,
            tool_args=tool_args,
            examples=examples
        )
        return result.response


class ToolMocker:
    """
    Mock tool responses based on patterns learned from traces.
    
    Uses exact matching first, then DSPy for generation.
    
    Attributes:
        tool_index: Mapping of tool_name -> list of (args, response) pairs
        module: DSPy module for generating novel responses
    """
    
    def __init__(self, traces: List[TraceSchema]):
        """
        Initialize mocker with parsed traces.
        
        Args:
            traces: List of TraceSchema objects to learn from
        """
        self.traces = traces
        self.tool_index: Dict[str, List[Dict[str, Any]]] = {}
        self.module = ToolResponseModule()
        self._lm_configured = False
        self._build_index()
    
    def _ensure_lm_configured(self):
        """Configure DSPy with Gemini on first use."""
        if not self._lm_configured:
            import os
            api_key = os.getenv("GOOGLE_API_KEY")
            # Use Gemini 3 Pro for complex generation tasks
            lm = dspy.LM("gemini/gemini-3-pro-preview", api_key=api_key)
            dspy.settings.configure(lm=lm)
            self._lm_configured = True
    
    def _build_index(self) -> None:
        """Extract all tool calls and responses from traces."""
        for trace in self.traces:
            for i, msg in enumerate(trace.messages):
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                        
                        # Find corresponding tool response
                        response = self._find_tool_response(trace.messages, i, tool_name)
                        
                        if tool_name not in self.tool_index:
                            self.tool_index[tool_name] = []
                        
                        self.tool_index[tool_name].append({
                            "args": tool_args,
                            "response": response,
                        })
    
    def _find_tool_response(
        self, messages: List, start_idx: int, tool_name: str
    ) -> Optional[Any]:
        """Find the tool response message following a tool call."""
        for msg in messages[start_idx + 1:]:
            if msg.role == "tool" and msg.name == tool_name:
                return msg.result
        return None
    
    def mock(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Return a mocked response for a tool call.
        
        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments passed to the tool
            
        Returns:
            Mocked tool response based on trace data or generated
        """
        # Try exact match first
        if tool_name in self.tool_index:
            for entry in self.tool_index[tool_name]:
                if entry["args"] == tool_args:
                    return entry["response"]
            
            # Return first available response for this tool as fallback
            if self.tool_index[tool_name]:
                return self.tool_index[tool_name][0]["response"]
        
        # Generate synthetic response using DSPy
        return self._generate_response(tool_name, tool_args)
    
    def _generate_response(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a synthetic tool response using DSPy.
        
        Args:
            tool_name: Name of the tool
            tool_args: Arguments to the tool
            
        Returns:
            Generated tool response
        """
        self._ensure_lm_configured()
        
        # Get examples from similar tools for context
        examples = []
        if tool_name in self.tool_index:
            examples = self.tool_index[tool_name][:3]
        
        examples_str = json.dumps(examples, indent=2) if examples else "No examples available"
        
        try:
            response_str = self.module(
                tool_name=tool_name,
                tool_args=json.dumps(tool_args),
                examples=examples_str
            )
            return json.loads(response_str)
        except (json.JSONDecodeError, Exception):
            # Fallback response
            return {"status": "success", "data": None}
    
    def get_available_tools(self) -> List[str]:
        """Return list of tool names seen in traces."""
        return list(self.tool_index.keys())
    
    def get_module(self) -> ToolResponseModule:
        """Return the DSPy module for optimization."""
        return self.module
    
    def set_module(self, module: ToolResponseModule) -> None:
        """Set an optimized DSPy module."""
        self.module = module
    
    def get_training_examples(self) -> List[dspy.Example]:
        """
        Convert tool index to DSPy Examples for optimization.
        
        Returns:
            List of dspy.Example objects
        """
        examples = []
        for tool_name, entries in self.tool_index.items():
            for entry in entries:
                if entry["response"] is not None:
                    examples.append(dspy.Example(
                        tool_name=tool_name,
                        tool_args=json.dumps(entry["args"]),
                        examples="",
                        response=json.dumps(entry["response"])
                    ).with_inputs("tool_name", "tool_args", "examples"))
        
        return examples
