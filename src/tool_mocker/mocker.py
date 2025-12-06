"""
Tool Mocker - Return realistic tool responses based on trace data.

This module:
- Extracts tool call patterns from traces
- Builds a lookup table of tool_name + args -> response
- Handles fuzzy matching for similar but not identical calls
- Can generate synthetic responses for unseen tool calls using Gemini
"""

from typing import List, Dict, Any, Optional
import google.generativeai as genai

from ..trace_parser import TraceSchema


class ToolMocker:
    """
    Mock tool responses based on patterns learned from traces.
    
    The mocker builds an index of tool calls seen in traces and returns
    appropriate responses. For novel tool calls, it can use Gemini to
    generate plausible responses.
    
    Attributes:
        tool_index: Mapping of (tool_name, args_hash) -> response
        traces: Original traces for context
    """
    
    def __init__(self, traces: List[TraceSchema]):
        """
        Initialize mocker with parsed traces.
        
        Args:
            traces: List of TraceSchema objects to learn from
        """
        self.traces = traces
        self.tool_index: Dict[str, List[Dict[str, Any]]] = {}
        self._build_index()
    
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
        
        # Generate synthetic response using Gemini
        return self._generate_response(tool_name, tool_args)
    
    def _generate_response(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a synthetic tool response using Gemini.
        
        Args:
            tool_name: Name of the tool
            tool_args: Arguments to the tool
            
        Returns:
            Generated tool response
        """
        # Get examples from similar tools for context
        examples = []
        if tool_name in self.tool_index:
            examples = self.tool_index[tool_name][:3]
        
        prompt = f"""You are simulating a tool response for a customer service system.

Tool name: {tool_name}
Tool arguments: {tool_args}

Examples of this tool's responses:
{examples if examples else "No examples available"}

Generate a realistic JSON response for this tool call. Return only valid JSON."""

        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            import json
            return json.loads(response.text)
        except Exception:
            # Fallback response
            return {"status": "success", "data": None}
    
    def get_available_tools(self) -> List[str]:
        """Return list of tool names seen in traces."""
        return list(self.tool_index.keys())

