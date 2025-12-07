"""
SGD Tool Mocker - Mock service calls using SGD dialogue data.

Provides realistic API responses based on the service_results in SGD traces.
Uses fuzzy matching to find appropriate responses for similar queries.
Optionally uses LLM to generate plausible results for unseen queries.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import dspy

from ..data.sgd_loader import SGDDialogue, SGDLoader, SGDSchema


@dataclass
class ServiceCall:
    """A recorded service call with its response."""
    method: str
    parameters: Dict[str, Any]
    results: List[Dict[str, Any]]
    dialogue_id: str


class SGDToolMocker:
    """
    Mock SGD service calls using recorded responses from dialogues.
    
    Uses exact matching first, then fuzzy parameter matching,
    then falls back to returning any response for the same method.
    
    Example:
        loader = SGDLoader("data/raw/schema_guided_dialogue")
        mocker = SGDToolMocker.from_loader(loader, "train")
        
        # Mock a service call
        results = mocker.call("FindRestaurants", {"city": "Oakland", "cuisine": "American"})
    """
    
    def __init__(
        self,
        dialogues: Optional[List[SGDDialogue]] = None,
        schemas: Optional[Dict[str, SGDSchema]] = None,
    ):
        """
        Initialize mocker with dialogues and schemas.
        
        Args:
            dialogues: List of SGD dialogues to extract service calls from
            schemas: Service schemas indexed by service name
        """
        self.schemas = schemas or {}
        
        # Index: method -> list of ServiceCall
        self._call_index: Dict[str, List[ServiceCall]] = defaultdict(list)
        
        # Index: (method, param_key, param_value) -> list of ServiceCall
        self._param_index: Dict[Tuple[str, str, str], List[ServiceCall]] = defaultdict(list)
        
        if dialogues:
            self._build_index(dialogues)
    
    @classmethod
    def from_loader(
        cls,
        loader: SGDLoader,
        split: str = "train",
        **filter_kwargs,
    ) -> "SGDToolMocker":
        """
        Create mocker from an SGDLoader.
        
        Args:
            loader: SGDLoader instance
            split: Data split to use
            **filter_kwargs: Filter options for dialogues
            
        Returns:
            Configured SGDToolMocker
        """
        dialogues = loader.load_dialogues(split, **filter_kwargs)
        schemas = loader.get_service_schemas(split)
        return cls(dialogues=dialogues, schemas=schemas)
    
    def _build_index(self, dialogues: List[SGDDialogue]) -> None:
        """Build indexes from dialogues."""
        for dialogue in dialogues:
            for turn in dialogue.turns:
                if turn.service_call and turn.service_results is not None:
                    call = ServiceCall(
                        method=turn.service_call["method"],
                        parameters=turn.service_call["parameters"],
                        results=turn.service_results,
                        dialogue_id=dialogue.dialogue_id,
                    )
                    
                    # Add to method index
                    self._call_index[call.method].append(call)
                    
                    # Add to parameter index for fuzzy matching
                    for key, value in call.parameters.items():
                        str_value = str(value).lower()
                        self._param_index[(call.method, key, str_value)].append(call)
    
    def call(
        self,
        method: str,
        parameters: Dict[str, Any],
        *,
        return_all_matches: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Mock a service call and return results.
        
        Args:
            method: Service method name (e.g., "FindRestaurants")
            parameters: Call parameters
            return_all_matches: If True, return results from all matching calls
            
        Returns:
            List of result entities (may be empty)
        """
        # Try exact match first
        exact = self._exact_match(method, parameters)
        if exact:
            if return_all_matches:
                return [r for call in exact for r in call.results]
            return exact[0].results
        
        # Try fuzzy parameter matching
        fuzzy = self._fuzzy_match(method, parameters)
        if fuzzy:
            if return_all_matches:
                return [r for call in fuzzy for r in call.results]
            return fuzzy[0].results
        
        # Fall back to any call with same method
        if method in self._call_index:
            calls = self._call_index[method]
            if calls:
                return calls[0].results
        
        # No match found
        return []
    
    def _exact_match(
        self,
        method: str,
        parameters: Dict[str, Any],
    ) -> List[ServiceCall]:
        """Find exact parameter matches."""
        if method not in self._call_index:
            return []
        
        matches = []
        for call in self._call_index[method]:
            if call.parameters == parameters:
                matches.append(call)
        
        return matches
    
    def _fuzzy_match(
        self,
        method: str,
        parameters: Dict[str, Any],
        min_match_ratio: float = 0.5,
    ) -> List[ServiceCall]:
        """
        Find calls with similar parameters.
        
        Uses parameter overlap to score matches.
        """
        if method not in self._call_index:
            return []
        
        # Score each call by parameter overlap
        scored: List[Tuple[float, ServiceCall]] = []
        
        for call in self._call_index[method]:
            score = self._compute_match_score(parameters, call.parameters)
            if score >= min_match_ratio:
                scored.append((score, call))
        
        # Sort by score descending
        scored.sort(key=lambda x: -x[0])
        
        return [call for _, call in scored]
    
    def _compute_match_score(
        self,
        query_params: Dict[str, Any],
        call_params: Dict[str, Any],
    ) -> float:
        """Compute overlap score between parameter sets."""
        if not query_params and not call_params:
            return 1.0
        
        all_keys = set(query_params.keys()) | set(call_params.keys())
        if not all_keys:
            return 1.0
        
        matches = 0
        for key in all_keys:
            q_val = str(query_params.get(key, "")).lower()
            c_val = str(call_params.get(key, "")).lower()
            
            if q_val == c_val:
                matches += 1
            elif q_val and c_val and (q_val in c_val or c_val in q_val):
                matches += 0.5
        
        return matches / len(all_keys)
    
    def get_available_methods(self) -> List[str]:
        """Get list of all available service methods."""
        return list(self._call_index.keys())
    
    def get_method_stats(self) -> Dict[str, int]:
        """Get number of recorded calls per method."""
        return {method: len(calls) for method, calls in self._call_index.items()}
    
    def get_tool_schema(self, method: str) -> Optional[Dict[str, Any]]:
        """
        Get tool schema for a method from service schemas.
        
        Args:
            method: Method name
            
        Returns:
            Tool definition dict or None
        """
        for schema in self.schemas.values():
            for intent in schema.intents:
                if intent.name == method:
                    # Build tool definition
                    tools = schema.to_tool_definitions()
                    for tool in tools:
                        if tool["name"] == method:
                            return tool
        return None
    
    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas from loaded services."""
        tools = []
        for schema in self.schemas.values():
            tools.extend(schema.to_tool_definitions())
        return tools
    
    def sample_call(self, method: str) -> Optional[ServiceCall]:
        """Get a sample call for a method (useful for testing)."""
        if method in self._call_index and self._call_index[method]:
            return self._call_index[method][0]
        return None


class SGDToolMockerWithHistory(SGDToolMocker):
    """
    Extended mocker that tracks call history and supports stateful mocking.
    
    Useful for multi-turn dialogues where subsequent calls should be
    consistent with previous ones.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._call_history: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = []
        self._entity_cache: Dict[str, Dict[str, Any]] = {}
    
    def call(
        self,
        method: str,
        parameters: Dict[str, Any],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Call with history tracking."""
        results = super().call(method, parameters, **kwargs)
        
        # Track call
        self._call_history.append((method, parameters, results))
        
        # Cache entities for consistency
        for result in results:
            # Use a unique identifier if available
            for key in ["restaurant_name", "hotel_name", "movie_name", "event_name"]:
                if key in result:
                    self._entity_cache[result[key]] = result
                    break
        
        return results
    
    def get_cached_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a previously returned entity by name."""
        return self._entity_cache.get(name)
    
    def get_call_history(self) -> List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
        """Get history of all calls made."""
        return self._call_history.copy()
    
    def reset(self) -> None:
        """Clear call history and entity cache."""
        self._call_history.clear()
        self._entity_cache.clear()


# =============================================================================
# LLM-Powered Tool Mocker
# =============================================================================

class GenerateToolResultSignature(dspy.Signature):
    """Generate realistic API results for a service call.
    
    You are simulating an API backend. Generate plausible, realistic results
    that match the query parameters. The results should be consistent with
    what a real API would return.
    """
    
    method_name: str = dspy.InputField(
        desc="The API method being called (e.g., 'FindRestaurants', 'SearchHotels')"
    )
    parameters: str = dspy.InputField(
        desc="The call parameters as JSON (e.g., {'city': 'Tokyo', 'cuisine': 'Japanese'})"
    )
    result_schema: str = dspy.InputField(
        desc="Expected fields in each result (e.g., 'restaurant_name, address, phone, cuisine, price_range')"
    )
    num_results: int = dspy.InputField(
        desc="Number of results to generate (typically 1-5)"
    )
    
    results_json: str = dspy.OutputField(
        desc="JSON array of result objects matching the schema and query"
    )


class LLMToolMocker(SGDToolMockerWithHistory):
    """
    Tool mocker that uses LLM to generate plausible results for unseen queries.
    
    Falls back to LLM generation when:
    1. No exact match found
    2. No fuzzy match found
    3. No fallback results available for the method
    
    This ensures the agent always receives semantically appropriate responses.
    
    Example:
        mocker = LLMToolMocker.from_loader(loader, "train", use_llm_fallback=True)
        
        # Even for unseen parameters, returns plausible results
        results = mocker.call("FindRestaurants", {"city": "Tokyo", "cuisine": "Sushi"})
        # Returns: [{"restaurant_name": "Sushi Dai", "city": "Tokyo", ...}]
    """
    
    def __init__(
        self,
        *args,
        use_llm_fallback: bool = True,
        llm_model: str = "gemini-2.0-flash",
        cache_llm_results: bool = True,
        **kwargs,
    ):
        """
        Initialize LLM-powered tool mocker.
        
        Args:
            use_llm_fallback: Whether to use LLM for generating results
            llm_model: Model to use for generation
            cache_llm_results: Cache LLM-generated results for reuse
            *args, **kwargs: Passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.use_llm_fallback = use_llm_fallback
        self.llm_model = llm_model
        self.cache_llm_results = cache_llm_results
        
        self._llm_configured = False
        self._generator = dspy.Predict(GenerateToolResultSignature)
        self._llm_cache: Dict[str, List[Dict[str, Any]]] = {}
    
    @classmethod
    def from_loader(
        cls,
        loader: SGDLoader,
        split: str = "train",
        use_llm_fallback: bool = True,
        **filter_kwargs,
    ) -> "LLMToolMocker":
        """Create LLM-powered mocker from an SGDLoader."""
        dialogues = loader.load_dialogues(split, **filter_kwargs)
        schemas = loader.get_service_schemas(split)
        return cls(
            dialogues=dialogues,
            schemas=schemas,
            use_llm_fallback=use_llm_fallback,
        )
    
    def _ensure_llm_configured(self):
        """Configure DSPy with LLM on first use."""
        if not self._llm_configured:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set - required for LLM fallback")
            
            lm = dspy.LM(f"gemini/{self.llm_model}", api_key=api_key)
            dspy.configure(lm=lm)
            self._llm_configured = True
    
    def call(
        self,
        method: str,
        parameters: Dict[str, Any],
        *,
        return_all_matches: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Mock a service call, using LLM fallback if needed.
        
        Priority:
        1. Exact match from recorded data
        2. Fuzzy match from recorded data
        3. LLM-generated results (if enabled)
        4. Any recorded result for the method
        5. Empty list
        """
        # Try exact match first
        exact = self._exact_match(method, parameters)
        if exact:
            results = exact[0].results if not return_all_matches else [r for call in exact for r in call.results]
            self._track_call(method, parameters, results)
            return results
        
        # Try fuzzy parameter matching (require high match - 80%+)
        fuzzy = self._fuzzy_match(method, parameters, min_match_ratio=0.8)
        if fuzzy:
            results = fuzzy[0].results if not return_all_matches else [r for call in fuzzy for r in call.results]
            self._track_call(method, parameters, results)
            return results
        
        # Try LLM fallback for partial/no matches
        if self.use_llm_fallback:
            llm_results = self._generate_with_llm(method, parameters)
            if llm_results:
                self._track_call(method, parameters, llm_results)
                return llm_results
        
        # Fall back to lower threshold fuzzy match
        fuzzy_low = self._fuzzy_match(method, parameters, min_match_ratio=0.5)
        if fuzzy_low:
            results = fuzzy_low[0].results
            self._track_call(method, parameters, results)
            return results
        
        # Fall back to any call with same method
        if method in self._call_index:
            calls = self._call_index[method]
            if calls:
                results = calls[0].results
                self._track_call(method, parameters, results)
                return results
        
        # No match found
        self._track_call(method, parameters, [])
        return []
    
    def _track_call(self, method: str, parameters: Dict[str, Any], results: List[Dict[str, Any]]):
        """Track call in history and cache entities."""
        self._call_history.append((method, parameters, results))
        
        for result in results:
            for key in ["restaurant_name", "hotel_name", "movie_name", "event_name"]:
                if key in result:
                    self._entity_cache[result[key]] = result
                    break
    
    def _generate_with_llm(
        self,
        method: str,
        parameters: Dict[str, Any],
        num_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate plausible results using LLM."""
        # Check cache first
        cache_key = f"{method}:{json.dumps(parameters, sort_keys=True)}"
        if self.cache_llm_results and cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        
        # Get schema for this method to know expected fields
        result_schema = self._get_result_schema(method)
        if not result_schema:
            return []
        
        try:
            self._ensure_llm_configured()
            
            result = self._generator(
                method_name=method,
                parameters=json.dumps(parameters),
                result_schema=result_schema,
                num_results=num_results,
            )
            
            # Parse JSON response
            results_json = result.results_json.strip()
            
            # Handle markdown code blocks
            if results_json.startswith("```"):
                lines = results_json.split("\n")
                results_json = "\n".join(lines[1:-1])
            
            results = json.loads(results_json)
            
            # Ensure it's a list
            if isinstance(results, dict):
                results = [results]
            
            # Ensure parameters are reflected in results
            for r in results:
                for key, value in parameters.items():
                    if key in r or key.replace("_", "") in [k.replace("_", "") for k in r.keys()]:
                        r[key] = value
            
            # Cache results
            if self.cache_llm_results:
                self._llm_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            # Silently fail and let caller use fallback
            return []
    
    def _get_result_schema(self, method: str) -> Optional[str]:
        """Get expected result fields for a method."""
        # Try to get from tool schema
        tool_schema = self.get_tool_schema(method)
        if tool_schema and "result_slots" in tool_schema:
            return ", ".join(tool_schema["result_slots"])
        
        # Infer from existing results
        if method in self._call_index and self._call_index[method]:
            sample = self._call_index[method][0].results
            if sample:
                return ", ".join(sample[0].keys())
        
        # Common schemas by method name pattern
        if "Restaurant" in method:
            return "restaurant_name, address, city, cuisine, phone_number, price_range, rating"
        elif "Hotel" in method:
            return "hotel_name, address, city, star_rating, price_per_night, phone_number"
        elif "Flight" in method:
            return "airline, flight_number, departure_time, arrival_time, price, origin, destination"
        elif "Movie" in method:
            return "movie_name, genre, director, rating, duration, theater_name"
        elif "Event" in method:
            return "event_name, venue, date, time, price, category"
        
        return None
    
    def clear_llm_cache(self):
        """Clear the LLM result cache."""
        self._llm_cache.clear()

