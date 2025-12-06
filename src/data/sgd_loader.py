"""
SGD Dataset Loader - Load and convert Schema-Guided Dialogue data.

This module handles:
- Loading raw SGD dialogues and schemas
- Converting to TraceSchema format
- Filtering by domain, complexity, etc.
- Extracting tool/service definitions
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Literal
from dataclasses import dataclass, field
from enum import Enum

from ..trace_parser import TraceSchema, Message


class Split(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


@dataclass
class SGDSlot:
    """A slot definition from SGD schema."""
    name: str
    description: str
    is_categorical: bool
    possible_values: List[str]


@dataclass
class SGDIntent:
    """An intent definition from SGD schema."""
    name: str
    description: str
    is_transactional: bool
    required_slots: List[str]
    optional_slots: Dict[str, Any]
    result_slots: List[str]


@dataclass
class SGDSchema:
    """Schema for a single SGD service/API."""
    service_name: str
    description: str
    slots: List[SGDSlot]
    intents: List[SGDIntent]
    
    def to_tool_definitions(self) -> List[Dict[str, Any]]:
        """Convert to tool definitions for the agent."""
        tools = []
        for intent in self.intents:
            # Build parameters schema
            properties = {}
            required = []
            
            for slot_name in intent.required_slots:
                slot = next((s for s in self.slots if s.name == slot_name), None)
                if slot:
                    prop = {"type": "string", "description": slot.description}
                    if slot.is_categorical and slot.possible_values:
                        prop["enum"] = slot.possible_values
                    properties[slot_name] = prop
                    required.append(slot_name)
            
            for slot_name, default in intent.optional_slots.items():
                slot = next((s for s in self.slots if s.name == slot_name), None)
                if slot:
                    prop = {"type": "string", "description": slot.description}
                    if slot.is_categorical and slot.possible_values:
                        prop["enum"] = slot.possible_values
                    if default is not None:
                        prop["default"] = default
                    properties[slot_name] = prop
            
            tools.append({
                "name": intent.name,
                "description": intent.description,
                "service": self.service_name,
                "is_transactional": intent.is_transactional,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
                "result_slots": intent.result_slots,
            })
        
        return tools


@dataclass
class SGDTurn:
    """A single turn in an SGD dialogue."""
    speaker: str  # "USER" or "SYSTEM"
    utterance: str
    frames: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def service_call(self) -> Optional[Dict[str, Any]]:
        """Get service call if present in any frame."""
        for frame in self.frames:
            if "service_call" in frame:
                return frame["service_call"]
        return None
    
    @property
    def service_results(self) -> Optional[List[Dict[str, Any]]]:
        """Get service results if present in any frame."""
        for frame in self.frames:
            if "service_results" in frame:
                return frame["service_results"]
        return None
    
    @property
    def state(self) -> Optional[Dict[str, Any]]:
        """Get dialogue state (user turns only)."""
        for frame in self.frames:
            if "state" in frame:
                return frame["state"]
        return None
    
    @property
    def actions(self) -> List[Dict[str, Any]]:
        """Get all actions from all frames."""
        all_actions = []
        for frame in self.frames:
            all_actions.extend(frame.get("actions", []))
        return all_actions


@dataclass
class SGDDialogue:
    """A complete SGD dialogue."""
    dialogue_id: str
    services: List[str]
    turns: List[SGDTurn]
    
    @property
    def num_domains(self) -> int:
        return len(self.services)
    
    @property
    def is_multi_domain(self) -> bool:
        return self.num_domains > 1
    
    @property
    def num_service_calls(self) -> int:
        return sum(1 for t in self.turns if t.service_call is not None)
    
    @property
    def intents(self) -> List[str]:
        """Extract all intents from the dialogue."""
        intents = []
        for turn in self.turns:
            if turn.state and turn.state.get("active_intent"):
                intent = turn.state["active_intent"]
                if intent != "NONE" and intent not in intents:
                    intents.append(intent)
        return intents
    
    def to_trace_schema(self) -> TraceSchema:
        """Convert to TraceSchema format."""
        messages = []
        
        for turn in self.turns:
            if turn.speaker == "USER":
                messages.append(Message(
                    role="user",
                    content=turn.utterance,
                ))
            else:  # SYSTEM
                # Check if there's a service call
                tool_calls = None
                if turn.service_call:
                    tool_calls = [{
                        "name": turn.service_call["method"],
                        "args": turn.service_call["parameters"],
                    }]
                
                messages.append(Message(
                    role="assistant",
                    content=turn.utterance,
                    tool_calls=tool_calls,
                ))
                
                # Add tool response message if there was a service call
                if turn.service_call and turn.service_results is not None:
                    messages.append(Message(
                        role="tool",
                        content="",
                        name=turn.service_call["method"],
                        result=turn.service_results,
                    ))
        
        # Determine outcome based on final actions
        outcome = self._determine_outcome()
        
        return TraceSchema(
            conversation_id=f"sgd_{self.dialogue_id}",
            messages=messages,
            outcome=outcome,
            metadata={
                "source": "sgd",
                "services": self.services,
                "intents": self.intents,
                "num_service_calls": self.num_service_calls,
                "is_multi_domain": self.is_multi_domain,
            },
        )
    
    def _determine_outcome(self) -> Optional[str]:
        """Determine dialogue outcome from actions."""
        # Check last few turns for outcome signals
        for turn in reversed(self.turns[-4:]):
            for action in turn.actions:
                act = action.get("act", "")
                if act == "NOTIFY_SUCCESS":
                    return "resolved"
                elif act == "NOTIFY_FAILURE":
                    return "failed"
                elif act == "GOODBYE":
                    continue  # Keep looking
        
        # Check if any transaction was successful
        for turn in self.turns:
            for action in turn.actions:
                if action.get("act") == "NOTIFY_SUCCESS":
                    return "resolved"
        
        return "completed"  # Default if conversation ended normally


class SGDLoader:
    """
    Load and manage SGD dataset.
    
    Example usage:
        loader = SGDLoader("data/raw/schema_guided_dialogue")
        
        # Load all training dialogues
        dialogues = loader.load_dialogues("train")
        
        # Convert to traces
        traces = loader.to_traces("train")
        
        # Filter by complexity
        simple = loader.load_dialogues("train", max_domains=1)
        complex = loader.load_dialogues("train", min_domains=2)
    """
    
    def __init__(self, base_path: str | Path):
        """
        Initialize loader with path to SGD dataset.
        
        Args:
            base_path: Path to the schema_guided_dialogue directory
        """
        self.base_path = Path(base_path)
        self._schema_cache: Dict[str, List[SGDSchema]] = {}
        self._dialogue_cache: Dict[str, List[SGDDialogue]] = {}
    
    def load_schemas(self, split: str = "train") -> List[SGDSchema]:
        """
        Load service schemas for a split.
        
        Args:
            split: One of "train", "dev", "test"
            
        Returns:
            List of SGDSchema objects
        """
        if split in self._schema_cache:
            return self._schema_cache[split]
        
        schema_file = self.base_path / split / "schema.json"
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        with open(schema_file) as f:
            raw_schemas = json.load(f)
        
        schemas = []
        for raw in raw_schemas:
            slots = [
                SGDSlot(
                    name=s["name"],
                    description=s["description"],
                    is_categorical=s["is_categorical"],
                    possible_values=s.get("possible_values", []),
                )
                for s in raw.get("slots", [])
            ]
            
            intents = [
                SGDIntent(
                    name=i["name"],
                    description=i["description"],
                    is_transactional=i["is_transactional"],
                    required_slots=i.get("required_slots", []),
                    optional_slots=i.get("optional_slots", {}),
                    result_slots=i.get("result_slots", []),
                )
                for i in raw.get("intents", [])
            ]
            
            schemas.append(SGDSchema(
                service_name=raw["service_name"],
                description=raw["description"],
                slots=slots,
                intents=intents,
            ))
        
        self._schema_cache[split] = schemas
        return schemas
    
    def load_dialogues(
        self,
        split: str = "train",
        *,
        min_domains: int = 1,
        max_domains: Optional[int] = None,
        min_turns: int = 1,
        max_turns: Optional[int] = None,
        services: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[SGDDialogue]:
        """
        Load dialogues with optional filtering.
        
        Args:
            split: One of "train", "dev", "test"
            min_domains: Minimum number of domains/services
            max_domains: Maximum number of domains/services
            min_turns: Minimum number of turns
            max_turns: Maximum number of turns
            services: Only include dialogues using these services
            limit: Maximum number of dialogues to return
            
        Returns:
            List of SGDDialogue objects
        """
        # Load all dialogues if not cached
        if split not in self._dialogue_cache:
            self._dialogue_cache[split] = self._load_all_dialogues(split)
        
        dialogues = self._dialogue_cache[split]
        
        # Apply filters
        filtered = []
        for d in dialogues:
            # Domain filter
            if d.num_domains < min_domains:
                continue
            if max_domains is not None and d.num_domains > max_domains:
                continue
            
            # Turn filter
            if len(d.turns) < min_turns:
                continue
            if max_turns is not None and len(d.turns) > max_turns:
                continue
            
            # Service filter
            if services is not None:
                if not any(s in d.services for s in services):
                    continue
            
            filtered.append(d)
            
            if limit is not None and len(filtered) >= limit:
                break
        
        return filtered
    
    def _load_all_dialogues(self, split: str) -> List[SGDDialogue]:
        """Load all dialogues from a split."""
        split_dir = self.base_path / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        dialogues = []
        for json_file in sorted(split_dir.glob("dialogues_*.json")):
            with open(json_file) as f:
                raw_dialogues = json.load(f)
            
            for raw in raw_dialogues:
                turns = [
                    SGDTurn(
                        speaker=t["speaker"],
                        utterance=t["utterance"],
                        frames=t.get("frames", []),
                    )
                    for t in raw["turns"]
                ]
                
                dialogues.append(SGDDialogue(
                    dialogue_id=raw["dialogue_id"],
                    services=raw["services"],
                    turns=turns,
                ))
        
        return dialogues
    
    def to_traces(
        self,
        split: str = "train",
        **filter_kwargs,
    ) -> List[TraceSchema]:
        """
        Load dialogues and convert to TraceSchema format.
        
        Args:
            split: One of "train", "dev", "test"
            **filter_kwargs: Passed to load_dialogues
            
        Returns:
            List of TraceSchema objects
        """
        dialogues = self.load_dialogues(split, **filter_kwargs)
        return [d.to_trace_schema() for d in dialogues]
    
    def get_tool_definitions(self, split: str = "train") -> List[Dict[str, Any]]:
        """
        Get all tool definitions from schemas.
        
        Args:
            split: One of "train", "dev", "test"
            
        Returns:
            List of tool definition dicts
        """
        schemas = self.load_schemas(split)
        tools = []
        for schema in schemas:
            tools.extend(schema.to_tool_definitions())
        return tools
    
    def get_service_schemas(self, split: str = "train") -> Dict[str, SGDSchema]:
        """Get schemas indexed by service name."""
        schemas = self.load_schemas(split)
        return {s.service_name: s for s in schemas}
    
    def iter_dialogues(
        self,
        split: str = "train",
        **filter_kwargs,
    ) -> Iterator[SGDDialogue]:
        """
        Iterate over dialogues without loading all into memory.
        
        Args:
            split: One of "train", "dev", "test"
            **filter_kwargs: Filtering options
            
        Yields:
            SGDDialogue objects
        """
        for dialogue in self.load_dialogues(split, **filter_kwargs):
            yield dialogue
    
    def get_stats(self, split: str = "train") -> Dict[str, Any]:
        """Get statistics for a split."""
        dialogues = self.load_dialogues(split)
        schemas = self.load_schemas(split)
        
        num_turns = [len(d.turns) for d in dialogues]
        num_domains = [d.num_domains for d in dialogues]
        num_calls = [d.num_service_calls for d in dialogues]
        
        return {
            "num_dialogues": len(dialogues),
            "num_services": len(schemas),
            "total_turns": sum(num_turns),
            "avg_turns": sum(num_turns) / len(dialogues) if dialogues else 0,
            "multi_domain_pct": sum(1 for d in dialogues if d.is_multi_domain) / len(dialogues) * 100 if dialogues else 0,
            "avg_service_calls": sum(num_calls) / len(dialogues) if dialogues else 0,
            "services": [s.service_name for s in schemas],
        }


def load_sgd_traces(
    base_path: str | Path,
    split: str = "train",
    **kwargs,
) -> List[TraceSchema]:
    """
    Convenience function to load SGD data as traces.
    
    Args:
        base_path: Path to SGD dataset
        split: One of "train", "dev", "test"
        **kwargs: Filter options passed to SGDLoader.to_traces
        
    Returns:
        List of TraceSchema objects
    """
    loader = SGDLoader(base_path)
    return loader.to_traces(split, **kwargs)

