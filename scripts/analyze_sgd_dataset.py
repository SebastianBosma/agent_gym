#!/usr/bin/env python3
"""
Analyze the Schema-Guided Dialogue (SGD) dataset for trace conversion.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import statistics


def load_dialogues(split_dir: Path) -> list[dict]:
    """Load all dialogues from a split directory."""
    dialogues = []
    for json_file in sorted(split_dir.glob("dialogues_*.json")):
        with open(json_file) as f:
            dialogues.extend(json.load(f))
    return dialogues


def load_schema(split_dir: Path) -> list[dict]:
    """Load schema for a split."""
    schema_file = split_dir / "schema.json"
    if schema_file.exists():
        with open(schema_file) as f:
            return json.load(f)
    return []


def analyze_dataset(base_path: Path):
    """Analyze the SGD dataset and print statistics."""
    
    splits = ["train", "dev", "test"]
    all_stats = {}
    
    for split in splits:
        split_dir = base_path / split
        if not split_dir.exists():
            continue
            
        dialogues = load_dialogues(split_dir)
        schema = load_schema(split_dir)
        
        stats = analyze_split(dialogues, schema, split)
        all_stats[split] = stats
        
    print_summary(all_stats)
    return all_stats


def analyze_split(dialogues: list[dict], schema: list[dict], split_name: str) -> dict:
    """Analyze a single split of the dataset."""
    
    stats = {
        "num_dialogues": len(dialogues),
        "num_turns": [],
        "num_user_turns": [],
        "num_system_turns": [],
        "services_per_dialogue": [],
        "service_calls": [],
        "service_results": [],
        "user_actions": Counter(),
        "system_actions": Counter(),
        "intents": Counter(),
        "slots_filled": [],
        "unique_services": set(),
        "multi_domain_dialogues": 0,
        "dialogues_with_service_calls": 0,
        "utterance_lengths": {"user": [], "system": []},
    }
    
    for dialogue in dialogues:
        services = dialogue.get("services", [])
        stats["services_per_dialogue"].append(len(services))
        stats["unique_services"].update(services)
        
        if len(services) > 1:
            stats["multi_domain_dialogues"] += 1
        
        turns = dialogue.get("turns", [])
        stats["num_turns"].append(len(turns))
        
        user_turns = 0
        system_turns = 0
        has_service_call = False
        
        for turn in turns:
            speaker = turn.get("speaker", "")
            utterance = turn.get("utterance", "")
            
            if speaker == "USER":
                user_turns += 1
                stats["utterance_lengths"]["user"].append(len(utterance.split()))
            else:
                system_turns += 1
                stats["utterance_lengths"]["system"].append(len(utterance.split()))
            
            for frame in turn.get("frames", []):
                # Count actions
                for action in frame.get("actions", []):
                    act = action.get("act", "")
                    if speaker == "USER":
                        stats["user_actions"][act] += 1
                        if act == "INFORM_INTENT":
                            for val in action.get("values", []):
                                stats["intents"][val] += 1
                    else:
                        stats["system_actions"][act] += 1
                
                # Count service calls (system turns only)
                if "service_call" in frame:
                    has_service_call = True
                    stats["service_calls"].append(frame["service_call"])
                
                # Count service results
                if "service_results" in frame:
                    results = frame["service_results"]
                    stats["service_results"].append(len(results))
                
                # Count slots in state (user turns only)
                state = frame.get("state", {})
                slot_values = state.get("slot_values", {})
                stats["slots_filled"].append(len(slot_values))
        
        stats["num_user_turns"].append(user_turns)
        stats["num_system_turns"].append(system_turns)
        
        if has_service_call:
            stats["dialogues_with_service_calls"] += 1
    
    return stats


def print_summary(all_stats: dict):
    """Print a formatted summary of the analysis."""
    
    print("=" * 80)
    print("SCHEMA-GUIDED DIALOGUE (SGD) DATASET ANALYSIS")
    print("=" * 80)
    
    # Overall stats
    total_dialogues = sum(s["num_dialogues"] for s in all_stats.values())
    total_turns = sum(sum(s["num_turns"]) for s in all_stats.values())
    all_services = set()
    for s in all_stats.values():
        all_services.update(s["unique_services"])
    
    print(f"\n📊 OVERALL STATISTICS")
    print("-" * 40)
    print(f"Total dialogues:     {total_dialogues:,}")
    print(f"Total turns:         {total_turns:,}")
    print(f"Unique services:     {len(all_services)}")
    print(f"Services: {sorted(all_services)}")
    
    # Per-split stats
    print(f"\n📁 PER-SPLIT BREAKDOWN")
    print("-" * 40)
    print(f"{'Split':<10} {'Dialogues':>12} {'Turns':>12} {'Avg Turns':>12} {'Multi-domain':>14}")
    print("-" * 60)
    
    for split, stats in all_stats.items():
        avg_turns = statistics.mean(stats["num_turns"]) if stats["num_turns"] else 0
        print(f"{split:<10} {stats['num_dialogues']:>12,} {sum(stats['num_turns']):>12,} {avg_turns:>12.1f} {stats['multi_domain_dialogues']:>14,}")
    
    # Turn statistics
    print(f"\n🔄 TURN STATISTICS")
    print("-" * 40)
    
    for split, stats in all_stats.items():
        if not stats["num_turns"]:
            continue
        print(f"\n{split.upper()}:")
        print(f"  Turns per dialogue:  min={min(stats['num_turns'])}, max={max(stats['num_turns'])}, avg={statistics.mean(stats['num_turns']):.1f}")
        print(f"  User turns/dialogue: avg={statistics.mean(stats['num_user_turns']):.1f}")
        print(f"  System turns/dialogue: avg={statistics.mean(stats['num_system_turns']):.1f}")
    
    # Action types
    print(f"\n🎯 USER ACTION TYPES (across all splits)")
    print("-" * 40)
    user_actions = Counter()
    for stats in all_stats.values():
        user_actions.update(stats["user_actions"])
    for act, count in user_actions.most_common():
        print(f"  {act:<20} {count:>8,}")
    
    print(f"\n🤖 SYSTEM ACTION TYPES (across all splits)")
    print("-" * 40)
    system_actions = Counter()
    for stats in all_stats.values():
        system_actions.update(stats["system_actions"])
    for act, count in system_actions.most_common():
        print(f"  {act:<20} {count:>8,}")
    
    # Service calls (tool usage)
    print(f"\n🔧 SERVICE CALLS (Tool Usage)")
    print("-" * 40)
    total_service_calls = sum(len(s["service_calls"]) for s in all_stats.values())
    total_with_calls = sum(s["dialogues_with_service_calls"] for s in all_stats.values())
    print(f"Total service calls: {total_service_calls:,}")
    print(f"Dialogues with service calls: {total_with_calls:,} ({100*total_with_calls/total_dialogues:.1f}%)")
    
    # Analyze service call methods
    method_counts = Counter()
    for stats in all_stats.values():
        for call in stats["service_calls"]:
            method_counts[call.get("method", "unknown")] += 1
    
    print(f"\nTop 15 service methods (intents):")
    for method, count in method_counts.most_common(15):
        print(f"  {method:<30} {count:>6,}")
    
    # Intent distribution
    print(f"\n💡 TOP 20 INTENTS")
    print("-" * 40)
    all_intents = Counter()
    for stats in all_stats.values():
        all_intents.update(stats["intents"])
    for intent, count in all_intents.most_common(20):
        print(f"  {intent:<30} {count:>6,}")
    
    # Utterance lengths
    print(f"\n📝 UTTERANCE LENGTHS (words)")
    print("-" * 40)
    for split, stats in all_stats.items():
        user_lens = stats["utterance_lengths"]["user"]
        sys_lens = stats["utterance_lengths"]["system"]
        if user_lens and sys_lens:
            print(f"{split}: User avg={statistics.mean(user_lens):.1f}, System avg={statistics.mean(sys_lens):.1f}")
    
    # Trace conversion recommendations
    print(f"\n" + "=" * 80)
    print("📋 TRACE CONVERSION RECOMMENDATIONS")
    print("=" * 80)
    
    print("""
PROPOSED TRACE STRUCTURE:
─────────────────────────
Each dialogue can be converted to a trace with the following mapping:

1. TRACE = 1 DIALOGUE
   - dialogue_id → trace_id
   - services → available_tools/APIs
   
2. TURN MAPPING:
   - USER turn → observation/user_input
   - SYSTEM turn → agent_response + optional tool_call
   
3. TOOL CALLS:
   - service_call.method → tool_name
   - service_call.parameters → tool_args
   - service_results → tool_response
   
4. STATE TRACKING:
   - state.active_intent → current_goal
   - state.slot_values → extracted_entities
   - state.requested_slots → pending_info_requests

5. ACTIONS (for reward/training):
   - User actions (INFORM, REQUEST, etc.) → ground truth signals
   - System actions → agent action labels

TRACE COMPLEXITY LEVELS:
────────────────────────
- Simple: Single-domain, no service calls
- Medium: Single-domain with service calls
- Complex: Multi-domain with multiple service calls

SUGGESTED SPLITS FOR TRAINING:
──────────────────────────────
- Use 'train' for main training
- Use 'dev' for validation/prompt optimization
- Use 'test' for final evaluation (has unseen domains!)
""")
    
    # Service results analysis
    print(f"\n📊 SERVICE RESULTS (API Responses)")
    print("-" * 40)
    all_results = []
    for stats in all_stats.values():
        all_results.extend(stats["service_results"])
    if all_results:
        print(f"Total API calls with results: {len(all_results):,}")
        print(f"Results per call: avg={statistics.mean(all_results):.1f}, max={max(all_results)}")
        zero_results = sum(1 for r in all_results if r == 0)
        print(f"Calls with zero results: {zero_results:,} ({100*zero_results/len(all_results):.1f}%)")


def main():
    base_path = Path(__file__).parent.parent / "data" / "raw" / "schema_guided_dialogue"
    
    if not base_path.exists():
        print(f"Dataset not found at {base_path}")
        return
    
    analyze_dataset(base_path)


if __name__ == "__main__":
    main()

