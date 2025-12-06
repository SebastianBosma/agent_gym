#!/usr/bin/env python3
"""
SGD Environment Demo - Demonstrates using the SGD dataset for RL training.

This example shows:
1. Loading the SGD dataset
2. Creating the RL environment
3. Running a simple episode
4. Inspecting rewards and observations
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SGDLoader
from src.environment import SGDEnvironment, AgentAction, create_sgd_environment
from src.environment.sgd_env import ActionType


def main():
    # Path to SGD dataset
    data_path = Path(__file__).parent.parent / "data" / "raw" / "schema_guided_dialogue"
    
    print("=" * 80)
    print("SGD Environment Demo (Quick Test)")
    print("=" * 80)
    
    # 1. Load dataset (minimal for quick test)
    print("\n📁 Loading SGD dataset (subset for quick test)...")
    loader = SGDLoader(data_path)
    
    # Quick stats from small sample
    sample_dialogues = loader.load_dialogues("train", limit=50)
    schemas = loader.load_schemas("train")
    print(f"   Loaded {len(sample_dialogues)} sample dialogues")
    print(f"   Services available: {len(schemas)}")
    
    # 2. Create environment with simple single-domain dialogues (small for speed)
    print("\n🎮 Creating environment...")
    env = create_sgd_environment(
        data_path,
        split="train",
        max_domains=1,  # Start simple with single-domain
        max_turns=14,   # Shorter dialogues
        limit=10,       # Small subset for quick test
    )
    print(f"   Loaded {env.get_num_dialogues()} dialogues")
    
    # 3. Show available tools for first dialogue
    print("\n🔧 Available tools for first dialogue:")
    obs = env.reset(dialogue_idx=0)
    for tool in obs.available_tools[:3]:  # Show first 3
        print(f"   - {tool['name']}: {tool['description'][:60]}...")
    
    # 4. Run a demo episode
    print("\n" + "=" * 80)
    print("📍 EPISODE DEMO")
    print("=" * 80)
    
    dialogue_info = env.get_dialogue_info(0)
    print(f"\nDialogue: {dialogue_info['dialogue_id']}")
    print(f"Services: {dialogue_info['services']}")
    print(f"Intents: {dialogue_info['intents']}")
    
    obs = env.reset(dialogue_idx=0)
    total_reward = 0
    step = 0
    
    while True:
        step += 1
        print(f"\n--- Step {step} ---")
        print(f"👤 User: {obs.user_message}")
        
        # Simple rule-based agent for demo
        action = create_demo_action(obs, env)
        
        print(f"🤖 Agent: {action.response[:100]}...")
        if action.tool_call:
            print(f"   Tool: {action.tool_call['name']}({action.tool_call.get('args', {})})")
        
        # Take step
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"   Reward: {reward:.2f}")
        if info.get("ground_truth_response"):
            print(f"   GT Response: {info['ground_truth_response'][:60]}...")
        
        if done:
            print(f"\n✅ Episode complete!")
            print(f"   Total reward: {total_reward:.2f}")
            if "outcome" in info:
                print(f"   Outcome: {info['outcome']}")
            break
        
        if step >= 4:  # Limit steps for quick test
            print("\n... (truncating for quick test)")
            break
    
    # 5. Show trace conversion
    print("\n" + "=" * 80)
    print("📄 TRACE CONVERSION DEMO")
    print("=" * 80)
    
    # Load a dialogue and convert to trace
    dialogues = loader.load_dialogues("train", limit=1)
    trace = dialogues[0].to_trace_schema()
    
    print(f"\nTrace ID: {trace.conversation_id}")
    print(f"Outcome: {trace.outcome}")
    print(f"Messages: {len(trace.messages)}")
    print(f"Metadata: {trace.metadata}")
    
    print("\nFirst 4 messages:")
    for msg in trace.messages[:4]:
        role = msg.role.upper()
        content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        print(f"  [{role}] {content}")
        if msg.tool_calls:
            print(f"         Tool: {msg.tool_calls[0]['name']}")
    
    # 6. Show tool mocker in action (reuse existing mocker from env)
    print("\n" + "=" * 80)
    print("🔧 TOOL MOCKER DEMO")
    print("=" * 80)
    
    mocker = env.tool_mocker
    print(f"\nAvailable methods: {len(mocker.get_available_methods())}")
    print(f"Top 5: {mocker.get_available_methods()[:5]}")
    
    # Try a mock call
    print("\nMocking FindRestaurants(city='San Jose', cuisine='Mexican'):")
    results = mocker.call("FindRestaurants", {"city": "San Jose", "cuisine": "Mexican"})
    if results:
        print(f"   Got {len(results)} results")
        print(f"   First: {results[0].get('restaurant_name', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("✨ Demo complete!")
    print("=" * 80)


def create_demo_action(obs, env) -> AgentAction:
    """Create a simple demo action based on observation."""
    gt_turn = env.ground_truth_turn
    
    if gt_turn:
        # Use ground truth for demo
        tool_call = None
        if gt_turn.service_call:
            tool_call = {
                "name": gt_turn.service_call["method"],
                "args": gt_turn.service_call["parameters"],
            }
        
        return AgentAction(
            type=ActionType.TOOL_CALL if tool_call else ActionType.RESPONSE,
            response=gt_turn.utterance,
            tool_call=tool_call,
        )
    
    # Fallback
    return AgentAction(
        type=ActionType.RESPONSE,
        response="I can help you with that.",
    )


if __name__ == "__main__":
    main()

