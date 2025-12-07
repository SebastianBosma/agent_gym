#!/usr/bin/env python3
"""
Demo: Create RL Environment and save outputs.

This script demonstrates:
1. Creating the RL environment (simulated user, tool mocker, reward fn)
2. Generating user messages from goals
3. Scoring agent responses with the reward function
4. Saving all outputs to a JSON file

Usage:
    python scripts/demo_environment.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import create_environment
from src.agent.simulated_user import SimulatedUserSignature, InitialUserMessageSignature


def main():
    print("=" * 70)
    print("AGENT GYM - RL ENVIRONMENT DEMO")
    print("=" * 70)
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n❌ GOOGLE_API_KEY not set. Please set it:")
        print("   export GOOGLE_API_KEY=your-key-here")
        sys.exit(1)
    
    data_path = Path(__file__).parent.parent / "data" / "raw" / "schema_guided_dialogue"
    output_path = Path(__file__).parent.parent / "checkpoints" / "demo_output.json"
    
    print(f"\nData path: {data_path}")
    print(f"Output path: {output_path}")
    
    # Create environment
    print("\n" + "=" * 70)
    print("1. CREATING ENVIRONMENT")
    print("=" * 70)
    
    simulated_user, tool_mocker, reward_fn = create_environment(
        str(data_path),
        split="train",
        limit=50,  # Limit for faster loading
    )
    
    print(f"✅ Environment created!")
    print(f"   - Simulated user: {type(simulated_user).__name__}")
    print(f"   - Tool mocker: {type(tool_mocker).__name__}")
    print(f"   - Reward function: {type(reward_fn).__name__}")
    print(f"   - Number of user goals: {simulated_user.get_num_goals()}")
    
    # Get simulated user prompts (signatures)
    print("\n" + "=" * 70)
    print("2. SIMULATED USER SIGNATURES (PROMPTS)")
    print("=" * 70)
    
    initial_sig_doc = InitialUserMessageSignature.__doc__
    respond_sig_doc = SimulatedUserSignature.__doc__
    
    print("\n--- Initial Message Signature ---")
    print(initial_sig_doc)
    
    print("\n--- Response Signature ---")
    print(respond_sig_doc)
    
    # Sample some user goals
    print("\n" + "=" * 70)
    print("3. SAMPLE USER GOALS")
    print("=" * 70)
    
    sample_goals = []
    for i in range(min(3, simulated_user.get_num_goals())):
        goal = simulated_user.get_goal(i)
        goal_dict = {
            "dialogue_id": goal.dialogue_id,
            "intents": goal.intents,
            "slot_values": goal.slot_values,
            "services": goal.services,
            "prompt_string": goal.to_prompt_string(),
        }
        sample_goals.append(goal_dict)
        print(f"\nGoal {i + 1}:")
        print(f"  Intents: {goal.intents}")
        print(f"  Slots: {goal.slot_values}")
        print(f"  Prompt: {goal.to_prompt_string()}")
    
    # Generate user messages
    print("\n" + "=" * 70)
    print("4. GENERATING USER MESSAGES")
    print("=" * 70)
    
    goal = simulated_user.get_goal(0)
    
    print(f"\nUsing goal: {goal.to_prompt_string()}")
    print("\nGenerating initial message...")
    
    initial_message = simulated_user.generate_initial_message(goal)
    print(f"✅ Initial user message: \"{initial_message}\"")
    
    # Simulate agent response
    agent_response = "Sure, I can help you with that. What city are you looking for?"
    print(f"\n[Simulated agent response]: \"{agent_response}\"")
    
    print("\nGenerating user response...")
    history = f"USER: {initial_message}\nASSISTANT: {agent_response}"
    
    user_response = simulated_user.generate_response(
        goal=goal,
        history=history,
        assistant_message=agent_response,
    )
    print(f"✅ User response: \"{user_response}\"")
    
    # Score the agent response with reward function
    print("\n" + "=" * 70)
    print("5. REWARD FUNCTION (SCORING AGENT RESPONSE)")
    print("=" * 70)
    
    print("\nScoring the agent's response...")
    print(f"  User said: \"{initial_message}\"")
    print(f"  Agent replied: \"{agent_response}\"")
    
    # Get available tools for the goal's services
    available_tools = []
    for service in goal.services:
        service_tools = [t for t in tool_mocker.get_all_tool_schemas() if t.get("service") == service]
        available_tools.extend(service_tools[:3])  # Limit tools shown
    
    result = reward_fn.evaluate(
        history=[],
        user_message=initial_message,
        response=agent_response,
        available_tools=available_tools,
        tool_call=None,
    )
    
    print(f"\n✅ Reward Function Results:")
    print(f"   Quality: {result.quality}")
    print(f"   Quality Score: {result.quality_score}")
    print(f"   Tool Correct: {result.tool_correct}")
    print(f"   Tool Args Score: {result.tool_args_score}")
    print(f"   Reasoning: {result.reasoning}")
    
    # Test with a tool call
    print("\n--- Testing with tool call ---")
    
    tool_call = {
        "name": "FindRestaurants",
        "args": {"city": "San Francisco", "cuisine": "Italian"}
    }
    
    result_with_tool = reward_fn.evaluate(
        history=[{"role": "user", "content": initial_message}],
        user_message="I want Italian food in San Francisco",
        response="Let me search for Italian restaurants in San Francisco.",
        available_tools=available_tools,
        tool_call=tool_call,
    )
    
    print(f"\n✅ Reward with Tool Call:")
    print(f"   Quality: {result_with_tool.quality}")
    print(f"   Quality Score: {result_with_tool.quality_score}")
    print(f"   Tool Correct: {result_with_tool.tool_correct}")
    print(f"   Tool Args Score: {result_with_tool.tool_args_score}")
    print(f"   Reasoning: {result_with_tool.reasoning}")
    
    # Tool mocker demo
    print("\n" + "=" * 70)
    print("6. TOOL MOCKER")
    print("=" * 70)
    
    print("\nAvailable methods:", tool_mocker.get_available_methods()[:10], "...")
    
    mock_results = tool_mocker.call("FindRestaurants", {"city": "San Francisco", "cuisine": "Italian"})
    print(f"\nMocked FindRestaurants call:")
    print(f"  Results: {len(mock_results)} restaurants found")
    if mock_results:
        print(f"  Sample: {mock_results[0]}")
    
    # Save all outputs
    print("\n" + "=" * 70)
    print("7. SAVING OUTPUTS")
    print("=" * 70)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "simulated_user_type": type(simulated_user).__name__,
            "tool_mocker_type": type(tool_mocker).__name__,
            "reward_fn_type": type(reward_fn).__name__,
            "num_user_goals": simulated_user.get_num_goals(),
        },
        "simulated_user_signatures": {
            "initial_message": initial_sig_doc,
            "response": respond_sig_doc,
        },
        "sample_goals": sample_goals,
        "conversation_demo": {
            "goal": goal.to_prompt_string(),
            "initial_user_message": initial_message,
            "agent_response": agent_response,
            "user_followup": user_response,
        },
        "reward_fn_demo": {
            "without_tool_call": {
                "user_message": initial_message,
                "agent_response": agent_response,
                "quality": str(result.quality),
                "quality_score": result.quality_score,
                "tool_correct": result.tool_correct,
                "tool_args_score": result.tool_args_score,
                "reasoning": result.reasoning,
            },
            "with_tool_call": {
                "user_message": "I want Italian food in San Francisco",
                "agent_response": "Let me search for Italian restaurants in San Francisco.",
                "tool_call": tool_call,
                "quality": str(result_with_tool.quality),
                "quality_score": result_with_tool.quality_score,
                "tool_correct": result_with_tool.tool_correct,
                "tool_args_score": result_with_tool.tool_args_score,
                "reasoning": result_with_tool.reasoning,
            },
        },
        "tool_mocker_demo": {
            "available_methods": tool_mocker.get_available_methods()[:20],
            "sample_call": {
                "method": "FindRestaurants",
                "args": {"city": "San Francisco", "cuisine": "Italian"},
                "num_results": len(mock_results),
                "sample_result": mock_results[0] if mock_results else None,
            },
        },
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Output saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print(f"\nView the full output: cat {output_path}")


if __name__ == "__main__":
    main()

