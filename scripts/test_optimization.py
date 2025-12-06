#!/usr/bin/env python3
"""
End-to-End Optimization Test

This script demonstrates the full optimization pipeline:
1. Creates baseline agent with dynamic tool catalog
2. Runs episodes to measure baseline performance
3. Optimizes using BootstrapFewShot
4. Runs episodes with optimized agent
5. Reports improvement metrics

Usage:
    python scripts/test_optimization.py
    python scripts/test_optimization.py --num-train 50 --num-episodes 5
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from dspy.teleprompt import BootstrapFewShot

from src.data import SGDLoader
from src.environment import create_sgd_environment, AgentAction
from src.environment.sgd_env import ActionType
from src.agent.sgd_agent import (
    SGDAgentModule,
    build_tool_catalog,
    extract_training_examples,
    create_metric,
)


def configure_dspy(model: str = "gemini-3-pro-preview"):
    """Configure DSPy with Gemini."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")
    
    lm = dspy.LM(f"gemini/{model}", api_key=api_key)
    dspy.configure(lm=lm)


def run_episodes(
    env,
    agent_module,
    num_episodes: int,
    max_steps: int = 8,
) -> Dict[str, Any]:
    """
    Run multiple episodes and collect metrics.
    
    Returns dict with aggregate metrics.
    """
    total_rewards = []
    avg_rewards = []
    steps_list = []
    outcomes = []
    
    for i in range(num_episodes):
        metrics = env.run_episode_with_agent(
            agent_module,
            dialogue_idx=i % env.get_num_dialogues(),
            max_steps=max_steps,
        )
        
        total_rewards.append(metrics["total_reward"])
        avg_rewards.append(metrics["avg_reward"])
        steps_list.append(metrics["steps"])
        outcomes.append(metrics["outcome"])
    
    return {
        "total_reward_mean": sum(total_rewards) / len(total_rewards),
        "total_reward_sum": sum(total_rewards),
        "avg_reward_mean": sum(avg_rewards) / len(avg_rewards),
        "steps_mean": sum(steps_list) / len(steps_list),
        "num_episodes": num_episodes,
        "outcomes": outcomes,
    }


def main():
    parser = argparse.ArgumentParser(description="Test SGD Agent Optimization")
    parser.add_argument(
        "--num-train",
        type=int,
        default=30,
        help="Number of training dialogues (default: 30)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes to run for evaluation (default: 5)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Max steps per episode (default: 6)",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-pro-preview",
        help="Gemini model to use (default: gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip optimization, just test baseline",
    )
    
    args = parser.parse_args()
    
    data_path = Path(__file__).parent.parent / "data" / "raw" / "schema_guided_dialogue"
    
    print("=" * 70)
    print("SGD AGENT OPTIMIZATION END-TO-END TEST")
    print("=" * 70)
    
    # Configure DSPy
    print(f"\n🔧 Configuring DSPy with {args.model}...")
    configure_dspy(args.model)
    
    # Load data
    print("\n📁 Loading data...")
    loader = SGDLoader(data_path)
    
    # Build tool catalog
    tool_catalog = build_tool_catalog(loader, "train")
    num_services = tool_catalog.count("##")
    print(f"  Tool catalog: {num_services} services")
    
    # Create environment
    print("\n🎮 Creating environment...")
    env = create_sgd_environment(
        str(data_path),
        split="train",
        reward_mode="ground_truth",
        max_domains=1,
        max_turns=14,
        limit=max(args.num_train, args.num_episodes * 2),
    )
    print(f"  Dialogues: {env.get_num_dialogues()}")
    
    # Create baseline agent
    print("\n🤖 Creating baseline agent...")
    baseline_agent = SGDAgentModule(tool_catalog)
    
    # Run baseline evaluation
    print(f"\n📊 Running {args.num_episodes} baseline episodes...")
    baseline_metrics = run_episodes(
        env, baseline_agent, args.num_episodes, args.max_steps
    )
    
    print(f"\n  BASELINE RESULTS:")
    print(f"    Total Reward (mean): {baseline_metrics['total_reward_mean']:.2f}")
    print(f"    Avg Reward (mean):   {baseline_metrics['avg_reward_mean']:.2f}")
    print(f"    Steps (mean):        {baseline_metrics['steps_mean']:.1f}")
    
    if args.skip_optimization:
        print("\n⏭️  Skipping optimization (--skip-optimization)")
        return
    
    # Extract training examples
    print(f"\n📚 Extracting training examples from {args.num_train} dialogues...")
    train_dialogues = loader.load_dialogues("train", limit=args.num_train)
    trainset = extract_training_examples(train_dialogues)
    print(f"  Training examples: {len(trainset)}")
    
    # Create metric
    metric_fn = create_metric()
    
    # Optimize
    print("\n⚡ Running BootstrapFewShot optimization...")
    print("  (This finds the best few-shot examples from training data)")
    
    optimizer = BootstrapFewShot(
        metric=metric_fn,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
    )
    
    # Use a subset for faster optimization
    train_subset = trainset[:min(50, len(trainset))]
    
    try:
        optimized_agent = optimizer.compile(
            baseline_agent,
            trainset=train_subset,
        )
        print("  ✅ Optimization complete!")
        
    except Exception as e:
        print(f"  ⚠️ Optimization error: {e}")
        print("  Using baseline agent for comparison...")
        optimized_agent = baseline_agent
    
    # Run optimized evaluation
    print(f"\n📊 Running {args.num_episodes} optimized episodes...")
    optimized_metrics = run_episodes(
        env, optimized_agent, args.num_episodes, args.max_steps
    )
    
    print(f"\n  OPTIMIZED RESULTS:")
    print(f"    Total Reward (mean): {optimized_metrics['total_reward_mean']:.2f}")
    print(f"    Avg Reward (mean):   {optimized_metrics['avg_reward_mean']:.2f}")
    print(f"    Steps (mean):        {optimized_metrics['steps_mean']:.1f}")
    
    # Calculate improvement
    baseline_score = baseline_metrics['total_reward_mean']
    optimized_score = optimized_metrics['total_reward_mean']
    improvement = optimized_score - baseline_score
    improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Baseline Score:   {baseline_score:.2f}")
    print(f"  Optimized Score:  {optimized_score:.2f}")
    print(f"  Improvement:      {improvement:+.2f} ({improvement_pct:+.1f}%)")
    
    if improvement > 0:
        print("\n  ✅ Optimization improved agent performance!")
    elif improvement == 0:
        print("\n  ➖ No change in performance")
    else:
        print("\n  ⚠️ Performance decreased (may need more training data)")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

