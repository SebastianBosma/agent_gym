#!/usr/bin/env python3
"""
SGD Agent Optimization Script

This script optimizes the SGD agent using DSPy's teleprompt optimizers:
- BootstrapFewShot: Fast, finds good few-shot examples (~5 min)
- MIPRO: Gemini rewrites prompt instructions (~30 min)

Usage:
    python scripts/optimize_agent.py --strategy bootstrap --num-train 100
    python scripts/optimize_agent.py --strategy mipro --num-train 50
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from dspy.teleprompt import BootstrapFewShot

from src.data import SGDLoader
from src.agent.sgd_agent import (
    SGDAgentModule,
    build_tool_catalog,
    extract_training_examples,
    create_metric,
    compute_response_similarity,
    compute_tool_accuracy,
)


def configure_dspy(model: str = "gemini-2.5-flash"):
    """Configure DSPy with Gemini."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")
    
    lm = dspy.LM(f"gemini/{model}", api_key=api_key)
    dspy.configure(lm=lm)
    print(f"Configured DSPy with {model}")


def evaluate_agent(
    agent: SGDAgentModule,
    evalset: list,
    metric_fn,
    num_samples: int = None,
) -> dict:
    """
    Evaluate an agent on a dataset.
    
    Args:
        agent: The agent module to evaluate
        evalset: List of DSPy examples
        metric_fn: Metric function
        num_samples: Number of samples to evaluate (None = all)
        
    Returns:
        Dict with scores
    """
    if num_samples:
        evalset = evalset[:num_samples]
    
    scores = []
    response_scores = []
    tool_scores = []
    
    for example in evalset:
        try:
            pred = agent(
                user_message=example.user_message,
                conversation_history=example.conversation_history,
            )
            
            # Compute individual scores
            resp_score = compute_response_similarity(
                getattr(pred, "response", ""),
                example.response,
            )
            tool_score = compute_tool_accuracy(
                getattr(pred, "tool_call", "none"),
                example.tool_call,
            )
            
            response_scores.append(resp_score)
            tool_scores.append(tool_score)
            scores.append(metric_fn(example, pred))
            
        except Exception as e:
            print(f"  Error evaluating example: {e}")
            scores.append(0.0)
            response_scores.append(0.0)
            tool_scores.append(0.0)
    
    return {
        "overall": sum(scores) / len(scores) if scores else 0.0,
        "response": sum(response_scores) / len(response_scores) if response_scores else 0.0,
        "tool": sum(tool_scores) / len(tool_scores) if tool_scores else 0.0,
        "num_samples": len(scores),
    }


def optimize_bootstrap(
    agent: SGDAgentModule,
    trainset: list,
    metric_fn,
    max_demos: int = 4,
) -> SGDAgentModule:
    """
    Optimize using BootstrapFewShot.
    
    This finds the best few-shot examples from the training set.
    """
    print(f"\nRunning BootstrapFewShot optimization...")
    print(f"  Training examples: {len(trainset)}")
    print(f"  Max demos: {max_demos}")
    
    optimizer = BootstrapFewShot(
        metric=metric_fn,
        max_bootstrapped_demos=max_demos,
        max_labeled_demos=max_demos,
    )
    
    optimized = optimizer.compile(agent, trainset=trainset)
    
    print("  Optimization complete!")
    return optimized


def optimize_mipro(
    agent: SGDAgentModule,
    trainset: list,
    metric_fn,
    num_candidates: int = 10,
) -> SGDAgentModule:
    """
    Optimize using MIPRO.
    
    This uses Gemini to propose and test different prompt instructions.
    """
    try:
        from dspy.teleprompt import MIPRO
    except ImportError:
        print("MIPRO not available in this DSPy version. Using BootstrapFewShot instead.")
        return optimize_bootstrap(agent, trainset, metric_fn)
    
    print(f"\nRunning MIPRO optimization...")
    print(f"  Training examples: {len(trainset)}")
    print(f"  Num candidates: {num_candidates}")
    print("  (This may take 20-30 minutes)")
    
    optimizer = MIPRO(
        metric=metric_fn,
        num_candidates=num_candidates,
        init_temperature=1.0,
    )
    
    optimized = optimizer.compile(
        agent,
        trainset=trainset,
        num_trials=num_candidates,
    )
    
    print("  Optimization complete!")
    return optimized


def main():
    parser = argparse.ArgumentParser(description="Optimize SGD Agent")
    parser.add_argument(
        "--strategy",
        choices=["bootstrap", "mipro"],
        default="bootstrap",
        help="Optimization strategy (default: bootstrap)",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=50,
        help="Number of training dialogues to use (default: 50)",
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=20,
        help="Number of evaluation examples (default: 20)",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=4,
        help="Max few-shot demos for bootstrap (default: 4)",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=10,
        help="Number of candidates for MIPRO (default: 10)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/sgd_agent_optimized.json",
        help="Output path for optimized agent",
    )
    
    args = parser.parse_args()
    
    # Setup
    data_path = Path(__file__).parent.parent / "data" / "raw" / "schema_guided_dialogue"
    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SGD AGENT OPTIMIZATION")
    print("=" * 70)
    print(f"Strategy: {args.strategy}")
    print(f"Training dialogues: {args.num_train}")
    print(f"Evaluation examples: {args.num_eval}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    
    # Configure DSPy
    configure_dspy(args.model)
    
    # Load data
    print("\n📁 Loading data...")
    loader = SGDLoader(data_path)
    
    # Build tool catalog
    tool_catalog = build_tool_catalog(loader, "train")
    print(f"  Tool catalog: {len(tool_catalog)} characters")
    
    # Load dialogues
    train_dialogues = loader.load_dialogues("train", limit=args.num_train)
    eval_dialogues = loader.load_dialogues("dev", limit=args.num_eval)
    
    print(f"  Train dialogues: {len(train_dialogues)}")
    print(f"  Eval dialogues: {len(eval_dialogues)}")
    
    # Extract training examples
    print("\n📊 Extracting training examples...")
    trainset = extract_training_examples(train_dialogues)
    evalset = extract_training_examples(eval_dialogues)
    
    print(f"  Train examples: {len(trainset)}")
    print(f"  Eval examples: {len(evalset)}")
    
    # Create metric
    metric_fn = create_metric()
    
    # Create baseline agent
    print("\n🤖 Creating baseline agent...")
    baseline_agent = SGDAgentModule(tool_catalog)
    
    # Evaluate baseline
    print("\n📈 Evaluating baseline...")
    baseline_scores = evaluate_agent(baseline_agent, evalset, metric_fn, num_samples=10)
    print(f"  Baseline overall: {baseline_scores['overall']:.3f}")
    print(f"  Baseline response: {baseline_scores['response']:.3f}")
    print(f"  Baseline tool: {baseline_scores['tool']:.3f}")
    
    # Optimize
    print("\n" + "=" * 70)
    if args.strategy == "bootstrap":
        optimized_agent = optimize_bootstrap(
            baseline_agent,
            trainset[:100],  # Use subset for speed
            metric_fn,
            max_demos=args.max_demos,
        )
    else:
        optimized_agent = optimize_mipro(
            baseline_agent,
            trainset[:50],  # Use smaller subset for MIPRO
            metric_fn,
            num_candidates=args.num_candidates,
        )
    
    # Evaluate optimized
    print("\n📈 Evaluating optimized agent...")
    optimized_scores = evaluate_agent(optimized_agent, evalset, metric_fn, num_samples=10)
    print(f"  Optimized overall: {optimized_scores['overall']:.3f}")
    print(f"  Optimized response: {optimized_scores['response']:.3f}")
    print(f"  Optimized tool: {optimized_scores['tool']:.3f}")
    
    # Calculate improvement
    improvement = optimized_scores['overall'] - baseline_scores['overall']
    improvement_pct = (improvement / baseline_scores['overall'] * 100) if baseline_scores['overall'] > 0 else 0
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline Score:  {baseline_scores['overall']:.3f}")
    print(f"Optimized Score: {optimized_scores['overall']:.3f}")
    print(f"Improvement:     {improvement:+.3f} ({improvement_pct:+.1f}%)")
    
    # Save
    print(f"\n💾 Saving optimized agent to {args.output}...")
    try:
        optimized_agent.save(str(output_path))
        print("  Saved successfully!")
    except Exception as e:
        print(f"  Warning: Could not save module: {e}")
        # Save as JSON fallback
        fallback_path = output_path.with_suffix(".meta.json")
        with open(fallback_path, "w") as f:
            json.dump({
                "strategy": args.strategy,
                "baseline_score": baseline_scores,
                "optimized_score": optimized_scores,
                "improvement": improvement,
                "improvement_pct": improvement_pct,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        print(f"  Saved metadata to {fallback_path}")
    
    print("\n✅ Optimization complete!")


if __name__ == "__main__":
    main()

