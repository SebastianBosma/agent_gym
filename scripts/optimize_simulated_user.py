#!/usr/bin/env python3
"""
Simulated User Optimization Script

This script optimizes the SimulatedUserAgent using DSPy's teleprompt optimizers
to generate more realistic user behavior that matches real users in the SGD dataset.

Usage:
    python scripts/optimize_simulated_user.py --strategy bootstrap --num-train 100
    python scripts/optimize_simulated_user.py --strategy mipro --num-train 50
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
from src.agent.simulated_user import (
    SimulatedUserModule,
    extract_user_training_examples,
    create_user_simulation_metric,
    compute_user_simulation_similarity,
)


def configure_dspy(model: str = "gemini-2.0-flash"):
    """Configure DSPy with Gemini."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")
    
    lm = dspy.LM(f"gemini/{model}", api_key=api_key)
    dspy.configure(lm=lm)
    print(f"Configured DSPy with {model}")


def evaluate_simulated_user(
    module: SimulatedUserModule,
    evalset: list,
    metric_fn,
    num_samples: int = None,
) -> dict:
    """
    Evaluate a simulated user module on a dataset.
    
    Args:
        module: The SimulatedUserModule to evaluate
        evalset: List of DSPy examples
        metric_fn: Metric function
        num_samples: Number of samples to evaluate (None = all)
        
    Returns:
        Dict with scores
    """
    if num_samples:
        evalset = evalset[:num_samples]
    
    scores = []
    initial_scores = []
    response_scores = []
    
    for example in evalset:
        try:
            # Check if this is an initial message or response
            has_assistant_response = hasattr(example, 'assistant_response') and example.assistant_response
            
            if has_assistant_response:
                pred = module(
                    user_goal=example.user_goal,
                    conversation_history=example.conversation_history,
                    assistant_response=example.assistant_response,
                )
                score = compute_user_simulation_similarity(
                    pred.user_message,
                    example.user_message,
                )
                response_scores.append(score)
            else:
                pred = module(
                    user_goal=example.user_goal,
                )
                score = compute_user_simulation_similarity(
                    pred.user_message,
                    example.user_message,
                )
                initial_scores.append(score)
            
            scores.append(score)
            
        except Exception as e:
            print(f"  Error evaluating example: {e}")
            scores.append(0.0)
    
    return {
        "overall": sum(scores) / len(scores) if scores else 0.0,
        "initial": sum(initial_scores) / len(initial_scores) if initial_scores else 0.0,
        "response": sum(response_scores) / len(response_scores) if response_scores else 0.0,
        "num_samples": len(scores),
    }


def optimize_bootstrap(
    module: SimulatedUserModule,
    trainset: list,
    metric_fn,
    max_demos: int = 4,
) -> SimulatedUserModule:
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
    
    optimized = optimizer.compile(module, trainset=trainset)
    
    print("  Optimization complete!")
    return optimized


def optimize_mipro(
    module: SimulatedUserModule,
    trainset: list,
    metric_fn,
    num_candidates: int = 10,
) -> SimulatedUserModule:
    """
    Optimize using MIPRO.
    
    This uses Gemini to propose and test different prompt instructions.
    """
    try:
        from dspy.teleprompt import MIPRO
    except ImportError:
        print("MIPRO not available in this DSPy version. Using BootstrapFewShot instead.")
        return optimize_bootstrap(module, trainset, metric_fn)
    
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
        module,
        trainset=trainset,
        num_trials=num_candidates,
    )
    
    print("  Optimization complete!")
    return optimized


def main():
    parser = argparse.ArgumentParser(description="Optimize Simulated User Agent")
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
        default="gemini-2.0-flash",
        help="Gemini model to use (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/simulated_user.json",
        help="Output path for optimized module",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to SGD dataset (default: data/raw/schema_guided_dialogue)",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = project_root / "data" / "raw" / "schema_guided_dialogue"
    
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SIMULATED USER OPTIMIZATION")
    print("=" * 70)
    print(f"Strategy: {args.strategy}")
    print(f"Training dialogues: {args.num_train}")
    print(f"Evaluation examples: {args.num_eval}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Data path: {data_path}")
    
    # Check data exists
    if not data_path.exists():
        print(f"\n❌ Data path not found: {data_path}")
        print("Please ensure the SGD dataset is downloaded.")
        sys.exit(1)
    
    # Configure DSPy
    configure_dspy(args.model)
    
    # Load data
    print("\n📁 Loading data...")
    loader = SGDLoader(data_path)
    
    # Load dialogues
    train_dialogues = loader.load_dialogues("train", limit=args.num_train)
    eval_dialogues = loader.load_dialogues("dev", limit=args.num_eval)
    
    print(f"  Train dialogues: {len(train_dialogues)}")
    print(f"  Eval dialogues: {len(eval_dialogues)}")
    
    # Extract training examples for user simulation
    print("\n📊 Extracting user simulation examples...")
    trainset = extract_user_training_examples(train_dialogues)
    evalset = extract_user_training_examples(eval_dialogues)
    
    # Count initial vs response examples
    initial_train = sum(1 for ex in trainset if not hasattr(ex, 'assistant_response') or not ex.assistant_response)
    response_train = len(trainset) - initial_train
    
    print(f"  Train examples: {len(trainset)} ({initial_train} initial, {response_train} responses)")
    print(f"  Eval examples: {len(evalset)}")
    
    # Create metric
    metric_fn = create_user_simulation_metric()
    
    # Create baseline module
    print("\n🤖 Creating baseline simulated user...")
    baseline_module = SimulatedUserModule()
    
    # Evaluate baseline
    print("\n📈 Evaluating baseline...")
    baseline_scores = evaluate_simulated_user(baseline_module, evalset, metric_fn, num_samples=20)
    print(f"  Baseline overall: {baseline_scores['overall']:.3f}")
    print(f"  Baseline initial: {baseline_scores['initial']:.3f}")
    print(f"  Baseline response: {baseline_scores['response']:.3f}")
    
    # Optimize
    print("\n" + "=" * 70)
    if args.strategy == "bootstrap":
        optimized_module = optimize_bootstrap(
            baseline_module,
            trainset[:100],  # Use subset for speed
            metric_fn,
            max_demos=args.max_demos,
        )
    else:
        optimized_module = optimize_mipro(
            baseline_module,
            trainset[:50],  # Use smaller subset for MIPRO
            metric_fn,
            num_candidates=args.num_candidates,
        )
    
    # Evaluate optimized
    print("\n📈 Evaluating optimized simulated user...")
    optimized_scores = evaluate_simulated_user(optimized_module, evalset, metric_fn, num_samples=20)
    print(f"  Optimized overall: {optimized_scores['overall']:.3f}")
    print(f"  Optimized initial: {optimized_scores['initial']:.3f}")
    print(f"  Optimized response: {optimized_scores['response']:.3f}")
    
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
    print(f"\n💾 Saving optimized module to {args.output}...")
    try:
        optimized_module.save(str(output_path))
        print("  Saved successfully!")
    except Exception as e:
        print(f"  Warning: Could not save module: {e}")
        # Save metadata as fallback
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
    
    print("\n✅ Simulated user optimization complete!")
    print(f"\nTo use the optimized module in the environment:")
    print(f"  from src.agent import SimulatedUserAgent")
    print(f"  agent = SimulatedUserAgent()")
    print(f"  agent.load_optimized('{args.output}')")


if __name__ == "__main__":
    main()

