#!/usr/bin/env python3
"""
Quick-run script for Agent Gym examples.

Usage:
    python scripts/run_example.py [example_name]
    
Examples:
    python scripts/run_example.py                    # Run customer_service
    python scripts/run_example.py customer_service   # Run customer_service
"""

import os
import sys
import subprocess
from pathlib import Path


# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def list_examples():
    """List available examples."""
    examples = [f.stem for f in EXAMPLES_DIR.glob("*.py") if f.stem != "__init__"]
    return examples


def run_example(name: str):
    """Run a specific example."""
    example_path = EXAMPLES_DIR / f"{name}.py"
    
    if not example_path.exists():
        print(f"Error: Example '{name}' not found.")
        print(f"Available examples: {', '.join(list_examples())}")
        sys.exit(1)
    
    print(f"Running example: {name}")
    print("-" * 40)
    
    # Run the example
    subprocess.run([sys.executable, str(example_path)], cwd=str(PROJECT_ROOT))


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
    else:
        example_name = "customer_service"  # Default example
    
    run_example(example_name)


if __name__ == "__main__":
    main()

