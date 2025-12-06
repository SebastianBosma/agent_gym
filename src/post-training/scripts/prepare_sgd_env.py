# src/post-training/scripts/prepare_sgd_env.py

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

# Add parent directory to path for imports when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from sgd_trace_parser import load_sgd_corpus
from env_compiler import compile_dialogue_to_env_samples, EnvSample
from jsonl_utils import write_jsonl
from google import genai

_gemini_client = None

def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client()
    return _gemini_client

def generate_rubric(system_prompt: str, user_prompt: str) -> str:
    """Generate rubric using Gemini."""
    client = get_gemini_client()
    import os
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    
    contents = f"{system_prompt}\n\n{user_prompt}"
    resp = client.models.generate_content(
        model=model,
        contents=contents,
    )
    return resp.text.strip()

def env_sample_to_dict(sample: EnvSample) -> dict:
    return {
        "id": sample.id,
        "dialogue_id": sample.dialogue_id,
        "turn_index": sample.turn_index,
        "prompt": sample.state,
        "ground_truth": sample.ground_truth,
        "rubric": sample.rubric,
        "task_description": sample.task_description,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgd_root", type=Path, required=True)
    ap.add_argument("--out_path", type=Path, required=True)
    ap.add_argument(
        "--with_rubric",
        action="store_true",
        help="If set, precompute rubrics using Gemini (costly).",
    )
    args = ap.parse_args()

    dialogues = load_sgd_corpus(args.sgd_root)

    all_samples: list[dict] = []
    for turns in dialogues:
        samples = compile_dialogue_to_env_samples(
            turns,
            generate_rubric_fn=generate_rubric if args.with_rubric else None,
        )
        all_samples.extend(env_sample_to_dict(s) for s in samples)

    write_jsonl(args.out_path, all_samples)
    print(f"Wrote {len(all_samples)} env samples to {args.out_path}")

if __name__ == "__main__":
    main()
