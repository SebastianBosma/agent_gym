from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sgd_grpo_demo.gemini_judge import GeminiJudge, RewardCache


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # Try to return “new” part only
    if decoded.startswith(prompt):
        return decoded[len(prompt):].strip()
    return decoded.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_jsonl", type=str, default="data/processed/eval.with_rubric.jsonl")
    ap.add_argument("--base_model", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--trained_model_dir", type=str, required=True)
    ap.add_argument("--gemini_model", type=str, default=os.getenv("GEMINI_MODEL", "gemini-3-pro-preview"))
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Missing GEMINI_API_KEY env var.")

    ds = load_dataset("json", data_files={"eval": args.eval_jsonl}, split="eval")
    ds = ds.select(range(min(args.n, len(ds))))

    # Load base + trained
    base_tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if base_tok.pad_token_id is None:
        base_tok.pad_token = base_tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    tr_tok = AutoTokenizer.from_pretrained(args.trained_model_dir, use_fast=True)
    if tr_tok.pad_token_id is None:
        tr_tok.pad_token = tr_tok.eos_token
    trained = AutoModelForCausalLM.from_pretrained(
        args.trained_model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    cache_path = str(Path(args.trained_model_dir) / "eval_rewards.cache.sqlite")
    judge = GeminiJudge(model=args.gemini_model, cache=RewardCache(cache_path))

    base_scores = []
    trained_scores = []

    for row in ds:
        prompt = row["prompt"]
        gt = row["ground_truth"]
        rubric = row.get("rubric", "- Be helpful\n- Be accurate\n- Ask for missing required info\n- Keep it concise")

        b = generate_one(base, base_tok, prompt)
        t = generate_one(trained, tr_tok, prompt)

        bs = judge.score_one(prompt, b, rubric, gt)
        ts = judge.score_one(prompt, t, rubric, gt)

        base_scores.append(bs)
        trained_scores.append(ts)

    def summarize(xs):
        return {
            "mean": float(statistics.mean(xs)),
            "stdev": float(statistics.pstdev(xs)),
            "min": float(min(xs)),
            "max": float(max(xs)),
        }

    print("\nBase scores:", summarize(base_scores))
    print("Trained scores:", summarize(trained_scores))
    print(f"Δ mean: {statistics.mean(trained_scores) - statistics.mean(base_scores):.4f}")


if __name__ == "__main__":
    main()
