#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from gemini_judge import GeminiJudge, RewardCache
from reward_fn import make_gemini_reward_func

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-step GRPO training: Gemma + Gemini reward on SGD prompts"
    )
    parser.add_argument(
        "--train_jsonl",
        type=str,
        default="data/processed/train.with_rubric.jsonl",
        help="JSONL file with at least: prompt, ground_truth, optional: rubric",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b-it",
        help="Base Gemma model to fine-tune.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Where to save trained model (default: runs/grpo_TIMESTAMP)",
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model name for reward (judge).",
    )

    # Training hyperparams (keep small for demos)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA for cheaper tuning
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # --------- sanity checks & paths ----------
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Missing GEMINI_API_KEY env var. Put it in .env and export it.")

    # Handle HuggingFace token (support both naming conventions)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_token

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"runs/grpo_{ts}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    reward_cache_path = str(Path(output_dir) / "rewards.cache.sqlite")

    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Using Gemini judge model: {args.gemini_model}")
    logger.info(f"Reward cache: {reward_cache_path}")

    # --------- load dataset ----------
    # Expects JSONL with: prompt, ground_truth, optional rubric
    ds = load_dataset("json", data_files={"train": args.train_jsonl}, split="train")
    logger.info(f"Loaded {len(ds)} training examples from {args.train_jsonl}")

    required_cols = {"prompt", "ground_truth"}
    missing = required_cols - set(ds.column_names)
    if missing:
        raise RuntimeError(f"train_jsonl missing required columns: {sorted(missing)}")

    if "rubric" not in ds.column_names:
        logger.warning("No 'rubric' column; reward will use a generic rubric.")

    # --------- tokenizer & model ----------
    logger.info(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # choose dtype based on GPU capability
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        torch_dtype = torch.bfloat16 if major >= 8 else torch.float16
        device_map = "auto"
    else:
        torch_dtype = None
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
    )

    # ensure generation config is sane
    gen_cfg = model.generation_config
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    gen_cfg.eos_token_id = tokenizer.eos_token_id
    if getattr(gen_cfg, "bos_token_id", None) is None and tokenizer.bos_token_id is not None:
        gen_cfg.bos_token_id = tokenizer.bos_token_id

    # --------- optional LoRA ----------
    peft_config = None
    if args.use_lora:
        logger.info("Using LoRA PEFT")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    # --------- Gemini judge + reward_fn ----------
    judge = GeminiJudge(
        model=args.gemini_model,
        cache=RewardCache(reward_cache_path),
    )
    reward_fn = make_gemini_reward_func(judge)

    # --------- GRPO config ----------
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=max(10, args.max_steps // 2),
        report_to="none",
        seed=args.seed,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
    )

    logger.info("Starting GRPO training...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        reward_funcs=reward_fn,      # our Gemini-based reward function
        processing_class=tokenizer,  # tokenizer for prompts/completions
        peft_config=peft_config,
    )

    trainer.train()

    logger.info("Saving model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Done. Trained artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
