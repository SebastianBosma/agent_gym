#!/usr/bin/env python
"""
Modal-enabled GRPO training script for running on cloud GPUs.
Wraps train_grpo.py functionality with Modal infrastructure.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

# Modal app configuration
app = modal.App("grpo-training")

# Create persistent volumes for data and checkpoints
training_data_volume = modal.Volume.from_name("training-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("grpo-checkpoints", create_if_missing=True)

# Define container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.46.0",
        "trl>=0.17.0",
        "datasets>=2.19.0",
        "accelerate>=0.33.0",
        "peft>=0.13.0",
        "google-genai>=1.0.0",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
    )
)

# Volume mount paths in container
DATA_DIR = "/data"
CHECKPOINT_DIR = "/checkpoints"
CODE_DIR = "/code"


@app.function(
    image=image,
    volumes={
        DATA_DIR: training_data_volume,
        CHECKPOINT_DIR: checkpoints_volume,
    },
    timeout=3600,  # 1 hour max
)
def upload_data(file_content_b64: str):
    """
    Upload training data to Modal volume.
    Receives file content as base64 encoded string.
    """
    import base64
    
    dest_path = Path(DATA_DIR) / "train.jsonl"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Uploading training data to Modal volume...")
    # Decode base64 and write
    file_content = base64.b64decode(file_content_b64)
    dest_path.write_bytes(file_content)
    
    training_data_volume.commit()
    print(f"✓ Upload complete: {dest_path}")
    
    # Verify
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"✓ Verified: {size_mb:.2f} MB")
    
    return str(dest_path)


@app.function(
    image=image,
    gpu="A10G",  # 24GB VRAM, ~$1.10/hour
    volumes={
        DATA_DIR: training_data_volume,
        CHECKPOINT_DIR: checkpoints_volume,
    },
    secrets=[
        modal.Secret.from_name("gemini-secret"),  # GEMINI_API_KEY
        modal.Secret.from_name("huggingface-secret"),  # HF_TOKEN
    ],
    timeout=7200,  # 2 hours max
)
def train(
    model_name: str = "gpt2",
    gemini_model: str = "gemini-3-pro-preview",
    max_steps: int = 40,
    learning_rate: float = 5e-6,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    num_generations: int = 2,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_prompt_length: int = 128,
    max_completion_length: int = 32,
    seed: int = 42,
):
    """
    Run GRPO training on Modal GPU.
    
    Example:
        modal run train_grpo_modal.py::train --max-steps 40 --use-lora
    """
    import logging
    import sys
    from datetime import datetime
    
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    
    # Embed reward modules inline (no file copying needed)
    import hashlib
    import json
    import re
    import sqlite3
    import threading
    import time
    from dataclasses import dataclass
    from typing import Any, Iterable, List, Optional
    
    from google import genai
    
    # ===== GeminiJudge and RewardCache (embedded) =====
    _SCORE_RE = re.compile(r"(-?\d+(?:\.\d+)?)")
    
    def _sha1(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()
    
    def _extract_score(text: str) -> Optional[float]:
        if not text:
            return None
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            t = re.sub(r"^\s*(json|JSON)\s*", "", t)
        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and "score" in obj:
                return float(obj["score"])
        except Exception:
            pass
        m = _SCORE_RE.search(t)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None
    
    class RewardCache:
        def __init__(self, path: str):
            self.path = path
            self._lock = threading.Lock()
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            with self._conn:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS rewards (k TEXT PRIMARY KEY, v REAL)"
                )
        
        def get(self, k: str) -> Optional[float]:
            with self._lock:
                cur = self._conn.execute("SELECT v FROM rewards WHERE k=?", (k,))
                row = cur.fetchone()
                return None if row is None else float(row[0])
        
        def set(self, k: str, v: float) -> None:
            with self._lock:
                with self._conn:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO rewards (k, v) VALUES (?, ?)",
                        (k, float(v)),
                    )
    
    @dataclass
    class GeminiJudge:
        model: str
        cache: RewardCache
        max_retries: int = 4
        base_sleep_s: float = 0.8
        
        def __post_init__(self) -> None:
            self.client = genai.Client()
        
        def score_one(self, prompt: str, completion: str, rubric: str, ground_truth: str) -> float:
            key_material = f"{self.model}\nPROMPT:{prompt}\nRUBRIC:{rubric}\nGT:{ground_truth}\nC:{completion}"
            k = _sha1(key_material)
            cached = self.cache.get(k)
            if cached is not None:
                return cached
            
            contents = (
                "You are a strict evaluator for a customer support assistant.\n\n"
                "Return ONLY valid JSON like: {\"score\": 0.73}\n"
                "Score range: 0.0 (bad) to 1.0 (excellent).\n"
                "Use this rubric as the main criteria; ground-truth is provided as a reference.\n\n"
                f"=== RUBRIC ===\n{rubric}\n\n"
                f"=== CONVERSATION PROMPT ===\n{prompt}\n\n"
                f"=== CANDIDATE ASSISTANT RESPONSE ===\n{completion}\n\n"
                f"=== REFERENCE (GROUND TRUTH) RESPONSE ===\n{ground_truth}\n\n"
                "Now output JSON."
            )
            
            last_err: Optional[Exception] = None
            for attempt in range(self.max_retries):
                try:
                    resp = self.client.models.generate_content(model=self.model, contents=contents)
                    text = (resp.text or "").strip()
                    score = _extract_score(text)
                    if score is None:
                        score = 0.0
                    score = max(0.0, min(1.0, float(score)))
                    self.cache.set(k, score)
                    return score
                except Exception as e:
                    last_err = e
                    sleep_s = self.base_sleep_s * (2 ** attempt)
                    time.sleep(sleep_s)
            
            self.cache.set(k, 0.0)
            return 0.0
        
        def score_batch(self, prompts: Iterable[str], completions: Iterable[str], 
                       rubrics: Iterable[str], ground_truths: Iterable[str]) -> List[float]:
            out: List[float] = []
            for p, c, r, gt in zip(prompts, completions, rubrics, ground_truths):
                out.append(self.score_one(p, c, r, gt))
            return out
    
    # ===== reward_fn (embedded) =====
    def _to_text(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, list):
            parts = []
            for item in x:
                if isinstance(item, dict) and "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(str(item))
            return "\n".join(parts).strip()
        if isinstance(x, dict) and "content" in x:
            return str(x["content"])
        return str(x)
    
    def make_gemini_reward_func(judge):
        def reward_func(prompts=None, completions=None, ground_truth=None, rubric=None, **kwargs) -> List[float]:
            prompts_l = prompts if isinstance(prompts, list) else [prompts] * len(completions)
            gt_l = ground_truth if isinstance(ground_truth, list) else [ground_truth] * len(completions)
            if rubric is None:
                rubric_l = ["- Be helpful\n- Be accurate\n- Ask for missing required info\n- Keep it concise"] * len(completions)
            else:
                rubric_l = rubric if isinstance(rubric, list) else [rubric] * len(completions)
            comps_l = completions if isinstance(completions, list) else [completions]
            prompts_txt = [_to_text(p) for p in prompts_l]
            comps_txt = [_to_text(c) for c in comps_l]
            gt_txt = [_to_text(g) for g in gt_l]
            rubric_txt = [_to_text(r) for r in rubric_l]
            return judge.score_batch(prompts_txt, comps_txt, rubric_txt, gt_txt)
        return reward_func
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    # Check environment
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY not found in Modal secrets")
    
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        logger.warning("No HF_TOKEN found - model download may fail if gated")
    
    # Setup paths
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{CHECKPOINT_DIR}/runs/grpo_{ts}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    reward_cache_path = f"{output_dir}/rewards.cache.sqlite"
    
    train_file = f"{DATA_DIR}/train.jsonl"
    if not Path(train_file).exists():
        raise FileNotFoundError(
            f"Training data not found: {train_file}\n"
            "Run: modal run train_grpo_modal.py::upload_data"
        )
    
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Using Gemini judge: {gemini_model}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    # Load dataset
    ds = load_dataset("json", data_files={"train": train_file}, split="train")
    logger.info(f"Loaded {len(ds)} training examples")
    
    required_cols = {"prompt", "ground_truth"}
    missing = required_cols - set(ds.column_names)
    if missing:
        raise RuntimeError(f"Dataset missing columns: {sorted(missing)}")
    
    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # GPU dtype selection
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        torch_dtype = torch.bfloat16 if major >= 8 else torch.float16
        device_map = "auto"
    else:
        torch_dtype = None
        device_map = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
    )
    
    # Configure generation
    gen_cfg = model.generation_config
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    gen_cfg.eos_token_id = tokenizer.eos_token_id
    if getattr(gen_cfg, "bos_token_id", None) is None and tokenizer.bos_token_id is not None:
        gen_cfg.bos_token_id = tokenizer.bos_token_id
    
    # Optional LoRA
    peft_config = None
    if use_lora:
        logger.info(f"Using LoRA: r={lora_r}, alpha={lora_alpha}")
        # Different models use different module names
        # GPT2: c_attn, c_proj
        # Gemma/Llama: q_proj, k_proj, v_proj, o_proj
        if "gpt2" in model_name.lower():
            target_modules = ["c_attn", "c_proj"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        logger.info(f"LoRA target modules: {target_modules}")
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
    
    # Gemini judge + reward function
    judge = GeminiJudge(
        model=gemini_model,
        cache=RewardCache(reward_cache_path),
    )
    reward_fn = make_gemini_reward_func(judge)
    
    # GRPO configuration
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        logging_steps=1,
        save_steps=max(10, max_steps // 2),
        report_to="none",
        seed=seed,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
    )
    
    logger.info("Starting GRPO training...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    trainer.train()
    
    logger.info("Saving model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Commit volumes
    checkpoints_volume.commit()
    
    logger.info(f"✓ Training complete!")
    logger.info(f"✓ Model saved to Modal volume: {output_dir}")
    logger.info(f"✓ Download with: modal volume get grpo-checkpoints {output_dir.replace(CHECKPOINT_DIR + '/', '')} ./local_output")
    
    return output_dir


@app.local_entrypoint()
def main(
    command: str = "train",
    model_name: str = "gpt2",
    max_steps: int = 5,
    use_lora: bool = True,
):
    """
    Local entrypoint for easy CLI usage.
    
    Examples:
        modal run train_grpo_modal.py              # Train with defaults
        modal run train_grpo_modal.py --max-steps 100
        modal run train_grpo_modal.py --command upload_data
    """
    if command == "upload_data":
        # Read local file and send to Modal
        import base64
        
        local_path = Path("data/processed/train.jsonl")
        if not local_path.exists():
            print(f"Error: Training data not found at {local_path}")
            print("Please ensure the file exists before uploading.")
            return
        
        print(f"Reading {local_path}...")
        file_content = local_path.read_bytes()
        print(f"Uploading {len(file_content) / (1024*1024):.2f} MB...")
        
        # Encode as base64 for Modal
        file_content_b64 = base64.b64encode(file_content).decode('utf-8')
        
        result = upload_data.remote(file_content_b64)
        print(f"✓ Data uploaded: {result}")
    elif command == "train":
        result = train.remote(
            model_name=model_name,
            max_steps=max_steps,
            use_lora=use_lora,
        )
        print(f"Training complete: {result}")
    else:
        print(f"Unknown command: {command}")
        print("Available: upload_data, train")
