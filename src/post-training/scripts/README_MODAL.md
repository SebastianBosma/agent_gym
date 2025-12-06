# Running GRPO Training on Modal

This guide explains how to run your GRPO training on Modal's cloud GPUs instead of your local machine.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **API Keys**: 
   - Gemini API key (for reward function)
   - HuggingFace token (for Gemma model access)

## Installation

```bash
# Install Modal SDK
pip install modal

# Authenticate with Modal (one-time setup)
modal setup
```

Follow the prompts to authenticate via browser.

## Configuration

### 1. Create Modal Secrets

Store your API keys as Modal secrets (secure, encrypted storage):

```bash
# Gemini API key
modal secret create gemini-secret GEMINI_API_KEY="your-gemini-api-key-here"

# HuggingFace token
modal secret create huggingface-secret HF_TOKEN="your-hf-token-here"
```

**Important**: Replace the placeholder values with your actual keys.

### 2. Upload Training Data

Upload your training data to Modal's persistent volume (one-time):

```bash
cd /Users/arapat/_PARA/Project/hackathon/agent_gym

# Upload data (from project root)
modal run src/post-training/scripts/train_grpo_modal.py::upload_data
```

Expected output:
```
Uploading data/processed/train.jsonl to Modal volume...
✓ Upload complete: /data/train.jsonl
✓ Verified: X.XX MB
```

## Running Training

### Basic Usage

```bash
# Run with default settings (40 steps, LoRA enabled, A10G GPU)
modal run src/post-training/scripts/train_grpo_modal.py
```

### Custom Parameters

```bash
# More training steps
modal run src/post-training/scripts/train_grpo_modal.py --max-steps 100

# Without LoRA (full fine-tuning, requires more memory)
modal run src/post-training/scripts/train_grpo_modal.py --use-lora false
```

### Advanced: Direct Function Call

For more control over hyperparameters:

```bash
modal run src/post-training/scripts/train_grpo_modal.py::train \
  --model-name "google/gemma-2-2b-it" \
  --gemini-model "gemini-1.5-pro-latest" \
  --max-steps 100 \
  --learning-rate 5e-6 \
  --per-device-train-batch-size 1 \
  --num-generations 2 \
  --use-lora true \
  --lora-r 16 \
  --lora-alpha 32
```

## Monitoring Training

Modal provides real-time logs in your terminal. You'll see:
- GPU allocation and initialization
- Model loading progress
- Training loss per step
- Checkpoint saves
- Completion status

Example output:
```
✓ Connected to Modal
✓ GPU allocated: NVIDIA A10G (24GB)
Loading model: google/gemma-2-2b-it...
Loaded 1234 training examples
Starting GRPO training...
Step 1/40: loss=1.234
Step 2/40: loss=1.189
...
✓ Training complete!
✓ Model saved to Modal volume: /checkpoints/runs/grpo_20251206_152800
```

## Downloading Results

After training completes, download your model from Modal:

```bash
# List available checkpoints
modal volume ls grpo-checkpoints

# Download specific run
modal volume get grpo-checkpoints runs/grpo_20251206_152800 ./local_models/grpo_run1
```

The downloaded directory contains:
- `adapter_model.bin` / `model.safetensors` - Trained weights
- `adapter_config.json` - LoRA configuration (if used)
- `tokenizer.json`, `tokenizer_config.json` - Tokenizer files
- `training_args.bin` - Training configuration
- `rewards.cache.sqlite` - Reward cache (for analysis)

## Cost Estimation

### GPU Pricing (approximate)
- **A10G (24GB)**: ~$1.10/hour
- **A100 (40GB)**: ~$4.00/hour
- **A100 (80GB)**: ~$6.00/hour

### Typical Training Costs
- **40 steps** (default): 15-30 minutes = **$0.30-0.60**
- **100 steps**: 30-60 minutes = **$0.60-1.20**
- **500 steps**: 2-3 hours = **$2.50-3.50**

*Note: Gemini API calls are minimal due to caching (<$0.10 total).*

## Troubleshooting

### "GEMINI_API_KEY not found"
```bash
# Recreate the secret
modal secret create gemini-secret GEMINI_API_KEY="your-key"
```

### "Training data not found"
```bash
# Re-upload data
modal run src/post-training/scripts/train_grpo_modal.py::upload_data
```

### "Model download failed" (HuggingFace)
```bash
# Verify token is set
modal secret create huggingface-secret HF_TOKEN="your-token"

# Or accept gated model on HuggingFace website
# Visit: https://huggingface.co/google/gemma-2-2b-it
```

### Out of Memory (OOM)
- Reduce `per_device_train_batch_size` to 1
- Enable LoRA: `--use-lora true`
- Use smaller model: `--model-name "google/gemma-2b-it"`

### Slow Training
- Increase `gradient_accumulation_steps` to 2 or 4
- Reduce `max_prompt_length` to 256
- Reduce `num_generations` to 1

## Architecture Details

### GPU Configuration
- **Default**: NVIDIA A10G (24GB VRAM)
- **Compute Capability**: 8.6 (supports bfloat16)
- **Optimizations**: Automatic mixed precision, gradient checkpointing

### Persistent Volumes
- `training-data`: Stores train.jsonl (uploaded once)
- `grpo-checkpoints`: Stores all training outputs (auto-versioned)

### Resource Limits
- **Timeout**: 2 hours max per training run
- **Memory**: 32GB RAM + 24GB VRAM (A10G)
- **Storage**: Unlimited (persistent volumes)

## Comparison: Local vs Modal

| Aspect | Local Machine | Modal (A10G) |
|--------|--------------|--------------|
| Setup Time | Hours (CUDA, drivers) | Minutes |
| Training Speed | Varies | 15-30 min (40 steps) |
| Cost | Electricity + GPU wear | ~$0.30-0.60 per run |
| Scalability | Limited | Instant multi-GPU |
| Maintenance | Manual updates | Auto-managed |

## Next Steps

After training:
1. **Evaluate** the model on test data
2. **Compare checkpoints** (step 20 vs 40)
3. **Tune hyperparameters** (learning rate, LoRA rank)
4. **Scale up** to more steps or examples

For evaluation:
```bash
# Download model
modal volume get grpo-checkpoints runs/grpo_TIMESTAMP ./my_model

# Run evaluation (local)
python src/post-training/scripts/eval.py --model-path ./my_model
```

## Support

- **Modal Docs**: https://modal.com/docs
- **Modal Slack**: https://modal.com/slack
- **TRL Docs**: https://huggingface.co/docs/trl

## Advanced: Multi-Run Experiments

Run multiple experiments in parallel:

```bash
# Terminal 1: Low learning rate
modal run train_grpo_modal.py::train --learning-rate 1e-6

# Terminal 2: High learning rate  
modal run train_grpo_modal.py::train --learning-rate 1e-5

# Terminal 3: More generations
modal run train_grpo_modal.py::train --num-generations 4
```

Each runs on a separate GPU in the cloud!
