# Quick Start: Modal GRPO Training

Get your GRPO training running on Modal's cloud GPUs in 5 minutes.

## One-Time Setup (5 minutes)

### 1. Install & Authenticate Modal

```bash
pip install modal
modal setup
```

This opens your browser to authenticate. Follow the prompts.

### 2. Configure API Keys

Create Modal secrets for your API keys:

```bash
# Gemini API key (required for reward function)
modal secret create gemini-secret GEMINI_API_KEY="your-gemini-key-here"

# HuggingFace token (optional, for gated models)
modal secret create huggingface-secret HF_TOKEN="your-hf-token-here"
```

**Where to get these:**
- **Gemini API**: https://aistudio.google.com/apikey
- **HuggingFace**: https://huggingface.co/settings/tokens

### 3. Upload Training Data

```bash
# From project root
cd /Users/arapat/_PARA/Project/hackathon/agent_gym

# Upload your training data (one-time)
modal run src/post-training/scripts/train_grpo_modal.py::upload_data
```

Expected output:
```
✓ Upload complete: /data/train.jsonl
✓ Verified: X.XX MB
```

## Running Training

### Basic Training (40 steps, ~$0.50)

```bash
modal run src/post-training/scripts/train_grpo_modal.py
```

That's it! Modal will:
1. Allocate an A10G GPU (24GB VRAM)
2. Build the container with all dependencies
3. Load your training data
4. Train the model with GRPO
5. Save checkpoints to persistent storage

### Custom Configuration

```bash
# More training steps
modal run src/post-training/scripts/train_grpo_modal.py --max-steps 100

# Disable LoRA (full fine-tuning)
modal run src/post-training/scripts/train_grpo_modal.py --use-lora false
```

## Monitoring

Watch real-time logs in your terminal:
```
✓ GPU allocated: NVIDIA A10G (24GB)
Loading model: google/gemma-2-2b-it...
Loaded 189345 training examples
Starting GRPO training...
Step 1/40: loss=1.234
Step 2/40: loss=1.189
...
✓ Training complete!
```

## Downloading Results

After training completes:

```bash
# List checkpoints
modal volume ls grpo-checkpoints

# Download specific run
modal volume get grpo-checkpoints runs/grpo_20251206_152800 ./my_trained_model
```

Your trained model is now in `./my_trained_model/`!

## Cost

- **40 steps**: ~$0.30-0.60 (15-30 min on A10G)
- **100 steps**: ~$0.60-1.20 (30-60 min on A10G)

## Automated Setup Script

For convenience, use the setup script:

```bash
bash src/post-training/scripts/setup_modal.sh
```

This automates all setup steps above.

## Troubleshooting

### "No such secret: gemini-secret"
```bash
modal secret create gemini-secret GEMINI_API_KEY="your-key"
```

### "Training data not found"
```bash
modal run src/post-training/scripts/train_grpo_modal.py::upload_data
```

### "Model download failed"
Accept Gemma model access: https://huggingface.co/google/gemma-2-2b-it

## Why Modal vs Local?

| Local Machine | Modal (A10G) |
|--------------|--------------|
| 5GB model download | Pre-cached in cloud |
| Slow on CPU/small GPU | Fast on 24GB A10G |
| Blocks your computer | Runs in background |
| Free (but slow) | ~$0.50 per run |

## Next Steps

1. **Evaluate** your trained model
2. **Experiment** with different hyperparameters
3. **Scale up** to 500+ steps for production

See [README_MODAL.md](./README_MODAL.md) for detailed documentation.

---

**Questions?** Check the full docs or Modal's documentation at https://modal.com/docs
