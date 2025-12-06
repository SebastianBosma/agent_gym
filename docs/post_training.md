# Gemini as Teacher → GRPO Fine-Tuning for Gemma

This pipeline implements **single-step GRPO (contextual bandit RL)** where a large model (Gemini) teaches a smaller model (Gemma) using dialogue traces from DSTC8 Schema-Guided Dialogue dataset.

## 🎯 Goal

**Input**: Raw dialogue traces (DSTC8 SGD)  
**Process**: Gemini manufactures environment + reward function  
**Output**: Fine-tuned Gemma that generates better responses according to Gemini's judgment

---

## 📋 Prerequisites

### 1. Install Dependencies

**With uv (recommended):**

```bash
cd src/post-training
uv sync
```

**Or with pip:**

```bash
cd src/post-training
pip install -e .
```

**Or with pip + requirements.txt (legacy):**

```bash
cd src/post-training
pip install -r requirements.txt
```

Required packages:
- `torch>=2.2.0`
- `transformers>=4.46.0`
- `trl>=0.17.0` (for GRPO)
- `google-genai>=1.0.0` (Gemini API)
- `peft>=0.13.0` (LoRA for efficient training)

### 2. Get API Keys

You'll need:
- **Gemini API Key**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Hugging Face Token** (optional): For downloading Gemma models

### 3. Set Environment Variables

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
HUGGINGFACE_HUB_TOKEN=your_hf_token_here  # optional
```

Load environment variables:
```bash
export $(cat .env | xargs)
```

### 4. Download DSTC8 SGD Dataset

```bash
# From project root
cd ../../data
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git
cd ../src/post-training
```

Your data should be at: `../../data/dstc8-schema-guided-dialogue/`

---

## 🚀 Quick Start (End-to-End)

### Step 1: Prepare Training Data

Convert DSTC8 dialogues into bandit-style training samples:

```bash
uv run python scripts/prepare_sgd_env.py \
  --sgd_root ../../data/dstc8-schema-guided-dialogue \
  --out_path ../../data/processed/train.jsonl
```

**What this does:**
- Parses DSTC8 dialogues
- Extracts each system turn as a training sample
- Creates `prompt` (conversation context) and `ground_truth` (reference response)
- Saves to JSONL format

**Optional - Add Gemini-Generated Rubrics** (costs API calls):

```bash
uv run python scripts/prepare_sgd_env.py \
  --sgd_root ../../data/dstc8-schema-guided-dialogue \
  --out_path ../../data/processed/train.with_rubric.jsonl \
  --with_rubric
```

This pre-generates quality rubrics for each sample using Gemini.

### Step 2: Train with GRPO

Fine-tune Gemma using Gemini as the reward function:

```bash
uv run python scripts/train_grpo.py \
  --train_jsonl ../../data/processed/train.jsonl \
  --model_name google/gemma-2-2b-it \
  --use_lora \
  --max_steps 100 \
  --learning_rate 5e-6 \
  --num_generations 2 \
  --output_dir ../../runs/grpo_experiment_1
```

**Key parameters:**
- `--train_jsonl`: Your prepared training data
- `--model_name`: Base Gemma model to fine-tune
- `--use_lora`: Use LoRA for memory-efficient training (recommended)
- `--max_steps`: Number of training steps (40-200 for experiments)
- `--num_generations`: Samples per prompt for GRPO (2-4 typical)
- `--output_dir`: Where to save trained model

**What happens during training:**
1. Gemma generates multiple responses for each prompt
2. Gemini judges each response (scores 0.0-1.0)
3. GRPO updates Gemma to prefer higher-scored responses
4. Rewards are cached in SQLite to avoid repeated API calls

**Training time**: ~10-30 minutes on GPU for quick experiments

### Step 3: Evaluate Results

Compare base model vs fine-tuned model:

```bash
# Prepare eval set (if not already done)
uv run python scripts/prepare_sgd_env.py \
  --sgd_root ../../data/dstc8-schema-guided-dialogue \
  --out_path ../../data/processed/eval.jsonl

# Run evaluation
uv run python scripts/eval.py \
  --eval_jsonl ../../data/processed/eval.jsonl \
  --base_model google/gemma-2-2b-it \
  --trained_model_dir ../../runs/grpo_experiment_1 \
  --n 50
```

**Output:**
```
Base scores: {'mean': 0.65, 'stdev': 0.18, 'min': 0.20, 'max': 0.95}
Trained scores: {'mean': 0.78, 'stdev': 0.14, 'min': 0.45, 'max': 0.98}
Δ mean: +0.13
```

This shows the improvement in Gemini's judgment of Gemma's responses.

---

## 🔧 Architecture Overview

```
┌─────────────────┐
│  DSTC8 Traces   │  Raw dialogue logs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ env_compiler.py │  Converts to bandit samples
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  train.jsonl    │  {prompt, ground_truth, rubric}
└────────┬────────┘
         │
         ▼
┌─────────────────┐         ┌──────────────────┐
│   GRPO Trainer  │◄────────┤  Gemini Judge    │
│   (TRL)         │  reward │  (reward_fn.py)  │
└────────┬────────┘         └──────────────────┘
         │
         ▼
┌─────────────────┐
│  Fine-tuned     │  Improved Gemma
│  Gemma Model    │
└─────────────────┘
```

### Key Components:

1. **`sgd_trace_parser.py`**: Loads DSTC8 dialogues into structured format
2. **`env_compiler.py`**: Converts dialogues into (state, action, reward) samples
3. **`gemini_judge.py`**: Gemini-based reward function with caching
4. **`reward_fn.py`**: TRL-compatible wrapper around Gemini judge
5. **`train_grpo.py`**: Main training loop using TRL's GRPOTrainer
6. **`eval.py`**: Before/after evaluation script

---

## 📊 Understanding the Data Format

### Input Format (train.jsonl)

Each line contains one training sample:

```json
{
  "id": "dialogue123-turn5",
  "dialogue_id": "dialogue123",
  "turn_index": 5,
  "prompt": "User: I need to book a restaurant\nAssistant: Sure! What cuisine?\nUser: Italian please",
  "ground_truth": "Great! What city and time would you like?",
  "rubric": "- Ask for missing required info\n- Be polite and helpful\n- Keep response concise",
  "task_description": "Help user book a restaurant"
}
```

### During Training

GRPO generates multiple completions and Gemini scores each:

```
Prompt: "User: I need to book a restaurant..."

Gemma samples:
1. "Great! What city?" → Gemini score: 0.85
2. "OK" → Gemini score: 0.40
3. "What city and time work?" → Gemini score: 0.92

GRPO update: Increase probability of sample 3, decrease sample 2
```

---

## 🎛️ Hyperparameter Tuning

### For Quick Experiments (5-10 min):
```bash
--max_steps 40
--per_device_train_batch_size 1
--gradient_accumulation_steps 2
--num_generations 2
--learning_rate 5e-6
```

### For Better Results (30-60 min):
```bash
--max_steps 200
--per_device_train_batch_size 2
--gradient_accumulation_steps 4
--num_generations 4
--learning_rate 3e-6
```

### Memory Management:
- Use `--use_lora` to reduce memory (recommended)
- Reduce `--max_prompt_length` and `--max_completion_length` if OOM
- Use `--per_device_train_batch_size 1` for limited GPU memory

---

## 🐛 Troubleshooting

### "ImportError: No module named 'src.post_training'"

Make sure you're in the correct directory and using `uv run`:
```bash
cd src/post-training
uv run python scripts/train_grpo.py ...
```

### "Missing GEMINI_API_KEY env var"

Set your API key:
```bash
export GEMINI_API_KEY="your_key_here"
```

### "Out of memory" during training

Reduce batch size and use LoRA:
```bash
--use_lora \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--max_prompt_length 256 \
--max_completion_length 64
```

### Gemini API rate limits

The code includes automatic retry with exponential backoff. If you hit limits:
- Use `--with_rubric` sparingly (only for small datasets)
- The reward cache prevents redundant API calls during training

---

## 📈 Expected Results

With 100-200 training steps on ~500 samples:
- **Base Gemma**: Average score ~0.60-0.70
- **Fine-tuned Gemma**: Average score ~0.75-0.85
- **Improvement**: +0.10 to +0.15 points

The model learns to:
1. Ask for missing information more naturally
2. Provide more helpful and grounded responses
3. Match the dialogue style better

---

## 🔬 Advanced Usage

### Use Pre-computed Rubrics

Pre-generate rubrics once to save costs:
```bash
uv run python scripts/prepare_sgd_env.py \
  --sgd_root ../../data/dstc8-schema-guided-dialogue \
  --out_path ../../data/processed/train.with_rubric.jsonl \
  --with_rubric
```

Then train multiple times reusing the same file.

### Split Data for Train/Eval

```python
# In Python
from src.post_training.jsonl_utils import read_jsonl, write_jsonl

data = read_jsonl("all_samples.jsonl")
train = data[:800]
eval_data = data[800:]

write_jsonl("train.jsonl", train)
write_jsonl("eval.jsonl", eval_data)
```

### Custom Reward Function

You can modify `gemini_judge.py` to change how responses are scored:
- Adjust the prompt template
- Add custom criteria
- Use different Gemini models

---

## 📚 References

- [DSTC8 Schema-Guided Dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
- [TRL Library](https://github.com/huggingface/trl)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [Gemini API Docs](https://ai.google.dev/docs)

---

## 🎉 Demo Tagline

> **"Give me traces → Gemini manufactures environment + reward → Gemma learns and improves."**

Your GRPO pipeline is now ready to turn any dialogue traces into training signal for smaller models!
