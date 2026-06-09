# ToolGrad SFT Training

This directory contains scripts for Supervised Fine-Tuning (SFT) of models on the ToolGrad dataset.

## Setup

Activate the python virtual environment at the root of the repository:

```bash
source .venv/bin/activate
```

## Launching Training

You can launch training using the `train_sft.py` script. The training dataset can be loaded from Hugging Face or local raw JSONL files.

### Reproducing ToolGrad Models (Training Recipes)

To train each model configuration as described in the paper, use the following commands:

#### 1. ToolGrad 1B
*   **Training Method:** Full Finetuning
*   **Base model:** `google/gemma-3-1b-it`
*   **Learning Rate:** `1e-5`
*   **Epochs:** `3`
*   **Seq Length:** `8192` (8k)
*   **Batch Size:** `per_device_train_batch_size=1` (kept at 1 to prevent VRAM Out-Of-Memory (OOM) errors during 8k context training, even with gradient checkpointing).
*   **Gradient Checkpointing:** Enabled (`--gradient_checkpointing`)
*   **Loss:** `assistant_only_loss=True` (enabled by default in script)
*   **Optimizer:** `adamw_8bit` (or `adamw_torch` / `adamw_torch_fused`)
*   *Note on Released Model:* The released ToolGrad-1B model was trained on **2 GPUs** using DDP (effective global batch size = 2). This took **900 steps** to complete 3 epochs on the 600-sample training dataset. 
*   *Running locally:* To match the exact global batch size of 2 on a single GPU, we use **gradient accumulation** (`--gradient_accumulation_steps 2`), which will run for **900 steps** to complete 3 epochs.

```bash
# Run SFT on a single GPU (reproducing global batch size 2 via gradient accumulation):
python src/train/train_sft.py \
  --model google/gemma-3-1b-it \
  --learning_rate 1e-5 \
  --num_epochs 3 \
  --seq_length 8192 \
  --gradient_checkpointing \
  --gradient_accumulation_steps 2
```

#### 2. ToolGrad 4B
*   **Training Method:** LoRA
*   **Base model:** `google/gemma-3-4b-it`
*   **Learning Rate:** `5e-6`
*   **LoRA Config:** rank `64`, alpha `16`, dropout `0.1`
*   **Epochs:** `2`
*   **Seq Length:** `8192` (8k)
*   **Batch Size:** `per_device_train_batch_size=1` (kept at 1 to prevent VRAM OOM during 8k context training).
*   **Loss:** `assistant_only_loss=True` (enabled by default in script)
*   **Gradient Checkpointing:** Enabled (`--gradient_checkpointing`)
*   *Note on Released Model:* The released ToolGrad-4B model was trained on **4 GPUs** using DDP (effective global batch size = 4). This took **300 steps** to complete 2 epochs on the 600-sample training dataset.
*   *Running locally:* To match the exact global batch size of 4 on a single GPU, we use **gradient accumulation** (`--gradient_accumulation_steps 4`), which will run for **300 steps** to complete 2 epochs.

```bash
# Run SFT on a single GPU (reproducing global batch size 4 via gradient accumulation):
python src/train/train_sft.py \
  --model google/gemma-3-4b-it \
  --use_lora \
  --lora_rank 64 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --learning_rate 5e-6 \
  --num_epochs 2 \
  --seq_length 8192 \
  --gradient_checkpointing \
  --gradient_accumulation_steps 4
```

#### 3. ToolGrad 12B
*   **Training Method:** LoRA
*   **Base model:** `google/gemma-3-12b-it`
*   **Learning Rate:** `2e-5`
*   **LoRA Config:** rank `64`, alpha `16`, dropout `0.1`
*   **Epochs:** `1`
*   **Seq Length:** `8192` (8k)
*   **Batch Size:** `per_device_train_batch_size=1`
*   **Loss:** `assistant_only_loss=True` (enabled by default in script)
*   **Gradient Checkpointing:** Enabled (`--gradient_checkpointing`)
*   *Note on Released Model:* The released ToolGrad-12B model was trained on a **single GPU** (effective global batch size = 1). This took **600 steps** to complete 1 epoch on the 600-sample training dataset.

```bash
# Run SFT on a single GPU:
python src/train/train_sft.py \
  --model google/gemma-3-12b-it \
  --use_lora \
  --lora_rank 64 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --learning_rate 2e-5 \
  --num_epochs 1 \
  --seq_length 8192 \
  --gradient_checkpointing
```

### Loading from Local Files

If you want to train on local JSONL datasets instead of Hugging Face, specify `--train_data` and `--val_data` and set `--dataset ""` to empty:

```bash
python src/train/train_sft.py \
  --model google/gemma-3-4b-it \
  --dataset "" \
  --train_data data/toolgrad_500/train.jsonl \
  --val_data data/toolgrad_500/test.jsonl \
  --use_lora
```

## Merging LoRA Weights

If you trained using LoRA, you can merge the adapters back into the base model:

```bash
python src/train/merge_lora.py \
  --base /path/to/base_model \
  --lora /path/to/lora_adapter_checkpoint \
  --out /path/to/output_merged_model
```
