"""
Script to merge LoRA adapters back into the base model to create a standalone model.
This is required for tools like vLLM that do not support hot-swapping adapters for certain models.

Usage:
    python merge_lora.py \
        --base /path/to/base_model \
        --lora /path/to/lora_adapter_checkpoint \
        --out /path/to/output_merged_model
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import PeftModel
import argparse
import os

def merge_lora(base_model_path, lora_path, output_path):
  print(f"Loading base model from {base_model_path}...")
  tokenizer = AutoProcessor.from_pretrained(base_model_path)
  
  model = AutoModelForCausalLM.from_pretrained(
      base_model_path,
      torch_dtype=torch.bfloat16,
      device_map="cpu", # Use CPU to avoid GPU memory issues during merge
      trust_remote_code=True,
  )
  
  print(f"Loading LoRA adapters from {lora_path}...")
  model = PeftModel.from_pretrained(model, lora_path)
  
  print("Merging weights...")
  model = model.merge_and_unload()
  
  print(f"Saving merged model to {output_path}...")
  model.save_pretrained(output_path)
  tokenizer.save_pretrained(output_path)
  print("Merge complete!")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--base", type=str, required=True)
  parser.add_argument("--lora", type=str, required=True)
  parser.add_argument("--out", type=str, required=True)
  args = parser.parse_args()
  
  merge_lora(args.base, args.lora, args.out)
