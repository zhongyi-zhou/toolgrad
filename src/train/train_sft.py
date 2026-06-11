"""SFT Trainer"""
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import os
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, AutoProcessor
import torch
import argparse
import logging
import transformers
import accelerate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set transformers and accelerate logging verbosity
transformers.utils.logging.set_verbosity_warning()
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_propagation()

try:
  from accelerate.logging import get_logger as get_accelerate_logger
  get_accelerate_logger("accelerate").setLevel(logging.INFO)
except ImportError:
  # Fallback if accelerate.logging is not available in this version
  pass

def get_chat_template(model_name: str) -> str:
  """Get the chat template for the given model name."""
  if "gemma" not in model_name.lower():
    raise ValueError(f"Model {model_name} is not supported. Only Gemma models are supported.")

  template_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "chat_templates", "gemma.jinja"
  )
  with open(template_path, "r") as f:
    return f.read()


def get_formatted_date_time(with_year: bool = False, utc_offset: int = 9) -> str:
  """Get the current date and time formatted as MMDDHHMM."""
  tz = datetime.timezone(datetime.timedelta(hours=utc_offset))

  now = datetime.datetime.now(tz)
  if with_year:
    return now.strftime("%Y%m%d_%H%M")
  else:
    return now.strftime("%m%d_%H%M")

def create_args():
  parser = argparse.ArgumentParser(description="")
  parser.add_argument(
      '--model',
      type=str,
      default="google/gemma-3-4b-it",
  )
  parser.add_argument(
      '--local_model',
      type=str,
      default=None,
  )
  parser.add_argument(
      '--dataset',
      type=str,
      default="zhongyi-zhou/toolgrad-500",
      help="Hugging Face dataset name.",
  )
  parser.add_argument(
      '--train_data',
      type=str,
      default=None,
      help="Path to train.jsonl.",
  )
  parser.add_argument(
      '--val_data',
      type=str,
      default=None,
      help="Path to test.jsonl.",
  )
  parser.add_argument(
      '--data_dir',
      type=str,
      default="./output/",
  )
  parser.add_argument('--use_wandb', action='store_true')
  parser.add_argument('--wandb_offline', action='store_true')
  parser.add_argument('--load_in_8bit', action='store_true')
  parser.add_argument('--optim', type=str, default="adamw_8bit")
  parser.add_argument('--num_epochs', type=int, default=3)
  parser.add_argument('--seq_length', type=int, default=8192)
  parser.add_argument('--gradient_checkpointing', action='store_true')
  parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
  parser.add_argument('--use_lora', action='store_true')
  parser.add_argument('--lora_rank', type=int, default=64)
  parser.add_argument('--lora_alpha', type=int, default=16)
  parser.add_argument('--lora_dropout', type=float, default=0.1)
  parser.add_argument('--learning_rate', type=float, default=5e-06)
  parser.add_argument('--lr_scheduler_type', type=str, default="constant")
  parser.add_argument('--warmup_ratio', type=float, default=0.1)
  parser.add_argument('--note', type=str, default=None, help="Additional note to append to output dir")
  parser.add_argument('--enable_tqdm', action='store_true')
  parser.add_argument('--preprocessed_dataset_path', type=str, default=None, help="Path to pre-tokenized dataset directory")
  parser.add_argument('--lora_add_embed', action='store_true', help="If true, save embeddings and lm_head")
  parser.add_argument('--utc_offset', type=int, default=9, help="UTC timezone offset in hours.")
  return parser.parse_args()


args = create_args()
logger.info(f"Script Arguments: {vars(args)}")

from accelerate import PartialState
from accelerate.utils import DistributedType
distributed_state = PartialState()

# FSDP + adamw_8bit often causes dtype mismatch AssertionErrors during checkpointing.
# Fallback to adamw_torch_fused or adamw_torch for better compatibility.
if distributed_state.distributed_type == DistributedType.FSDP and args.optim == "adamw_8bit":
  logger.warning("FSDP + adamw_8bit detected. Switching to adamw_torch_fused for better checkpointing compatibility.")
  args.optim = "adamw_torch_fused"

if distributed_state.num_processes > 1:
  device_map = None  # FSDP/DeepSpeed/DDP will handle device placement and sharding
else:
  device_map = "auto"

training_type = "lora" if args.use_lora else "full"
note_str = f"_{args.note}" if args.note else ""
model_name_clean = os.path.basename(args.model.rstrip("/"))
output_dir = os.path.join(
    args.data_dir,
    "sft",
    model_name_clean,
    f"{training_type}{note_str}_{get_formatted_date_time(with_year=True, utc_offset=args.utc_offset)}"
)

if args.use_wandb:
  os.environ["WANDB_PROJECT"] = f"toolgrad-sft-{model_name_clean}"
  if args.wandb_offline:
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

from datasets import load_from_disk

if args.dataset:
  logger.info(f"Loading dataset '{args.dataset}' from Hugging Face")
  dataset = load_dataset(args.dataset)
  train_set = dataset["train"]
  val_set = dataset["test"]
elif args.train_data and args.val_data:
  logger.info(f"Loading raw dataset from local files: train={args.train_data}, test={args.val_data}")
  val_set = load_dataset(
      "json",
      data_files=args.val_data,
      split="train",
  )
  train_set = load_dataset(
      "json",
      data_files=args.train_data,
      split="train",
  )
elif args.preprocessed_dataset_path:
  logger.info(f"Loading preprocessed dataset from {args.preprocessed_dataset_path}")
  train_set = load_from_disk(os.path.join(args.preprocessed_dataset_path, "train_dataset"))
  val_set = load_from_disk(os.path.join(args.preprocessed_dataset_path, "eval_dataset"))
else:
  raise ValueError("Must specify either --dataset, --train_data/--val_data, or --preprocessed_dataset_path")

def format_dataset(sample):
  return {
      "messages": [
          {"role": "system", "content": sample["system"].strip()},
          {"role": "user", "content": sample["user"].strip()},
          {"role": "assistant", "content": sample["assistant"].strip()}
      ]
  }

if "messages" not in train_set.column_names:
  logger.info("Formatting dataset columns ('system', 'user', 'assistant') into conversational 'messages' list.")
  train_set = train_set.map(format_dataset, remove_columns=train_set.column_names)
  val_set = val_set.map(format_dataset, remove_columns=val_set.column_names)



training_args = SFTConfig(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_ratio=args.warmup_ratio,
    num_train_epochs=args.num_epochs,
    max_length=args.seq_length,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    output_dir=output_dir,
    bf16=True,
    optim=args.optim,
    logging_steps=50,
    disable_tqdm=not args.enable_tqdm,
    # do_eval=True,
    assistant_only_loss=True,
    eval_strategy="epoch",
    report_to=["tensorboard", "wandb"] if args.use_wandb else ["tensorboard"],
    data_seed=1234,
    save_strategy="epoch",
    eval_on_start=True,
    gradient_checkpointing=args.gradient_checkpointing,
    gradient_checkpointing_kwargs={'use_reentrant': False} if args.gradient_checkpointing else None,
)

logger.info(f"DEBUG: distributed_state.distributed_type = {distributed_state.distributed_type}")
logger.info(f"DEBUG: os.environ.get('ACCELERATE_USE_FSDP') = {os.environ.get('ACCELERATE_USE_FSDP')}")
logger.info(f"DEBUG: training_args.gradient_checkpointing = {training_args.gradient_checkpointing}")


# flash_attention_2 needs the flash-attn package, which has no prebuilt wheels on
# some platforms (e.g. Windows). Fall back to PyTorch's SDPA kernel when it is missing.
import importlib.util
if importlib.util.find_spec("flash_attn") is not None:
  attn_implementation = "flash_attention_2"
else:
  attn_implementation = "sdpa"
  logger.warning(
      "flash-attn not installed; falling back to attn_implementation='sdpa'.")


if args.local_model and os.path.exists(args.local_model):
  model_path = args.local_model
else:
  if args.local_model:
    logger.warning(f"Local model path {args.local_model} does not exist. Falling back to {args.model}")
  model_path = args.model


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation=attn_implementation,
    dtype=torch.bfloat16,
    device_map=device_map,
    max_length=args.seq_length,
)

if "gemma-3-1b" in model_path.lower() or "toolgrad_1b" in model_path.lower():
  logger.info("Gemma-3 1B model detected. Using google/gemma-3-4b-it tokenizer as a fallback.")
  tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
else:
  tokenizer = AutoTokenizer.from_pretrained(model_path)

# Custom Chat Template to support assistant_only_loss in TRL
# Adds {% generation %} tags around model output.
chat_template = get_chat_template(args.model)
if chat_template:
  logger.info(f"Applying custom chat template for {args.model} [v2-robust]")
  tokenizer.chat_template = chat_template


if args.use_lora:
  peft_config = LoraConfig(
      r=args.lora_rank,
      lora_alpha=args.lora_alpha,
      lora_dropout=args.lora_dropout,
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
      bias="none",
      task_type="CAUSAL_LM",
      modules_to_save=["lm_head", "embed_tokens"] if args.lora_add_embed else None,
  )
else:
    peft_config = None

class SaveConfigCallback(TrainerCallback):
  """Callback to save base model and processor configs in LoRA checkpoints."""
  def on_save(self, args, state, control, **kwargs):
    checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
    # Always save the config and generation_config
    if kwargs.get("model") is not None:
      model = kwargs["model"]
      model.config.save_pretrained(checkpoint_folder)
      if hasattr(model, "generation_config"):
        model.generation_config.save_pretrained(checkpoint_folder)
      
    # Also save the processing_class (tokenizer/processor)
    # This ensures processor_config.json etc are present
    if kwargs.get("processing_class") is not None:
      kwargs["processing_class"].save_pretrained(checkpoint_folder)

callbacks = [SaveConfigCallback()] if args.use_lora else None

trainer = SFTTrainer(
    model,
    train_dataset=train_set,
    eval_dataset=val_set,
    args=training_args,
    peft_config=peft_config,
    processing_class=tokenizer,
    callbacks=callbacks,
)
trainer.train()
