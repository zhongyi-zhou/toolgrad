import argparse
import json
import logging
import os
import sys
import traceback
import warnings

import gin
from typing import Any, Dict, List
from toolgrad.utils.langchain import create_callback_handler

import toolgrad as tog

# Clean up warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"^pydantic\.")


# GeminiCallbackHandler and create_callback_handler imported from toolgrad.utils.langchain


def create_args():
  """Create command line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--num_apis',
      type=int,
      default=50,
      help='Number of APIs to sample.',
  )
  parser.add_argument(
      '--iter',
      type=int,
      default=5,
      help='Number of iterations to run.',
  )
  parser.add_argument(
      '--seed',
      type=int,
      default=123,
      help='Seed for sampling APIs.',
  )
  parser.add_argument(
      '--cfg',
      type=str,
      default='configs/gemini-2.5-lite.gin',
      help='gin config file path',
  )
  parser.add_argument(
      '--num_proposals',
      type=int,
      default=3,
      help='Number of API proposals to generate.',
  )
  parser.add_argument(
      '--output_dir',
      type=str,
      required=True,
      help='Output directory for data and logs.',
  )
  return parser.parse_args()


if __name__ == '__main__':
  if "TOOLBENCH_KEY" not in os.environ:
    raise ValueError(
        "Please set the environment variable TOOLBENCH_KEY to your ToolBench API key."
    )
  args = create_args()
  cfg_path = args.cfg
  output_dir = args.output_dir

  # Create output directories
  os.makedirs(f"{output_dir}/data", exist_ok=True)
  os.makedirs(f"{output_dir}/log", exist_ok=True)
  os.makedirs(f"{output_dir}/metadata", exist_ok=True)
  os.makedirs(f"{output_dir}/trace", exist_ok=True)

  # Configure logging to file
  log_path = f"{output_dir}/log/{args.seed:05d}.log"
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
      handlers=[
          logging.FileHandler(log_path),
          logging.StreamHandler()  # Also log to console
      ])
  logging.getLogger("httpx").disabled = True
  logging.getLogger("asyncio").setLevel(logging.CRITICAL)
  logging.getLogger("google_genai._api_client").setLevel(logging.ERROR)

  gin.parse_config_file(cfg_path)
  app = tog.prebuilt.create_graph_on_toolbench(
      sample_seed=args.seed,
      num_apis=args.num_apis,
      num_iterations=args.iter,
      num_proposals=args.num_proposals,
  )
  cb = create_callback_handler()
  cfg = {
      "configurable": {
          "thread_id": 42
      },
      "callbacks": [cb],
      "recursion_limit": 1000,
  }

  # Create tracer
  from toolgrad.utils import trace_utils
  tracer = trace_utils.ExecutionTracer(output_dir=output_dir, seed=args.seed)

  initial = tog.modules.ToolGradState(
      workflow_cur=None,
      api_proposals=None,
      api_reports=None,
      api_selection=None,
      step=0,
      sampled_apis=[],
      total_tool_calls=0,
      tracer=tracer,
  )

  try:
    final_state = app.invoke(initial, config=cfg)
  except Exception as e:
    # Log the full error with traceback
    error_msg = f"\n{'='*80}\nERROR during graph execution for seed {args.seed}\n{'='*80}\n"
    logging.error(error_msg)
    logging.error(f"Error type: {type(e).__name__}")
    logging.error(f"Error message: {str(e)}")
    logging.error(
        f"\nFull traceback:\n{''.join(traceback.format_exception(type(e), e, e.__traceback__))}"
    )
    logging.error(f"{'='*80}\n")

    # Save trace up to the point of failure
    try:
      tracer.save()
      logging.info(
          f"Partial trace saved to {output_dir}/trace/{args.seed:05d}.json")
    except Exception as trace_error:
      logging.error(f"Failed to save trace: {trace_error}")

    # Re-raise the exception to ensure the process exits with error code
    raise

  # Save trace
  tracer.save()

  logging.info(f"LLM cost: {cb}")

  # Get accumulated tool call count from the state
  tool_call_count = final_state.get("total_tool_calls", 0)

  # Collect metadata
  metadata = {
      "seed": args.seed,
      "num_apis": args.num_apis,
      "iterations": args.iter,
      "num_proposals": args.num_proposals,
      "config": cfg_path,
      "llm_costs": {
          "total_tokens": cb.total_tokens,
          "prompt_tokens": cb.prompt_tokens,
          "prompt_tokens_cached": cb.prompt_tokens_cached,
          "completion_tokens": cb.completion_tokens,
          "reasoning_tokens": cb.reasoning_tokens,
          "successful_requests": cb.successful_requests,
          "total_cost_usd": cb.total_cost,
      },
      "tool_call_count": tool_call_count,
  }

  # Save metadata to JSON file
  metadata_path = f"{output_dir}/metadata/{args.seed:05d}.json"
  with open(metadata_path, "w") as meta_file:
    json.dump(metadata, meta_file, indent=2)
  logging.info(f"Metadata saved to {metadata_path}")

  # Save the generated (query, workflow, response) to a JSON file
  data_sample = final_state["workflow_cur"]
  save_path = f"{output_dir}/data/{args.seed:05d}.json"
  if not data_sample:
    logging.error(f"Empty data sample for {save_path}, skipping save.")
    raise ValueError("Empty data sample")

  with open(save_path, "w") as json_file:
    json.dump(data_sample.model_dump(), json_file, indent=2)

  logging.info(f"Results saved to {save_path}")
  logging.info(f"Logs saved to {log_path}")
