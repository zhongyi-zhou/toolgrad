import argparse
import json
import logging
import os
import warnings

import gin
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

import toolgrad as tog

# Clean up warnings and logging
warnings.filterwarnings("ignore", category=UserWarning, module=r"^pydantic\.")
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").disabled = True
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


def create_args():
  """Create command line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--num_apis',
      type=int,
      default=5,
      help='Number of APIs to sample.',
  )
  parser.add_argument(
      '--iter',
      type=int,
      default=3,
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
  return parser.parse_args()


if __name__ == '__main__':
  args = create_args()
  # Make config path absolute relative to script directory
  script_dir = os.path.dirname(os.path.abspath(__file__))
  cfg_path = os.path.join(script_dir, args.cfg)
  output_dir = os.path.join(script_dir, "outputs/")

  gin.parse_config_file(cfg_path)
  app = tog.prebuilt.create_graph_on_mcp(
      sample_seed=args.seed,
      num_apis=args.num_apis,
      num_iterations=args.iter,
      mcp_dict=tog.utils.mcp.get_default_mcp_dict())
  cb = OpenAICallbackHandler()
  cfg = {
      "configurable": {
          "thread_id": 42
      },
      "callbacks": [cb],
      "recursion_limit": 1000,
  }
  # Use the Runnable
  tracer = tog.utils.trace_utils.ExecutionTracer(output_dir=output_dir, seed=args.seed)
  initial = tog.modules.ToolGradState(
      workflow_cur=None,
      api_proposals=None,
      api_reports=None,
      api_selection=None,
      step=0,
      sampled_apis=[],
      tracer=tracer,
  )
  final_state = app.invoke(initial, config=cfg)
  tracer.save()
  logging.info(f"LLM cost: {cb}")

  # Save the generated (query, workflow, response) to a JSON file
  data_sample = final_state["workflow_cur"]
  save_path = f"{output_dir}/seed={args.seed}__iter={args.iter}__num_apis={args.num_apis}.json"
  os.makedirs(output_dir, exist_ok=True)
  with open(save_path, "w") as json_file:
    json.dump(data_sample.model_dump(), json_file, indent=2)
