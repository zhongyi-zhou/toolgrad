
import os
import glob
import random
import logging
import multiprocessing
from typing import Any, Optional
import tqdm
import toolgrad as tog
from src.data.data_format_lib import DataFormatter, get_tool_list_from_sample, process_one_apiusechain

class NegativeSampler:
  """Helper class to generate negative (irrelevance) SFT samples."""

  def __init__(self, formatter: DataFormatter, seed: int = 42):
    self.formatter = formatter
    self.rng = random.Random(seed)
    self.api_to_tool_map = {}
    self._build_tool_maps()

  def _build_tool_maps(self):
    """Pre-compute API -> Tool Name mapping for stricter filtering."""
    for api_name in self.formatter.toolname_all:
      try:
        # Use the utility from toolgrad to parse hierarchy
        # Structure: {'category_name': ..., 'tool_name': ..., 'api_name': ...}
        parsed = tog.utils.toolbench.toolbench_langchain_utils.parse_category_tool_api(
            api_name, hash_api_name=False
        )
        # We only care about the tool_name for exclusion
        self.api_to_tool_map[api_name] = parsed['tool_name']
      except Exception as e:
        # If parsing fails, we assume it's an isolated tool or log warning
        # For safety, we map it to itself so it's only excluded if exactly matched
        self.api_to_tool_map[api_name] = api_name

  def process_one_negative_sample(
      self,
      workflow: tog.states.ApiUseWorkflow,
      num_tool_candidates: int,
      output_format: str = 'json',
      is_openai_format: bool = True
  ) -> Optional[dict[str, Any]]:
    """
    Creates a 'Negative' version of the given workflow with strict Tool-level exclusion.
    
    Logic:
    1. Extract the 'Positive Tools'.
    2. Identify the parent 'Tool' for each positive API.
    3. Exclude ALL APIs that belong to these parent Tools from the candidate pool.
       (e.g., if 'weather_get' is used, exclude 'weather_set' too).
    4. Sample candidates from the remaining valid pool.
    
    Args:
        workflow: The original workflow.
        num_tool_candidates: How many tools to present.
        output_format: 'json' or 'python'.
        
    Returns:
        SFT sample dict or None.
    """
    try:
      # 1. Analyze the original (positive) workflow
      chains = workflow.api_use_chains
      preds_original = {k[8:]: process_one_apiusechain(v) for k, v in chains.items()}
      pos_apis = get_tool_list_from_sample(preds_original)
      query = workflow.query

      # 2. Identify forbidden TOOL families
      forbidden_tools = set()
      for api in pos_apis:
        t_name = self.api_to_tool_map.get(api, api)
        forbidden_tools.add(t_name)
      
      # 3. Filter candidates: Exclude any API whose parent Tool is forbidden
      valid_candidates = []
      for api in self.formatter.toolname_all:
        t_name = self.api_to_tool_map.get(api, api)
        if t_name not in forbidden_tools:
          valid_candidates.append(api)

      # 4. Sample strict negatives
      if len(valid_candidates) < num_tool_candidates:
        logging.warning("Not enough candidates after strict tool-level filtering.")
        return None

      irrelevant_tools_names = self.rng.sample(valid_candidates, num_tool_candidates)
      
      # Wrap them with arguments for the prompt
      neg_tools_objs = self.formatter.wrap_apilist_with_args(irrelevant_tools_names)
      
      # 5. Construct the SFT Data
      empty_preds = {} # No tools called
      empty_pos_tools = [] # No correct tools available

      sft_sample = self.formatter.format_sft_data(
          query=query,
          pos_tools=empty_pos_tools,
          neg_tools=neg_tools_objs,
          api_preds=empty_preds,
          num_tool_candidates=num_tool_candidates,
          is_openai_format=is_openai_format,
          output_format=output_format,
      )
      
      return sft_sample

    except Exception as e:
      logging.warning(f"Failed to generate negative sample: {e}")
      return None


def process_single_negative(args_tuple):
  """Multiprocessing worker: generate one negative SFT sample and save it."""
  json_path, output_dir, num_tools, output_format, seed, max_desc_len = args_tuple
  json_data = tog.utils.data.read_json(json_path)
  workflow = tog.states.ApiUseWorkflow(**json_data)
  formatter = DataFormatter(seed=seed, max_description_length=max_desc_len)
  sampler = NegativeSampler(formatter, seed=seed)
  sample = sampler.process_one_negative_sample(
      workflow=workflow,
      num_tool_candidates=num_tools,
      output_format=output_format,
      is_openai_format=True,
  )
  if sample:
    name_no_ext = os.path.splitext(os.path.basename(json_path))[0]
    tog.utils.data.save_json(sample, os.path.join(output_dir, f"{name_no_ext}_neg.json"))
    return True
  return False


def generate_negatives_main(
    input_dir: str,
    output_dir: str,
    ratio: float,
    num_tools: int,
    output_format: str,
    seed: int,
    num_processes: int,
    file_list: list = None,
):
  """Generate negative SFT samples from an input directory of workflow JSONs."""
  random.seed(seed)
  os.makedirs(output_dir, exist_ok=True)

  if file_list is not None:
    selected_files = file_list
    logging.info(f"Using provided file list of {len(selected_files)} for negative samples.")
  else:
    all_json_files = glob.glob(os.path.join(input_dir, "*.json"))
    if not all_json_files:
      logging.error(f"No JSON files found in {input_dir}")
      return
    target_count = int(len(all_json_files) * ratio)
    logging.info(f"Found {len(all_json_files)} source files. Generating {target_count} negative samples (ratio={ratio}).")
    selected_files = random.sample(all_json_files, min(target_count, len(all_json_files)))

  process_args = [
      (fpath, output_dir, num_tools, output_format, seed + i, 256)
      for i, fpath in enumerate(selected_files)
  ]
  with multiprocessing.Pool(processes=num_processes) as pool:
    results = list(tqdm.tqdm(pool.imap(process_single_negative, process_args), total=len(process_args)))

  logging.info(f"Successfully generated {sum(results)} negative samples in {output_dir}")
