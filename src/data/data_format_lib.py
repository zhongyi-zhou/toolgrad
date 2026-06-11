import ast
import sys
import os
import random
import json
import glob
import argparse
import multiprocessing
import logging
from typing import Any, Optional, Union
from pydantic import BaseModel

# Add current directory to sys.path
sys.path.append(os.getcwd())
import tqdm
import toolgrad as tog
from src.data import prompt_template

def format_python_call(tool_name: str, tool_input: Union[str, dict]) -> str:
  """Formats a tool call as a Python-like function call string using the AST module.
     e.g. tool_name(arg1='val1', arg2='val2')
  """
  if isinstance(tool_input, str):
    try:
      tool_input = json.loads(tool_input)
    except json.JSONDecodeError:
      logging.error(f"tool_input for {tool_name} is a string but not valid JSON: {tool_input}")
      return f"{tool_name}()"

  if isinstance(tool_input, dict):
    # Construct AST for the call: func(key=value, ...)
    keywords = []
    for k, v in tool_input.items():
      try:
        val_node = ast.parse(repr(v)).body[0].value
      except Exception:
        val_node = ast.Constant(value=str(v))
      
      keywords.append(ast.keyword(arg=k, value=val_node))
    
    call_node = ast.Call(
      func=ast.Name(id=tool_name, ctx=ast.Load()),
      args=[],
      keywords=keywords
    )
    
    # unparse was added in 3.9
    if sys.version_info >= (3, 9):
      return ast.unparse(call_node)
    else:
      args_str = ", ".join([f"{k}={repr(v)}" for k, v in tool_input.items()])
      return f"{tool_name}({args_str})"

  return f"{tool_name}()"



class ApiuseAction(BaseModel):
  tool: str
  tool_input: Union[str, dict]


class ApiArgs(BaseModel):
  description: str
  required: list[dict]
  optional: list[dict]


# A Graph consists of maximum 10 chains.
_CHAIN_NAMES = ["chain_" + str(i) for i in range(1, 11)]


def format_openai_sft_data(
    user_prompt: str,
    ai_resp: str,
    system_prompt: Optional[str],
) -> dict:
  msg_all = []
  if system_prompt:
    msg_all.append({
      "role": "system",
      "content": str(system_prompt),
    })
  else:
    logging.info("No system prompt is provided.")
  msg_all.append({
    "role": "user",
    "content": str(user_prompt),
  })
  msg_all.append({
    "role": "assistant",
    "content": str(ai_resp),
  })
  return {"messages": msg_all}


def process_one_apiusechain(
    api_use_chain: tog.states.ApiUseChain,) -> list[ApiuseAction]:
  """Process one api use chain.
  
  It returns a list of dictionaries, each of which contains the tool and tool input.
  """
  chain_record = []
  for step in api_use_chain.intermediate_steps:
    chain_record.append(
      ApiuseAction(
        tool=step[0].tool,
        tool_input=step[0].tool_input,
      ))
  return chain_record


def get_tool_list_from_sample(
    sample: dict[str, list[ApiuseAction]]) -> list[str]:
  """Get the list of tools from the sample.
  
  The data format of the sample is Dict[str, ApiuseAction].
  """
  tool_list = []
  for key, chain in sample.items():
    for step in chain:
      tool_list.append(step.tool)
  return tool_list


class DataFormatter:

  def __init__(self, seed: int = 1234, max_description_length: int = 256):
    self.toolname_all = self.get_all_toolname()
    random.seed(seed)
    self.max_description_length = max_description_length

  def get_all_toolname(self) -> list[str]:
    # Use the mapping from toolbench_langchain_utils
    toolname_all = list(tog.utils.toolbench.toolbench_langchain_utils.TOOLNAME_TO_HASH.keys())
    return toolname_all

  def get_api_args(self, name: str) -> ApiArgs:
    assert name in self.toolname_all
    category, tool_name, api_name = tog.utils.toolbench.toolbench_langchain_utils.parse_category_tool_api(
      name, hash_api_name=False).values()
    
    # toolbench_utils.LIBRARY_ROOT is not defined, so we define it here or get from env
    LIBRARY_ROOT = os.getenv("TOOLBENCH_LIBRARY_ROOT", None)
    if LIBRARY_ROOT is None:
      raise ValueError("TOOLBENCH_LIBRARY_ROOT is not set.")
    tool_cfg_path = f"{LIBRARY_ROOT}/{category}/{tool_name}.json"
    
    tool_cfg = tog.utils.data.read_json(tool_cfg_path)
    api_cfg = tog.utils.toolbench.toolbench_utils.get_target_api_cfg_from_api_list(
      tool_cfg['api_list'],
      api_name,
    )
    if api_cfg is None:
      return ApiArgs(
        description="",
        required=[],
        optional=[],
      )
    else:
      required = api_cfg.get('required_parameters', [])
      optional = api_cfg.get('optional_parameters', [])
      description = api_cfg.get('description', '') or ""
      if len(description) > self.max_description_length:
        description = description[:self.max_description_length]
      
      for param in required + optional:
        if 'description' in param and len(param['description']) > self.max_description_length:
          param['description'] = param['description'][:self.max_description_length]
          
      return ApiArgs(
        description=description,
        required=required,
        optional=optional,
      )

  def wrap_apilist_with_args(
      self,
      apilist: list[str],
  ) -> list[dict[str, ApiArgs]]:
    if len(apilist) == 0:
      return []
    assert isinstance(apilist, list) and all(
      [api in self.toolname_all for api in apilist])
    return [{name: self.get_api_args(name)} for name in apilist]

  def sample_neg_tools(
      self,
      full_sample_size: int,
      pos_tools: list[str],
  ) -> list[str]:
    if full_sample_size == -1:
      return []
    num_neg_samples = full_sample_size - len(set(pos_tools))
    neg_tools = random.sample(
      [tool for tool in self.toolname_all if tool not in pos_tools],
      num_neg_samples,
    )
    assert not set(neg_tools) & set(pos_tools)
    return neg_tools

  def process_one_workflow(
      self,
      workflow: tog.states.ApiUseWorkflow,
      num_tool_candidates: int,
  ) -> dict[str, Any]:
    chains = workflow.api_use_chains
    preds = {k[8:]: process_one_apiusechain(v) for k, v in chains.items()}
    query = workflow.query
    pos_tools = get_tool_list_from_sample(preds)
    neg_tools = self.sample_neg_tools(
      full_sample_size=num_tool_candidates,
      pos_tools=pos_tools,
    )
    return {
      "query": query,
      "pos_tools": self.wrap_apilist_with_args(pos_tools),
      "neg_tools": self.wrap_apilist_with_args(neg_tools),
      "api_preds": preds,
    }

  def _convert_to_bfcl_tool_def(self, name: str, args: ApiArgs) -> dict:
    """Converts ApiArgs (custom format) to BFCL/OpenAI standard JSON tool definition."""
    parameters = {
      "type": "object",
      "properties": {},
      "required": []
    }
    
    for param in args.required:
      param_name = param.get("name")
      if not param_name: continue
      parameters["properties"][param_name] = {
        "type": param.get("type", "string"),
        "description": param.get("description", "")
      }
      parameters["required"].append(param_name)
      
    for param in args.optional:
      param_name = param.get("name")
      if not param_name: continue
      parameters["properties"][param_name] = {
        "type": param.get("type", "string"),
        "description": param.get("description", "")
      }
    
    return {
      "name": name,
      "description": args.description,
      "parameters": parameters
    }

  def get_candidate_pools(self, tools: list[dict[str, ApiArgs]]) -> list[dict]:
    # Flatten the list of single-entry dicts {tool_name: ApiArgs}
    all_tool_dict = {key: value for d in tools for key, value in d.items()}
    
    # Convert to BFCL list format
    bfcl_tools = []
    for name, args in all_tool_dict.items():
      bfcl_tools.append(self._convert_to_bfcl_tool_def(name, args))
      
    return bfcl_tools

  def format_ai_reponse(self, ai_preds: dict[str, list[ApiuseAction]], output_format: str = 'json') -> dict:
    if output_format == 'python':
      # take all actions from all chains into a single list.
      all_actions = []
      for chain in ai_preds.values():
        for action in chain:
          all_actions.append(format_python_call(action.tool, action.tool_input))
      # Format: "[func1(...), func2(...)]"
      return "[" + ", ".join(all_actions) + "]"
    
    if output_format == 'json':
      # Standard BFCL JSON format: [{"function": "name", "parameters": {...}}]
      if not ai_preds:
        return []
          
      # Get the first chain (values are lists of ApiuseAction)
      first_chain = next(iter(ai_preds.values()))
      
      bfcl_response = []
      for action in first_chain:
        # Convert tool_input: if it's a string JSON, load it.
        params = action.tool_input
        if isinstance(params, str):
          try:
            params = json.loads(params)
          except:
            pass # keep as string if not valid json
                  
        bfcl_response.append({
          "function": action.tool,
          "parameters": params
        })
      
      return bfcl_response
    
    return {
      _CHAIN_NAMES[i]: [
        api_use_chain.model_dump() for api_use_chain in v
      ] for i, v in enumerate(ai_preds.values())
    }

  def format_sft_data(
      self,
      query: str,
      pos_tools: list[dict[str, ApiArgs]],
      neg_tools: list[dict[str, ApiArgs]],
      api_preds: dict[str, list[ApiuseAction]],
      num_tool_candidates: int,
      is_openai_format: bool = True,
      output_format: str = 'json',
  ) -> dict:

    if output_format == 'python':
      base_system_prompt = prompt_template.SYSTEM_PROMPT_PYTHON
    elif output_format == 'json':
      schema_str = json.dumps(prompt_template.RESPONSE_SCHEMA_DICT)
      base_system_prompt = prompt_template.SYSTEM_PROMPT + prompt_template.RETURN_SCHEMA.format(
        response_json_schema=schema_str
      )
    else:
      raise ValueError(f"Unsupported output format: {output_format}")

    if num_tool_candidates < 0:
      system_prompt = base_system_prompt
    else:
      system_prompt = base_system_prompt + prompt_template.SELECTION_POOL.format(
        selection_pool=self.get_candidate_pools(pos_tools + neg_tools))
    user_prompt = query
    ai_resp = self.format_ai_reponse(api_preds, output_format=output_format)
    if is_openai_format:
      return format_openai_sft_data(
        user_prompt=user_prompt,
        ai_resp=str(ai_resp), # Ensure string conversion happens here if it's a dict
        system_prompt=system_prompt,
      )
    else:
      return {
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'ai_resp': ai_resp,
      }

  def run_format_data(
      self,
      json_path: str,
      num_tool_candidates: int,
      is_openai_format: bool = True,
      save_dir: Optional[str] = None,
      output_format: str = 'json',
      max_description_length: int = 256,
  ) -> tuple[dict, bool]:
    try:
      json_data = tog.utils.data.read_json(json_path)
    except Exception as e:
      logging.warning(f"Failed to read JSON from {json_path}: {e}")
      return {}, False
    try:
      workflow = tog.states.ApiUseWorkflow(**json_data)
    except Exception as e:
      logging.error(f"Failed to parse workflow from {json_path}: {e}")
      return {}, False
    query, pos_tools, neg_tools, api_preds = (self.process_one_workflow(
      workflow, num_tool_candidates)).values()

    sft_sample = self.format_sft_data(
      query,
      pos_tools,
      neg_tools,
      api_preds,
      num_tool_candidates,
      is_openai_format=is_openai_format,
      output_format=output_format,
    )
    if save_dir:
      os.makedirs(save_dir, exist_ok=True)
      filename = os.path.basename(json_path)
      tog.utils.data.save_json(sft_sample, f"{save_dir}/{filename}")
      metadata_dir = os.path.join(save_dir, "metadata")
      os.makedirs(metadata_dir, exist_ok=True)
      pos_tools_dict = self.get_candidate_pools(pos_tools)
      neg_tools_dict = self.get_candidate_pools(neg_tools)
      if num_tool_candidates != -1:
        assert len(pos_tools_dict) + len(neg_tools_dict) == num_tool_candidates
      api_preds = {
        k: [v.model_dump() for v in chain]
        for k, chain in api_preds.items()
      }
      tog.utils.data.save_json(
        {
          "pos_tools": pos_tools_dict,
          "neg_tools": neg_tools_dict,
          "api_preds": api_preds,
        }, f"{metadata_dir}/{filename}")
    return sft_sample, True


def process_json(json_path, seed, num_tool_candidates, outdir, max_description_length, output_format):
  # Each process creates its own instance of DataFormatter.
  formatter = DataFormatter(seed=seed, max_description_length=max_description_length)
  _, is_success = formatter.run_format_data(
    json_path=json_path,
    num_tool_candidates=num_tool_candidates,
    is_openai_format=True,
    save_dir=outdir,
    output_format=output_format,
  )
  return is_success


def process_json_wrapper(args):
  json_path, seed, num_tool_candidates, outdir, max_description_length, output_format = args
  return process_json(json_path, seed, num_tool_candidates, outdir, max_description_length, output_format)




def create_args():
  parser = argparse.ArgumentParser(description="Format tool execution data for SFT.")
  parser.add_argument(
    '--input_dir',
    type=str,
    required=True,
    help='Input directory containing JSON files.',
  )
  parser.add_argument(
    '--output_dir',
    type=str,
    required=True,
    help='Output directory for formatted data.',
  )
  parser.add_argument(
    '--num_tools',
    type=int,
    default=20,
    choices=[20, -1],
    help='Number of tool candidates (positive + negative).',
  )
  parser.add_argument(
    '--num_processes',
    type=int,
    default=16,
    help='Number of processes for parallel processing.',
  )
  parser.add_argument(
    '--seed',
    type=int,
    default=1234,
    help='Random seed.',
  )
  parser.add_argument(
    '--max_description_length',
    type=int,
    default=256,
    help='Maximum length of the tool description.',
  )
  parser.add_argument(
    '--output_format',
    type=str,
    default='json',
    choices=['json', 'python'],
    help='Output format for the AI response.',
  )
  args = parser.parse_args()
  return args


def format_data_main(
    input_dir: str,
    output_dir: str,
    num_tools: int,
    num_processes: int,
    seed: int,
    max_description_length: int,
    output_format: str = 'json',
    file_list: list[str] = None,
):
  random.seed(seed)
  
  if file_list is not None:
    json_path_all = file_list
    logging.info(f"Using provided file list of {len(json_path_all)} for SFT formatting.")
  else:
    # Find all JSON files in the input directory
    json_path_all = glob.glob(os.path.join(input_dir, "*.json"))
    if not json_path_all:
      logging.warning(f"No JSON files found in {input_dir}")
      return
        
    logging.info(f"Found {len(json_path_all)} files in {input_dir}")
  
  # Generate seeds for each file
  seeds_all = random.sample(range(1000000), len(json_path_all))

  os.makedirs(output_dir, exist_ok=True)
  
  # Prepare arguments for parallel processing
  process_args = [(j, s, num_tools, output_dir, max_description_length, output_format)
      for j, s in zip(json_path_all, seeds_all)]

  with multiprocessing.Pool(processes=num_processes) as pool:
    results = list(
      tqdm.tqdm(pool.imap(process_json_wrapper, process_args), total=len(process_args)))
  
  num_success = sum(results)
  logging.info(f"Total number of successful samples: {num_success} out of {len(json_path_all)}")


if __name__ == '__main__':
  args = create_args()
  format_data_main(
    input_dir=args.input_dir,
    output_dir=args.output_dir,
    num_tools=args.num_tools,
    num_processes=args.num_processes,
    seed=args.seed,
    max_description_length=args.max_description_length,
    output_format=args.output_format,
  )
