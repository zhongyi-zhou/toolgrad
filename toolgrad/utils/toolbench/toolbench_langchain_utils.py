from collections.abc import Awaitable
from importlib import resources
import logging
import os
from typing import Any, Callable, Literal, Tuple, Type, Union
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import functools
# from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import create_model
from pydantic import Field

from toolgrad.utils import data
from toolgrad.utils.toolbench import templates
from toolgrad.utils.toolbench import toolbench_utils
from toolgrad.utils.toolbench import toolbench_data_utils
from toolgrad import states

def get_toolname_to_hash(version: str = toolbench_data_utils.DEFAULT_VERSION):
  """Get toolname to hash mapping for a specific version."""
  return toolbench_data_utils.get_tool_to_hash_prefix(version=version)


def get_hash_to_toolname(version: str = toolbench_data_utils.DEFAULT_VERSION):
  """Get hash to toolname mapping for a specific version."""
  return toolbench_data_utils.get_hash_to_tool_prefix(version=version)


HASH_TO_TOOLNAME = get_hash_to_toolname()
TOOLNAME_TO_HASH = get_toolname_to_hash()


def process_tool(
    tool_candidate: dict,
    version: str,
    name_is_hash: bool = True,
) -> StructuredTool:
  category = tool_candidate["category"]
  tool = tool_candidate["tool"]
  api = tool_candidate["api"]
  return create_one_langchain_tool(
      category=category,
      tool_name=tool,
      api_name=api,
      hash_api_name=name_is_hash,
      version=version,
  )


def get_all_apis(
    num_apis: int = 50,
    name_is_hash: bool = True,
    max_workers: int = 32,
    version: str = toolbench_data_utils.DEFAULT_VERSION,
) -> list[StructuredTool]:
  tool_library = os.getenv("LIBRARY_ROOT")
  pkg = f"toolgrad.data.toolbench.{version}"
  with resources.path(pkg, "data.jsonl") as json_path:
    filter_apis = data.read_jsonl(str(json_path))
  if num_apis > len(filter_apis) or num_apis <= 0:
    warnings.warn(
        f"num_tools should be between 1 and {len(filter_apis)}, got {num_apis}.\n Using all tools."
    )
  else:
    filter_apis = random.sample(filter_apis, num_apis)
  processed_apis = []

  process_tool_hash = functools.partial(process_tool, name_is_hash=name_is_hash)
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_tool = {
        executor.submit(process_tool_hash, tool_candidate, version):
            tool_candidate for tool_candidate in filter_apis
    }

    for future in as_completed(future_to_tool):
      tool_candidate = future_to_tool[future]
      try:
        result = future.result()
        if result is not None:
          processed_apis.append(result)
      except Exception as exc:
        warnings.warn(f"API {tool_candidate} generated an exception: {exc}")

  logging.info(
      f"Successfully processed {len(processed_apis)} tools out of {len(filter_apis)}."
  )
  return processed_apis


def process_one_intermediate_step(
    intermediate_step: Tuple[dict, Union[dict, str]],
    remove_msglog: bool = True,
    hash_to_name: bool = True,
) -> Tuple[dict, Union[dict, str]]:
  if not remove_msglog and not hash_to_name:
    return intermediate_step
  action, result = intermediate_step
  if isinstance(result, tuple):
    if len(result) == 1:
      result = result[0]
    else:
      result = str(result)
  if isinstance(result, list):
    result = str(result)
    
  if remove_msglog:
    action.message_log = []
  if hash_to_name:
    api_hashname = action.tool
    api_name = HASH_TO_TOOLNAME.get(api_hashname, '')
    if api_name == '':
      # Check if api_hashname is already in full name format (LLM error case)
      if api_hashname in HASH_TO_TOOLNAME.values():
        logging.warning(
            f"API tool name '{api_hashname}' is already in full format, expected hash format like 'tool_xxx'. Using as-is."
        )
        api_name = api_hashname
      else:
        logging.error(
            f"Invalid hash '{api_hashname}': not found in hash-to-tool mapping. Returning empty step to continue execution."
        )
        return None
    action.tool = api_name
    action.log = str(
        data.replace_keyword(
            action.log,
            old=api_hashname,
            new=api_name,
        ))
    result = data.replace_keyword(
        result,
        old=api_hashname,
        new=api_name,
    )
  return (action, result)


def parse_category_tool_api(
    name: str,
    hash_api_name: bool = True,
) -> dict[str, str]:
  if hash_api_name:
    if name not in HASH_TO_TOOLNAME.keys():
      logging.error(f"invalid hash {name}")
      return {
          'category': '',
          'tool_name': '',
          'api_name': '',
      }
    name = HASH_TO_TOOLNAME[name]
  split_result = name.split('-')
  if len(split_result) != 3:
    warnings.warn(
        f"Invalid parsed list: {split_result}. Expected 3 elements, but got {len(split_result)}\n Solving by returning all ''."
    )
    return {'category': '', 'tool_name': '', 'api_name': ''}
  return {
      'category': split_result[0],
      'tool_name': split_result[1],
      'api_name': split_result[2]
  }


class ToolbenchStructuredTool(StructuredTool):
  api_name: str = Field(..., description="The name of the API.")
  tool_name: str = Field(..., description="The name of the tool.")
  category: str = Field(..., description="The category of the tool.")

  @classmethod
  def from_function(
      cls,
      api_name: str,
      tool_name: str,
      category: str,
      func: Callable | None = None,
      coroutine: Callable[..., Awaitable[Any]] | None = None,
      name: str | None = None,
      description: str | None = None,
      return_direct: bool = False,
      args_schema: type[BaseModel] | None = None,
      infer_schema: bool = True,
      *,
      response_format: Literal["content", "content_and_artifact"] = "content",
      parse_docstring: bool = False,
      error_on_invalid_docstring: bool = False,
      **kwargs: Any,
  ) -> StructuredTool:
    return cls(name=name,
               description=description,
               func=func,
               args_schema=args_schema,
               return_direct=return_direct,
               api_name=api_name,
               tool_name=tool_name,
               category=category)


def create_pydantic_model(
    spec: dict[str, Any],
    model_name: str,
) -> Type[BaseModel]:
  fields = {}
  for param in spec.get('required_parameters', []):
    param_type, comment = toolbench_utils.convert_api_args_type_into_python_type(
        param['type'])
    fields[param['name']] = (
        param_type,
        Field(..., description=param.get('description', '') + '\n' + comment),
    )

  for param in spec.get('optional_parameters', []):
    param_type, comment = toolbench_utils.convert_api_args_type_into_python_type(
        param['type'])
    fields[param['name']] = (
        param_type,
        Field(..., description=param.get('description', '') + '\n' + comment),
    )
  return create_model(model_name, **fields)


def create_one_langchain_tool(
    category: str,
    tool_name: str,
    api_name: str,
    model_name: str | None = None,
    hash_api_name: bool = True,
    version: str = toolbench_data_utils.DEFAULT_VERSION,
) -> StructuredTool | None:
  """Create a LangChain tool from a toolbench API."""

  tool_cfg = toolbench_data_utils.read_tool_cfg(
      category=category,
      tool_name=tool_name,
  )
  if tool_cfg is None:
    raise ValueError(
        f"API configuration not found for {category}, {tool_name}, {api_name}")
  api_cfg = toolbench_utils.get_target_api_cfg_from_api_list(
      tool_cfg['api_list'],
      api_name,
  )
  if model_name is None:
    model_name = f"{tool_name}-{api_name}-Args"
  if api_cfg is None:
    raise ValueError(f"api_cfg is None for {tool_name} {api_name}")
  if api_cfg is None:
    raise ValueError(f"api_cfg is None for {tool_name} {api_name}")
  ArgsModel = create_pydantic_model(api_cfg, model_name)
  func = toolbench_utils.create_api_call_function(
      category=category,
      tool_name=tool_name,
      api_name=api_name,
  )

  formatted_description = templates.ApiDescriptionTemplate.format(
      api_name=api_name,
      tool_name=tool_name,
      category=category,
      api_description=api_cfg.get('description', ''),
      tool_description=tool_cfg.get('description', ''),
  )
  if formatted_description.__len__() > 1024:
    formatted_description = formatted_description[:1024]

  name = f"{category}-{tool_name}-{api_name}"
  if hash_api_name:
    if version == toolbench_data_utils.DEFAULT_VERSION:
      mapping = TOOLNAME_TO_HASH
    else:
      mapping = get_toolname_to_hash(version=version)
    if name not in mapping.keys():
      raise ValueError(f"{name} is not a valid name")
    name = mapping.get(name, '')

  tool = StructuredTool.from_function(
      func=func,
      name=name,
      description=formatted_description,
      args_schema=ArgsModel,
      return_direct=True,
  )
  return tool


def postprocess_proposalall(
    proposals: states.ApiProposalAll,
    convert_hash_to_name: bool = True,
    version: str = toolbench_data_utils.DEFAULT_VERSION,
) -> states.ApiProposalAll:
  """
  Post-process proposals to convert hash names to real names.
  
  This is critical for Gemini models which have a ~64 character limit on tool names.
  The proposer sees hashed names and generates instructions with those hashes.
  We need to:
  1. Convert api.name from hash to real name
  2. Replace hash names in instruction text with real names
  
  This ensures the instruction matches the actual tool names available to the executor.
  """
  if convert_hash_to_name:
    for proposal in proposals.proposals:
      # Build mapping of hash -> real name for this proposal
      hash_to_name_mapping = {}
      mapping = get_hash_to_toolname(version=version)
      
      valid_apis = []
      for api in proposal.api:
        if api.name not in mapping.keys():
          logging.warning(f"Skipping invalid API hash: {api.name}")
          continue
        
        hash_to_name_mapping[api.name] = mapping[api.name]
        # Convert the API name field
        api.name = mapping[api.name]
        valid_apis.append(api)
      
      # Update the proposal with only valid APIs
      proposal.api = valid_apis

      # Replace all hash names in the instruction text
      for hash_name, real_name in hash_to_name_mapping.items():
        proposal.instruction = proposal.instruction.replace(
            hash_name, real_name)

  return proposals
