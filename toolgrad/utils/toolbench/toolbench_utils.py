from importlib import resources
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import httpx
import tqdm

from toolgrad import states
from toolgrad.utils import data
from toolgrad.utils.toolbench import toolbench_data_utils

_VALID_API_NAMES = set()
_VALID_TOOL_NAMES = set()

_TOOL_API_TO_CATEGORY = {}

_VALID_TOOLBENCH_ENCODED_TOOLNAMES = []


def filter_querys(query_dict):
  """Filter queries based on the tool names in the query dictionary."""
  global _VALID_TOOLBENCH_ENCODED_TOOLNAMES
  if _VALID_TOOLBENCH_ENCODED_TOOLNAMES.__len__() == 0:
    valid_api_list = toolbench_data_utils.get_valid_api_dict_list()
    _VALID_TOOLBENCH_ENCODED_TOOLNAMES = [
        f"{one_api['category']}-{one_api['tool']}-{one_api['api']}"
        for one_api in valid_api_list
    ]
  filtered_queries = []
  for query in query_dict:
    if query["tool_name"] in _VALID_TOOLBENCH_ENCODED_TOOLNAMES:
      filtered_queries.append(query)
  return filtered_queries


def get_rightmost_chain(root, skip_root=False):
  # start at root
  node = root
  chain = []
  while True:
    # copy everything except 'children'
    cleaned = {k: v for k, v in node.items() if k != "children"}
    chain.append(cleaned)

    children = node.get("children", [])
    if not children:
      break
    # go to last child
    node = children[-1]

  # if you want to drop the root (depth 0) and start at depth 1:
  return chain[1:] if skip_root else chain


def get_api_call_config(
    category: str,
    tool_name: str,
    api_name: str,
    tool_input: dict | str,
    strip: str = "",
):
  URL = 'http://8.130.32.149:8080/rapidapi'
  TOOLBENCH_KEY = os.environ['TOOLBENCH_KEY']

  headers = {
      'accept': 'application/json',
      'Content-Type': 'application/json',
      'toolbench_key': TOOLBENCH_KEY,
  }

  data = {
      "category": category,
      "tool_name": tool_name,
      "api_name": api_name,
      "tool_input": tool_input,
      "strip": strip,
      "toolbench_key": TOOLBENCH_KEY
  }
  return URL, headers, data


def api_caller(
    category: str,
    tool_name: str,
    api_name: str,
    tool_input: Dict[str, Any],
    strip: str = "",
) -> Any:
  URL, headers, data = get_api_call_config(
      category=category,
      tool_name=tool_name,
      api_name=api_name,
      tool_input=tool_input,
      strip=strip,
  )

  with httpx.Client() as client:
    response = client.post(URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()


async def api_caller_async(
    category: str,
    tool_name: str,
    api_name: str,
    tool_input: str | dict,
    strip: str = "",
    timeout: float = 10.0,
):
  """Asynchronous ToolBench API caller function."""
  URL, headers, data = get_api_call_config(
      category=category,
      tool_name=tool_name,
      api_name=api_name,
      tool_input=tool_input,
      strip=strip,
  )

  async with httpx.AsyncClient(timeout=httpx.Timeout(
      timeout=timeout)) as client:
    response = await client.post(URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()


def convert_api_args_type_into_python_type(
    type_str: str) -> Tuple[Type[Any], str]:
  if type_str.upper() == "STRING":
    return (str, '')
  elif type_str.upper() == "BOOLEAN":
    return (bool, '')
  elif type_str.upper() == "NUMBER":
    return (Union[float, int], '')
  # There is some limitation in the current implementation below
  # I have not walked though the whole dataset to verify whether
  # the elements in the array are all string
  elif type_str == 'ARRAY':
    return (List[str], '')
  elif type_str == "DATE (YYYY-MM-DD)":
    return (str, "date in the format of 'YYYY-MM-DD'")
  elif type_str == "GEOPOINT (latitude, longitude)":
    return (str, "geopoint in the format of 'latitude, longitude'")
  elif type_str == 'TIME (24-hour HH:MM)':
    return (str, "time in the format of '24-hour HH:MM'")
  else:
    return (str, '')


def create_api_call_function(
    category: str,
    tool_name: str,
    api_name: str,
    strip: str = "",
) -> Callable:

  def api_call(**kwargs):
    tool_input = {k: v for k, v in kwargs.items()}
    return api_caller(
        category=category,
        tool_name=tool_name,
        api_name=api_name,
        tool_input=tool_input,
        strip=strip,
    )

  return api_call


def get_target_api_cfg_from_api_list(api_list: list[dict], apiname: str):
  for api_cfg in api_list:
    name_std = standardize(api_cfg['name'])
    if name_std == apiname:
      return api_cfg
  return None


def get_category_from_tool_api(tool_name: str, api_name: str) -> str | None:
  if _TOOL_API_TO_CATEGORY == {}:
    logging.info("Loading valid tool-api-category mapping...")
    valid_cat_tool_api_dict_list = toolbench_data_utils.get_valid_api_dict_list(
    )
    for cat_tool_api_dict in tqdm.tqdm(valid_cat_tool_api_dict_list):
      _TOOL_API_TO_CATEGORY[(
          cat_tool_api_dict['tool'],
          cat_tool_api_dict['api'],
      )] = cat_tool_api_dict['category']
  if (tool_name, api_name) in _TOOL_API_TO_CATEGORY:
    return _TOOL_API_TO_CATEGORY[(tool_name, api_name)]
  else:
    logging.warning(
        f"Category not found for tool: {tool_name}, api: {api_name}")
    return None


def create_workflow_node_from_chain_node(
    chain_node: states.ApiUseChainNode,
    ids: List[str],
    length: int = 6,
) -> states.ApiUseChainNode:
  category = get_category_from_tool_api(
      chain_node.tool_name,
      chain_node.api_name,
  )
  docstring, _ = get_api_docstring(category, chain_node.tool_name,
                                   chain_node.api_name)
  description = {
      'docstring': docstring,
      'category': category,
      'tool_name': chain_node.tool_name,
      'api_name': chain_node.api_name,
  }
  node = states.ApiUseWorkflowNode(
      query=chain_node.query,
      observation=chain_node.observation,
      id=data.get_random_unique_id(length=length, existing_ids=ids),
      description=json.dumps(description),
  )
  ids.append(node.id)
  return node, ids


def jsonl_to_workflow_state(jsonlpath: str) -> states.ApiUseWorkflow:
  chain = [
      states.ApiUseChainNode(**chain_node)
      for chain_node in data.read_jsonl(jsonlpath)
  ]

  if len(chain) == 0:
    return []
  else:
    ids = []
    workflow = states.ApiUseWorkflow()
    for i, chain_node in enumerate(chain):
      node, ids = create_workflow_node_from_chain_node(chain_node, ids=ids)
      workflow.add_node(node)
      if i > 0:
        workflow.add_edge(i - 1, i)
  return workflow


def get_formatted_time():
  return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_valid_names():
  print("Getting valid names...")
  with resources.path("toolgrad.data.metadata",
                      "filtered_v1.json") as json_path:
    data = data.read_jsonl(str(json_path))
  for category in data.keys():
    for tool in data[category].keys():
      _VALID_TOOL_NAMES.add(tool)
      for api in data[category][tool]:
        _VALID_API_NAMES.add(api)
  print("Done.")


def is_valid_tool_name(tool_name):
  if len(_VALID_TOOL_NAMES) == 0:
    get_valid_names()
  return tool_name in _VALID_TOOL_NAMES


def is_valid_api_name(api_name):
  if len(_VALID_API_NAMES) == 0:
    get_valid_names()
  return api_name in _VALID_API_NAMES


def standardize_category(category):
  save_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
  while " " in save_category or "," in save_category:
    save_category = save_category.replace(" ", "_").replace(",", "_")
  save_category = save_category.replace("__", "_")
  return save_category


def standardize(string):
  res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
  string = res.sub("_", string)
  string = re.sub(r"(_)\1+", "_", string).lower()
  while True:
    if len(string) == 0:
      return string
    if string[0] == "_":
      string = string[1:]
    else:
      break
  while True:
    if len(string) == 0:
      return string
    if string[-1] == "_":
      string = string[:-1]
    else:
      break
  if string[0].isdigit():
    string = "get_" + string
  return string


def parse_api_for_tool(api_for_tool: str) -> Optional[Dict[str, str]]:
  length = api_for_tool.split("_for_").__len__()
  if length < 2:
    logging.warning(
        f"Invalid tool name: {api_for_tool}, it has too few parts: {length}")
    return None

  elif length < 10:
    parts = api_for_tool.split("_for_")
    for i in range(1, length):
      api_name = "_for_".join(parts[:i])
      tool_name = "_for_".join(parts[i:])
      if is_valid_api_name(api_name) and is_valid_tool_name(tool_name):
        return {'api': api_name, 'tool': tool_name}
  logging.warning(f"No valid split found for tool_name: {api_for_tool}")
  return None
