from importlib import resources
from toolgrad.utils import data
import logging
import importlib
import os
import warnings
from dotenv import load_dotenv

load_dotenv()


def get_valid_api_dict_list() -> list[dict[str, str]]:
  """Retrieve a list of valid API dictionaries.

  Returns:
      list[dict[str, str]]: A list of dictionaries containing category, tool, and api.
  """
  with resources.path(
      "toolgrad.data.toolbench",
      "filtered_v1.jsonl",
  ) as jsonl_path:
    json_list = data.read_jsonl(jsonl_path)
  return json_list


def get_tool_to_hash_prefix() -> dict[str, str]:
  """Retrieve a mapping of tool names to their hash prefixes."""
  with resources.path("toolgrad.data.toolbench",
                      "tool_to_hash_prefix.json") as json_path:
    return data.read_json(str(json_path))


def get_hash_to_tool_prefix() -> dict[str, str]:
  """Retrieve a mapping of hash prefixes to their tool names."""
  with resources.path("toolgrad.data.toolbench",
                      "hash_to_tool_prefix.json") as json_path:
    return data.read_json(str(json_path))


def get_api_docstring(category, tool, api_name_standardized):
  library_root = os.getenv("TOOLBENCH_LIBRARY_ROOT")
  if library_root is None:
    raise ValueError("TOOLBENCH_LIBRARY_ROOT environment variable is not set.")
  py_path = library_root + "/" + category + "/" + tool + "/" + "api.py"
  modulename = f"{category}/{tool}"
  spec = importlib.util.spec_from_file_location(modulename, py_path)
  api_module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(api_module)

  try:
    func = getattr(api_module, api_name_standardized)
  except AttributeError:
    logging.warning(
        f"AttributeError: {api_name_standardized} not found in {py_path}")
    return None
  return {"docstring": func.__doc__.strip(), "function": func}


def read_tool_cfg(category: str, tool_name: str) -> dict | None:
  """Read the tool configuration file for a given category and tool name."""
  library_root = os.getenv("TOOLBENCH_LIBRARY_ROOT")
  if library_root is None:
    raise ValueError("TOOLBENCH_LIBRARY_ROOT environment variable is not set.")
  tool_cfg_path = f"{library_root}/{category}/{tool_name}.json"
  if not os.path.exists(tool_cfg_path):
    warnings.warn(f"Tool config not found for {tool_name, category}")
    return None
  return data.read_json(tool_cfg_path)
