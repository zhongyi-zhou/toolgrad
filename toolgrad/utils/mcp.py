# from langchain.tools import StructuredTool
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None
from langchain_core.tools import StructuredTool, ToolException
from typing import Any
import asyncio
import random
import os


def get_default_mcp_dict() -> dict:
  filesystem_path = get_example_filesystem_dir()
  return {
      "filesystem": {
          "command": "npx",
          "args": [
              "-y",
              "@modelcontextprotocol/server-filesystem",
              filesystem_path,
          ],
          "transport": "stdio",
      },
  }


def get_example_filesystem_dir() -> str:
  import toolgrad
  toolgrad_root = toolgrad.__path__[0]
  filesystem_path = os.path.join(toolgrad_root, "utils", "filesystem")
  return filesystem_path


# Read-only access
ALLOWED_APIS = [
    'read_file', 'list_directory', 'read_text_file', 'directory_tree',
    'read_multiple_files'
]


class JSONWrappingTool(StructuredTool):

  def run(self, tool_input, config=None, **kwargs):
    try:
      result = super().run(tool_input, config, **kwargs)
      # Handle MCP results that may be arrays of content blocks
      if isinstance(result, list):
        # Extract text content from MCP response format
        content = "\n".join([
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in result
        ])
        return {"error": "", "response": content}
      return {"error": "", "response": str(result)}
    except ToolException as te:
      return {"error": str(te), "response": ""}
    except Exception as e:
      return {"error": str(e), "response": ""}

  async def arun(self, tool_input, config=None, **kwargs):
    try:
      result = await super().arun(tool_input, config=config, **kwargs)
      # Handle MCP results that may be arrays of content blocks
      if isinstance(result, list):
        # Extract text content from MCP response format
        content = "\n".join([
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in result
        ])
        return {"error": "", "response": content}
      return {"error": "", "response": str(result)}
    except ToolException as te:
      return {"error": str(te), "response": ""}
    except Exception as e:
      return {"error": str(e), "response": ""}


def _wrap_with_path_prefix(tool: StructuredTool, base_path: str) -> StructuredTool:
  """Wrap a tool's function to automatically prepend base_path to 'path' or 'paths' arguments if they are relative."""
  
  original_func = tool.func
  original_coroutine = tool.coroutine if hasattr(tool, 'coroutine') else None

  def wrap_args(kwargs):
    new_kwargs = kwargs.copy()
    if 'path' in new_kwargs and isinstance(new_kwargs['path'], str):
      if not os.path.isabs(new_kwargs['path']):
        new_kwargs['path'] = os.path.join(base_path, new_kwargs['path'])
    if 'paths' in new_kwargs and isinstance(new_kwargs['paths'], list):
       new_kwargs['paths'] = [os.path.join(base_path, p) if not os.path.isabs(p) else p for p in new_kwargs['paths']]
    return new_kwargs

  def new_func(**kwargs):
    return original_func(**wrap_args(kwargs))

  if original_coroutine:
    async def new_coroutine(**kwargs):
      return await original_coroutine(**wrap_args(kwargs))
  else:
    new_coroutine = None

  return StructuredTool(
      name=tool.name,
      description=tool.description,
      func=new_func,
      coroutine=new_coroutine,
      args_schema=tool.args_schema,
  )


def get_mcp_apis(
    mcp_dict: dict,
    num_apis: int = 5,
    seed: int = 42,
) -> list[StructuredTool]:

  client = MultiServerMCPClient(mcp_dict)
  tools = asyncio.run(client.get_tools())
  tools = [tool for tool in tools if tool.name in ALLOWED_APIS]

  # Wrap tools to prepend filesystem path
  filesystem_path = get_example_filesystem_dir()
  tools = [_wrap_with_path_prefix(tool, filesystem_path) for tool in tools]

  print(f"Found {len(tools)} APIs in MCP.")
  if num_apis > len(tools) or num_apis <= 0:
    raise ValueError(
        f"num_apis should be between 1 and {len(tools)}, got {num_apis}.")
  random.seed(seed)
  return random.sample(tools, num_apis)
