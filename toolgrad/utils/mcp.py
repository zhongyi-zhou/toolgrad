from langchain.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import StructuredTool, ToolException
from typing import Any
import asyncio
import random
import os


def get_default_mcp_dict() -> dict:
  this_dir = os.path.dirname(__file__)
  return {
      "filesystem": {
          "command": "npx",
          "args": [
              "-y", "@modelcontextprotocol/server-filesystem",
              os.path.join(this_dir, "filesystem")
          ],
          "transport": "stdio",
      },
  }


# Read-only access
BLOCKED_APIS = [
    "write_file",
    "edit_file",
    "create_directory",
    "move_file",
]


class JSONWrappingTool(StructuredTool):

  def run(self, tool_input, config=None, **kwargs):
    try:
      result = super().run(tool_input, config, **kwargs)
      return {"error": "", "response": result[0]}
    except ToolException as te:
      return {"error": str(te), "response": ""}

  async def arun(self, tool_input, config=None, **kwargs):
    try:
      result = await super().arun(tool_input, config=config, **kwargs)
      return {
          "error": "",
          "response": result[0],
      }
    except ToolException as te:
      return {"error": str(te), "response": ""}


def get_mcp_apis(
    mcp_dict: dict,
    num_apis: int = 5,
    seed: int = 42,
) -> list[StructuredTool]:

  client = MultiServerMCPClient(mcp_dict)
  tools = asyncio.run(client.get_tools())
  tools = [tool for tool in tools if tool.name not in BLOCKED_APIS]
  tools = [
      JSONWrappingTool(
          name=my_tool.name,
          description=my_tool.description,
          args_schema=my_tool.args_schema,
          func=my_tool.func,
          coroutine=my_tool.coroutine,  # carry over the MCP async entrypoint
      ) for my_tool in tools
  ]
  print(f"Found {len(tools)} APIs in MCP.")
  if num_apis > len(tools) or num_apis <= 0:
    raise ValueError(
        f"num_apis should be between 1 and {len(tools)}, got {num_apis}.")
  random.seed(seed)
  return random.sample(tools, num_apis)
