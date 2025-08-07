import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import ValidationError
import httpx

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain.tools import StructuredTool
from langchain_community.callbacks.manager import OpenAICallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from toolgrad import states
from toolgrad.modules import prompt_lib
from toolgrad.utils import data
from toolgrad.utils import langchain
from toolgrad.utils.toolbench import toolbench_langchain_utils
from toolgrad.utils import log_utils
import asyncio

from langchain_core.language_models.chat_models import BaseChatModel

_MAX_TOOL_AGENT_ITERATIONS = 3
_JST = datetime.timezone(datetime.timedelta(hours=+9), 'JST')


def get_success_execution_id(
    intermediate_steps: list[Tuple[ToolAgentAction, dict]],
    is_toolbench: bool = True,
) -> int | None:
  """Return the index of the last successful step in the intermediate steps or None if none of them are successful."""
  success_step_id = None
  if is_toolbench:
    # ToolBench API returns a dict in the format of {''error': ..., 'response': ...}
    for i, intermediate_step in enumerate(intermediate_steps):
      action, result = intermediate_step
      assert isinstance(result, dict)
      if result.get('error', None) == "":
        success_step_id = i
  else:
    # For MCP APIs without standard dict structure, we use heuristic to determine the success step.
    success_step_id = intermediate_steps.__len__() - 1
  return success_step_id


def select_object_by_id(objects: List, id: str):
  for obj in objects:
    if not hasattr(obj, 'id'):
      raise ValueError(f"Object {obj} does not have an id attribute.")
    if obj.id == id:
      return obj
  logging.warning(f"Object with id {id} not found.")
  return None


def generate_chain_description_from_intermediate_steps(
    intermediate_steps: List[Tuple[ToolAgentAction, Optional[dict]]],) -> str:
  prompt = prompt_lib.CHAIN_SUMMARIZER
  llm = langchain.create_llm()
  return (prompt | llm).invoke({
      "api_execution_steps": intermediate_steps,
  })


def update_io_from_intermediate_steps(chain_dict: Dict[str, Any],):
  intermediate_steps = chain_dict.get('intermediate_steps', [])
  if intermediate_steps is []:
    logging.warning(
        "No intermediate steps found in the chain. Returning empty inputs and outputs."
    )
    chain_dict['input'] = {}
    chain_dict['output'] = {}
    return chain_dict

  # equivalent to the following pseudocode:
  # assert isinstance(intermediate_steps, List[Tuple[ToolAgentAction, Optional[dict]]])
  assert isinstance(intermediate_steps, list)
  assert all(
      isinstance(item, tuple) and len(item) == 2 and
      isinstance(item[0], ToolAgentAction)
      for item in intermediate_steps)  # Check tuple structure
  chain_dict['input'] = intermediate_steps[0][0].tool_input
  chain_dict['output'] = intermediate_steps[-1][1]
  return chain_dict


def convert_dedicated_apiproposals_to_standard_apiproposals(
    dedicated_api_proposals: states.ApiProposalAll,) -> states.ApiProposalAll:
  """Convert the dedicated api proposals to standard api proposals."""
  assert dedicated_api_proposals.__class__.__name__ == "DedicatedApiProposalAll"
  return states.ApiProposalAll(**dedicated_api_proposals.model_dump())


def get_api_use_chains_without_intermediate_steps(
    api_use_chains: Dict[str, states.ApiUseChain],
) -> Dict[str, Dict[str, states.ApiUseChainBase]]:
  """Get the api use chains without intermediate steps."""

  return {
      chain_id:
          states.ApiUseChainBase({
              "chain_id": chain.chain_id,
              "description": chain.description,
          }) for chain_id, chain in api_use_chains.items()
  }


def find_tool_by_name(
    name: str,
    tools: List[StructuredTool],
) -> StructuredTool | None:
  for tool in tools:
    if tool.name == name:
      return tool
  return None


def save_generation(
    output_dir: str,
    seed: int,
    workflow: Optional[states.ApiUseWorkflow] = None,
    cb: Optional[OpenAICallbackHandler] = None,
    format_digits: int = 8,
    **kwargs,
):
  formatted_seed = data.format_int_to_k_digits(seed, format_digits)

  # Save the workflow
  if not isinstance(workflow, states.ApiUseWorkflow):
    logging.critical("Invalid workflow found. Saving an empty file.")
    workflow = {}
  else:
    workflow = workflow.model_dump()
  workflow_path = f"{output_dir}/data/{formatted_seed}.json"
  os.makedirs(os.path.dirname(workflow_path), exist_ok=True)
  data.save_json(workflow, workflow_path)

  # Save Metadata
  cost = {k: v for k, v in cb.__dict__.items() if k != "_lock"}

  metadata = {
      'cost': cost,
      'time': datetime.datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S"),
      'config': {
          **kwargs
      },
  }

  metadata_path = f"{output_dir}/metadata/{formatted_seed}.json"
  os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
  data.save_json(
      metadata,
      metadata_path,
  )


def get_literal_api_proposals_pydantic_model(
    apis: List[StructuredTool]) -> states.ApiProposalAll:
  """Get a more strict Pydantic model of `ApiProposal` with literal constraints.
    
    The `Literal` constrains what it can propose based on self.apis values in the current timstamp."""
  apiuse_chainnode_customized = states.create_dedicated_apiusechainnode(
      allowed_values=[api.name for api in apis])
  api_proposal_customized = states.create_dedicated_apiproposal(
      api_node_model=apiuse_chainnode_customized)
  return states.create_dedicated_apiproposalall(
      api_proposal_model=api_proposal_customized)


def create_api_proposer(apis: List[StructuredTool]) -> BaseChatModel:
  llm = langchain.create_llm()

  # Dynamically create the output structure.
  customized_api_proposals = get_literal_api_proposals_pydantic_model(apis)
  api_proposer = prompt_lib.TOOLUSE_PROPOSER_WOO | llm.with_structured_output(
      customized_api_proposals)
  return api_proposer


def get_all_tool_annotations(
    apis: List[StructuredTool],) -> dict[str, dict[str, Any]]:

  def get_one_tool_annotation(tool: StructuredTool):
    if isinstance(tool.args_schema, dict):
      args_schema = tool.args_schema
    elif issubclass(tool.args_schema, BaseModel):
      args_schema = tool.args_schema.model_json_schema()
    else:
      raise ValueError(
          f"Invalid args_schema type: {type(tool.args_schema)} for tool {tool.name}"
      )
    return {
        "name": tool.name,
        "description": tool.description,
        "args_schema": args_schema,
    }

  return {tool.name: get_one_tool_annotation(tool) for tool in apis}


# Executor Modules
def create_api_executor(
    apis_in_one_proposal: List[StructuredTool],) -> AgentExecutor:
  llm = langchain.create_llm()
  prompt = ChatPromptTemplate.from_messages([
      ("system", "You are a helpful assistant"),
      ("human", prompt_lib.API_EXECUTOR.format(plan="{plan}")),
      ("placeholder", "{agent_scratchpad}"),
  ])
  agent = create_tool_calling_agent(llm, apis_in_one_proposal, prompt)
  agent_executor = AgentExecutor(
      agent=agent,
      tools=apis_in_one_proposal,
      return_intermediate_steps=True,
      max_iterations=_MAX_TOOL_AGENT_ITERATIONS,
  )
  return agent_executor


def create_api_execution_validator(key_list: List[str]):
  llm = langchain.create_llm()
  api_execution_validation = states.create_apiexecutionvalidation(
      allowed_values=key_list)
  api_execution_validator = prompt_lib.API_EXECUTION_VALIDATOR | llm.with_structured_output(
      api_execution_validation)
  return api_execution_validator


async def single_api_execute_step(
    proposal: states.ApiProposal,
    is_hash_api_name: bool = False,
    is_toolbench: bool = True,
    sampled_apis: List[StructuredTool] | None = None,
) -> states.ApiReport:

  def failure_report() -> states.ApiReport:
    return states.ApiReport(
        id=proposal.id,
        success=False,
        api_call_step=None,
        description="",
    )

  if is_toolbench:
    apis = [
        toolbench_langchain_utils.create_one_langchain_tool(
            **toolbench_langchain_utils.parse_category_tool_api(
                api_proposal.name, hash_api_name=is_hash_api_name))
        for api_proposal in proposal.api
        if api_proposal is not None
    ]
  else:
    if sampled_apis is None:
      raise ValueError(
          "sampled_apis must be provided when is_toolbench is False")
    apis = [
        valid_api for api in proposal.api
        if (valid_api := find_tool_by_name(api.name, sampled_apis)) is not None
    ]
  api_executor = create_api_executor(apis_in_one_proposal=apis)
  try:
    api_executions = await api_executor.ainvoke({
        "plan": proposal.instruction,
    })
  except Exception as e:
    logging.info(
        f"Failed to execute the API: {proposal.id}, adding a placeholder report."
    )
    return failure_report()

  if 'intermediate_steps' not in api_executions:
    return failure_report()
  processed_steps = [
      toolbench_langchain_utils.process_one_intermediate_step(
          inter_step, remove_msglog=True, hash_to_name=is_toolbench)
      for inter_step in api_executions['intermediate_steps']
  ]

  processed_steps = [e for e in processed_steps if e is not None and e != ()]
  if processed_steps.__len__() == 0:
    return failure_report()

  last_success_step_id = get_success_execution_id(processed_steps)

  if last_success_step_id is None:
    logging.warning(f"One API execution fails.")
    return failure_report()

  api_report = {
      'id': proposal.id,
      'success': True,
      'api_call_step': processed_steps[last_success_step_id],
      'description': None,
  }
  return states.ApiReport(**api_report)


@log_utils.log_time
def api_execute_step(
    state: states.ApiUseState,
    is_hash_api_name: bool = False,
    is_toolbench: bool = True,
) -> dict[str, states.ApiReport]:
  proposals = state.api_proposals
  if proposals == None:
    raise ValueError("No valid proposal to execute")
  assert isinstance(proposals, states.ApiProposalAll)

  coros = [
      single_api_execute_step(
          proposal,
          is_hash_api_name=is_hash_api_name,
          is_toolbench=is_toolbench,
          sampled_apis=state.sampled_apis,
      ) for proposal in proposals.proposals
  ]
  loop = asyncio.new_event_loop()
  try:
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(asyncio.gather(*coros))
  finally:
    loop.close()

  api_reports = {
      proposal.id: result
      for proposal, result in zip(proposals.proposals, results)
  }
  return api_reports


# Selector modules
def create_api_selector(
    output_schema=states.ApiSelection) -> states.ApiSelection:
  llm = langchain.create_llm()
  api_selector = prompt_lib.SELECTOR | llm.with_structured_output(output_schema)
  return api_selector


# Workflow modules
def get_valid_chainids(max_num_chains: int = 10) -> list[str]:
  return ["api_use_chain_" + str(i) for i in range(1, max_num_chains + 1)]


def chain_update_step(
    workflow_cur: states.ApiUseWorkflow,
    which_chain: str,
    api_step: Tuple[ToolAgentAction, Union[dict, str]],
    num_saturated_warnings: int | None = 3,
    max_num_chains: int = 10,
) -> dict[str, Any]:
  """Update or create a chain in the workflow.

  Args:
      workflow_cur (states.ApiUseWorkflow): the current workflow state.
      which_chain (str): the chain ID to update or '-1' to create a new chain.
      api_step (Tuple[ToolAgentAction, Union[dict, str]]): the API step to add to the chain.
      num_saturated_warnings (int | None): Avoid the chains to be created too many times. Defaults to 3.
      max_num_chains (int, optional): max number of chains to create. Defaults to 10.

  Returns:
      dict[str, Any]: updated chain or None if no valid chain can be created.
  """
  valid_chainids = get_valid_chainids(max_num_chains=max_num_chains)
  if workflow_cur is None or which_chain == '-1':
    logging.info("Creating a new chain.")
    # Get a valid new chain ID
    if workflow_cur is None:
      chain_id = valid_chainids[0]
    else:
      # check whether the current number of chains exceeds the limit
      # if exceeds, return None
      if num_saturated_warnings is not None:
        num_chains = workflow_cur.api_use_chains.keys().__len__()
        if num_chains >= len(valid_chainids):
          logging.warning("No valid chain ID to use.")
          num_saturated_warnings -= 1
          if num_saturated_warnings == 0:
            logging.warning(
                "Too many chains have been created. Stop creating new chains.")
            return {'workflow_cur': None}
          return {'workflow_cur': workflow_cur}
      chain_id = valid_chainids[num_chains]
    intermediate_steps = [api_step]
  elif which_chain in workflow_cur.api_use_chains:
    chain_id = which_chain
    logging.info(f"Updating chain {which_chain}.")
    intermediate_steps = workflow_cur.api_use_chains[
        which_chain].intermediate_steps + [api_step]
  else:
    raise ValueError(
        f"No valid chain to update. Got {which_chain}, expected a valid chain ID {workflow_cur.api_use_chains.keys()} or '-1'."
    )

  return {
      'chain':
          states.ApiUseChain(
              intermediate_steps=intermediate_steps,
              chain_id=chain_id,
          )
  }


def create_workflow_updater(output_schema: BaseModel = states.ApiUseWorkflow,):
  predict_workflow_prompt = prompt_lib.PREDICT_WORKFLOW
  llm = langchain.create_llm()
  workflow_updater = predict_workflow_prompt | llm.with_structured_output(
      output_schema)
  return workflow_updater
