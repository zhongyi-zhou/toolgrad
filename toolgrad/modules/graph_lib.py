"""This file contains main graph-level modules in the ToolGrad framework, as shown in Figure 2 of the ToolGrad paper."""
import logging
import random
from typing import Literal, Union, Optional
from enum import Enum

from langchain_core.tools import StructuredTool
from pydantic import ConfigDict

from toolgrad.modules import module_lib
from toolgrad.utils import langchain
from toolgrad.utils import log_utils
from toolgrad.utils.toolbench import toolbench_langchain_utils
from toolgrad import states
from toolgrad.utils import mcp
from toolgrad.utils import trace_utils

__all__ = [
    'ToolGradState',
]


class SpecialStates(Enum):
  """Special states for the ToolGrad graph."""
  NEXT_STEP = '__next_step__'
  NEXT_ITER = '__next_iter__'
  END = '__end__'


# State
class ToolGradState(states.ApiUseState):
  """A state class for ToolGrad, inheriting from ApiUseState."""
  model_config = ConfigDict(arbitrary_types_allowed=True)

  step: int = 0
  sampled_apis: list[StructuredTool] = []
  should_early_next_iter: bool = False
  total_tool_calls: int = 0
  tracer: Optional[trace_utils.ExecutionTracer] = None


# Modules
@log_utils.log_time
def sample_apis_or_end(
    toolgrad_state: ToolGradState,
    api_sample_seed,
    max_iter: int = 10,
    num_apis: int = 50,
    database: Literal['mcp', 'toolbench'] = 'toolbench',
    mcp_dict: dict = None,
) -> Union[ToolGradState, str]:
  """count the number of iterations."""
  # Initialization
  toolgrad_state.step += 1
  toolgrad_state.should_early_next_iter = False
  random.seed(api_sample_seed * max_iter + toolgrad_state.step)

  # Start iteration trace
  if toolgrad_state.tracer:
    toolgrad_state.tracer.start_iteration(toolgrad_state.step)
  # Sample APIs
  if database == 'mcp':
    mcp_dict = mcp.get_default_mcp_dict() if mcp_dict is None else mcp_dict
    toolgrad_state.sampled_apis = mcp.get_mcp_apis(
        mcp_dict=mcp_dict,
        num_apis=num_apis,
        seed=api_sample_seed,
    )
  else:
    toolgrad_state.sampled_apis = toolbench_langchain_utils.get_all_apis(
        num_apis)
  return toolgrad_state


@log_utils.log_time
def api_proposer(
    toolgrad_state: ToolGradState,
    is_toolbench: bool = True,
    num_proposals: int = 3,
) -> Union[ToolGradState, str]:
  """Propose APIs based on the current state.
  
  We use hash names for APIs in ToolBench because some names exceed the the text length limit supported by OpenAI.
  """

  api_proposal_model = module_lib.create_api_proposer(
      toolgrad_state.sampled_apis,
      is_mcp=not is_toolbench,
      num_proposals=num_proposals,
  )

  all_api_annotations = module_lib.get_all_tool_annotations(
      toolgrad_state.sampled_apis)
  try:
    dedicated_api_proposals = api_proposal_model.invoke({
        "workflow_cur": toolgrad_state.workflow_cur,
        "api_all": all_api_annotations,
    })
  except Exception as e:
    logging.warning(
        f"API proposal model invocation failed: {e}. Skip this step.")
    toolgrad_state.should_early_next_iter = True
    return toolgrad_state

  # Validate the output before conversion
  if dedicated_api_proposals is None:
    logging.warning("API proposal model returned None. Skip this step.")
    toolgrad_state.should_early_next_iter = True
    return toolgrad_state

  api_proposals = module_lib.convert_dedicated_apiproposals_to_standard_apiproposals(
      dedicated_api_proposals)
  if api_proposals is None or api_proposals == {}:
    logging.warning("No API proposals were generated. Skip this step.")
    toolgrad_state.should_early_next_iter = True
    return toolgrad_state

  # Post-process proposals: convert hash names to real names in both API names and instructions
  if is_toolbench:
    api_proposals = toolbench_langchain_utils.postprocess_proposalall(
        api_proposals, convert_hash_to_name=True)

  toolgrad_state.api_proposals = api_proposals

  # Trace proposals
  if toolgrad_state.tracer:
    toolgrad_state.tracer.log_proposals(api_proposals)

  return toolgrad_state


@log_utils.log_time
def api_executor(
    toolgrad_state: ToolGradState,
    is_toolbench: bool = True,
) -> ToolGradState | str:
  """Execute the proposed APIs and update the state."""
  api_reports = module_lib.api_execute_step(
      toolgrad_state,
      is_hash_api_name=False,
      is_toolbench=is_toolbench,
  )
  toolgrad_state.api_reports = api_reports
  if api_reports is None or api_reports == {}:
    logging.warning("No API reports are generated. Skip the iteration.")
    toolgrad_state.should_early_next_iter = True
    return toolgrad_state
  # Count all tool calls from all proposals (both success and failure)
  for report in api_reports.values():
    toolgrad_state.total_tool_calls += report.num_tool_calls

  # Trace executions
  if toolgrad_state.tracer:
    toolgrad_state.tracer.log_executions(api_reports)

  if all(not report.success for report in api_reports.values()):
    logging.warning("All APIs failed to execute. Skip the iteration.")
    toolgrad_state.should_early_next_iter = True
    return toolgrad_state
  return toolgrad_state


@log_utils.log_time
def api_selector(toolgrad_state: ToolGradState,) -> ToolGradState | str:
  api_reports = toolgrad_state.api_reports
  assert isinstance(api_reports, dict) and all(
      isinstance(v, states.ApiReport) for v in api_reports.values())
  api_reports = {k: v for k, v in api_reports.items() if v.success}
  workflow_cur = toolgrad_state.workflow_cur
  if api_reports.keys().__len__() == 0:
    logging.info("No valid api reports to select from.")
    return {}
  id_list = [rep.id for rep in api_reports.values()]
  if workflow_cur is None:
    chain_keys = []
  else:
    assert isinstance(workflow_cur, states.ApiUseWorkflow)
    chain_keys = list(workflow_cur.api_use_chains.keys())
  api_selector = module_lib.create_api_selector(
      output_schema=states.create_dedicated_apiselection(
          id_list=id_list,
          chain_keys=chain_keys,
      ))
  try:
    api_selection = api_selector.invoke({
        "workflow_cur": toolgrad_state.workflow_cur,
        "api_reports": api_reports,
    })
  except Exception as ve:
    logging.warning(
        f"API selection validation error: {ve}. Skip the iteration.")
    # return EARLY_NEXT_STEP
    toolgrad_state.should_early_next_iter = True
    return toolgrad_state

  if api_selection is None:
    logging.warning("No valid API selection made.")
    # return EARLY_NEXT_STEP
    toolgrad_state.should_early_next_iter = True
    return toolgrad_state
  elif api_selection.id is None:
    logging.warning("No valid API selection made.")
    # return EARLY_NEXT_STEP
    toolgrad_state.should_early_next_iter = True
    return toolgrad_state
  # logging.info(f"Selected APIs: {api_selection}")
  toolgrad_state.api_selection = api_selection

  # Trace selection
  if toolgrad_state.tracer:
    selected_report = api_reports[api_selection.id]
    toolgrad_state.tracer.log_selection(api_selection, selected_report)

  return toolgrad_state


@log_utils.log_time
def inverse_predictor(toolgrad_state: ToolGradState,) -> ToolGradState:
  """Inverse predictor to update the state."""
  # workflow_cur = toolgrad_state.get("workflow_cur", None)
  workflow_cur = toolgrad_state.workflow_cur
  assert isinstance(workflow_cur, states.ApiUseWorkflow) or workflow_cur is None
  api_selection = toolgrad_state.api_selection
  id = api_selection.id
  api_reports = toolgrad_state.api_reports

  selected_report = api_reports[id]
  assert isinstance(selected_report, states.ApiReport)

  # Start workflow update
  # Step 1: choose which chain to process
  which_chain = api_selection.which_chain

  chain_update_result = module_lib.chain_update_step(
      workflow_cur,
      which_chain,
      api_step=selected_report.api_call_step,
  )
  if "workflow_cur" in chain_update_result:
    toolgrad_state.workflow_cur = chain_update_result["workflow_cur"]
    return toolgrad_state

  # Step 2: update the corresponding chain in the workflow
  new_chain = chain_update_result["chain"]
  assert isinstance(new_chain, states.ApiUseChain)

  if workflow_cur is None:
    # Create a new workflow from a single chain if the workflow does not exist
    chains_new = {new_chain.chain_id: new_chain}
  else:
    chains_cur = workflow_cur.api_use_chains
    chains_new = chains_cur.copy()
    if which_chain == '-1':
      # Create a new chain in the existing workflow
      chains_new[new_chain.chain_id] = new_chain
    elif which_chain in workflow_cur.api_use_chains:
      # Update an existing chain in the workflow
      chains_new[which_chain] = new_chain
    else:
      raise ValueError(
          f"No valid chain to update. Got {which_chain}, expected a valid chain ID {workflow_cur.api_use_chains.keys()} or '-1'."
      )
  # LLM generates only query and response (not api_use_chains)
  workflow_updater = module_lib.create_workflow_updater()
  llm_output = workflow_updater.invoke({
      "api_use_chains": chains_new,  # Passed for context only
  })

  # Validate the output before using it
  if llm_output is None:
    logging.warning("Workflow updater model returned None. Skip this step.")
    toolgrad_state.should_early_next_iter = True
    return toolgrad_state

  # Combine LLM-generated query/response with actual execution chains
  workflow_cur = states.ApiUseWorkflow(
      api_use_chains=chains_new,  # Use actual execution chains
      query=llm_output.query,  # LLM-generated query
      response=llm_output.response  # LLM-generated response
  )
  toolgrad_state.workflow_cur = workflow_cur

  # Trace workflow update
  if toolgrad_state.tracer:
    toolgrad_state.tracer.log_workflow_update(workflow_cur)

  return toolgrad_state


# Edges
def should_early_next_iter_or_end(
    toolgrad_state: ToolGradState,
    max_iter: int = 10,
) -> Literal[
    SpecialStates.NEXT_STEP,
    SpecialStates.NEXT_ITER,
    SpecialStates.END,
]:
  """Check if the current state should move forward, continue a new iteration, or end."""
  if toolgrad_state.should_early_next_iter:
    if toolgrad_state.step >= max_iter:
      return SpecialStates.END
    else:
      return SpecialStates.NEXT_ITER
  else:
    return SpecialStates.NEXT_STEP


def should_end(
    toolgrad_state: ToolGradState,
    max_iter: int = 10,
) -> SpecialStates:
  """Check if the current state indicates an early next step."""

  if toolgrad_state.step >= max_iter:
    return SpecialStates.END
  else:
    return SpecialStates.NEXT_ITER
