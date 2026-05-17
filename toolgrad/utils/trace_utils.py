"""Utilities for tracing ToolGrad execution for debugging and analysis."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from toolgrad import states


class ExecutionTracer:
  """Trace execution details for each iteration."""

  def __init__(self, output_dir: Optional[str] = None, seed: int = 0):
    """Initialize tracer.
        
        Args:
            output_dir: Base output directory. Trace will be saved to {output_dir}/trace/
            seed: Seed number for filename
        """
    self.output_dir = output_dir
    self.seed = seed
    self.traces = []
    self.current_iteration = 0

    if output_dir:
      self.trace_dir = Path(output_dir) / "trace"
      self.trace_dir.mkdir(parents=True, exist_ok=True)

  def start_iteration(self, iteration: int):
    """Start a new iteration."""
    self.current_iteration = iteration
    self.traces.append({
        "iteration": iteration,
        "timestamp": None,
        "proposals": None,
        "executions": None,
        "selection": None,
        "workflow_update": None,
    })

  def log_proposals(self, api_proposals: states.ApiProposalAll):
    """Log proposed APIs."""
    if not self.traces:
      return

    proposals_info = []
    for proposal in api_proposals.proposals:
      proposal_info = {
          "id":
              proposal.id,
          "instruction":
              proposal.instruction,
          "apis": [{
              "name": api.name,
              "description": getattr(api, 'description', 'N/A'),
          } for api in proposal.api],
      }
      proposals_info.append(proposal_info)

    self.traces[-1]["proposals"] = proposals_info

    # Log to console
    logging.info(
        f"[Iteration {self.current_iteration}] Proposed {len(proposals_info)} API proposals"
    )
    for i, prop in enumerate(proposals_info, 1):
      logging.info(f"  Proposal {i} ({prop['id']}): {prop['instruction']}")
      for api in prop['apis']:
        logging.info(f"    └─ {api['name']}")

  def log_executions(self, api_reports: Dict[str, states.ApiReport]):
    """Log execution results."""
    if not self.traces:
      return

    executions_info = []
    for proposal_id, report in api_reports.items():
      execution_info = {
          "proposal_id": proposal_id,
          "success": report.success,
          "num_tool_calls": report.num_tool_calls,
          "api_call_step": None,
      }

      if report.success and report.api_call_step:
        action, result = report.api_call_step
        execution_info["api_call_step"] = {
            "tool":
                action.tool if hasattr(action, 'tool') else str(action),
            "tool_input":
                action.tool_input if hasattr(action, 'tool_input') else {},
            "result_preview":
                self._truncate_result(result),
        }
      elif not report.success and report.error_message:
        execution_info["error_message"] = report.error_message

      executions_info.append(execution_info)

    self.traces[-1]["executions"] = executions_info

    # Log to console
    successful = sum(1 for e in executions_info if e['success'])
    logging.info(
        f"[Iteration {self.current_iteration}] Executed {len(executions_info)} proposals: {successful} successful, {len(executions_info) - successful} failed"
    )
    for exec_info in executions_info:
      status = "✓" if exec_info['success'] else "✗"
      logging.info(
          f"  {status} {exec_info['proposal_id']}: {exec_info['num_tool_calls']} tool call(s)"
      )
      if exec_info['success'] and exec_info['api_call_step']:
        step = exec_info['api_call_step']
        logging.info(f"      Tool: {step['tool']}")
        logging.info(f"      Input: {step['tool_input']}")
      elif not exec_info['success'] and exec_info.get('error_message'):
        logging.info(f"      Error: {exec_info['error_message']}")

  def log_selection(self, api_selection: states.ApiSelection,
                    selected_report: states.ApiReport):
    """Log the selected API."""
    if not self.traces:
      return

    selection_info = {
        "selected_id":
            api_selection.id,
        "action":
            "create_new_chain" if api_selection.which_chain == '-1' else
            f"update_chain_{api_selection.which_chain}",
        "which_chain":
            api_selection.which_chain,
        "tool_used":
            None,
    }

    if selected_report.api_call_step:
      action, _ = selected_report.api_call_step
      selection_info["tool_used"] = action.tool if hasattr(
          action, 'tool') else str(action)

    self.traces[-1]["selection"] = selection_info

    # Log to console
    action_desc = "Creating new chain" if api_selection.which_chain == '-1' else f"Updating {api_selection.which_chain}"
    logging.info(
        f"[Iteration {self.current_iteration}] Selected: {api_selection.id} → {action_desc}"
    )
    if selection_info["tool_used"]:
      logging.info(f"  └─ Tool: {selection_info['tool_used']}")

  def log_workflow_update(self, workflow: states.ApiUseWorkflow):
    """Log workflow state after update."""
    if not self.traces:
      return

    workflow_info = {
        "num_chains": len(workflow.api_use_chains),
        "chains": {
            chain_id: len(chain.intermediate_steps)
            for chain_id, chain in workflow.api_use_chains.items()
        },
    }

    self.traces[-1]["workflow_update"] = workflow_info

    # Log to console
    logging.info(
        f"[Iteration {self.current_iteration}] Workflow updated: {workflow_info['num_chains']} chain(s)"
    )
    for chain_id, steps in workflow_info['chains'].items():
      logging.info(f"  └─ {chain_id}: {steps} step(s)")

  def save(self):
    """Save trace to file."""
    if not self.output_dir:
      return

    trace_file = self.trace_dir / f"{self.seed:05d}.json"

    # Create summary
    summary = {
        "seed": self.seed,
        "total_iterations": len(self.traces),
        "iterations": self.traces,
    }

    with open(trace_file, 'w') as f:
      json.dump(summary, f, indent=2)

    logging.info(f"Execution trace saved to {trace_file}")

  def _truncate_result(self, result: Any, max_length: int = 200) -> str:
    """Truncate result for logging."""
    result_str = str(result)
    if len(result_str) > max_length:
      return result_str[:max_length] + "..."
    return result_str
