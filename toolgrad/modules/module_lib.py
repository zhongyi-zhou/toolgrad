import datetime
import json
import logging

logging.getLogger("langchain_google_genai._function_utils").setLevel(
    logging.ERROR)
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import ValidationError
import httpx
import functools
import inspect

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_community.callbacks.manager import OpenAICallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from toolgrad import states
from toolgrad.modules import prompt_lib
from toolgrad.utils import data
from toolgrad.utils import langchain as langchain_utils
from toolgrad.utils.toolbench import toolbench_langchain_utils
from toolgrad.utils import log_utils
import asyncio
import langgraph as lg
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
import toolgrad as tog

_MAX_TOOL_AGENT_ITERATIONS = 3
_JST = datetime.timezone(datetime.timedelta(hours=+9), 'JST')

# ============================================================================
# Helper functions for API response validation
# ============================================================================


def _is_empty_value(val: Any) -> bool:
  """Checks if a value is effectively empty (None, empty list/dict/str)."""
  if val is None:
    return True
  if isinstance(val, (str, list, dict, tuple, set)) and len(val) == 0:
    return True
  return False


def _is_recursively_empty(val: Any) -> bool:
  """Checks if a value is recursively empty (e.g. {'a': [], 'b': {}})."""
  if _is_empty_value(val):
    return True

  if isinstance(val, dict):
    # It's empty if all its values are recursively empty
    return all(_is_recursively_empty(v) for v in val.values())

  if isinstance(val, list):
    # It's empty if all its items are recursively empty
    return all(_is_recursively_empty(v) for v in val)

  return False


def _is_single_key_empty_value(parsed: dict) -> bool:
  """Checks for pattern {'key': {}} or {'key': []}."""
  if len(parsed) != 1:
    return False
  val = list(parsed.values())[0]
  return _is_recursively_empty(val)


def _has_binary_data(response: str) -> bool:
  """Detect binary data (images, PDFs, etc.) in response."""
  # Check for common binary file signatures
  if response.startswith('\x89PNG') or response.startswith('PNG') or response.startswith('�PNG'):
    return True
  if response.startswith('\xff\xd8\xff'):  # JPEG
    return True
  if response.startswith('%PDF'):
    return True

  # Check for high percentage of non-printable characters
  if len(response) > 50:
    non_printable = sum(
        1 for c in response[:100] if ord(c) < 32 and c not in '\n\r\t')
    if non_printable > 20:  # More than 20% non-printable
      return True

  return False


def _is_html_or_xml(response: str) -> bool:
  """Check if response is HTML or XML error page instead of valid data."""
  response_stripped = response.strip()

  # Check for XML - but only flag if it contains error indicators
  # Some APIs legitimately return XML as their data format
  if response_stripped.lower().startswith('<?xml'):
    response_lower = response.lower()
    xml_error_patterns = [
        '<error>', '</error>', 'error code', 'error message', '<fault>',
        '</fault>', 'soap:fault', 'faultcode', 'faultstring'
    ]
    if any(pattern in response_lower for pattern in xml_error_patterns):
      return True

  # Check for HTML - only flag HTML pages with error indicators
  if response_stripped.startswith('<!DOCTYPE html>') or \
     response_stripped.startswith('<html'):
    response_lower = response.lower()
    # Only flag HTML pages that contain specific error indicators
    html_error_patterns = [
        'error page', 'page not found', '404 error', '404 not found',
        '500 error', '500 internal server error', 'deployment not found',
        'deployment could not be found', 'service unavailable',
        'temporarily unavailable', 'server error', '>not found<'
    ]
    if any(pattern in response_lower for pattern in html_error_patterns):
      return True

  return False


def _contains_error_keywords(text: str) -> tuple[bool, str]:
  """
  Check if text contains error keywords.
  Shared logic for both plain text and pattern-based error detection.

  Returns:
    (has_error, error_reason): Tuple indicating if error found and the reason
  """
  if not text:
    return False, ""

  text_lower = text.lower()

  # Common error patterns with their descriptions
  error_patterns = [
      # Validation errors
      ("not valid", "Validation error"),
      ("invalid", "Invalid input"),
      ("must be a", "Validation error"),
      ("must be an", "Validation error"),

      # Parsing errors
      ("could not parse", "Parsing error"),
      ("failed to parse", "Parsing error"),
      ("cannot parse", "Parsing error"),

      # Rate limiting
      ("rate limit", "Rate limit exceeded"),
      ("maximum request limit", "Rate limit exceeded"),
      ("too many requests", "Rate limit exceeded"),

      # Authentication/Authorization
      ("authentication failed", "Authentication failed"),
      ("authentication required", "Authentication required"),
      ("unauthorized", "Unauthorized access"),
      ("access denied", "Access denied"),
      ("forbidden", "Access forbidden"),

      # Resource not found
      ("404 page not found", "404 error - page not found"),
      ("404 not found", "404 error - not found"),
      ("404 error", "404 error"),
      ("not found", "Resource not found"),
      ("does not exist", "Resource not found"),
      ("doesn't exist", "Resource not found"),
      ("can't find", "Resource not found"),
      ("cannot find", "Resource not found"),

      # API/Endpoint errors
      ("api doesn't exist", "API endpoint not found"),
      ("api does not exist", "API endpoint not found"),
      ("invalid api", "Invalid API endpoint"),
      ("endpoint does not exist", "API endpoint not found"),
      ("endpoint doesn't exist", "API endpoint not found"),
      ("endpoint is disabled", "Endpoint disabled"),

      # Network/Infrastructure errors
      ("blocked host:", "Blocked host error"),
      ("service unavailable", "Service unavailable"),
      ("internal server error", "Internal server error"),
      ("deployment not found", "Deployment not found"),
      ("deployment could not be found", "Deployment not found"),
      ("timed out", "API timeout"),

      # Subscription/Plan errors
      ("disabled for your subscription", "Subscription required"),
      ("subscription required", "Subscription required"),
      ("upgrade your plan", "Plan upgrade required"),

      # Generic errors
      ("error", "API error"),
      ("failed", "API request failed"),
  ]

  for pattern, reason in error_patterns:
    if pattern in text_lower:
      return True, reason

  return False, ""


def _check_error_fields_case_insensitive(parsed: dict) -> tuple[bool, str]:
  """
  Check for error-indicating fields in JSON, case-insensitively.

  Returns:
    (has_error, error_reason): Tuple indicating if error found and the reason
  """
  # Create case-insensitive key lookup
  lower_keys = {k.lower(): k for k in parsed.keys()}

  # Define error-indicating field names to check
  error_field_names = ['error', 'message', 'messages', 'detail', 'notice']

  # Extract subdict of error-related fields (case-insensitive)
  error_fields = {
      field_name: parsed[lower_keys[field_name]]
      for field_name in error_field_names
      if field_name in lower_keys
  }

  # Check each error field for error keywords
  for field_name, value in error_fields.items():
    # Handle string values
    if isinstance(value, str) and value:
      # Special handling for 'notice' field - check for subscription/plan patterns
      if field_name == 'notice' and len(value) < 200:
        value_lower = value.lower()
        if ('available' in value_lower and 'plan' in value_lower) or \
           ('upgrade' in value_lower and ('plan' in value_lower or 'subscription' in value_lower)):
          return True, "Subscription or plan upgrade required"

      # Apply length limits for certain fields to avoid false positives
      max_len = 200 if field_name == 'notice' else 100
      if len(value) <= max_len or field_name in ['error', 'detail']:
        has_error, reason = _contains_error_keywords(value)
        if has_error:
          return True, f"API error: {value}" if field_name == 'detail' else reason

    # Handle Pydantic-style validation errors (list of dicts)
    elif isinstance(value, list) and len(value) > 0 and field_name == 'detail':
      first_error = value[0]
      if isinstance(first_error, dict) and 'msg' in first_error:
        return True, f"Validation error: {first_error.get('msg', 'Invalid input')}"

  # Check success/status flags
  if 'success' in lower_keys and parsed[lower_keys['success']] is False:
    msg = parsed.get(lower_keys.get('message', 'message'), 'Unknown error')
    return True, f"API failed: {msg}"

  if 'status' in lower_keys and parsed[lower_keys['status']] == 'error':
    msg = parsed.get(lower_keys.get('message', 'message'), 'Unknown error')
    return True, f"API failed: {msg}"

  return False, ""


def _is_single_key_empty_value(parsed: dict) -> bool:
  """
  Check if response is a dict with exactly one key whose value is empty.
  Examples: {'us': {}}, {'real_time_quotes': []}, {'list': []}, {'data': ''}
  """
  if not isinstance(parsed, dict) or len(parsed) != 1:
    return False

  # Get the single value
  value = next(iter(parsed.values()))
  return _is_empty_value(value)


def _check_empty_structures(parsed: dict) -> tuple[bool, str]:
  """
  Check for various empty structure patterns that indicate API failures.

  Returns:
    (is_empty, error_reason): Tuple indicating if response is effectively empty
  """
  # Check for objects containing only empty arrays/lists (likely errors)
  # Examples: {"articles": []}, {"data": []}, {"results": []}
  if len(parsed) <= 2:  # Small objects are suspicious
    all_empty_arrays = True
    has_array = False
    for key, value in parsed.items():
      if isinstance(value, list):
        has_array = True
        if len(value) > 0:
          all_empty_arrays = False
          break
      elif value not in [None, '', 0, {}]:  # Ignore null/empty/zero values
        all_empty_arrays = False
        break

    if has_array and all_empty_arrays:
      return True, "API returned object with only empty arrays"

  # Check for empty results arrays in successful responses
  if parsed.get('status') == 'success' or parsed.get('success') is True:
    data = parsed.get('data', parsed)
    if isinstance(data, dict):
      results = data.get('results')
      if isinstance(results, list) and len(results) == 0:
        # Check if this is intentionally empty or an error
        # If there's no other data besides empty results, consider it empty
        if len(
            data
        ) <= 3:  # Usually just 'results', maybe 'message' and one more field
          return True, "Empty results returned"

  return False, ""


def validate_api_response(
    response: str,
    api_name: str,
    error_msg: str,
) -> tuple[bool, str]:
  """
  Validate if an API response is actually successful.

  Many ToolBench APIs return valid HTTP responses (error="") but embed error
  messages in the response content. This function detects such hidden errors.

  Validation sequence:
    1. Check explicit error_msg parameter
    2. Check for empty/whitespace responses
    3. Detect binary data (images, PDFs, etc.)
    4. Detect plain text errors (before JSON parsing)
    5. Check for HTML/XML error responses
    6. Parse and validate JSON structure:
       - Bare empty arrays/objects
       - Single-key empty values
       - Error fields (case-insensitive)
       - Empty structure patterns
    7. Fallback pattern matching for edge cases

  Args:
    response: The API response content
    api_name: Name of the API being called
    error_msg: The error field from the result dict

  Returns:
    (is_valid, error_reason): Tuple indicating if response is valid and why if not
  """
  # 1. If there was already an exception, it's definitely an error
  if error_msg:
    return False, error_msg

  # 2. Empty response - consider it a failure
  if not response or response.strip() in ["", "{}", "[]"]:
    return False, "Empty response from API"

  # 2.1 Check for "None" string (Python None converted to string)
  if response.strip() == 'None':
    return False, "Response is Python 'None' string"

  # 3. Check for binary data (images, PDFs, etc.)
  if _has_binary_data(response):
    return False, "Binary data is not supported in API response"

  # 4. Check for plain text error messages (before JSON parsing)
  # Common pattern: APIs return error as plain text instead of JSON
  response_stripped = response.strip()

  # 4.1 Check for HTML/XML error pages (enhanced)
  # Check for common HTML/XML start tags
  if response_stripped.startswith(('<!DOCTYPE', '<html', '<head', '<body', '<?xml', '<QUOTE', '<ERROR')):
    return False, "HTML/XML error page returned instead of JSON"

  # Also check for specific HTML tags if they appear at start (ignoring whitespace)
  if response_stripped.lower().startswith(('<html>', '<!doctype html>', '<?xml')):
    return False, "HTML/XML error page returned instead of JSON"

  # 4.2 Check for explicit error strings
  explicit_error_strings = [
      'Payment required',
      'DEPLOYMENT_DISABLED',
      'API request missing api_key or valid OAuth parameters',
      'Internal Server Error',
      '404 Not Found',
      '403 Forbidden',
      'The endpoint has been disabled',
      'API rate limit exceeded',
      'Invalid Stock Ticker',
      'could not parse location:',
      'Value ETHBTC for symbol is not valid',
      'You have reached maximum request limit.',
      'An API key is required to',
      'API has been moved',
      'Wrong value passed',
      'This file is no longer',
      'No Any Results',
      'Missing coordinates',
      'could not be found',
      'No data found',
      'abbreviation: CDT', # Metadata-only plain text response
  ]
  for error_str in explicit_error_strings:
    if error_str in response_stripped:
      return False, f"Explicit error message found: '{error_str}'"

  if not response_stripped.startswith(('{', '[')):
    # Likely plain text, check for error keywords
    has_error, reason = _contains_error_keywords(response_stripped)
    if has_error:
      return False, reason

    # If it's not JSON and not flagged by keywords, it might still be an error if it's just a short string
    # But we should be careful not to flag valid plain text responses if they exist
    # For now, relying on _contains_error_keywords and explicit_error_strings

  # 5. Check for HTML/XML responses (infrastructure errors, not data)
  # Keeping original check as backup
  if _is_html_or_xml(response):
    return False, "HTML/XML error page returned instead of JSON"

  # 6. Try JSON validation (for structured APIs)
  parsed = None
  try:
    import json
    parsed = json.loads(response)
  except (json.JSONDecodeError, TypeError):
    # Not valid JSON, try ast.literal_eval for Python string representations
    try:
      import ast
      if response.strip().startswith(('{', '[')):
        parsed = ast.literal_eval(response)
    except (ValueError, SyntaxError):
      pass

  if parsed is not None:
    is_valid, reason = _validate_parsed_json(parsed)
    if not is_valid:
      return False, reason

  # 7. Pattern-based error detection (fallback for edge cases)
  # This catches errors that might be embedded in larger text
  has_error, reason = _contains_error_keywords(response)
  if has_error:
    return False, reason

  # 8. If nothing flagged it, consider it valid
  return True, ""


def _validate_parsed_json(parsed: Any) -> tuple[bool, str]:
  """Helper to validate parsed JSON/dict structure."""
  # Check for bare empty array (very likely an error)
  if isinstance(parsed, list) and len(parsed) == 0:
    return False, "API returned empty array instead of structured response"

  # Check for bare empty object (very likely an error)
  if isinstance(parsed, dict) and len(parsed) == 0:
    return False, "API returned empty object instead of structured response"

  if isinstance(parsed, dict):
    # Check for single-key empty value pattern: {'us': {}}, {'real_time_quotes': []}
    if _is_single_key_empty_value(parsed):
      return False, "API returned single-key object with empty value"

    # Stricter check: Ignore metadata keys and check if remaining content is empty
    # e.g. {'status': 'OK', 'data': {}} -> effectively empty
    metadata_keys = {
        'status', 'code', 'statuscode', 'status_code', 'cod', 't-status',
        'success', 'ok',
        'request_id', 'requestid',
        'message', 'msg', # Often just "Success" or "OK" if status is good
        'at', 'method', 'hostname', # Default server response metadata
        'last_page', 'total', 'count', 'offset', 'limit', 'page', # Pagination metadata
    }

    # Filter out metadata keys
    content_keys = [k for k in parsed.keys() if str(k).lower() not in metadata_keys]

    if len(content_keys) == 0:
      # Only metadata keys present.
      # If we are here, it means no explicit error was found (e.g. status was 200/OK).
      # But if there's no data, it's likely a useless response for the model.
      return False, "API returned only metadata/status fields with no content"

    if len(content_keys) == 1:
      # Only one content key, check if it's empty
      key = content_keys[0]
      val = parsed[key]
      if _is_recursively_empty(val):
         return False, f"API returned metadata and single empty key '{key}'"

    # Check error fields (case-insensitive)
    has_error, reason = _check_error_fields_case_insensitive(parsed)
    if has_error:
      return False, reason

    # Enhanced JSON Validation
    # Check for success/ok/status fields
    # Check 'success': False/false
    if 'success' in parsed and str(parsed['success']).lower() == 'false':
      return False, "JSON 'success' field is false"

    # Check 'ok': False/false
    if 'ok' in parsed and str(parsed['ok']).lower() == 'false':
      return False, "JSON 'ok' field is false"

    # Check 'status' or 'code' indicating error
    # Common error codes: 4xx, 5xx, 0 (sometimes), "error", "failed"
    for status_field in ['status', 'code', 'statusCode', 'status_code', 'cod', 't-status']:
      if status_field in parsed:
        val = parsed[status_field]
        val_str = str(val).lower()
        if val_str in ['error', 'failed', 'failure']:
           return False, f"JSON '{status_field}' indicates error: {val}"

        # Check for numeric error codes (400+)
        try:
          val_int = int(val)
          if val_int >= 400:
            return False, f"JSON '{status_field}' is error code: {val_int}"
          # Some APIs use 0 for failure if accompanied by data: "No result found"
          if val_int == 0 and isinstance(parsed.get('data'), str) and "No result" in parsed.get('data', ''):
             return False, f"JSON '{status_field}' is 0 with no result"
        except (ValueError, TypeError):
          pass

    # Check nested meta status (common in some APIs)
    # e.g. {'meta': {'status': 401, ...}}
    if 'meta' in parsed and isinstance(parsed['meta'], dict):
      meta = parsed['meta']
      if 'status' in meta:
        try:
          if int(meta['status']) >= 400:
            return False, f"JSON 'meta.status' is error code: {meta['status']}"
        except (ValueError, TypeError):
          pass

    # Check for empty structures
    is_empty, reason = _check_empty_structures(parsed)
    if is_empty:
      return False, reason

  return True, ""


def get_success_execution_id_legacy(
    intermediate_steps: list[Tuple[dict, dict]],
    is_toolbench: bool = True,
) -> int | None:
  """Return the index of the last successful step in the intermediate steps or None if none of them are successful."""
  success_step_id = None
  if is_toolbench:
    # ToolBench API returns a dict in the format of {'error': ..., 'response': ...}
    for i, intermediate_step in enumerate(intermediate_steps):
      action, result = intermediate_step
      assert isinstance(result, dict)
      error_msg = result.get('error', None) or ""
      response = result.get('response', "")

      # Validate the response content, not just the error field
      is_valid, error_reason = validate_api_response(
          response=str(response),
          api_name=action.tool,
          error_msg=error_msg,
      )

      if is_valid:
        success_step_id = i
      elif error_reason:
        # Store the error reason in the result for later reporting
        result['validation_error'] = error_reason
  else:
    # For MCP APIs: use simple keyword detection on string result
    for i, intermediate_step in enumerate(intermediate_steps):
      action, result = intermediate_step
      result_str = str(result).lower()
      # Heuristic: if result doesn't contain error keywords, consider it successful
      if "error" not in result_str and "failed" not in result_str and "access denied" not in result_str:
        success_step_id = i
  return success_step_id


def track_tool_invocation(tracker: dict):
  """Decorator that wraps a tool function to track when it's called.

  Args:
    tracker: A dict with 'called' key to mark invocation (mutable state)
  """

  def decorator(func):
    # Handle both sync and async functions
    if inspect.iscoroutinefunction(func):

      @functools.wraps(func)
      async def async_wrapper(*args, **kwargs):
        tracker['called'] = True
        return await func(*args, **kwargs)

      return async_wrapper
    else:

      @functools.wraps(func)
      def sync_wrapper(*args, **kwargs):
        tracker['called'] = True
        return func(*args, **kwargs)

      return sync_wrapper

  return decorator


def wrap_tool_with_tracking(tool: StructuredTool,
                            tracker: dict) -> StructuredTool:
  """Wrap a LangChain tool with invocation tracking using a decorator.

  Args:
    tool: The original StructuredTool
    tracker: A dict with 'called' key to track invocation

  Returns:
    A new StructuredTool with decorated function
  """
  # Apply the tracking decorator to the tool's function
  tracked_func = track_tool_invocation(tracker)(tool.func)

  # Create a new tool with the decorated function
  return StructuredTool(
      name=tool.name,
      description=tool.description,
      func=tracked_func,
      args_schema=tool.args_schema,
      coroutine=tool.coroutine if hasattr(tool, 'coroutine') else None,
  )


def select_object_by_id(objects: List, id: str):
  for obj in objects:
    if not hasattr(obj, 'id'):
      raise ValueError(f"Object {obj} does not have an id attribute.")
    if obj.id == id:
      return obj
  logging.warning(f"Object with id {id} not found.")
  return None


def generate_chain_description_from_intermediate_steps(
    intermediate_steps: List[Tuple[dict, Optional[dict]]],) -> str:
  prompt = prompt_lib.CHAIN_SUMMARIZER
  llm = langchain_utils.create_llm()
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
  # assert isinstance(intermediate_steps, List[Tuple[dict, Optional[dict]]])
  assert isinstance(intermediate_steps, list)
  assert all(
      isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict)
      for item in intermediate_steps)  # Check tuple structure
  chain_dict['input'] = intermediate_steps[0][0].tool_input
  chain_dict['output'] = intermediate_steps[-1][1]
  return chain_dict


def convert_dedicated_apiproposals_to_standard_apiproposals(
    dedicated_api_proposals: states.ApiProposalAll,) -> states.ApiProposalAll:
  """Convert the dedicated api proposals to standard api proposals."""
  # Defensive check: some models (e.g., Gemini) may not always return properly structured output
  if dedicated_api_proposals is None:
    logging.warning("Received None for dedicated_api_proposals.")
    return None

  if dedicated_api_proposals.__class__.__name__ != "DedicatedApiProposalAll":
    logging.warning(
        f"Expected 'DedicatedApiProposalAll' but got '{dedicated_api_proposals.__class__.__name__}'. "
        "This may indicate the LLM failed to produce properly structured output."
    )
    return None

  try:
    return states.ApiProposalAll(**dedicated_api_proposals.model_dump())
  except Exception as e:
    # Log details about validation errors, especially for 'too_long' errors
    if "too_long" in str(e).lower():
      logging.error(
          f"API proposal validation failed: LLM returned too many APIs in a proposal. "
          f"Expected max {states._NUM_API_PROPOSALS} APIs per proposal. "
          f"Error: {e}")
    else:
      logging.error(f"Failed to convert dedicated API proposals: {e}")
    return None


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


def create_api_proposer(
    apis: List[StructuredTool],
    is_mcp: bool = False,
    num_proposals: int = 3,
) -> BaseChatModel:
  llm = langchain_utils.create_llm()

  # Dynamically create the output structure.
  customized_api_proposals = get_literal_api_proposals_pydantic_model(apis)

  # Use MCP-specific prompt when in MCP mode to inject filesystem constraints
  if is_mcp:
    import toolgrad as tog
    toolgrad_root = tog.__path__[0]
    filesystem_workspace = os.path.join(toolgrad_root, "utils", "filesystem")
    api_proposer = prompt_lib.TOOLUSE_PROPOSER_WOO_MCP.partial(
        filesystem_workspace=filesystem_workspace,
        num_proposals=num_proposals,
    ) | llm.with_structured_output(customized_api_proposals)
  else:
    api_proposer = prompt_lib.TOOLUSE_PROPOSER_WOO.partial(
        num_proposals=num_proposals,
    ) | llm.with_structured_output(customized_api_proposals)
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
    apis_in_one_proposal: List[StructuredTool],
    is_mcp: bool = False,
) -> AgentExecutor:
  llm = langchain_utils.create_llm()

  if is_mcp:
    import toolgrad as tog
    # Use the toolgrad package path to locate the filesystem directory
    toolgrad_root = tog.__path__[0]
    filesystem_workspace = os.path.join(toolgrad_root, "utils", "filesystem")
    system_prompt = f"""You are a helpful assistant.

IMPORTANT: Your filesystem tools ONLY have access to files within the allowed directory.

You MUST use simple filenames or relative paths.
DO NOT use absolute paths - they will be rejected."""
  else:
    system_prompt = "You are a helpful assistant."
  prompt = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("human", prompt_lib.API_EXECUTOR.messages[0].prompt.template),
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
  llm = langchain_utils.create_llm()
  api_execution_validation = states.create_apiexecutionvalidation(
      allowed_values=key_list)
  api_execution_validator = prompt_lib.API_EXECUTION_VALIDATOR | llm.with_structured_output(
      api_execution_validation)
  return api_execution_validator


# TODO: refactor to use langchain 1.0 create_agent
async def single_api_execute_step_wip(
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
        if (valid_api := find_tool_by_name(api.name, tools=sampled_apis)
           ) is not None
    ]

  model = langchain_utils.create_llm()
  # Note: We cannot use response_format=states.ApiReport because ApiReport contains
  # complex types (Tuple[ToolAgentAction, ...]) that Gemini cannot serialize in function calling
  api_executor = create_agent(
      model=model,
      tools=apis,
      system_prompt=
      "You are a helpful assistant that uses tool by following the user request.",
  )
  try:
    api_executions = await api_executor.ainvoke({
        "messages": [{
            "role":
                "user",
            "content":
                prompt_lib.API_EXECUTOR.format(plan=proposal.instruction),
        }]
    })
  except Exception as e:
    print(f"API execution failed for proposal {proposal.id}: \n{e}")
    logging.info(
        f"Failed to execute the API in the proposal {proposal.id}: adding a placeholder report."
    )
    return failure_report()
  return api_executions


async def single_api_execute_step(
    proposal: states.ApiProposal,
    is_hash_api_name: bool = False,
    is_toolbench: bool = True,
    sampled_apis: List[StructuredTool] | None = None,
) -> states.ApiReport:

  def failure_report(error_msg: str | None = None,
                     num_calls: int = 0) -> states.ApiReport:
    return states.ApiReport(
        id=proposal.id,
        success=False,
        api_call_step=None,
        description="",
        num_tool_calls=num_calls,
        error_message=error_msg,
    )

  # Create invocation tracker (shared mutable state)
  invocation_tracker = {'called': False}

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
        if (valid_api := find_tool_by_name(api.name, tools=sampled_apis)
           ) is not None
    ]

  # Wrap all tools with tracking decorator
  tracked_apis = [
      wrap_tool_with_tracking(tool, invocation_tracker) for tool in apis
  ]

  api_executor = create_api_executor(
      apis_in_one_proposal=tracked_apis,
      is_mcp=not is_toolbench,
  )

  try:
    # Legacy AgentExecutor
    api_executions = await api_executor.ainvoke({
        "plan": proposal.instruction,
    })
  except Exception as e:
    error_msg = str(e)
    print(f"API execution failed for proposal {proposal.id}: \n{error_msg}")
    logging.info(
        f"Failed to execute the API in the proposal {proposal.id}: adding a placeholder report."
    )
    # Check if tool was called before exception, OR if this is a validation error
    # ValidationError occurs when LLM generates arguments but they fail Pydantic validation
    # (e.g., missing required fields, wrong types). These should count as tool call attempts
    # since the LLM tried to invoke the tool, it just failed argument validation.
    is_validation_error = isinstance(e, ValidationError)
    num_calls = 1 if (invocation_tracker['called'] or
                      is_validation_error) else 0
    return failure_report(error_msg, num_calls=num_calls)
  if 'intermediate_steps' not in api_executions:
    return failure_report("No intermediate steps in execution result")

  # Track the total number of tool calls made
  num_tool_calls = len(api_executions['intermediate_steps'])

  processed_steps = [
      toolbench_langchain_utils.process_one_intermediate_step(
          inter_step, remove_msglog=True, hash_to_name=is_toolbench)
      for inter_step in api_executions['intermediate_steps']
  ]

  processed_steps = [e for e in processed_steps if e is not None and e != ()]
  if processed_steps.__len__() == 0:
    return failure_report("Execution produced no valid steps",
                          num_calls=num_tool_calls)

  last_success_step_id = get_success_execution_id_legacy(
      processed_steps, is_toolbench)

  if last_success_step_id is None:
    # Log detailed error information for debugging
    logging.warning(f"API execution failed for proposal {proposal.id}.")
    error_details = []
    for i, (action, result) in enumerate(processed_steps):
      # Check for validation errors first (from validate_api_response)
      if isinstance(result, dict) and result.get('validation_error'):
        validation_error = result.get('validation_error')
        logging.warning(
            f"  Step {i}: {action.tool} - Validation failed: {validation_error}"
        )
        error_details.append(f"{action.tool}: {validation_error}")
      elif isinstance(result, dict) and result.get('error'):
        error_msg = result.get('error')
        logging.warning(f"  Step {i}: {action.tool} - Error: {error_msg}")
        error_details.append(f"{action.tool}: {error_msg}")
      elif isinstance(result, str) and ('error' in result.lower() or
                                        'failed' in result.lower()):
        logging.warning(f"  Step {i}: {action.tool} - Result: {result[:200]}")
        error_details.append(f"{action.tool}: {result[:200]}")
    error_message = " | ".join(
        error_details) if error_details else "All tool executions failed"
    return failure_report(error_message, num_calls=num_tool_calls)

  api_report = {
      'id': proposal.id,
      'success': True,
      'api_call_step': processed_steps[last_success_step_id],
      'description': None,
      'num_tool_calls': num_tool_calls,
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
    for result in results:
      assert isinstance(result, states.ApiReport)
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
  llm = langchain_utils.create_llm()
  api_selector = prompt_lib.SELECTOR | llm.with_structured_output(output_schema)
  return api_selector


# Workflow modules
def get_valid_chainids(max_num_chains: int = 10) -> list[str]:
  return ["api_use_chain_" + str(i) for i in range(1, max_num_chains + 1)]


def chain_update_step(
    workflow_cur: states.ApiUseWorkflow,
    which_chain: str,
    api_step: Tuple[dict, Union[dict, str]],
    num_saturated_warnings: int | None = 3,
    max_num_chains: int = 10,
) -> dict[str, Any]:
  """Update or create a chain in the workflow.

  Args:
      workflow_cur (states.ApiUseWorkflow): the current workflow state.
      which_chain (str): the chain ID to update or '-1' to create a new chain.
      api_step (Tuple[dict, Union[dict, str]]): the API step to add to the chain.
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


def create_workflow_updater():
  """Create a workflow updater that generates query and response only.

  Note: This returns only query/response fields, not api_use_chains.
  The caller must combine LLM output with actual execution chains.
  """
  predict_workflow_prompt = prompt_lib.PREDICT_WORKFLOW
  llm = langchain_utils.create_llm()

  workflow_updater = predict_workflow_prompt | llm.with_structured_output(
      states.WorkflowQueryResponse)
  return workflow_updater
