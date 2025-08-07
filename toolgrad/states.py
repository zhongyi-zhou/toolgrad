from typing import List, Union, Tuple, Dict, Optional, Literal
from pydantic import BaseModel, Field, create_model
from langchain.agents.output_parsers.tools import ToolAgentAction


class ApiUseChainNode(BaseModel):
  name: Optional[str] = Field(
      description=
      "name of the selected api. It is a hashed code provided in each tool configuration but without 'functions.' in the front. For example, it should be `9f6e2309` instead of `functions.9f6e2309`"
  )


def create_dedicated_apiusechainnode(allowed_values: List[str]):
  # literal_type = Literal[tuple(allowed_values + [None])]
  literal_type = Optional[Literal[tuple(allowed_values)]]

  # Since we are not using agent executor, there is no need to clarify `functions.XXX`.
  description = "the name of the selected api."

  DedicatedApiUseChainNode = create_model(
      'DedicatedApiUseChainNode',  # Name of the new model
      name=(literal_type, Field(description=description)),  # Override 
      __base__=ApiUseChainNode  # inherit from OriginalModel
  )
  return DedicatedApiUseChainNode


class ApiSelection(BaseModel):
  id: Optional[str] = Field(
      description=
      "The id of ApiReport selected. The id should match the id in the ApiReport or None if none of them is sense-making."
  )
  reason: str = Field(description="the reasoning process of this selection.")
  which_chain: str = Field(
      description=
      "If you want to create a new chain, use -1. Otherwises, select the api-use chain_id in the current workflow that you want to augument with the selected api. If you choose not to select, use -1 here."
  )


def create_dedicated_apiselection(id_list: List[str], chain_keys: List[str]):

  DedicatedApiSelection = create_model(
      'DedicatedApiSelection',  # Name of the new model
      id=(Optional[Literal[tuple(id_list)]],
          Field(description=ApiSelection.model_fields['id'].description)),
      which_chain=(Literal[tuple(chain_keys + ['-1'])],
                   Field(description=ApiSelection.model_fields['which_chain'].
                         description)),
      __base__=ApiSelection  # inherit from OriginalModel
  )
  return DedicatedApiSelection


class ApiUseChainNodeReport(BaseModel):
  id: str = Field(
      description=
      "the id of the node function; the id must be unique within the workflow.")
  input: Optional[str] = Field(
      description="the query input of the api, if any.")
  output: Optional[str] = Field(description="the output of the api")


class ApiUseEdge(BaseModel):
  source: str = Field(description="the id of the source node")
  destination: str = Field(description="the id of the destination node")


class ApiProposal(BaseModel):
  """The proposed API for enhancing the api-use workflow."""
  id: str = Field(description="a unique id for the proposal.")
  instruction: str = Field(
      description='a detailed instruction on how to use the proposed API(s).')
  api: List[ApiUseChainNode] = Field(description='The proposed API list')


def create_dedicated_apiproposal(api_node_model: ApiUseChainNode):
  # literal_type = Literal[tuple(allowed_values + [None])]
  description = ApiProposal.model_fields['api'].description
  return create_model(
      'DedicatedApiProposal',  # Name of the new model
      api=(List[api_node_model], Field(description=description)),  # Override 
      __base__=ApiProposal  # inherit from OriginalModel
  )
  # return DedicatedApiProposal


class ApiProposalAll(BaseModel):
  """proposed APIs for enhancing the api-use workflow."""
  reason: str = Field(
      description=
      'The reasoning process of the workflow extension. Think step by step, and explain why you choose the proposed APIs.'
  )
  proposals: List[ApiProposal] = Field(description='A list of proposed APIs')


def create_dedicated_apiproposalall(api_proposal_model: ApiProposal):
  description = ApiProposalAll.model_fields['proposals'].description
  return create_model(
      'DedicatedApiProposalAll',  # Name of the new model
      proposals=(List[api_proposal_model],
                 Field(description=description)),  # Override 
      __base__=ApiProposalAll  # inherit from OriginalModel
  )


class ApiExecutionStep(ApiUseChainNode):
  input: Optional[Dict[str, str]] = Field(
      description=
      "the query input of the api, if any. It should be in a dictionary format in which the keys are the parameters of this api and the values are stingified input. For example, to specify the input of a function that takes two api parameters, you can write {'param1': 'value1', 'param2': 'value2'}."
  )
  name: str = Field(
      description=
      "name of the selected api. It should be in the format of 'category-api_name-tool_name' instead of 'functions.category-api_name-tool_name'."
  )
  output: Optional[str] = Field(description="the output of the api execution")


class ApiExecutionValidatorBase(BaseModel):
  success: bool = Field(
      description="whether the API execution result follows the plan.")
  step_key: Optional[str] = Field(
      description=
      "the seletced key in the api execution dictionary that contains the selected execution step. Return None if the execution is not successful."
  )
  description: Optional[str] = Field(
      description=
      "the description of the selected api execution step. Return None if the execution is not successful."
  )


def create_apiexecutionvalidation(allowed_values: List[str]):
  literal_type = Optional[Literal[tuple(allowed_values)]]

  description = ApiExecutionValidatorBase.model_fields['step_key'].description

  ApiExecutionValidation = create_model(
      'ApiExecutionValidation',  # Name of the new model
      step_key=(literal_type, Field(description=description)),  # Override 
      __base__=ApiExecutionValidatorBase  # inherit from OriginalModel
  )
  return ApiExecutionValidation


class ApiUseChainBase(BaseModel):
  """The base class for an api-use chain."""
  intermediate_steps: Optional[List[Tuple[ToolAgentAction, Union[
      dict, str]]]] = Field(
          default=[],
          description=
          'the intermediate steps of the api execution, if any. Use [] if there is no intermediate step.'
      )
  description: str = Field(
      description=
      "the description of this api-use chain that elaborate what each api does in the `intermediate_steps`."
  )


class ApiUseChainBase(BaseModel):
  chain_id: str = Field(
      description="the unique id of the api-use chain in a workflow.")


class ApiUseChain(ApiUseChainBase):
  intermediate_steps: Optional[List[Tuple[ToolAgentAction, Union[
      dict, str]]]] = Field(
          default=[],
          description=
          'the intermediate steps of the api execution, if any. Use [] if there is no intermediate step.'
      )


class ApiUseWorkflow(BaseModel):
  query: str = Field(
      description=
      "the user query. This query requires the LLM to use apis for providing answers."
  )
  api_use_chains: Dict[str, ApiUseChain] = Field(
      description="a dictionary of api-use chains")
  response: str = Field(
      description=
      "the response to the user query by refering to the results from api-use chains."
  )


class ApiReport(BaseModel):
  id: str = Field(
      default="-1",
      description=
      ("A unique identifier for the report. This ID must match the proposal's ID. "
       "Avoid using the default '-1' except when logging an execution error."))
  success: bool = Field(
      description="Indicates whether the API execution chain was successful.")
  api_call_step: Optional[Tuple[ToolAgentAction, Union[dict, str]]] = Field(
      default=None,
      description=
      "The API execution step represented as a tuple: (api_calling_action, api_response)."
  )
  description: str | None = Field(
      default=None,
      description="A natural language explanation of the api_call_step.",
  )

  class Config:
    extra = "forbid"  # Disallow additional properties


class ApiUseState(BaseModel):
  workflow_cur: Optional[ApiUseWorkflow] = Field(
      default=None,
      description="the api-use workflow in the current iteration.")
  api_proposals: Optional[ApiProposalAll] = Field(
      default=None,
      description='the proposed API usage in the current iteration.')
  api_reports: Optional[Dict[str, ApiReport]] = Field(
      default=None,
      description='A dictionary of api execution reports of the proposed APIs.')
  api_selection: Optional[ApiSelection] = Field(
      default=None,
      description='the selected API usage in the current iteration.')
