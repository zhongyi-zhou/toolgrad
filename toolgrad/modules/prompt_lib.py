from langchain_core.prompts import ChatPromptTemplate

SCHEMA_INCONTEXT_EXAMPLE = """As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted. You should also ensure the output contains all the required fields in the schema."""

TOOLUSE_PROPOSER_WOO = ChatPromptTemplate.from_template(
    """You are tasked with augmenting an API-use workflow with more APIs from a given library so that it can serve for more advanced tasks.
Given the following information that provides the context, please make three API-use proposals to augment the current workflow.

The current workflow:
{workflow_cur}

The following is a pool of APIs that you can use:
{api_all}

Notes:
- Please reply in the required data structure.
- To select an API, you should return its name.
- If you do not have any additional tools to propose, you can respond with None.
""")

TOOLUSE_PROPOSER = ChatPromptTemplate.from_template(
    """You are tasked with augumenting an API-use workflow with more APIs from a given library so that it can serve for more advanced tasks.
Given the following information that provides the context, please make three API-use proposals to augument the current workflow.
To do so, you should first recall what APIs you can use in the library and then make the proposals.
The output should be formatted as a JSON instance that conforms to the JSON schema explained below.

The current workflow:
{workflow_cur}

The proposed APIs in the previous iteration:
{api_proposals}

When you decide to respond to the user, you should output a json object by following the schema below:
{json_schema}

Notes:
- You can have multiple or just one API(s) in each proposal.
- If you do not have any additional tools to propose, you can respond with None.
- When selecting an API, you should use the name after "functions." only. For example, it should be `9f6e2309` instead of `functions.9f6e2309`"".
""")

CHAIN_SUMMARIZER = ChatPromptTemplate.from_template(
    """You are an API-use chain summarizer.
Given a following list of API execution steps, you need to describe the overall functionality of the API-use chain.
The goal is to make the description informative so that another agent can understand the functionality of the API-use chain without reading the api execution steps.

API execution steps:
{api_execution_steps}
""")

STRUCTURER_TEMPLATE = ChatPromptTemplate.from_template(
    """Given the following conversation between the user an AI assistant, please summarize it into a structured format.
Conversation hitstory:
{messages}
""")

API_EXECUTOR = ChatPromptTemplate.from_template(
    """You are tasked with exploring an API based on a given plan.
The following shows a guide for you to follow:
1. verify whether the API-calling result follows the plan.
2. report `success = False`, if you fail to get the expected result, and explain why.
3. report `success = True`, if you get the expected result, and provide justification for the success.
4. if you report `success = True`, you should also report which function calling step leads to the success.


Chances are that the API may return bad results or fail to execute in one attempt.
In such cases, you should do another try by changing the input.
If it still fails, you should report `success = False`.

The following is the plan:
{plan}

Notes:
- If you consider the API execution provides a reasonable result of a given plan, you should report `success = True`.
- If an API fails to execute, you should report `success = False`.
""")

API_EXECUTION_VALIDATOR = ChatPromptTemplate.from_template(
    """You are tasked with verifying an API execution agent based on a given plan.

You will be provided with a dictionary of tuples.
Each tuple store one agent api execution step and its corresponding output.

You should
- verify whether the API-calling result follows the plan.
- report whether the API execution is successful or not.
- select the api calling step that leads to the success if the API execution is successful.
- provide a description of the API execution if it is successful.

The following is the plan:
{plan}

The following are the API execution steps:
{api_execution_steps}
""")

SELECTOR = ChatPromptTemplate.from_template("""You are an API selector.

You need to select one API or refuse to select any API from the given list of APIs to augument the current workflow.

The current api-use workflow:
{workflow_cur}

Reports from the proposed APIs:
{api_reports}

When you select an api, you need to make the following decisions:
1. whether any API can be used to augment the current workflow.
2. if yes, select one API to augment the current workflow.
3. decide whether you want to append the selected API to a api-use chain or create a new api-use chain with this API. 
    3.1 When the `tool_input` value in `ToolAgentAction` of this API is dependent on any API execution `response` in an api-use chain, choose the append operation. Examples include the `tool_input` reuse any information in the `response`. When you decide to append, you should also select which api-use chain you want to append the selected API to. 
    3.2 If the `tool_input` value in `ToolAgentAction` is not dependent of any API execution `response` in any api-use chain of the current workflow, you should create a new api-use chain with this API. For example, if the `tool_input` is empty, you should always choose to create a new api-use chain.
""")

PREDICT_WORKFLOW = ChatPromptTemplate.from_template(
    """Given the following API usage chains: {api_use_chains}, your task is to:

1. *Infer the user query* that would have triggered all the API-calling events. The query should be sufficiently detailed to ensure an LLM can trigger all API calls in the provided chains.
2. *Predict the agent's response* to the user after executing all API calls in the workflow. The response should reflect the results of the executed APIs in a natural and informative way.
Notes:
- The inferred user query must be comprehensive enough to guide the LLM in generating all API calls (including the input and the selection of api/tool name) across the given API-use chains.
- Ensure that the agent's response accurately summarizes or presents the results of the API executions."""
)

PREDICT_WORKFLOW_LAST_STEP = ChatPromptTemplate.from_template(
    """As a reference, the following is the triplet (user query, api use chain, response) in the last step: 
{last_basemodel}
""")
