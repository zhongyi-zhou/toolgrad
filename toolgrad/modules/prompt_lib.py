from langchain_core.prompts import ChatPromptTemplate

SCHEMA_INCONTEXT_EXAMPLE = """As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted. You should also ensure the output contains all the required fields in the schema."""

TOOLUSE_PROPOSER_WOO = ChatPromptTemplate.from_template(
    """You are tasked with augmenting an API-use workflow with more APIs from a given library so that it can serve for more advanced tasks.
Given the following information that provides the context, please make {num_proposals} API-use proposals to augment the current workflow.

The current workflow:
{workflow_cur}

The following is a pool of APIs that you can use:
{api_all}

CRITICAL - Instruction Format Requirements:
- Each proposal's "instruction" field must be an ACTIONABLE COMMAND that tells an agent executor HOW to use the proposed API(s).
- Instructions must be IMPERATIVE commands (e.g., "Get...", "Use...", "Retrieve..."), NOT descriptive summaries.
- DO NOT reference previous executions or results (e.g., avoid "This tool was used..." or "It returned...").
- The instruction will be given to an agent executor that will use it to decide which tools to call and with what inputs.

Notes:
- Please reply in the required data structure.
- To select an API, you should return its name.
- If you do not have any additional tools to propose, you can respond with None.
""")

TOOLUSE_PROPOSER_WOO_MCP = ChatPromptTemplate.from_template(
    """You are tasked with augmenting an API-use workflow with more APIs from a given library so that it can serve for more advanced tasks.
Given the following information that provides the context, please make {num_proposals} API-use proposals to augment the current workflow.

The current workflow:
{workflow_cur}

The following is a pool of APIs that you can use:
{api_all}

⚠️ CRITICAL FILESYSTEM CONSTRAINT:
All filesystem APIs can ONLY access files within the allowed directory.
You MUST use simple filenames or relative paths.
DO NOT use absolute paths - they will fail.

CRITICAL - Instruction Format Requirements:
- Each proposal's "instruction" field must be an ACTIONABLE COMMAND that tells an agent executor HOW to use the proposed API(s).
- Instructions must be IMPERATIVE commands (e.g., "Get...", "Use...", "Retrieve..."), NOT descriptive summaries.
- DO NOT reference previous executions or results (e.g., avoid "This tool was used..." or "It returned...").
- The instruction will be given to an agent executor that will use it to decide which tools to call and with what inputs.

Notes:
- Please reply in the required data structure.
- To select an API, you should return its name.
- If you do not have any additional tools to propose, you can respond with None.
""")

CHAIN_SUMMARIZER = ChatPromptTemplate.from_template(
    """You are an API-use chain summarizer.
Given the following list of API execution steps, you need to describe the overall functionality of the API-use chain.
The goal is to make the description informative so that another agent can understand the functionality of the API-use chain without reading the API execution steps.

API execution steps:
{api_execution_steps}
""")

STRUCTURER_TEMPLATE = ChatPromptTemplate.from_template(
    """Given the following conversation between the user and an AI assistant, please summarize it into a structured format.
Conversation history:
{messages}
""")

API_EXECUTOR = ChatPromptTemplate.from_template(
    """You are tasked with exploring an API based on a given plan.
The following shows a guide for you to follow:
1. Verify whether the API-calling result provides a reasonable response for the given plan.
2. Report `success = False` if you fail to get a reasonable result, and explain why.
3. Report `success = True` if you get a reasonable result that addresses the plan, and provide justification for the success.
4. If you report `success = True`, you should also report which function calling step leads to the success.

Retry Logic:
- The API may return bad results (e.g., error messages, completely irrelevant data, or responses that don't address the plan).
- In such cases, you should try again with different input parameters if reasonable alternatives exist.
- You have a maximum number of iterations to complete the task. Use them wisely.
- If retries fail or no reasonable alternative inputs exist, report `success = False`.

The following is the plan:
{plan}

Notes:
- Success criterion: If the API execution provides a reasonable result that addresses the given plan, report `success = True`.
- If an API fails to execute or returns unusable results after reasonable attempts, report `success = False`.
""")

API_EXECUTOR_SYSTEM_PROMPT = """You are tasked with exploring an API based on a plan given by the user.
The following shows a guide for you to follow:
1. Verify whether the API-calling result provides a reasonable response for the given plan.
2. Report `success = False` if you fail to get a reasonable result, and explain why.
3. Report `success = True` if you get a reasonable result that addresses the plan, and provide justification for the success.
4. If you report `success = True`, you should also report which function calling step leads to the success.

Retry Logic:
- The API may return bad results (e.g., error messages, completely irrelevant data, or responses that don't address the plan).
- In such cases, you should try again with different input parameters if reasonable alternatives exist.
- You have a maximum number of iterations to complete the task. Use them wisely.
- If retries fail or no reasonable alternative inputs exist, report `success = False`.

Notes:
- Success criterion: If the API execution provides a reasonable result that addresses the given plan, report `success = True`.
- If an API fails to execute or returns unusable results after reasonable attempts, report `success = False`.
"""

API_EXECUTION_VALIDATOR = ChatPromptTemplate.from_template(
    """You are tasked with verifying an API execution agent based on a given plan.

You will be provided with a dictionary of tuples.
Each tuple stores one agent API execution step and its corresponding output.

You should:
- Verify whether the API-calling result follows the plan.
- Report whether the API execution is successful or not.
- Select the API calling step that leads to the success if the API execution is successful.
- Provide a description of the API execution if it is successful.

The following is the plan:
{plan}

The following are the API execution steps:
{api_execution_steps}
""")

SELECTOR = ChatPromptTemplate.from_template("""You are an API selector.

You need to select one API or refuse to select any API from the given list of APIs to augment the current workflow.

The current API-use workflow:
{workflow_cur}

Reports from the proposed APIs:
{api_reports}

When you select an API, you need to make the following decisions:
1. Determine whether any API can be used to augment the current workflow.
2. If yes, select one API to augment the current workflow.
3. Decide whether to append the selected API to an existing API-use chain or create a new API-use chain:
    3.1 **Append to existing chain**: Choose this when the API logically should follow another call in an existing chain. This includes cases where:
        - The `tool_input` contains data from a previous API's `response`
        - The API's purpose depends on results from a previous call
        - Even if `tool_input` is minimal, the API conceptually continues work from a previous step
        When you decide to append, you should also select which API-use chain to append to.
    3.2 **Create new chain**: Choose this when the API is independent and does not logically depend on any existing API execution. For example:
        - The API addresses a completely separate aspect of the workflow
        - The `tool_input` is self-contained and doesn't rely on previous results
""")

PREDICT_WORKFLOW = ChatPromptTemplate.from_template(
    """You are generating training data for a tool-use language model. Given API execution traces, create a natural user query that would trigger these API calls, followed by an appropriate response.

**API Execution Details:**
{api_use_chains}

**Task:** Generate (1) a natural user query and (2) the agent's response based on the API execution traces above.

**Important:** You will receive the API execution chains for context, but you should NOT return them in your output. Only return the query and response fields.

---

**CRITICAL: User Query Requirements**

1. **✅ DO**: Write queries like a real human would ask
   - "What's the weather forecast for London next week?"
   - "I'm researching Tesla stock. Show me recent performance and news."
   - "Find me Italian restaurants near Golden Gate Park with good ratings."

2. **❌ NEVER**: Mention APIs, tools, functions, or technical implementation
   - Never say: "call the weather API", "use get_forecast", "invoke the search tool"
   - Never ask: "which API should I use", "can you run this function"
   - Never include: tool names, API endpoints, function signatures

3. **✅ DO**: Provide specific, concrete details
   - Include exact values from tool_input (locations, IDs, names, numbers)
   - Use specific examples: "123 Main St, Oakland CA" not "an address"
   - Mention precise entities: "Tesla stock" not "a company's stock"

4. **✅ DO**: Create realistic scenarios
   - Explain WHY the user needs this information:
     * "I'm planning a trip..."
     * "I'm writing a research report on..."
     * "I need to prepare for a meeting about..."
   - Make the request feel natural and purposeful

5. **✅ DO**: Cover ALL API calls implicitly
   - If 3 APIs were called, the query should naturally require all 3
   - Don't list them ("do A, B, and C"), weave them into a cohesive need
   - Example: Instead of "Get weather, find hotels, search restaurants"
     → "I'm visiting Paris this weekend. What should I expect, and where should I stay and eat?"

---

**Response Requirements:**
- Synthesize all API execution results into a helpful, natural response
- Present information clearly without mentioning APIs or tools
- Reference concrete data from the execution outputs
- Sound like a knowledgeable assistant answering a user's question

---

**Examples:**

**Example 1:**
API Chains: [weather(city="Tokyo"), currency_convert(from="JPY", to="USD", amount=5000)]
Query: "I'm traveling to Tokyo next month. What's the current weather like, and how much is 5,000 yen in US dollars?"
Response: "Tokyo is currently experiencing mild temperatures around 18°C with partly cloudy skies. As for the currency conversion, 5,000 Japanese yen is approximately 33 US dollars."

**Example 2:**
API Chains: [github_search(topic="ML"), github_get_repo(id="tensorflow/tensorflow"), github_get_contributors(id="tensorflow/tensorflow")]
Query: "I'm researching popular machine learning projects on GitHub. Can you tell me about TensorFlow—how active is the project and who are the main contributors?"
Response: "TensorFlow is one of the most popular machine learning frameworks on GitHub with over 180,000 stars. The project is very active with regular updates. The main contributors include members of the Google Brain team, with key developers like Jeff Dean and Rajat Monga being significant contributors."

---

Make the query sound like something a real person would ask in a conversation or search bar.
""")
