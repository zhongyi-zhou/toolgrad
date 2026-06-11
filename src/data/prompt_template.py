SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
"""

SYSTEM_PROMPT_PYTHON = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
"""

# Aligned with BFCL's JSON format
RESPONSE_SCHEMA_DICT = [
    {
        "function": "api_name",
        "parameters": {
            "param1": "value1",
            "param2": "value2"
        }
    }
]

RETURN_SCHEMA = """The following is the response json schema you should follow:
```json
{response_json_schema}
```
"""

SELECTION_POOL = """The following is a list of apis in the library that you can use.
Each api is a JSON object with 'name', 'description' and 'parameters'.

```json
{selection_pool}
```
"""
