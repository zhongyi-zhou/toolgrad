from langchain_core.prompts import ChatPromptTemplate

JUDGE = ChatPromptTemplate.from_template("""Task Overview:
You are tasked with evaluating the quality of a response to a user query. The response is ground on a tool use trace, which is a list of (api_use_request, api_response) tuples.
Your evaluation should produce a score between 0 and 100, based on how well the response addresses each aspect of the user query compared to the provided ground truth response.


Evaluation Criteria:
1. Coverage of Requests:
- User Requests Count: Identify the number of distinct requests or tasks contained in the user query.
- Response Count: Determine how many of these requests the response addresses.
- If a request is not addressed at all, that aspect should receive a score of 0.
2. Quality of Each Response:
- For each request/task that the request addresses, rate the quality of the response on a scale from 0 to 100.
- If all API calls related to the request are failed, then the score is 0.
- If there is successful API call related to the request, then the score can be greater than 0.
- socre = 100 means the response is 1) grounded on successful API call, 2) the response can respond to the user query similar to the ground truth response.
3. Final Score Calculation:
- Compute the final score by averaging the individual scores for each aspect of the query.
- For example, if the user query requests 5 tasks, the AI response only does 3 tasks, and the quality of the response is 80, 90, 70, then the final score is (80 + 90 + 70 + 0 + 0) / 5 = 48.

Input Data:
User query:
{query}

Tool use trace:
{tool_use_trace}

The response to evaluate:
{pred}

Ground truth response:
{gt}
""")
