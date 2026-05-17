import os
import gin
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_genai import (ChatGoogleGenerativeAI, HarmBlockThreshold,
                                    HarmCategory)
from langchain_google_vertexai import ChatVertexAI
from typing import TypedDict
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from anthropic import AnthropicVertex
from typing import Union, Literal
from pydantic import BaseModel, Field, create_model
import google.auth
import google.auth.transport.requests


from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any

__all__ = [
    "CustomChatOpenAI", "CustomAzureChatOpenAI", "create_llm",
    "remove_fields_from_model", "IterState", "MaxIters",
    "GeminiCallbackHandler", "create_callback_handler"
]

class GeminiCallbackHandler(BaseCallbackHandler):
  """Callback handler for tracking Gemini token usage and costs."""

  def __init__(self,
               prompt_token_cost_per_million=1.0,
               completion_token_cost_per_million=4.0):
    """Initialize the callback handler with configurable pricing.
    
    Args:
      prompt_token_cost_per_million: Cost in USD per 1 million prompt tokens
      completion_token_cost_per_million: Cost in USD per 1 million completion tokens
    """
    super().__init__()
    self.total_tokens = 0
    self.prompt_tokens = 0
    self.completion_tokens = 0
    self.prompt_tokens_cached = 0
    self.reasoning_tokens = 0
    self.successful_requests = 0
    self.total_cost = 0.0

    # Convert per-million pricing to per-token pricing
    self.prompt_token_cost = prompt_token_cost_per_million / 1_000_000
    self.completion_token_cost = completion_token_cost_per_million / 1_000_000

  def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
    """Extract token usage from LLM response."""
    # Token usage is in the AIMessage object within generations
    if response.generations and len(response.generations) > 0:
      gen = response.generations[0][0]
      if hasattr(gen, 'message') and hasattr(gen.message, 'usage_metadata'):
        usage = gen.message.usage_metadata
        if usage:
          prompt_tokens = usage.get('input_tokens', 0)
          completion_tokens = usage.get('output_tokens', 0)
          token_details = usage.get('input_token_details', {})
          cached_tokens = token_details.get('cache_read', 0)

          self.prompt_tokens += prompt_tokens
          self.completion_tokens += completion_tokens
          self.prompt_tokens_cached += cached_tokens
          self.total_tokens += prompt_tokens + completion_tokens

          # Calculate cost (adjust pricing as needed)
          self.total_cost += (prompt_tokens * self.prompt_token_cost +
                              completion_tokens * self.completion_token_cost)

          self.successful_requests += 1

  def __str__(self) -> str:
    return (f"Tokens Used: {self.total_tokens}\n"
            f"\tPrompt Tokens: {self.prompt_tokens}\n"
            f"\t\tPrompt Tokens Cached: {self.prompt_tokens_cached}\n"
            f"\tCompletion Tokens: {self.completion_tokens}\n"
            f"\t\tReasoning Tokens: {self.reasoning_tokens}\n"
            f"Successful Requests: {self.successful_requests}\n"
            f"Total Cost (USD): ${self.total_cost:.2f}")


@gin.configurable
def create_callback_handler(prompt_token_cost_per_million=1.0,
                            completion_token_cost_per_million=4.0):
  """Create a GeminiCallbackHandler with configurable pricing.
  
  Args:
    prompt_token_cost_per_million: Cost in USD per 1 million prompt tokens
    completion_token_cost_per_million: Cost in USD per 1 million completion tokens
    
  Returns:
    GeminiCallbackHandler instance with configured pricing
  """
  return GeminiCallbackHandler(
      prompt_token_cost_per_million=prompt_token_cost_per_million,
      completion_token_cost_per_million=completion_token_cost_per_million)

_VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
class CustomChatOpenAI(ChatOpenAI):

  def structured_output(self, *args, **kwargs):
    # If 'strict' isn’t provided in kwargs, default it to True.
    kwargs.setdefault("strict", True)
    return super().with_structured_output(*args, **kwargs)


class CustomAzureChatOpenAI(AzureChatOpenAI):

  def structured_output(self, *args, **kwargs):
    # If 'strict' isn’t provided in kwargs, default it to True.
    kwargs.setdefault("strict", True)
    return super().with_structured_output(*args, **kwargs)


@gin.configurable
def get_llm_codename(name: str) -> str:
  return name


class VertexClient:
  def __init__(self, project_id: str = None, location: str = "us-central1"):
    self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    self.location = location
    
    if not self.project_id:
        raise ValueError("project_id must be provided or set in GOOGLE_CLOUD_PROJECT env var")

    # 1. Initialize shared credentials
    self.credentials, _ = google.auth.default()
    self.auth_req = google.auth.transport.requests.Request()

  def _get_access_token(self) -> str:
    """Refreshes and returns the Google Cloud access token."""
    self.credentials.refresh(self.auth_req)
    return self.credentials.token

  def get_claude(self, model_name: str = "claude-haiku-4-5@20251001", **kwargs):
    """Returns a LangChain ChatAnthropicVertex instance."""
    client = AnthropicVertex(
        region="global", # Often needs to be global for the client itself
        project_id=self.project_id
    )
    
    return ChatAnthropicVertex(
        client=client,
        model_name=model_name,
        # location="global", # Explicitly set to global as verified
        **kwargs
    )

  def get_openai_compatible(self, model_name: str, location: str = None, **kwargs):
    """
    Returns a LangChain ChatOpenAI instance configured for Vertex AI MaaS.
    Used for DeepSeek, Qwen, Llama, etc. hosted on Vertex Model Garden MaaS.
    """
    # Use provided location or default to client location
    loc = location or self.location
    
    # Vertex AI OpenAI-compatible endpoint
    base_url = f"https://{loc}-aiplatform.googleapis.com/v1beta1/projects/{self.project_id}/locations/{loc}/endpoints/openapi"
    
    return ChatOpenAI(
        model=model_name,
        api_key=self._get_access_token(),
        base_url=base_url,
        **kwargs
    )

  # Some working examples below
  def get_deepseek(self, model_name: str = "deepseek-ai/deepseek-v3.1-maas", location: str = "us-west2", **kwargs):
    return self.get_openai_compatible(model_name, location=location, **kwargs)

  def get_qwen(self, model_name: str = "qwen/qwen3-235b-a22b-instruct-2507-maas", location: str = "us-south1", **kwargs):
    return self.get_openai_compatible(model_name, location=location, **kwargs)

  def get_llama(self, model_name: str, location: str = None, **kwargs):
    return self.get_openai_compatible(model_name, location=location, **kwargs)



@gin.configurable
def create_llm(
    temp: float,
    provider: Literal['openai', 'azure', 'google', 'vertex'] = 'vertex',
    model: str = "gemini-2.5-flash-lite",
    location: str = "us-central1",
) -> BaseChatModel:
  if provider == 'vertex':
    if os.getenv("GOOGLE_CLOUD_PROJECT") is None:
      raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set.")
    if model.startswith("gemini-2.5-flash"):
      return ChatGoogleGenerativeAI(
          model=model,
          temperature=temp,
          max_tokens=None,
          project=os.getenv("GOOGLE_CLOUD_PROJECT"),
          location=location,
          vertexai=True,
          safety_settings={
              HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                  HarmBlockThreshold.BLOCK_NONE,
          },
          thinking_budget=0,
      )
    elif model.startswith("gemini-3-flash"):
      return ChatGoogleGenerativeAI(
          model=model,
          temperature=temp,
          max_tokens=None,
          project=os.getenv("GOOGLE_CLOUD_PROJECT"),
          location=location,
          vertexai=True,
          safety_settings={
              HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                  HarmBlockThreshold.BLOCK_NONE,
          },
          thinking_level="minimal"
      )
    elif model.startswith("gemini"):
      raise ValueError(f"Unknown model: {model}\nIn Vertex AI Gemini models, We only support 2.5/3 flash (lite) model")
    else:
      client = VertexClient()
      if model.startswith("claude"):
        return client.get_claude(model, location=location, temperature=temp)
      else:
        return client.get_openai_compatible(model, location=location, temperature=temp, max_tokens=None)
  if model.startswith("gpt-"):
    if provider == 'openai':
      return ChatOpenAI(
          api_key=os.environ["OPENAI_API_KEY"],
          model=model,
          max_tokens=None,
          timeout=None,
          temperature=temp,
      )
    elif provider == 'azure':
      return CustomAzureChatOpenAI(
          azure_endpoint=os.environ["AZURE_O3_MINI_ENDPOINT"],
          api_key=os.environ["AZURE_O3_MINI_API_KEY"],  # type: ignore
          azure_deployment=model,
          api_version="2025-02-01-preview",
          max_tokens=None,
          timeout=None,
          temperature=temp,
      )
  elif model.startswith("gemini-"):
    if provider == 'google':
      return ChatGoogleGenerativeAI(
          model=model,
          temperature=temp,
          max_tokens=None,
          thinking_budget=0,
          timeout=None,
          safety_settings={
              HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                  HarmBlockThreshold.BLOCK_NONE,
          },
      )
    else: 
      raise ValueError("Gemini models are only supported with Google or Vertex provider.")


def remove_fields_from_model(
    model: BaseModel,
    exclude_fields: list[str],
    name: str | None = None,
) -> BaseModel:
  new_fields = {
      k: (v.annotation, Field(description=v.description))
      for k, v in model.model_fields.items()
      if k not in exclude_fields
  }
  if name is None:
    name = f"{model.__name__}Without{''.join(exclude_fields)}"
  return create_model(name, **new_fields)


class IterState(TypedDict, total=False):
  iters: int


class MaxIters(AgentMiddleware[IterState, None]):
  name = "max_iters"
  state_schema = IterState  # merged via _resolve_schema

  def __init__(self, max_iters: int = 2):
    self.max_iters = max_iters

  # after each model run, bump a counter and possibly end
  def after_model(self, state, runtime):
    print("iteration number: ", state.get("iters", 0) + 1)
    state["iters"] = state.get("iters", 0) + 1
    if state["iters"] >= self.max_iters:
      state["jump_to"] = "end"
    return state
