import os
import gin
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from typing import Union, Literal
from pydantic import BaseModel, Field, create_model

__all__ = [
    "CustomChatOpenAI",
    "CustomAzureChatOpenAI",
    "create_llm",
    "remove_fields_from_model",
]


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
def create_llm(
    temp: float,
    provider: Literal['openai', 'azure'],
    model: str = "gpt-4.1-mini",
) -> ChatOpenAI:
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
