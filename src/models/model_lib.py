"""This file contains functions to create various LLMs using the LangChain library.

=== Important Note ===
We may use Azure services for OpenAI models.
"""
import os
from enum import Enum

from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import (ChatGoogleGenerativeAI, HarmBlockThreshold,
                                    HarmCategory)
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI, ChatOpenAI

_TIME_OUT = None


class ModelName(str, Enum):
  O4_MINI = 'o4-mini'
  GPT_4O_MINI = 'gpt-4o-mini'
  GPT_4O = 'gpt-4o'
  GPT_4_1 = 'gpt-4.1'
  GPT_4_1_MINI = 'gpt-4.1-mini'
  GPT_4_1_NANO = 'gpt-4.1-nano'
  GPT_O4_MINI = 'o4-mini'
  GEMINI_2_0_FLASH = 'gemini-2.0-flash'
  GEMINI_2_0_FLASH_THINKING = 'gemini-2.0-flash-thinking'
  GEMINI_2_5_FLASH_BASE = 'gemini-2.5-flash-base'
  GEMINI_2_5_FLASH_THINKING_8K = 'gemini-2.5-flash-thinking-8k'
  GEMINI_2_5_PRO_BASE = 'gemini-2.5-pro-base'
  DEEPSEEK_V3 = 'deepseek-v3'
  DEEPSEEK_R1 = 'deepseek-r1'
  CLAUDE_3_5_SONNET = 'claude-3.5-sonnet'
  CLAUDE_3_7_SONNET_BASE = 'claude-3.7-sonnet-base'
  CLAUDE_3_7_SONNET_REASON = 'claude-3.7-sonnet-reason'
  LLAMA_3_3_70B = 'llama-3.3-70b'
  LLAMA_4_MAVERICK = 'llama-4-maverick'
  LLAMA_4_SCOUT = 'llama-4-scout'


def create_llm(modelname: str, temperature: float = 0):
  if modelname in ModelName:
    return create_offtheshelf_llm(modelname, temperature)
  elif modelname.startswith("google/gemma-3"):
    # We assume that the base gemma is always a vllm model and use 8020 port.
    return ChatOpenAI(
        base_url="http://localhost:8020/v1",
        model=modelname,
        temperature=temperature,
    )
  elif modelname.startswith("vllm:"):
    # We assume that the ft model is always a vllm model and use 8010 port.
    modelname = modelname[len("vllm:"):]
    return ChatOpenAI(
        base_url="http://localhost:8010/v1",
        model=modelname,
    )
  elif modelname.startswith("ft:gpt"):
    # OpenAI fine-tuned model
    return ChatOpenAI(
        temperature=temperature,
        model=modelname,
    )
  else:
    raise NotImplementedError(f"Unknown model name: {modelname}")


def create_offtheshelf_llm(
    modelname: str,
    temperature: float = 0,
):
  assert modelname in ModelName, f"Unknown model name: {modelname}"
  if modelname.startswith('gpt-4'):
    return ChatOpenAI(
        model=modelname,
        temperature=temperature,
        max_tokens=None,
        timeout=_TIME_OUT,
    )
  elif modelname == 'o4-mini':
    return AzureChatOpenAI(
        azure_deployment="o4-mini",  # or your deployment
        api_version="2025-02-01-preview",
        max_tokens=None,
        timeout=None,
    )
  elif modelname == 'gemini-2.5-flash-base':
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        temperature=temperature,
        thinking_budget=0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                HarmBlockThreshold.BLOCK_NONE,
        },
    )
  elif modelname == 'gemini-2.5-flash-thinking-8k':
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        temperature=temperature,
        thinking_budget=8192,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                HarmBlockThreshold.BLOCK_NONE,
        },
    )
  elif modelname == 'gemini-2.0-flash':
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        temperature=temperature,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                HarmBlockThreshold.BLOCK_NONE,
        },
    )
  elif modelname == 'gemini-2.0-flash-thinking':
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-thinking-exp-01-21",
        temperature=temperature,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                HarmBlockThreshold.BLOCK_NONE,
        },
    )
  elif modelname == 'gemini-2.5-pro-base':
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-preview-05-06",
        temperature=temperature,
        thinking_budget=0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                HarmBlockThreshold.BLOCK_NONE,
        },
    )
  elif modelname == 'deepseek-v3':
    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=temperature,
    )
  elif modelname == 'deepseek-r1':
    # Commented out because the model is unstable.
    return ChatDeepSeek(
        model="deepseek-reasoner",
        temperature=temperature,
    )
  elif modelname == 'claude-3.5-sonnet':
    return ChatAnthropic(
        model_name="claude-3-5-sonnet-20241022",
        temperature=temperature,
        timeout=None,
        stop=None,
    )
  elif modelname == 'claude-3.7-sonnet-reason':
    return ChatAnthropic(
        model_name="claude-3-7-sonnet-20250219",
        temperature=temperature,
        thinking={
            "type": "enabled",
            "budget_tokens": 8192,
        },
        timeout=None,
        stop=None,
    )
  elif modelname == 'claude-3.7-sonnet-base':
    return ChatAnthropic(
        model_name="claude-3-7-sonnet-20250219",
        temperature=temperature,
        thinking={
            "type": "disabled",
        },
        timeout=None,
        stop=None,
    )
  elif modelname == 'llama-3.3-70b':
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
    )
  elif modelname == 'llama-4-maverick':
    return ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=temperature,
    )
  elif modelname == 'llama-4-scout':
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=temperature,
    )
  raise ValueError(f"Unknown model name: {modelname}")
