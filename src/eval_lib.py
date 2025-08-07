from pydantic import BaseModel
from pydantic import Field

from src import prompts
from src.models import model_lib


def clip_value(value, min_value, max_value):
  return max(min_value, min(value, max_value))


class EvaluationResult(BaseModel):
  num_requests: int = Field(
      ..., description="The number of requests the user requested.")
  reason: str = Field(
      ...,
      description="reason the scoring logic and analyze how to score.",
  )
  response_scores: list[int] = Field(
      ...,
      description="The scores of each response. From 0 to 100.",
  )
  final_score: int = Field(
      ...,
      description=
      "The final score of the AI response, which is an averaged value of the response_score in each aspect. From 0 - 100.",
  )


def create_llm_judge(model_name: str):
  print("judge model:", model_name)
  llm = model_lib.create_offtheshelf_llm(
      modelname=model_name,
      temperature=0,
  )
  return prompts.JUDGE | llm.with_structured_output(
      EvaluationResult,
      strict=True,
  )
