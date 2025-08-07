from src.models import model_lib

from pydantic import BaseModel, Field


class ResponseFormatter(BaseModel):
  """Always use this tool to structure your response to the user."""
  answer: str = Field(description="The answer to the user's question")
  followup_question: str = Field(
      description="A followup question the user could ask")


llm = model_lib.create_llm("gpt-4.1-mini", temperature=0)
prompt = "What is your name? Do you know what much is 23* 12 +3.22?"
res = llm.invoke(prompt)
# res_structured = llm_structured.invoke(prompt)
