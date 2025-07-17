from typing import List

from pydantic import BaseModel, Field


class QuestionAnswer(BaseModel):
    question: str = Field(..., description="The text of the question")
    answer: str = Field(..., description="The answer to the question")


class QuestionAnswerSet(BaseModel):
    qa_set: List[QuestionAnswer]
