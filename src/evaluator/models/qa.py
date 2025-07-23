from typing import List

from pydantic import BaseModel, Field


class QA(BaseModel):
    question: str = Field(..., description="The text of the question")
    answer: str = Field(..., description="The answer to the question")


class QASet(BaseModel):
    qa_set: List[QA]
