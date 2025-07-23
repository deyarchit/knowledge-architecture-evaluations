from typing import Dict

from pydantic import BaseModel, Field


class QA(BaseModel):
    question: str = Field(..., description="The text of the question")
    answer: str = Field(..., description="The answer to the question")


class QACollection(BaseModel):
    qa_map: Dict[int, QA]
