from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    answer: str = Field(..., description="The answer to the question")
