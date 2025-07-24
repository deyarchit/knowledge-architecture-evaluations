from litellm import CustomStreamWrapper, completion
from litellm.types.utils import StreamingChoices
from ratelimit import limits, sleep_and_retry  # type: ignore

from evaluator.models.llm import LLMResponse


class LLMAnswerGenerator:
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature

        calls, period = self._get_llm_rate_limits()
        # Dynamically apply the rate limit decorator to the instance's generate method

        self.generate = sleep_and_retry(  # type: ignore [method-assign]
            limits(calls=calls, period=period)(self.generate)
        )

    def _get_llm_rate_limits(self) -> tuple:
        # Determine rate limit parameters based on the model name
        if self.model_name.startswith("gemini"):
            calls, period = 8, 60
        elif self.model_name.startswith("openai") or self.model_name.startswith("gpt"):
            calls, period = 8, 60
        else:
            calls, period = 60, 60
        return (calls, period)

    def generate(self, question: str) -> LLMResponse:
        # The decorator applied in __init__ handles rate limiting automatically.
        # No 'with' block is needed.
        response = completion(
            model=self.model_name,
            response_format=LLMResponse,
            messages=[
                {"content": self.system_prompt, "role": "system"},
                {"content": question, "role": "user"},
            ],
            temperature=self.temperature,
        )

        if isinstance(response, CustomStreamWrapper):
            raise TypeError("Expected Non-Streaming response but got streaming response")

        if isinstance(response.choices[0], StreamingChoices):
            raise TypeError("Expected Non-Streaming response but got streaming response")

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM response content is None")

        return LLMResponse.model_validate_json(content)
