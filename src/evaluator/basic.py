from litellm import CustomStreamWrapper, completion
from litellm.types.utils import StreamingChoices
from ratelimit import limits, sleep_and_retry

from evaluator.loader import load_ap_history_qa_set
from evaluator.models.llm import LLMResponse
from evaluator.models.qa import QASet


def ap_history_evaluation():
    # This is only needed once
    # process_ap_history_data()

    # Load the processed data
    qa_set: QASet = load_ap_history_qa_set()

    # Store the subset of items you are actually evaluating
    evaluated_qa_items = qa_set.qa_set

    # Get the total number of items evaluated
    total_evaluated_questions = len(evaluated_qa_items)
    incorrect_answers = 0

    for qa in evaluated_qa_items:
        response = generate_answer(qa.question)
        if response.answer.upper() != qa.answer.upper():
            print(
                f"Incorrect answer: LLM Response: {response.answer.upper()} Correct Response: {qa.answer.upper()}"
            )
            incorrect_answers += 1

    pass_percentage = (total_evaluated_questions - incorrect_answers) / total_evaluated_questions
    print(f"Total Questions: {total_evaluated_questions} Pass Percentage: {pass_percentage:.2%}")


@sleep_and_retry
@limits(calls=8, period=60)
def generate_answer(question: str) -> LLMResponse:
    response = completion(
        # model="gemini/gemini-2.5-flash",
        model="ollama/granite3.3:2b",
        response_format=LLMResponse,
        messages=[
            {"content": system_prompt, "role": "system"},
            {"content": question, "role": "user"},
        ],
        temperature=0.0,
    )

    if isinstance(response, CustomStreamWrapper):
        raise TypeError("Expected Non-Streaming response but got streaming response")

    if isinstance(response.choices[0], StreamingChoices):
        raise TypeError("Expected Non-Streaming response but got streaming response")

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM response content is None")

    parsed_response: LLMResponse = LLMResponse.model_validate_json(content)
    return parsed_response


system_prompt = """
    You are an expert in multiple-choice questions. For each question provided, respond with only the letter corresponding to the correct answer.
    Do not include any additional text, explanations, or the content of the answer option itself.

    Example Input Format:
    Question: What is the capital of France?
    A) Berlin
    B) Madrid
    C) Paris
    D) Rome

    Example Output Format:
    C

    Your Task:
    Answer the following multiple-choice questions by providing only the letter of the correct option.
    /no_think
    """
