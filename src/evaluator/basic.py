from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Dict, Optional

from litellm import CustomStreamWrapper, completion
from litellm.types.utils import StreamingChoices
from ratelimit import limits, sleep_and_retry

from evaluator.loader import (
    load_ap_history_qa_set,
    process_ap_history_data,
    read_json_from_file,
    write_json_to_file,
)
from evaluator.models.llm import LLMResponse
from evaluator.models.qa import QA, QACollection, default_qa
from evaluator.utils import get_data_path

eval_dir = Path("evals/basic")


def ap_history_pre_process_data():
    process_ap_history_data()


def ap_history_evaluation(model_name: str, max_questions: Optional[int] = None):
    # Load the processed data
    qa_collection: QACollection = load_ap_history_qa_set()
    qa_set = qa_collection.qa_map

    # Collection to store model outputs
    model_response_set: Dict[int, QA] = defaultdict(default_qa)

    # Get the total number of items evaluated
    if max_questions:
        total_evaluated_questions = max_questions
    else:
        total_evaluated_questions = len(qa_set)

    # Capture responses
    for q_number in islice(qa_set, total_evaluated_questions):
        qa = qa_set[q_number]
        response = generate_answer(model_name, qa.question)
        model_response_set[q_number].answer = response.answer.upper()

    output_file = get_data_path(f"{eval_dir}/{get_normalized_model_name(model_name)}.json")

    model_response_collection: QACollection = QACollection(qa_map=model_response_set)
    if write_json_to_file(output_file, model_response_collection):
        print("Eval completed")


def get_normalized_model_name(model_name: str) -> str:
    return "".join(model_name.split("/")[-1])


def score_model_outputs() -> Dict[str, float]:
    evals = get_data_path(f"{eval_dir}")
    ground_truth: QACollection = load_ap_history_qa_set()
    qa_set = ground_truth.qa_map

    scores: Dict[str, float] = {}

    for model_file in evals.glob("*.json"):
        model_name = model_file.stem
        model_collection = read_json_from_file(model_file, QACollection)
        if not model_collection:
            print(f"Skipping {model_file} due to load error.")
            continue

        model_qa_map = model_collection.qa_map

        correct = 0
        total = 0

        for q_number, expected_qa in qa_set.items():
            if q_number not in model_qa_map:
                continue

            predicted_qa = model_qa_map[q_number]

            # Exact match scoring, case-insensitive
            if predicted_qa.answer.strip().lower() == expected_qa.answer.strip().lower():
                correct += 1

            total += 1

        accuracy = correct / total if total else 0.0
        scores[model_name] = accuracy
        print(f"{model_name}: {correct}/{total} correct ({accuracy:.2%})")

    return scores


@sleep_and_retry
@limits(calls=8, period=60)
def generate_answer(model_name: str, question: str) -> LLMResponse:
    response = completion(
        model=model_name,
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
