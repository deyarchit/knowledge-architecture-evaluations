from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Dict, Optional

from evaluator.llm import LLMAnswerGenerator
from evaluator.loader import (
    load_ap_history_qa_set,
    process_ap_history_data,
    read_json_from_file,
    write_json_to_file,
)
from evaluator.models.qa import QA, QACollection, default_qa
from evaluator.utils import get_data_path

eval_dir = Path("evals/basic")


def ap_history_pre_process_data():
    process_ap_history_data()


def ap_history_evaluation(model_name: str, max_questions: Optional[int] = None):
    model_name_str = get_normalized_model_name(model_name)
    print(f"Starting evaluation for model: {model_name_str}")
    # Load the processed data
    qa_collection: QACollection = load_ap_history_qa_set()
    qa_set = qa_collection.qa_map

    # Init llm
    gen = LLMAnswerGenerator(model_name, system_prompt)

    # Path for output file
    output_file = get_data_path(f"{eval_dir}/{model_name_str}.json")

    model_response_set: Dict[int, QA] = defaultdict(default_qa)
    # Load existing responses if any
    if output_file.exists():
        existing_collection = read_json_from_file(output_file, QACollection)
        if existing_collection:
            model_response_set = existing_collection.qa_map

    # Determine total number of questions to evaluate
    total_evaluated_questions = max_questions or len(qa_set)

    # Capture responses for unanswered questions
    for q_number in islice(qa_set, total_evaluated_questions):
        if (
            q_number in model_response_set and model_response_set[q_number].answer != ""
        ):  # Skip if already answered
            continue

        qa = qa_set[q_number]
        response = gen.generate(qa.question)
        model_response_set[q_number] = QA(question="", answer=response.answer.upper())

    # Save updated response set
    model_response_collection = QACollection(qa_map=model_response_set)
    if write_json_to_file(output_file, model_response_collection):
        print(f"Eval completed for model: {model_name_str}")


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
