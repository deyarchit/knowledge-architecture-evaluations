from pathlib import Path
from typing import Dict

from evaluator.data import read_json_from_file
from evaluator.models.qa import QACollection


def score_model_outputs(ground_truth: QACollection, eval_path: Path) -> Dict[str, float]:
    qa_set = ground_truth.qa_map

    scores: Dict[str, float] = {}

    for model_file in eval_path.glob("*.json"):
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
