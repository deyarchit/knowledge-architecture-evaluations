from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional

from rich.progress import track

from evaluator.data.file_io import (
    read_json_from_file,
    write_json_to_file,
)
from evaluator.evals.scoring import score_model_outputs
from evaluator.models.qa import QA, QACollection, default_qa
from evaluator.utils import get_normalized_model_name


class BasicEval:
    def __init__(
        self,
        models: List[str],
        qa_collection: QACollection,
        answer_generator,
        output_dir: Path,
        max_questions: Optional[int] = None,
    ) -> None:
        self.models = models

        # Load processed data
        self._qa_collection = qa_collection
        self._qa_set: Dict[int, QA] = qa_collection.qa_map

        # Determine total number of questions to evaluate
        self.max_questions: int = max_questions or len(qa_collection.qa_map)

        # Dir where the eval outputs will be stored
        self._eval_dir = Path(f"{output_dir}/basic")

        self._answer_generator = answer_generator

        # Configure prompt for this evaluation
        self._system_prompt = """
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

    def run_eval(self):
        print(f"Running {self.__class__.__name__}")
        for model_name in self.models:
            self._generate_answers(model_name)

        self._score()

    def _generate_answers(self, model_name: str):
        model_name_str = get_normalized_model_name(model_name)
        print(f"{model_name_str}: Starting evaluation")

        # Path for output file
        output_file = Path(f"{self._eval_dir}/{model_name_str}.json")

        # Init llm
        gen = self._answer_generator(model_name, self._system_prompt)

        model_response_set: Dict[int, QA] = defaultdict(default_qa)
        # Load existing responses if any
        if output_file.exists():
            existing_collection = read_json_from_file(output_file, QACollection)
            if existing_collection:
                model_response_set = existing_collection.qa_map
                print(f"{model_name_str}: Loaded the already existing output")

        questions_to_answer = [
            q_number for q_number in islice(self._qa_set, self.max_questions)
        ]

        for q_number in track(
            questions_to_answer, description=f"{model_name_str}: Generating answers"
        ):
            if (
                q_number in model_response_set
                and model_response_set[q_number].answer != ""
            ):  # Skip if already answered
                continue

            qa = self._qa_set[q_number]
            response = gen.generate(qa.question)
            model_response_set[q_number] = QA(
                question="", answer=response.answer.upper()
            )

        # Save updated response set
        model_response_collection = QACollection(qa_map=model_response_set)
        if write_json_to_file(output_file, model_response_collection):
            print(f"{model_name_str}: Eval completed")

    def _score(self):
        ground_truth: QACollection = self._qa_collection
        evals = self._eval_dir
        score_model_outputs(ground_truth, evals)
