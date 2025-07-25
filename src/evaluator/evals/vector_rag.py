from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional

from evaluator.data.file_io import (
    load_ap_history_qa_set,
    read_json_from_file,
    write_json_to_file,
)
from evaluator.data.vector_search import VectorSearch
from evaluator.evals.scoring import score_model_outputs
from evaluator.llm import LLMAnswerGenerator
from evaluator.models.qa import QA, QACollection, default_qa
from evaluator.utils import get_data_path, get_normalized_model_name


@dataclass
class Strategy:
    name: str
    description: str
    max_results: int = 3
    enable_reranking: bool = False


class VectorRAGEval:
    def __init__(
        self, models: List[str], strategy: str, max_questions: Optional[int] = None
    ) -> None:
        self.models = models

        # Load processed data
        qa_collection: QACollection = load_ap_history_qa_set()
        self._qa_set: Dict[int, QA] = qa_collection.qa_map

        # Determine total number of questions to evaluate
        self.max_questions: int = max_questions or len(qa_collection.qa_map)

        # Load strategy
        strategy_details = self._get_strategy_details(strategy)
        if not strategy_details:
            raise RuntimeError(f"Invalid strategy: {strategy}")

        # Dir where the eval outputs will be stored
        self._eval_dir = Path(f"evals/vector_rag/{strategy_details.name}")

        # Configure prompt for this evaluation
        self._system_prompt = """
            You are an expert in multiple-choice questions. For each question provided, consider the accompanying context as primary information. If the context does not directly provide the answer, you may use your general knowledge to select the correct option. Respond with only the letter corresponding to the correct answer.

            Do not include any additional text, explanations, or the content of the answer option itself.

            Example Input Format:
            Context: ["Paris is the capital and most populous city of France, with an estimated population of 2,141,000 residents in 2020."]
            Question: What is the capital of France?
            A) Berlin
            B) Madrid
            C) Paris
            D) Rome

            Example Output Format:
            C

            Your Task:
            Answer the following multiple-choice questions by providing only the letter of the correct option, prioritizing information from the provided context, but supplementing with your general knowledge if necessary.
            /no_think
            """

        self.vector_search = VectorSearch(max_results=strategy_details.max_results)

    def _get_strategy_details(self, strategy_name: str) -> Optional[Strategy]:
        strategies: Dict[str, Strategy] = {
            "strategy_baseline": Strategy(
                name="strategy_baseline", description="", max_results=3, enable_reranking=False
            )
        }
        if strategy_name not in strategies:
            return None

        return strategies[strategy_name]

    def run_eval(self):
        print(f"Running {self.__class__.__name__}")
        for model_name in self.models:
            self._generate_answers(model_name)

        self._score()

    def _generate_answers(self, model_name: str):
        model_name_str = get_normalized_model_name(model_name)
        print(f"{model_name_str}: Starting evaluation")

        # Path for output file
        output_file = get_data_path(f"{self._eval_dir}/{model_name_str}.json")

        # Init llm
        gen = LLMAnswerGenerator(model_name, self._system_prompt)

        model_response_set: Dict[int, QA] = defaultdict(default_qa)
        # Load existing responses if any
        if output_file.exists():
            existing_collection = read_json_from_file(output_file, QACollection)
            if existing_collection:
                model_response_set = existing_collection.qa_map
                print(f"{model_name_str}: Loaded the already existing output")

        # Capture responses for unanswered questions
        for q_number in islice(self._qa_set, self.max_questions):
            if (
                q_number in model_response_set and model_response_set[q_number].answer != ""
            ):  # Skip if already answered
                continue

            qa = self._qa_set[q_number]
            context_docs = self.vector_search.query(qa.question)
            response = gen.generate(qa.question, context_docs)
            model_response_set[q_number] = QA(question="", answer=response.answer.upper())

        # Save updated response set
        model_response_collection = QACollection(qa_map=model_response_set)
        if write_json_to_file(output_file, model_response_collection):
            print(f"{model_name_str}: Eval completed")

    def _score(self):
        evals = get_data_path(f"{self._eval_dir}")
        ground_truth: QACollection = load_ap_history_qa_set()
        score_model_outputs(ground_truth, evals)
