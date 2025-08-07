from dotenv import load_dotenv

import evaluator.data.file_io as data
from evaluator.evals import basic, vector_rag

from evaluator.data.file_io import (
    load_ap_history_qa_set,
)

from evaluator.utils import get_data_path
from evaluator.llm import LLMAnswerGenerator
from evaluator.models.qa import QACollection
from evaluator.data.vector_search import SearchConfiguration, VectorSearch

load_dotenv()


def pre_process_data():
    data.process_ap_history_data()
    data.process_ap_history_solution_guide()


def main():
    print("Running knowledge evaluations...")
    model_list = [
        "ollama/granite3.3:2b",
        "ollama/phi4-mini:3.8b",
        "ollama/qwen3:4b",
        "ollama/gemma3:1b",
        "ollama/gemma3:4b",
        "ollama/qwen3:1.7b",
    ]

    def llm_answer_generator(model_name, system_prompt):
        return LLMAnswerGenerator(model_name, system_prompt)

    evals_root = get_data_path("evals")

    qa_collection: QACollection = load_ap_history_qa_set()

    basic_evaluator = basic.BasicEval(
        models=model_list,
        qa_collection=qa_collection,
        answer_generator=llm_answer_generator,
        output_dir=evals_root,
    )
    basic_evaluator.run_eval()

    def vector_search_factory(config: SearchConfiguration):
        return VectorSearch(config)

    vector_rag_evaluator = vector_rag.VectorRAGEval(
        models=model_list,
        qa_collection=qa_collection,
        answer_generator=llm_answer_generator,
        output_dir=evals_root,
        vector_search_factory=vector_search_factory,
        strategy=vector_rag.strategy_baseline,
    )
    vector_rag_evaluator.run_eval()

    vector_rag_evaluator = vector_rag.VectorRAGEval(
        models=model_list,
        qa_collection=qa_collection,
        answer_generator=llm_answer_generator,
        output_dir=evals_root,
        vector_search_factory=vector_search_factory,
        strategy=vector_rag.strategy_with_reranking,
    )
    vector_rag_evaluator.run_eval()

    vector_rag_evaluator = vector_rag.VectorRAGEval(
        models=model_list,
        qa_collection=qa_collection,
        answer_generator=llm_answer_generator,
        output_dir=evals_root,
        vector_search_factory=vector_search_factory,
        strategy=vector_rag.strategy_with_reranking_with_basic_chunking,
    )
    vector_rag_evaluator.run_eval()


if __name__ == "__main__":
    main()
