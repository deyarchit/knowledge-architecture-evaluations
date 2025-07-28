from dotenv import load_dotenv

import evaluator.data.file_io as data
from evaluator.evals import basic, vector_rag

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

    basic_evaluator = basic.BasicEval(models=model_list)
    basic_evaluator.run_eval()

    vector_rag_evaluator = vector_rag.VectorRAGEval(
        models=model_list, strategy=vector_rag.strategy_baseline
    )
    vector_rag_evaluator.run_eval()

    vector_rag_evaluator = vector_rag.VectorRAGEval(
        models=model_list, strategy=vector_rag.strategy_with_reranking
    )
    vector_rag_evaluator.run_eval()

    vector_rag_evaluator = vector_rag.VectorRAGEval(
        models=model_list,
        strategy=vector_rag.strategy_with_reranking_with_basic_chunking,
    )
    vector_rag_evaluator.run_eval()


if __name__ == "__main__":
    main()
