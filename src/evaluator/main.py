import evaluator.data as data
from evaluator.evals import basic


def pre_process_data():
    # process_ap_history_data()
    data.process_ap_history_solution_guide()


def main():
    print("Running knowledge evaluations...")
    model_list = [
        "ollama/granite3.3:2b",
        "ollama/qwen3:4b",
        "ollama/gemma3:1b",
        "ollama/gemma3:4b",
        "ollama/qwen3:1.7b",
    ]

    basic_evaluator = basic.BasicEval(models=model_list)
    basic_evaluator.run_eval()


if __name__ == "__main__":
    main()
