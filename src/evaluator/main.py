from evaluator.basic import ap_history_evaluation, score_model_outputs


def main():
    print("Running knowledge evaluations...")
    model_list = [
        "ollama/granite3.3:2b",
        "ollama/qwen3:4b",
        "ollama/gemma3:1b",
        "ollama/gemma3:4b",
        "ollama/qwen3:1.7b",
    ]
    for model in model_list:
        ap_history_evaluation(model)

    score_model_outputs()


if __name__ == "__main__":
    main()
