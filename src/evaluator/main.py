from evaluator.basic import ap_history_evaluation, score_model_outputs


def main():
    print("Running knowledge evaluations...")
    ap_history_evaluation("ollama/granite3.3:2b", 5)
    score_model_outputs()


if __name__ == "__main__":
    main()
