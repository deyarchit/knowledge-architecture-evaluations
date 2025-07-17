from litellm import completion

from data.loader import load_ap_history_qa_set
from models.llm import LLMResponse
from models.qa import QASet


def ap_history_evaluation():
    # This is only needed once
    # process_ap_history_data()

    # Load the processed data
    qa_set: QASet = load_ap_history_qa_set()

    # Store the subset of items you are actually evaluating
    evaluated_qa_items = qa_set.qa_set

    # Get the total number of items evaluated
    total_evaluated_questions = len(evaluated_qa_items)
    incorrect_answers = 0

    for qa in evaluated_qa_items:
        response = generate_answer(qa.question)
        if response.answer.upper() != qa.answer.upper():
            print(
                f"Incorrect answer: LLM Response: {response.answer.upper()} Actual Response: {qa.answer.upper()}"
            )
            incorrect_answers += 1

    pass_percentage = (total_evaluated_questions - incorrect_answers) / total_evaluated_questions
    print(f"Total Questions: {total_evaluated_questions} Pass Percentage: {pass_percentage:.2%}")


def generate_answer(question: str) -> LLMResponse:
    response = completion(
        # model="gemini/gemini-2.0-flash",
        model="ollama/gemma3:1b",
        response_format=LLMResponse,
        messages=[
            {"content": system_prompt, "role": "system"},
            {"content": question, "role": "user"},
        ],
        temperature=0.0,
    )

    parsed_response: LLMResponse = LLMResponse.model_validate_json(
        response.choices[0].message.content
    )
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
    """
