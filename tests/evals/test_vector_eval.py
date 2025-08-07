import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluator.evals import vector_rag
from evaluator.models.qa import QACollection, QA
from evaluator.models.llm import LLMResponse


def mock_write_json_to_file(file_path, data):
    return True


@pytest.fixture
def sample_qa_collection():
    return QACollection(
        qa_map={
            1: QA(
                question="Q1",
                answer="A",
            ),
            2: QA(
                question="Q2",
                answer="B",
            ),
            3: QA(
                question="Q3",
                answer="C",
            ),
        }
    )


@pytest.fixture
def mock_llm_response():
    """A mock LLMResponse that acts like a real string for testing."""
    mock_response = Mock(spec=LLMResponse)
    mock_response.answer = "A"
    return mock_response


@pytest.fixture
def mock_answer_generator(mock_llm_response):
    """A mock AnswerGenerator that returns a predefined mock response."""
    mock_gen = Mock(spec=vector_rag.AnswerGenerator)
    mock_gen.generate.return_value = mock_llm_response
    return mock_gen


@pytest.fixture
def mock_context_retriever():
    """A mock ContextRetriever with a predefined list of mock documents."""
    mock_retriever = Mock(spec=vector_rag.ContextRetriever)
    mock_retriever.query.return_value = [
        "mock_doc_1",
        "mock_doc_2",
        "mock_doc_3",
    ]
    return mock_retriever


@pytest.fixture
def mock_answer_generator_factory(mock_answer_generator):
    """A mock factory that returns the mock AnswerGenerator."""
    return Mock(
        spec=vector_rag.AnswerGeneratorFactory, return_value=mock_answer_generator
    )


@pytest.fixture
def mock_context_retriever_factory(mock_context_retriever):
    """A mock factory that returns the mock ContextRetriever."""
    return Mock(
        spec=vector_rag.ContextRetrieverFactory, return_value=mock_context_retriever
    )


# Patch the external functions to avoid real file I/O and imports
@patch("evaluator.evals.vector_rag.write_json_to_file", new=mock_write_json_to_file)
class TestVectorEval:
    def test_generate_answers_writes_to_file(
        self,
        sample_qa_collection,
        mock_answer_generator,
        mock_answer_generator_factory,
        mock_context_retriever_factory,
    ):
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # Pass the factories to the VectorRAGEval class
            evaluator = vector_rag.VectorRAGEval(
                models=["test_model"],
                qa_collection=sample_qa_collection,
                answer_generator=mock_answer_generator_factory,
                vector_search_factory=mock_context_retriever_factory,
                strategy=vector_rag.strategy_baseline,
                output_dir=output_dir,
            )

            with patch(
                "evaluator.evals.vector_rag.write_json_to_file", return_value=True
            ) as mock_write:
                evaluator._generate_answers("test_model")

                # Assert that the generator was called for each question
                assert mock_answer_generator.generate.call_count == len(
                    sample_qa_collection.qa_map
                )
                mock_answer_generator.generate.assert_any_call(
                    "Q1",
                    [
                        "mock_doc_1",
                        "mock_doc_2",
                        "mock_doc_3",
                    ],
                )
                mock_answer_generator.generate.assert_any_call(
                    "Q2",
                    [
                        "mock_doc_1",
                        "mock_doc_2",
                        "mock_doc_3",
                    ],
                )
                mock_answer_generator.generate.assert_any_call(
                    "Q3",
                    [
                        "mock_doc_1",
                        "mock_doc_2",
                        "mock_doc_3",
                    ],
                )

                # Assert that the write function was called once with the correct path
                expected_output_path = (
                    output_dir
                    / "vector_rag"
                    / vector_rag.strategy_baseline.name
                    / "test_model.json"
                )
                mock_write.assert_called_once_with(
                    expected_output_path,
                    QACollection(
                        qa_map={
                            1: QA(
                                question="",
                                answer="A",
                            ),
                            2: QA(
                                question="",
                                answer="A",
                            ),
                            3: QA(
                                question="",
                                answer="A",
                            ),
                        }
                    ),
                )

    def test_run_eval_calls_generate_answers_for_each_model(
        self,
        sample_qa_collection,
        mock_answer_generator_factory,
        mock_context_retriever_factory,
    ):
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            evaluator = vector_rag.VectorRAGEval(
                models=["model_a", "model_b"],
                qa_collection=sample_qa_collection,
                answer_generator=mock_answer_generator_factory,
                vector_search_factory=mock_context_retriever_factory,
                strategy=vector_rag.strategy_baseline,
                output_dir=output_dir,
            )

            with patch.object(evaluator, "_generate_answers") as mock_generate_answers:
                evaluator.run_eval()

                # Assert that the method was called for each model
                assert mock_generate_answers.call_count == 2
                mock_generate_answers.assert_any_call("model_a")
                mock_generate_answers.assert_any_call("model_b")
