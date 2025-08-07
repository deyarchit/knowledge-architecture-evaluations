import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluator.evals import basic
from evaluator.models.qa import QACollection, QA


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
def mock_answer_generator():
    mock_gen = Mock()
    mock_gen.generate.side_effect = lambda q: Mock(answer="A")
    return mock_gen


@pytest.fixture
def mock_answer_generator_factory(mock_answer_generator):
    return Mock(return_value=mock_answer_generator)


# Patch the external functions to avoid real file I/O and imports
@patch("evaluator.evals.basic.write_json_to_file", new=mock_write_json_to_file)
class TestBasicEval:
    def test_generate_answers_writes_to_file(
        self, sample_qa_collection, mock_answer_generator, mock_answer_generator_factory
    ):
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            evaluator = basic.BasicEval(
                models=["test_model"],
                qa_collection=sample_qa_collection,
                answer_generator=mock_answer_generator_factory,
                output_dir=output_dir,
            )

            with patch(
                "evaluator.evals.basic.write_json_to_file", return_value=True
            ) as mock_write:
                evaluator._generate_answers("test_model")

                # Assert that the generator was called for each question
                assert mock_answer_generator.generate.call_count == len(
                    sample_qa_collection.qa_map
                )
                mock_answer_generator.generate.assert_any_call("Q1")
                mock_answer_generator.generate.assert_any_call("Q2")
                mock_answer_generator.generate.assert_any_call("Q3")

                # Assert that the write function was called once with the correct path
                expected_output_path = output_dir / "basic" / "test_model.json"
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
        self, sample_qa_collection, mock_answer_generator_factory
    ):
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            evaluator = basic.BasicEval(
                models=["model_a", "model_b"],
                qa_collection=sample_qa_collection,
                answer_generator=mock_answer_generator_factory,
                output_dir=output_dir,
            )

            with patch.object(evaluator, "_generate_answers") as mock_generate_answers:
                evaluator.run_eval()

                # Assert that the method was called for each model
                assert mock_generate_answers.call_count == 2
                mock_generate_answers.assert_any_call("model_a")
                mock_generate_answers.assert_any_call("model_b")
