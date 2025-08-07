from pathlib import Path

import pytest
from pydantic import BaseModel

from evaluator.data.file_io import read_json_from_file, write_json_to_file


class SampleModel(BaseModel):
    name: str
    value: int


@pytest.mark.parametrize(
    "model_instance,expected_content",
    [
        (SampleModel(name="foo", value=1), '"name": "foo"'),
        (SampleModel(name="bar", value=99), '"value": 99'),
    ],
)
def test_write_json_to_file_success(tmp_path: Path, model_instance, expected_content):
    output_file = tmp_path / "data.json"

    success = write_json_to_file(output_file, model_instance)

    assert success is True
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert expected_content in content


def test_write_json_to_file_failure(monkeypatch, tmp_path: Path):
    model = SampleModel(name="fail", value=0)
    output_file = tmp_path / "data.json"

    def fail_write_text(*args, **kwargs):
        raise OSError("simulated failure")

    monkeypatch.setattr(Path, "write_text", fail_write_text)

    success = write_json_to_file(output_file, model)
    assert success is False


@pytest.mark.parametrize(
    "json_content,expected_result",
    [
        ('{"name": "foo", "value": 1}', SampleModel(name="foo", value=1)),
        ('{"name": "bar", "value": 42}', SampleModel(name="bar", value=42)),
    ],
)
def test_read_json_from_file_valid(tmp_path: Path, json_content, expected_result):
    file = tmp_path / "input.json"
    file.write_text(json_content, encoding="utf-8")

    result = read_json_from_file(file, SampleModel)

    assert isinstance(result, SampleModel)
    assert result == expected_result


def test_read_json_from_file_missing_file(tmp_path: Path):
    file = tmp_path / "notfound.json"
    result = read_json_from_file(file, SampleModel)
    assert result is None


@pytest.mark.parametrize(
    "bad_json",
    [
        '{"name": "incomplete"',
        '{"name": 123, "value": "wrong_type"}',
        "not even json",
    ],
)
def test_read_json_from_file_invalid_json(tmp_path: Path, bad_json):
    file = tmp_path / "bad.json"
    file.write_text(bad_json, encoding="utf-8")

    result = read_json_from_file(file, SampleModel)
    assert result is None
