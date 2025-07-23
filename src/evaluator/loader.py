import re
from pathlib import Path
from typing import Dict, Optional, Type, TypeVar

from pydantic import BaseModel
from pypdf import PdfReader

from evaluator.models.qa import QA, QACollection
from evaluator.utils import get_data_path

output_json = get_data_path("processed/ap_history_qa.json")


def load_ap_history_qa_set() -> QACollection:
    output_file = Path(output_json)
    existing_qa = QACollection.model_validate_json(output_file.read_text(encoding="utf-8"))
    return existing_qa


def process_ap_history_data():
    input_pdfs = [
        "data/raw/MC_1450-1750.pdf",
        "data/raw/MC_600-1450.pdf",
        "data/raw/MC_periods.pdf",
    ]
    for pdf in input_pdfs:
        process_raw_data(pdf, str(output_json))


def process_raw_data(pdf_path: str, output_json_path: str):
    output_file = Path(output_json_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing QA set (if any) and calculate index offset
    offset: int = 0
    existing_qa_set: Dict[int, QA] = {}
    existing_qa_collection: QACollection | None = read_json_from_file(output_file, QACollection)

    if existing_qa_collection:
        existing_qa_set = existing_qa_collection.qa_map

    offset = max(existing_qa_set.keys(), default=-1) + 1

    # Parse new PDF and offset question numbers during parsing
    reader = PdfReader(pdf_path)
    qa_set: Dict[int, QA] = {}

    q_idx = offset
    for idx, page in enumerate(reader.pages[1:]):
        text = page.extract_text()

        if idx % 2 == 0:
            qa_set[q_idx] = QA(question=_cleanup_text(text), answer="")
        else:
            qa_set[q_idx].answer = _extract_answer_option(text)
            q_idx += 1

    #  Merge and dump
    merged_qa_set = {**existing_qa_set, **qa_set}
    final_qa_set = QACollection(qa_map=merged_qa_set)

    write_json_to_file(output_file, final_qa_set)


T = TypeVar("T", bound=BaseModel)


def read_json_from_file(file: Path, model_cls: Type[T]) -> Optional[T]:
    if not file.exists():
        print(f"File does not exist: {file}")
        return None

    try:
        data = file.read_text(encoding="utf-8")
        return model_cls.model_validate_json(data)
    except Exception as e:
        print(f"Error loading content from {file}: {e}")
        return None


def write_json_to_file(output_file: Path, model_instance: BaseModel) -> bool:
    try:
        output_file.write_text(
            model_instance.model_dump_json(indent=2, exclude_none=True), encoding="utf-8"
        )
        print(f"Successfully saved content to {output_file}")
        return True
    except OSError as e:
        print(f"Error saving content to {output_file}: {e}")
        return False


def _cleanup_text(text: str) -> str:
    # Replace common explicit newlines/tabs with space
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Replace ALL sequences of whitespace characters
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace from the page
    text = text.strip()
    return text


def _extract_answer_option(text: str) -> str:
    pattern = r"Answer:\s*([a-zA-Z])\s*"

    match = re.search(pattern, text)

    if match:
        # Group 1 (index 1) refers to the content within the first parentheses in the pattern
        return match.group(1)
    else:
        return ""
