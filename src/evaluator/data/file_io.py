import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Type, TypeVar

from pydantic import BaseModel
from pypdf import PdfReader
from unstructured.chunking.basic import chunk_elements
from unstructured.partition.pdf import partition_pdf

from evaluator.models.qa import QA, Concepts, QACollection
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


def process_ap_history_solution_guide():
    pdf_file = get_data_path("raw/ap_history_guide.pdf")
    chunks = _use_unstructured(pdf_file)

    concept_chunks: Dict[str, str] = defaultdict(str)
    for i, chunk in enumerate(chunks):
        concept_chunks[str(i)] = chunk

    concepts: Concepts = Concepts(chunks=concept_chunks)
    output_file = get_data_path("processed/ap_history_concepts.json")
    write_json_to_file(output_file, concepts)


def _use_pypdf(pdf_file: Path):
    reader = PdfReader(pdf_file)
    pages = []

    for page in reader.pages:
        pages.append(normalize_text(page.extract_text()))

    return pages


def _use_unstructured(pdf_file: Path):
    elements = partition_pdf(
        filename=str(pdf_file),
        strategy="fast",
        infer_table_structure=False,
        include_metadata=False,
    )

    elements_cleaned = []
    for element in elements:
        if hasattr(element, "text"):
            original_text = element.text
            cleaned_text = clean_cid_chars(original_text)

            element.text = cleaned_text

        elements_cleaned.append(element)

    # chunks = chunk_by_title(
    #     elements_cleaned, combine_text_under_n_chars=50, max_characters=1000, new_after_n_chars=800
    # )

    chunks = chunk_elements(
        elements_cleaned,
        max_characters=1000,
        new_after_n_chars=800,  # soft-max
        overlap=100,  # helps retain semantic meaning from surrounding sentences
    )

    return (chunk.text for chunk in chunks)


def process_raw_data(pdf_path: str, output_json_path: str):
    output_file = Path(output_json_path)

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
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_file.write_text(
            model_instance.model_dump_json(indent=2, exclude_none=True), encoding="utf-8"
        )
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


def normalize_text(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def clean_cid_chars(text: str) -> str:
    """
    Cleans common (cid:X) characters and broken ligatures from text.
    """

    cid_replacements = {
        "(cid:35)": "ff",
        "(cid:69)": "f",
        "(cid:44)": "fi",
        "(cid:48)": "fi",
        "(cid:49)": "fi",
        "(cid:59)": "fi",
        # Add other common ones if encountered:
        # "(cid:80)": "i",
        # "(cid:81)": "l",
        # "(cid:82)": "t",
        # "(cid:83)": "s",
        # "(cid:84)": "r",
        # "(cid:85)": "e",
    }

    for cid_pattern, replacement_char in cid_replacements.items():
        text = text.replace(cid_pattern, replacement_char)

    # 2. Handle common broken ligatures (ff, fi, fl, ffi, ffl)
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬀ", "ff")
    text = text.replace("ﬂ", "fl")
    text = text.replace("ﬃ", "ffi")
    text = text.replace("ﬄ", "ffl")

    # 3. Remove any remaining (cid:X) patterns (fallback for unknown CIDs)
    # This regex looks for (cid: followed by one or more digits and a closing parenthesis
    text = re.sub(r"\(cid:\d+\)", "", text)

    # Optional: Clean up extra spaces that might result from removals
    text = re.sub(r"\s+", " ", text).strip()

    return text
