from typing import Dict, List, Optional

import chromadb

from evaluator.data.file_io import (
    read_json_from_file,
)
from evaluator.models.qa import Concepts
from evaluator.utils import get_data_path


class VectorSearch:
    def __init__(self, max_results: int = 3) -> None:
        client = chromadb.Client()
        self.collection = client.get_or_create_collection(name="ap_history_concepts")

        chunk_file = get_data_path("processed/ap_history_concepts.json")
        concepts: Concepts | None = read_json_from_file(chunk_file, Concepts)
        if not concepts:
            raise RuntimeError(f"No concepts found in file: {chunk_file}")
        self._build_vector_search_engine(concepts.chunks)

        self.max_results: int = max_results

    def _build_vector_search_engine(self, chunks: Dict[str, str]):
        ids: list = []
        documents: list = []
        for concept_id, concept in chunks.items():
            documents.append(concept)
            ids.append(concept_id)

        self.collection.upsert(documents=documents, ids=ids)

    def query(self, query: str) -> Optional[List[str]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=self.max_results,
            include=["documents", "distances", "metadatas"],
        )
        selected_results: List[str] = []
        if not results["documents"]:
            return None

        for doc in results["documents"][0]:
            selected_results.append(doc)
