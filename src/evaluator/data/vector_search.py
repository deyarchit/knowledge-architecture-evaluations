from typing import Dict, List, Optional

import chromadb
from sentence_transformers.cross_encoder import CrossEncoder

from evaluator.data.file_io import (
    read_json_from_file,
)
from evaluator.models.qa import Concepts
from evaluator.utils import get_data_path


class VectorSearch:
    def __init__(self, max_results: int = 3, enable_reranking: bool = False) -> None:
        client = chromadb.Client()
        self.collection = client.get_or_create_collection(name="ap_history_concepts")

        chunk_file = get_data_path("processed/ap_history_concepts.json")
        concepts: Concepts | None = read_json_from_file(chunk_file, Concepts)
        if not concepts:
            raise RuntimeError(f"No concepts found in file: {chunk_file}")
        self._build_vector_search_engine(concepts.chunks)

        self.max_results: int = max_results  # Total results to return
        self.search_results: int = 25 if enable_reranking else max_results  # Total results to query

        self.enable_reranking = enable_reranking
        self.cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    def _build_vector_search_engine(self, chunks: Dict[str, str]):
        print("Building vector search")
        ids: list = []
        documents: list = []
        for concept_id, concept in chunks.items():
            documents.append(concept)
            ids.append(concept_id)

        self.collection.upsert(documents=documents, ids=ids)

    def query(self, query: str) -> Optional[List[str]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=self.search_results,
            include=["documents", "distances", "metadatas"],
        )

        documents = results.get("documents", [])
        if not documents or not documents[0]:
            return None

        # Store the documents corresponding to the first and only search query
        search_results = documents[0]

        if not self.enable_reranking:
            return search_results[: self.max_results]

        ranks = self.cross_encoder_model.rank(query, search_results)

        final_results: List[str] = []
        for rank in ranks[: self.max_results]:
            element_idx: int | float | str = rank["corpus_id"]
            try:
                element_idx = int(element_idx)
            except ValueError:
                raise RuntimeError(
                    f"'corpus_id' is not a valid integer. Using original value: {element_idx}"
                )
            final_results.append(search_results[element_idx])

        return final_results
