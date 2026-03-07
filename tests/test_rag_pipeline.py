"""
Unit tests for rag_pipeline.py

Tests cover pure-Python logic (prompt building, retriever construction,
error handling). No GPU, no Llama model download required — the Generator
class is not instantiated in any of these tests.
"""

import json
import os
import pytest
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def make_doc(
    product="NUST Savings Account", question="What is the rate?", answer="5%", score=0.9
):
    return {
        "product": product,
        "question": question,
        "answer": answer,
        "score": score,
        "source": "excel:Sheet1",
    }


# ──────────────────────────────────────────────────────────────────────────────
# build_prompt
# ──────────────────────────────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_contains_query(self):
        from src.rag_pipeline import build_prompt

        result = build_prompt("How do I open an account?", [make_doc()])
        assert "How do I open an account?" in result

    def test_contains_product_name(self):
        from src.rag_pipeline import build_prompt

        result = build_prompt("query", [make_doc(product="NUST Current Account")])
        assert "NUST Current Account" in result

    def test_contains_answer_text(self):
        from src.rag_pipeline import build_prompt

        result = build_prompt("query", [make_doc(answer="The profit rate is 7%")])
        assert "The profit rate is 7%" in result

    def test_multiple_docs_all_included(self):
        from src.rag_pipeline import build_prompt

        docs = [
            make_doc(product="Product A", question="Q1?", answer="A1"),
            make_doc(product="Product B", question="Q2?", answer="A2"),
            make_doc(product="Product C", question="Q3?", answer="A3"),
        ]
        result = build_prompt("query", docs)
        assert "Product A" in result
        assert "Product B" in result
        assert "Product C" in result

    def test_empty_docs_does_not_crash(self):
        from src.rag_pipeline import build_prompt

        result = build_prompt("query", [])
        assert "query" in result

    def test_numbered_context_sections(self):
        from src.rag_pipeline import build_prompt

        docs = [make_doc(), make_doc(product="Other Bank")]
        result = build_prompt("q", docs)
        assert "[1]" in result
        assert "[2]" in result


# ──────────────────────────────────────────────────────────────────────────────
# Retriever — file-not-found error handling
# ──────────────────────────────────────────────────────────────────────────────


class TestRetrieverErrorHandling:
    def test_raises_on_missing_faiss_index(self, tmp_path):
        """Retriever should raise FileNotFoundError with helpful message if index absent."""
        import src.rag_pipeline as rp

        doc_mapping = [make_doc()]
        mapping_path = tmp_path / "doc_mapping.json"
        mapping_path.write_text(json.dumps(doc_mapping), encoding="utf-8")

        original_faiss = rp.FAISS_INDEX_PATH
        original_mapping = rp.DOC_MAPPING_PATH
        try:
            rp.FAISS_INDEX_PATH = str(tmp_path / "nonexistent.bin")
            rp.DOC_MAPPING_PATH = str(mapping_path)
            with pytest.raises(FileNotFoundError, match="build_index.py"):
                rp.Retriever()
        finally:
            rp.FAISS_INDEX_PATH = original_faiss
            rp.DOC_MAPPING_PATH = original_mapping

    def test_raises_on_missing_doc_mapping(self, tmp_path, monkeypatch):
        """Retriever should raise FileNotFoundError with helpful message if mapping absent."""
        import src.rag_pipeline as rp

        # Provide a real (tiny) FAISS index so it doesn't fail on that first
        import faiss

        index = faiss.IndexFlatIP(4)
        index_path = tmp_path / "faiss_index.bin"
        faiss.write_index(index, str(index_path))

        original_faiss = rp.FAISS_INDEX_PATH
        original_mapping = rp.DOC_MAPPING_PATH
        try:
            rp.FAISS_INDEX_PATH = str(index_path)
            rp.DOC_MAPPING_PATH = str(tmp_path / "nonexistent.json")
            with pytest.raises(FileNotFoundError, match="build_index.py"):
                rp.Retriever()
        finally:
            rp.FAISS_INDEX_PATH = original_faiss
            rp.DOC_MAPPING_PATH = original_mapping


# ──────────────────────────────────────────────────────────────────────────────
# Retriever — retrieve method (with mocked FAISS + embedding model)
# ──────────────────────────────────────────────────────────────────────────────


class TestRetrieverRetrieve:
    def _make_retriever(self, tmp_path, docs):
        """Build a Retriever backed by a real (tiny) FAISS index and stub embeddings."""
        import faiss
        import src.rag_pipeline as rp
        from unittest.mock import MagicMock, patch

        dim = 4
        # Create embeddings: one per doc, normalized
        vecs = np.random.rand(len(docs), dim).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms

        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        index_path = tmp_path / "faiss_index.bin"
        mapping_path = tmp_path / "doc_mapping.json"
        faiss.write_index(index, str(index_path))
        mapping_path.write_text(json.dumps(docs), encoding="utf-8")

        # Stub the SentenceTransformer so it returns a fixed vector
        mock_model = MagicMock()
        mock_model.encode.return_value = vecs[0:1]  # returns first vector

        with patch("src.rag_pipeline.SentenceTransformer", return_value=mock_model):
            original_faiss = rp.FAISS_INDEX_PATH
            original_mapping = rp.DOC_MAPPING_PATH
            rp.FAISS_INDEX_PATH = str(index_path)
            rp.DOC_MAPPING_PATH = str(mapping_path)
            retriever = rp.Retriever()
            rp.FAISS_INDEX_PATH = original_faiss
            rp.DOC_MAPPING_PATH = original_mapping

        retriever.model = mock_model
        return retriever

    def test_retrieve_returns_top_k_results(self, tmp_path):
        docs = [
            make_doc(question=f"Question {i}?", answer=f"Answer {i}") for i in range(5)
        ]
        retriever = self._make_retriever(tmp_path, docs)
        results = retriever.retrieve("test query", top_k=3)
        assert len(results) == 3

    def test_retrieve_results_have_score_field(self, tmp_path):
        docs = [make_doc(question=f"Q{i}?", answer=f"A{i}") for i in range(3)]
        retriever = self._make_retriever(tmp_path, docs)
        results = retriever.retrieve("test query", top_k=2)
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_retrieve_results_have_expected_fields(self, tmp_path):
        docs = [make_doc()]
        retriever = self._make_retriever(tmp_path, docs)
        results = retriever.retrieve("test query", top_k=1)
        assert len(results) == 1
        for field in ("product", "question", "answer", "score"):
            assert field in results[0]
