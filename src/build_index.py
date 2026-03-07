"""
Build Embedding Index
=====================
Generates sentence embeddings for all processed documents and builds a
FAISS index for fast similarity search.

Usage:
    python src/build_index.py
"""

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import (
    PROCESSED_DOCS_PATH,
    FAISS_INDEX_PATH,
    DOC_MAPPING_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
)


def load_documents(path: str) -> list[dict]:
    """Load the processed document corpus."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Processed corpus not found at '{path}'.\n"
            "Run 'python src/data_preprocessing.py' first to generate it."
        )


def generate_embeddings(
    texts: list[str], model_name: str
) -> tuple[np.ndarray, SentenceTransformer]:
    """
    Generate sentence embeddings using a SentenceTransformer model.

    Returns:
        Tuple of (embeddings array, loaded model) — model is returned so the
        caller can reuse it without loading it a second time.
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Generating embeddings for {len(texts)} documents...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,  # Normalize for cosine similarity via inner product
        batch_size=64,
    )
    return np.array(embeddings, dtype="float32"), model


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index using inner product (cosine similarity on normalized vectors).
    IndexFlatIP is exact search — perfectly fine for our ~300 document corpus.
    """
    dim = embeddings.shape[1]
    assert dim == EMBEDDING_DIMENSION, (
        f"Embedding dimension mismatch: got {dim}, expected {EMBEDDING_DIMENSION}. "
        "Check EMBEDDING_DIMENSION in config/settings.py."
    )
    print(f"Building FAISS index: {embeddings.shape[0]} vectors, dim={dim}")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, index_path: str) -> None:
    """Save the FAISS index to disk."""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")


def save_doc_mapping(documents: list[dict], mapping_path: str) -> None:
    """
    Save a mapping from FAISS index position → document.
    This lets us look up the original document after a similarity search.
    """
    mapping = []
    for i, doc in enumerate(documents):
        mapping.append(
            {
                "id": i,
                "question": doc["question"],
                "answer": doc["answer"],
                "product": doc["product"],
                "source": doc["source"],
            }
        )
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"Document mapping saved to {mapping_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load processed documents
    documents = load_documents(PROCESSED_DOCS_PATH)
    print(f"Loaded {len(documents)} documents")

    # 2. Extract text chunks for embedding
    texts = [doc["text"] for doc in documents]

    # 3. Generate embeddings (model is returned for reuse in the sanity check)
    embeddings, embedding_model = generate_embeddings(texts, EMBEDDING_MODEL_NAME)
    print(f"Embeddings shape: {embeddings.shape}")

    # 4. Build and save FAISS index
    index = build_faiss_index(embeddings)
    save_index(index, FAISS_INDEX_PATH)

    # 5. Save document mapping
    save_doc_mapping(documents, DOC_MAPPING_PATH)

    # 6. Quick sanity check — search for a test query (reuses already-loaded model)
    print("\n── Sanity Check ──")
    test_query = "What is the daily transfer limit?"
    query_embedding = embedding_model.encode([test_query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, k=3)
    print(f"Query: '{test_query}'")
    print(f"Top 3 results:")
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        doc = documents[idx]
        print(f"  {rank}. [score={dist:.4f}] ({doc['product']})")
        print(f"     Q: {doc['question']}")
        print(f"     A: {doc['answer'][:100]}...")
