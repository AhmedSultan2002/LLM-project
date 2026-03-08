"""
Centralized configuration for the NUST Bank LLM project.
All paths, model names, and hyperparameters in one place.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw data
RAW_EXCEL_PATH = os.path.join(PROJECT_ROOT, "NUST Bank-Product-Knowledge.xlsx")
RAW_JSON_PATH = os.path.join(PROJECT_ROOT, "funds_transfer_app_features_faq.json")

# Processed data
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DOCS_PATH = os.path.join(DATA_DIR, "processed_documents.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
DOC_MAPPING_PATH = os.path.join(DATA_DIR, "doc_mapping.json")

# ─── Embedding Model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ─── LLM ──────────────────────────────────────────────────────────────────────
LLM_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.9
LLM_REPETITION_PENALTY = 1.1
LLM_MAX_INPUT_LENGTH = 2048  # Tokenizer truncation limit
LLM_USE_4BIT = (
    True  # Enable quantization where supported (CUDA only; ignored on MPS/CPU)
)


# ─── Hardware / Platform Detection ────────────────────────────────────────────
def _detect_device() -> str:
    """
    Detect the best available compute device.

    Returns one of:
        "cuda"  – NVIDIA GPU with CUDA support
        "mps"   – Apple Silicon GPU (Metal Performance Shaders)
        "cpu"   – fallback
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


DEVICE = _detect_device()

# bitsandbytes NF4 quantization only works on CUDA.
# On MPS/CPU we disable it automatically regardless of LLM_USE_4BIT.
QUANTIZATION_ENABLED = LLM_USE_4BIT and (DEVICE == "cuda")

# ─── RAG ──────────────────────────────────────────────────────────────────────
RAG_TOP_K = 3  # Number of retrieved chunks for context
