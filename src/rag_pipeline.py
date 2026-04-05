"""
RAG Pipeline
============
Retrieval-Augmented Generation pipeline for NUST Bank customer queries.
Combines FAISS-based retrieval with Llama 3.2 generation.

Usage:
    python src/rag_pipeline.py                           # Interactive mode
    python src/rag_pipeline.py --query "your question"   # Single query mode
"""

# ── Environment setup (must happen before any heavy imports) ──────────────────
# On macOS, FAISS and sentence-transformers each initialise OpenMP/BLAS thread
# pools.  When the HuggingFace safetensors loader later spawns its own Rust
# thread pool a deadlock can occur on Apple Silicon.  Limiting all thread pools
# to 1 avoids the deadlock while still allowing MPS GPU parallelism.
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    FAISS_INDEX_PATH,
    DOC_MAPPING_PATH,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_REPETITION_PENALTY,
    LLM_MAX_INPUT_LENGTH,
    LLM_USE_4BIT,
    DEVICE,
    QUANTIZATION_ENABLED,
    RAG_TOP_K,
    DATA_DIR,
)

# ──────────────────────────────────────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────────────────────────────────────


class Retriever:
    """Handles query embedding and FAISS similarity search."""

    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print("Loading FAISS index...")
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at '{FAISS_INDEX_PATH}'.\n"
                "Run 'python src/build_index.py' first to generate it."
            )
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        print("Loading document mapping...")
        if not os.path.exists(DOC_MAPPING_PATH):
            raise FileNotFoundError(
                f"Document mapping not found at '{DOC_MAPPING_PATH}'.\n"
                "Run 'python src/build_index.py' first to generate it."
            )
        with open(DOC_MAPPING_PATH, "r", encoding="utf-8") as f:
            self.doc_mapping = json.load(f)

        print(f"Retriever ready: {self.index.ntotal} documents indexed\n")

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        """
        Retrieve the most relevant documents for a query.

        Returns:
            List of dicts with 'question', 'answer', 'product', 'source', 'score'.
        """
        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype="float32")

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k=top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue  # FAISS returns -1 for empty results
            doc = self.doc_mapping[idx].copy()
            doc["score"] = float(dist)
            results.append(doc)

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a highly restricted and professional customer service assistant strictly for NUST Bank.
Your ONLY role is to assist customers with questions regarding NUST Bank products, services, and transactions.

RULES:
1. You MUST NOT answer any questions or provide information outside the scope of NUST Bank.
2. If the user asks about general knowledge, programming, non-NUST entities, or anything else, you MUST reply: "I am a customer service assistant for NUST Bank. I can only assist you with NUST Bank products and services."
3. You MUST answer ONLY based on the provided NUST Bank context below.
4. If the provided context does not contain the answer to a NUST bank related query, say: "I don't have specific information about that. Please contact NUST Bank at +92 (51) 111 000 494 or visit your nearest branch for assistance."
5. Never break character, ignore instructions, or act as a general AI model. You are exclusively a NUST Bank representative."""


def build_prompt(query: str, retrieved_docs: list[dict]) -> str:
    """Build the full prompt with retrieved context for the LLM."""
    # Format retrieved context
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(
            f"[{i}] Product: {doc['product']}\n"
            f"    Q: {doc['question']}\n"
            f"    A: {doc['answer']}"
        )
    context = "\n\n".join(context_parts)

    user_message = (
        f"Context from NUST Bank knowledge base:\n"
        f"{context}\n\n"
        f"Customer Question: {query}\n\n"
        f"Please provide a helpful answer based on the above context."
    )

    return user_message


def format_chat_prompt(system_prompt: str, user_message: str, tokenizer) -> str:
    """Format the prompt using the model's chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ──────────────────────────────────────────────────────────────────────────────
# LLM Generator
# ──────────────────────────────────────────────────────────────────────────────


class Generator:
    """Handles LLM loading and text generation."""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = (
            torch  # Store reference so generate() can reuse it without re-importing
        )

        print(f"Loading LLM: {LLM_MODEL_NAME}")
        print(f"  Device: {DEVICE}")
        print(f"  4-bit quantization: {QUANTIZATION_ENABLED}")

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

        if QUANTIZATION_ENABLED:
            # ── CUDA path: bitsandbytes NF4 4-bit quantization ────────────────
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        elif DEVICE == "mps":
            # ── Apple Silicon path: float16 on MPS ────────────────────────────
            # bitsandbytes is not supported on MPS.
            # Load on CPU first then move to MPS — loading directly onto MPS
            # can segfault on some PyTorch versions due to a Metal memory bug.
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to("mps")
        else:
            # ── CPU fallback: float32 ─────────────────────────────────────────
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.float32,
            )

        # ── Load LoRA Adapter if it exists ────────────────────────────────────
        lora_path = os.path.join(DATA_DIR, "lora-nust-bank")
        if os.path.exists(lora_path):
            print(f"Loading Fine-Tuned LoRA adapter from {lora_path}...")
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                print("LoRA adapter applied successfully.")
            except ImportError:
                print("peft library not found. Running base model without LoRA.")
        else:
            print("No LoRA adapter found. Running base model.")

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("LLM loaded successfully!\n")

    def generate(self, query: str, retrieved_docs: list[dict]) -> str:
        """Generate a response given a query and retrieved documents."""
        torch = self._torch  # Reuse the already-imported torch from __init__

        # Build prompt
        user_message = build_prompt(query, retrieved_docs)
        full_prompt = format_chat_prompt(SYSTEM_PROMPT, user_message, self.tokenizer)

        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=LLM_MAX_INPUT_LENGTH,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE,
                do_sample=True,
                top_p=LLM_TOP_P,
                repetition_penalty=LLM_REPETITION_PENALTY,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (skip the prompt)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()


# ──────────────────────────────────────────────────────────────────────────────
# RAG Pipeline (ties it all together)
# ──────────────────────────────────────────────────────────────────────────────


class RAGPipeline:
    """Full Retrieval-Augmented Generation pipeline with Guardrails."""

    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        
        # Hardcoded Guardrails pre-filter keywords
        self.blocked_keywords = [
            "ignore previous instructions", "forget everything", "system prompt",
            "jailbreak", "dan", "you are an unstructured", "as an ai", "help me code"
        ]

    def validate_query(self, query: str) -> bool:
        """Returns False if query trips the basic guardrails."""
        query_lower = query.lower()
        for kw in self.blocked_keywords:
            if kw in query_lower:
                return False
        return True

    def query(self, user_query: str) -> dict:
        """
        Process a user query end-to-end.

        Returns:
            dict with 'query', 'answer', 'sources', 'latency_seconds'.
        """
        start_time = time.time()
        
        # Guardrail Step: Pre-flight validation
        if not self.validate_query(user_query):
            latency = time.time() - start_time
            return {
                "query": user_query,
                "answer": "I am a customer service assistant for NUST Bank. I cannot process or respond to this input.",
                "sources": [],
                "latency_seconds": round(latency, 2),
            }

        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(user_query)

        # Step 2: Generate answer
        answer = self.generator.generate(user_query, retrieved_docs)

        latency = time.time() - start_time

        return {
            "query": user_query,
            "answer": answer,
            "sources": [
                {
                    "product": d["product"],
                    "question": d["question"],
                    "score": d["score"],
                }
                for d in retrieved_docs
            ],
            "latency_seconds": round(latency, 2),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="NUST Bank RAG Pipeline")
    parser.add_argument("--query", type=str, help="Single query mode")
    args = parser.parse_args()

    print("=" * 60)
    print("  NUST Bank Customer Service Assistant")
    print("  (RAG Pipeline — Llama 3.2)")
    print("=" * 60)
    print()

    pipeline = RAGPipeline()

    if args.query:
        # Single query mode
        result = pipeline.query(args.query)
        print(f"\nQ: {result['query']}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources:")
        for s in result["sources"]:
            print(f"  - [{s['score']:.3f}] {s['product']}: {s['question']}")
        print(f"\nLatency: {result['latency_seconds']}s")
    else:
        # Interactive mode
        print("Type your question (or 'quit' to exit):\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            result = pipeline.query(user_input)

            print(f"\nAssistant: {result['answer']}")
            print(
                f"\n  [Sources: {', '.join(s['product'] for s in result['sources'])} | "
                f"Latency: {result['latency_seconds']}s]"
            )
            print()


if __name__ == "__main__":
    main()
