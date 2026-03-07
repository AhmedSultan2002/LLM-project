# NUST Bank Customer Service Assistant

An intelligent customer service chatbot for NUST Bank, powered by **Llama 3.2** with **Retrieval-Augmented Generation (RAG)**.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Sentence-Transformer│  ──→  Query Embedding (384-dim)
│  (all-MiniLM-L6-v2) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    FAISS Index       │  ──→  Top-K Relevant Documents
│  (314 bank Q&A docs) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Prompt Builder     │  ──→  System Prompt + Context + Query
│  (Banking domain)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Llama 3.2 (3B)     │  ──→  Generated Response
│  (4-bit quantized)   │
└─────────────────────┘
```

## Setup

### Prerequisites
- Python 3.10+
- NVIDIA GPU with ≥6GB VRAM (for 4-bit quantized Llama 3.2 3B)
- CUDA toolkit

### Installation

1. **Install PyTorch with CUDA support** (Important for Windows compatibility):
   ```bash
   pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
   ```

2. **Install other dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Hugging Face Access
Llama 3.2 requires access approval. Make sure you:
1. Accept the license at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
2. Log in to your terminal using your Hugging Face access token:
   ```bash
   # Use this python script to avoid PATH issues on Windows
   python -c "from huggingface_hub import login; login()"
   ```

## Usage

### Step 1: Preprocess Data
```bash
python src/data_preprocessing.py
```
Parses the NUST Bank Excel workbook and JSON FAQ into a cleaned document corpus.

### Step 2: Build Vector Index
```bash
python src/build_index.py
```
Generates embeddings and builds the FAISS similarity search index.

### Step 3: Run the Assistant
```bash
# Interactive mode
python src/rag_pipeline.py

# Single query
python src/rag_pipeline.py --query "What is the daily transfer limit?"
```

## Project Structure

```
├── config/
│   └── settings.py              # Centralized configuration
├── src/
│   ├── data_preprocessing.py    # Excel + JSON data parsing
│   ├── build_index.py           # Embedding generation + FAISS
│   └── rag_pipeline.py          # RAG pipeline (retrieval + generation)
├── data/                        # Generated data (gitignored)
│   ├── processed_documents.json
│   ├── faiss_index.bin
│   └── doc_mapping.json
├── NUST Bank-Product-Knowledge.xlsx
├── funds_transfer_app_features_faq.json
├── requirements.txt
└── README.md
```

## Tech Stack

| Component       | Technology                          |
|----------------|-------------------------------------|
| LLM            | Llama 3.2 3B Instruct (4-bit)     |
| Embeddings     | all-MiniLM-L6-v2 (384-dim)        |
| Vector Store   | FAISS (IndexFlatIP)               |
| Quantization   | bitsandbytes (NF4)                |
| Language       | Python 3.10                        |

## Team
- Ahmed Sultan
