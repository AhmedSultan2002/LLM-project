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
│  CUDA: 4-bit NF4     │
│  MPS:  float16       │
└─────────────────────┘
```

## Setup

### Prerequisites
- Python 3.10
- conda (recommended for environment management)
- **NVIDIA GPU** with ≥6 GB VRAM — runs with 4-bit NF4 quantization (bitsandbytes)
- **OR Apple Silicon Mac** (M1/M2/M3) — runs in float16 on MPS, no quantization library needed

### Installation

1. **Create and activate a conda environment**:
   ```bash
   conda create -n nust-bank python=3.10 -y
   conda activate nust-bank
   ```

2. **Install PyTorch**:

   - **NVIDIA GPU (CUDA 12.4)**:
     ```bash
     pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
         --index-url https://download.pytorch.org/whl/cu124
     ```
   - **Apple Silicon / CPU**:
     ```bash
     pip install torch==2.5.1
     ```

3. **Install project dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **NVIDIA only — install bitsandbytes for 4-bit quantization**:
   ```bash
   pip install bitsandbytes==0.45.0
   ```

### Hugging Face Access
Llama 3.2 is a gated model. You must:
1. Accept the license at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
2. Authenticate in your terminal:
   ```bash
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

> **Note (macOS):** Do not run via `conda run` — it causes a segfault on macOS due to
> process isolation. Always activate the environment first (`conda activate nust-bank`)
> and run with `python` directly.

> **Note:** The interactive mode is **stateless** — each question is processed independently
> with no memory of previous turns. Follow-up questions like "tell me more about that"
> will not work as expected. Each question should be self-contained.

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

| Component       | Technology                                        |
|----------------|---------------------------------------------------|
| LLM            | Llama 3.2 3B Instruct                            |
| Embeddings     | all-MiniLM-L6-v2 (384-dim)                       |
| Vector Store   | FAISS (IndexFlatIP)                              |
| Quantization   | bitsandbytes NF4 (CUDA only) / float16 (MPS)    |
| Language       | Python 3.10                                       |

## Team
- Ahmed Sultan
- Muhammad Saad Ashraf

