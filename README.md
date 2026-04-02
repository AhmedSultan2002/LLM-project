# NUST Bank Customer Service Assistant

An intelligent customer service chatbot for NUST Bank, powered by **Llama 3.2** (fine-tuned with **QLoRA**) and **Retrieval-Augmented Generation (RAG)**.

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

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\Activate.ps1
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install PyTorch with CUDA support** (must be done separately as it is not in `requirements.txt`):

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

### Step 3: Generate Fine-Tuning Dataset
```bash
python src/generate_finetune_data.py
```
Converts the processed Q&A corpus into Llama 3 chat template format. Output: `data/finetune_dataset.json`.

### Step 4: Fine-Tune the Model (QLoRA)
```bash
python src/finetune.py
```
Fine-tunes Llama 3.2 3B using 4-bit QLoRA on the generated dataset (~10 min on RTX 4060).
The LoRA adapter weights are saved to `data/lora-nust-bank/`. The RAG pipeline will automatically load these adapters if they exist.

### Step 5: Run the Assistant
```bash
# Interactive mode
python src/rag_pipeline.py

# Single query
python src/rag_pipeline.py --query "What is the daily transfer limit?"
```

> **Note:** The interactive mode is **stateless** — each question is processed independently
> with no memory of previous turns. Each question should be self-contained.

## Project Structure

```
├── config/
│   └── settings.py                # Centralized configuration
├── src/
│   ├── data_preprocessing.py      # Excel + JSON data parsing
│   ├── build_index.py             # Embedding generation + FAISS
│   ├── generate_finetune_data.py  # Fine-tune dataset generation
│   ├── finetune.py                # QLoRA fine-tuning script
│   └── rag_pipeline.py            # RAG pipeline + guardrails
├── data/                          # Generated data (gitignored)
│   ├── processed_documents.json
│   ├── faiss_index.bin
│   ├── doc_mapping.json
│   ├── finetune_dataset.json
│   └── lora-nust-bank/            # Fine-tuned LoRA adapter weights
├── NUST Bank-Product-Knowledge.xlsx
├── funds_transfer_app_features_faq.json
├── requirements.txt
└── README.md
```

## Tech Stack

| Component       | Technology                                        |
|----------------|---------------------------------------------------|
| LLM            | Llama 3.2 3B Instruct (QLoRA fine-tuned)         |
| Fine-Tuning    | PEFT / TRL SFTTrainer (4-bit QLoRA)              |
| Embeddings     | all-MiniLM-L6-v2 (384-dim)                       |
| Vector Store   | FAISS (IndexFlatIP)                              |
| Quantization   | bitsandbytes NF4 (CUDA) / float16 (MPS)          |
| Language       | Python 3.10                                       |

## Team
- Ahmed Sultan
- Muhammad Saad Ashraf

