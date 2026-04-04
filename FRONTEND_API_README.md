# NUST Bank Customer Service - Frontend & API

## Quick Start

### Option 1: Run Locally (Recommended for Development)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API server:**
   ```bash
   python -m uvicorn src.api.main:app --reload
   ```
   - API runs at: `http://localhost:8000`
   - API docs at: `http://localhost:8000/docs`

3. **Start the Streamlit frontend (in another terminal):**
   ```bash
   streamlit run frontend/app.py
   ```
   - Frontend runs at: `http://localhost:8501`

---

### Option 2: Docker Compose

1. **Build and run:**
   ```bash
   docker-compose up --build
   ```

2. **Access:**
   - Frontend: `http://localhost:8501`
   - API: `http://localhost:8000`
   - API Docs: `http://localhost:8000/docs`

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root info |
| `/api/v1/health` | GET | Health check |
| `/api/v1/query` | POST | Submit a query |
| `/api/v1/sources` | GET | Get available sources |

### Example: Submit a Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the daily transfer limit?"}'
```

Response:
```json
{
  "query": "What is the daily transfer limit?",
  "answer": "The daily transfer limit for NUST Bank...",
  "sources": [
    {
      "product": "Funds Transfer",
      "question": "What is the daily transfer limit?",
      "score": 0.892
    }
  ],
  "latency_seconds": 2.45
}
```

---

## Project Structure

```
LLM-project/
├── frontend/
│   └── app.py              # Streamlit web interface
├── src/
│   └── api/
│       ├── main.py         # FastAPI application
│       ├── routes.py       # API endpoints
│       ├── models.py       # Request/Response models
│       └── service.py     # RAG pipeline service
├── requirements.txt        # Python dependencies
├── Dockerfile             # API container
├── Dockerfile.frontend    # Frontend container
└── docker-compose.yml     # Orchestration
```

---

## Notes

- The API uses **lazy loading** - the LLM loads only when the first query is submitted
- First query may take 30-60 seconds due to model loading
- Ensure data files exist: `data/faiss_index.bin`, `data/doc_mapping.json`
