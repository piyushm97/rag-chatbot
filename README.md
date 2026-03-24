# 🧠 RAG Chatbot

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload documents (PDF or TXT) and ask questions about them. Built with FastAPI, OpenAI, and a clean web UI — deployable via Docker.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)
![CI](https://github.com/YOUR_USERNAME/rag-chatbot/actions/workflows/ci.yml/badge.svg)

---

## ✨ Features

- **Document ingestion** — Upload PDF or TXT files; they are automatically chunked and embedded
- **Semantic search** — Cosine similarity over OpenAI embeddings retrieves the most relevant context
- **Grounded answers** — GPT-4o-mini answers strictly from your documents, no hallucination on off-topic questions
- **Source citations** — Every answer shows which documents were used
- **Latency tracking** — Response time displayed per query
- **REST API** — Clean FastAPI endpoints, fully documented at `/docs`
- **Docker-ready** — Single command to run anywhere
- **CI pipeline** — GitHub Actions runs tests and linting on every push

---

## 🏗️ Architecture

```
User
 │
 ▼
FastAPI  ──────────────────────────────────────────────────────────┐
 │                                                                  │
 ├─► /upload  → parse → chunk → embed (text-embedding-3-small)     │
 │                                    ↓                            │
 │                             In-memory store                     │
 │                          (swap for Pinecone / pgvector)         │
 │                                    ↑                            │
 └─► /chat   → embed query → cosine similarity → top-k chunks      │
                              → GPT-4o-mini → answer + sources ────┘
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your OpenAI key
```

### 3. Run

```bash
uvicorn app:app --reload
```

Open `http://localhost:8000` — you'll see the chat UI.

---

## 🐳 Docker

```bash
docker build -t rag-chatbot .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-your-key rag-chatbot
```

---

## 📡 API Reference

Full interactive docs at `http://localhost:8000/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload a PDF or TXT file |
| `POST` | `/chat` | Ask a question |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents` | Clear knowledge base |
| `GET` | `/health` | Health check |

### Example: Upload a document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@report.pdf"
```

### Example: Ask a question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "top_k": 3}'
```

**Response:**
```json
{
  "answer": "The key findings include...",
  "sources": ["report.pdf"],
  "latency_ms": 843.2,
  "docs_used": 3
}
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

Tests use mocking — no real API calls or keys needed.

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py              # FastAPI app & routes
├── rag_engine.py       # Core RAG logic (chunking, embedding, retrieval, generation)
├── static/
│   └── index.html      # Chat UI
├── tests/
│   └── test_rag.py     # Unit tests
├── .github/
│   └── workflows/
│       └── ci.yml      # GitHub Actions CI
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔧 Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | **Required** |
| `CHUNK_SIZE` | Characters per chunk (in `rag_engine.py`) | `500` |
| `CHUNK_OVERLAP` | Overlap between chunks | `100` |
| `EMBED_MODEL` | Embedding model | `text-embedding-3-small` |
| `CHAT_MODEL` | Chat model | `gpt-4o-mini` |

---

## 🗺️ Roadmap / What I'd Add Next

- [ ] Swap in-memory store for **Pinecone** or **pgvector** for persistence
- [ ] Add **streaming** responses (SSE)
- [ ] Support more file types: DOCX, HTML, Markdown
- [ ] Add **conversation history** (multi-turn)
- [ ] Rate limiting and auth middleware
- [ ] Evaluation pipeline (RAGAS metrics)

---

## 📄 License

MIT — use it, modify it, ship it.
