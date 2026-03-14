import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag_engine import RAGEngine

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGEngine()

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

# ---------- Models ----------

class ChatRequest(BaseModel):
    question: str
    top_k: int = 3

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float
    docs_used: int

# ---------- Routes ----------

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or TXT document to the knowledge base."""
    allowed_types = ["application/pdf", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    content = await file.read()
    result = rag.add_document(file.filename, content, file.content_type)
    return {"message": f"Document '{file.filename}' indexed successfully.", "chunks": result["chunks"]}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Ask a question against the uploaded documents."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.time()
    result = rag.query(req.question, top_k=req.top_k)
    latency = round((time.time() - start) * 1000, 2)

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=latency,
        docs_used=result["docs_used"],
    )

@app.get("/documents")
def list_documents():
    """List all indexed documents."""
    return {"documents": rag.list_documents()}

@app.delete("/documents")
def clear_documents():
    """Clear the entire knowledge base."""
    rag.clear()
    return {"message": "Knowledge base cleared."}

@app.get("/health")
def health():
    return {"status": "ok", "docs_indexed": len(rag.list_documents())}
