import os
import io
import re
import numpy as np
from typing import Optional
from openai import OpenAI

# Optional PDF support
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"
CHUNK_SIZE  = 500   # characters
CHUNK_OVERLAP = 100


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def embed(texts: list[str]) -> np.ndarray:
    """Get embeddings for a list of texts."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in response.data], dtype="float32")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and a matrix."""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b @ a


class RAGEngine:
    """
    A simple in-memory RAG engine.
    For production, swap self._store with a vector DB like Pinecone or pgvector.
    """

    def __init__(self):
        self._store: list[dict] = []   # [{text, embedding, source}]
        self._documents: list[str] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_document(self, filename: str, content: bytes, content_type: str) -> dict:
        """Parse, chunk, embed, and store a document."""
        text = self._parse(content, content_type)
        chunks = chunk_text(text)
        embeddings = embed(chunks)

        for chunk, emb in zip(chunks, embeddings):
            self._store.append({"text": chunk, "embedding": emb, "source": filename})

        if filename not in self._documents:
            self._documents.append(filename)

        return {"chunks": len(chunks)}

    def _parse(self, content: bytes, content_type: str) -> str:
        if content_type == "application/pdf":
            if not PDF_SUPPORT:
                raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        else:
            return content.decode("utf-8", errors="ignore")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question: str, top_k: int = 3) -> list[dict]:
        """Return the top_k most relevant chunks."""
        if not self._store:
            return []

        q_emb = embed([question])[0]
        matrix = np.stack([item["embedding"] for item in self._store])
        scores = cosine_similarity(q_emb, matrix)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self._store[i] for i in top_indices]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 3) -> dict:
        """Retrieve context and generate an answer."""
        chunks = self.retrieve(question, top_k=top_k)

        if not chunks:
            return {
                "answer": "I don't have any documents in my knowledge base yet. Please upload a document first.",
                "sources": [],
                "docs_used": 0,
            }

        context = "\n\n---\n\n".join(c["text"] for c in chunks)
        sources  = list(dict.fromkeys(c["source"] for c in chunks))  # deduplicated

        system_prompt = (
            "You are a helpful assistant. Answer the user's question using ONLY the context below. "
            "If the answer is not in the context, say: 'I could not find this information in the provided documents.' "
            "Be concise and accurate. Cite the source document when helpful."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=600,
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": sources,
            "docs_used": len(chunks),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def list_documents(self) -> list[str]:
        return self._documents

    def clear(self):
        self._store.clear()
        self._documents.clear()
