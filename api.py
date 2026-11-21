"""FastAPI service: /search (retrieval only) and /ask (grounded answer with citations).

Run:  uvicorn api:app --reload
"""
from __future__ import annotations

import os
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from answer import ask
from retrieval import load_folder

DOCS_DIR = os.environ.get("DOCQA_DOCS", "sample_docs")
index = load_folder(DOCS_DIR)
app = FastAPI(title="DocQA Agent", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(index.chunks)}


@app.get("/search")
def search(q: str, k: int = 5) -> list[dict]:
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    return [{"source": c.ref, "score": round(s, 3), "text": c.text} for c, s in index.search(q, k=k)]


@app.post("/ask")
def ask_endpoint(req: AskRequest) -> dict:
    return asdict(ask(req.question, index, k=req.k))
