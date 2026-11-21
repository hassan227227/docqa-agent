# DocQA Agent

Ask questions about your own documents and get answers that cite the exact passages they came from. Built with Python, BM25 retrieval, Claude (with citations), FastAPI and an MCP server.

```
documents ──► chunker ──► BM25 index ──┬──► /search  (FastAPI)
                                       ├──► /ask ──► Claude + citations ──► grounded answer
                                       └──► MCP tools: search_docs, get_passage, list_documents
```

## Why

Fintech and operations teams answer the same policy questions every day: lease terms, credit thresholds, required documents. DocQA retrieves the relevant passages and has Claude answer **only** from them, with a citation for every claim, so an analyst can check the answer in seconds. If the documents don't contain the answer, it says so.

## Features

- **Retrieval (RAG):** paragraph-aware chunking with overlap and an Okapi BM25 index in pure Python. No vector database needed to get started, and it's fully unit-tested.
- **Grounded answers with citations:** retrieved chunks go to Claude as `document` blocks with citations enabled; the response carries the quoted source text.
- **Safety fallback:** if the model declines a request, the API retries on a fallback model automatically.
- **HTTP API:** `GET /search` for retrieval only and `POST /ask` for answers (FastAPI, validated inputs).
- **MCP server:** exposes the index to any MCP client (Claude Desktop, Claude Code, your own agents) as `search_docs`, `get_passage` and `list_documents`.
- **CLI:** `python answer.py "question"`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...                         # Windows: set ANTHROPIC_API_KEY=...

# CLI
python answer.py "What is the early termination fee?"

# HTTP API
uvicorn api:app --reload
curl "localhost:8000/search?q=credit+score+approval"
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Which documents does an applicant need?"}'

# Tests (no API key needed)
pytest -q
```

Point it at your own files with `DOCQA_DOCS=/path/to/folder` (`.md` and `.txt`). The sample documents in `sample_docs/` are fictional policies.

## Use it from Claude Desktop (MCP)

Add this to your Claude Desktop config:

```json
{
  "mcpServers": {
    "docqa": {
      "command": "python",
      "args": ["/absolute/path/to/docqa-agent/mcp_server.py"],
      "env": { "DOCQA_DOCS": "/absolute/path/to/your/docs" }
    }
  }
}
```

## Project layout

| File | What it does |
|---|---|
| `retrieval.py` | Tokenizer, chunker, BM25 index, folder loader |
| `answer.py` | Calls Claude with retrieved passages and returns the answer plus citations; also the CLI |
| `api.py` | FastAPI app: `/health`, `/search`, `/ask` |
| `mcp_server.py` | MCP server with `search_docs`, `get_passage`, `list_documents` |
| `test_retrieval.py` | Unit tests for chunking and ranking |

## Roadmap

- Embedding search (pgvector) alongside BM25, with rank fusion
- PDF ingestion
- An evaluation set measuring answer accuracy and citation coverage

## Author

Hassan Raza · [Portfolio](https://hassan227227.github.io) · [LinkedIn](https://www.linkedin.com/in/iamhassan890)
