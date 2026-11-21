"""MCP server exposing the document index as tools.

Any MCP client (Claude Desktop, Claude Code, an agent you build) can then
search your documents and fetch passages directly.

Run over stdio:  python mcp_server.py
"""
from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

from retrieval import load_folder

DOCS_DIR = os.environ.get("DOCQA_DOCS", "sample_docs")
index = load_folder(DOCS_DIR)
_by_ref = {c.ref: c for c in index.chunks}

server = MCPServer(
    name="docqa",
    instructions="Search a local document collection. Use search_docs first, then get_passage for full text.",
)


@server.tool()
def search_docs(query: str, k: int = 5) -> list[dict]:
    """Search the documents with BM25 and return the best-matching passages."""
    k = max(1, min(k, 20))
    return [
        {"source": c.ref, "score": round(s, 3), "preview": c.text[:300]}
        for c, s in index.search(query, k=k)
    ]


@server.tool()
def get_passage(source: str) -> str:
    """Return the full text of a passage by its source id, e.g. 'leasing_policy.md#2'."""
    chunk = _by_ref.get(source)
    if chunk is None:
        raise ValueError(f"Unknown source id: {source}")
    return chunk.text


@server.tool()
def list_documents() -> list[str]:
    """List the documents in the index."""
    return sorted({c.doc_id for c in index.chunks})


if __name__ == "__main__":
    server.run("stdio")
