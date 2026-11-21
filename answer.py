"""Grounded answers with citations from Claude.

Retrieved chunks are sent as `document` blocks with citations enabled, so every
claim in the answer points back to the exact passage it came from.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import anthropic

from retrieval import BM25Index, Chunk, load_folder

MODEL = os.environ.get("DOCQA_MODEL", "claude-opus-5-5")
SYSTEM = (
    "You answer questions using only the documents provided. "
    "Cite the passages you rely on. If the documents don't contain the answer, "
    "say that plainly instead of guessing. Keep answers short and direct."
)


@dataclass
class Citation:
    source: str
    quote: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    sources: list[str]
    refused: bool = False


def _document_blocks(hits: list[tuple[Chunk, float]]) -> list[dict]:
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": chunk.text},
            "title": chunk.ref,
            "citations": {"enabled": True},
        }
        for chunk, _ in hits
    ]


def ask(question: str, index: BM25Index, k: int = 5, client: anthropic.Anthropic | None = None) -> Answer:
    hits = index.search(question, k=k)
    if not hits:
        return Answer(text="I couldn't find anything about that in the documents.", citations=[], sources=[])

    client = client or anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "medium"},
        # On a safety decline, the API re-runs the request on a fallback model it picks.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[
            {
                "role": "user",
                "content": [*_document_blocks(hits), {"type": "text", "text": question}],
            }
        ],
    )

    if response.stop_reason == "refusal":
        return Answer(text="The model declined to answer this question.", citations=[], sources=[], refused=True)

    parts: list[str] = []
    citations: list[Citation] = []
    for block in response.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        for c in getattr(block, "citations", None) or []:
            citations.append(Citation(source=c.document_title or "", quote=c.cited_text))

    return Answer(
        text="".join(parts).strip(),
        citations=citations,
        sources=[chunk.ref for chunk, _ in hits],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about a folder of documents.")
    parser.add_argument("question")
    parser.add_argument("--docs", default="sample_docs", help="folder of .md/.txt files")
    parser.add_argument("-k", type=int, default=5, help="number of passages to retrieve")
    args = parser.parse_args()

    result = ask(args.question, load_folder(args.docs), k=args.k)
    print(result.text)
    if result.citations:
        print("\nSources:")
        for c in result.citations:
            print(f"  [{c.source}] \"{c.quote.strip()[:120]}\"")


if __name__ == "__main__":
    main()
