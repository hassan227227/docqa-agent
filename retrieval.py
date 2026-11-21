"""Document loading, chunking and BM25 retrieval.

Pure Python with no external services, so retrieval can be tested offline and
the same index can back the HTTP API, the CLI and the MCP server.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were will with".split()
)
SUPPORTED_SUFFIXES = {".md", ".txt"}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: int
    text: str

    @property
    def ref(self) -> str:
        return f"{self.doc_id}#{self.chunk_id}"


def chunk_text(text: str, doc_id: str, max_words: int = 180, overlap: int = 40) -> list[Chunk]:
    """Split text into overlapping word windows, keeping paragraph breaks where possible."""
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap < max_words:
        raise ValueError("overlap must be >= 0 and smaller than max_words")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    words: list[str] = []
    for p in paragraphs:
        words.extend(p.split())
        words.append("\n\n")

    chunks: list[Chunk] = []
    step = max_words - overlap
    start = 0
    while start < len(words):
        window = words[start : start + max_words]
        body = " ".join(window).replace(" \n\n ", "\n\n").replace("\n\n ", "\n\n").strip()
        if body:
            chunks.append(Chunk(doc_id=doc_id, chunk_id=len(chunks), text=body))
        if start + max_words >= len(words):
            break
        start += step
    return chunks


@dataclass
class BM25Index:
    """Okapi BM25 over chunks (k1/b are the usual defaults)."""

    k1: float = 1.5
    b: float = 0.75
    chunks: list[Chunk] = field(default_factory=list)
    _tfs: list[Counter] = field(default_factory=list, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)
    _avgdl: float = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            tf = Counter(tokenize(c.text))
            self.chunks.append(c)
            self._tfs.append(tf)
            self._df.update(tf.keys())
        total = sum(sum(tf.values()) for tf in self._tfs)
        self._avgdl = total / len(self._tfs) if self._tfs else 0.0

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []
        scored: list[tuple[Chunk, float]] = []
        for chunk, tf in zip(self.chunks, self._tfs):
            dl = sum(tf.values()) or 1
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    score += self._idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def load_folder(folder: str | Path, max_words: int = 180, overlap: int = 40) -> BM25Index:
    """Index every .md / .txt file under a folder."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Docs folder not found: {root}")
    index = BM25Index()
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            index.add(chunk_text(text, doc_id=path.relative_to(root).as_posix(), max_words=max_words, overlap=overlap))
    return index
