from pathlib import Path

import pytest

from retrieval import BM25Index, chunk_text, load_folder, tokenize


def test_tokenize_drops_stopwords_and_case():
    assert tokenize("The Lease TERM is 36 months") == ["lease", "term", "36", "months"]


def test_chunk_text_overlaps_and_covers_all_words():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, "doc.md", max_words=100, overlap=20)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]
    assert "w499" in chunks[-1].text


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", "d", max_words=10, overlap=10)


def test_bm25_ranks_relevant_chunk_first():
    index = BM25Index()
    index.add(chunk_text("Early termination of a lease costs three monthly payments.", "lease.md"))
    index.add(chunk_text("Our office is open Monday to Friday.", "office.md"))
    index.add(chunk_text("Credit scores above 700 are approved automatically.", "credit.md"))
    hits = index.search("what does early lease termination cost?")
    assert hits and hits[0][0].doc_id == "lease.md"


def test_search_with_no_matches_is_empty():
    index = BM25Index()
    index.add(chunk_text("alpha beta gamma", "a.md"))
    assert index.search("zzz") == []
    assert index.search("the and of") == []


def test_load_folder_indexes_sample_docs():
    index = load_folder(Path(__file__).parent / "sample_docs")
    docs = {c.doc_id for c in index.chunks}
    assert {"leasing_policy.md", "credit_policy.md"} <= docs
    top = index.search("minimum credit score for automatic approval", k=1)[0][0]
    assert top.doc_id == "credit_policy.md"
