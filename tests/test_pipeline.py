"""Run from project root: python -m pytest -q"""
from io import BytesIO

import pytest
from pypdf import PdfWriter

from document_pipeline import Passage, PdfSearchIndex, parse_pdf, split_text, offline_answer


def test_chunk_overlap_and_last_chunk():
    text = " ".join(f"word{i}" for i in range(52))
    pieces = split_text(text, chunk_words=30, overlap=10)
    assert len(pieces) == 3
    assert pieces[0].split()[-10:] == pieces[1].split()[:10]
    assert pieces[-1].split()[-1] == "word51"


def test_pdf_search_ranks_matching_topic():
    passages = [
        Passage("biology.pdf", 2, "Photosynthesis converts sunlight into sugars in plants."),
        Passage("chemistry.pdf", 7, "Ionic bonds involve electron transfer between atoms."),
    ]
    index = PdfSearchIndex(passages)
    results = index.search("How do plants use sunlight during photosynthesis?")
    assert results
    assert results[0].passage.filename == "biology.pdf"
    assert "biology.pdf" in offline_answer(results)


def test_unmatched_query_returns_no_sources():
    index = PdfSearchIndex([Passage("test.pdf", 1, "An apple contains vitamin C")])
    assert index.search("neutron star magnetosphere") == []


def test_invalid_pdf_is_rejected():
    with pytest.raises(ValueError, match="not a valid PDF"):
        parse_pdf(b"not a PDF", "wrong.pdf")


def test_blank_pdf_has_no_text_chunks():
    writer = PdfWriter()
    writer.add_blank_page(width=400, height=500)
    buf = BytesIO()
    writer.write(buf)
    assert parse_pdf(buf.getvalue(), "empty.pdf") == []
