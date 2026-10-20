"""PDF parsing, chunking and lightweight ML-based retrieval for StudyingBud."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
from typing import Iterable

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MAX_FILE_BYTES = 15 * 1024 * 1024   # 15 MB per file; starter-project safety limit
MAX_FILES = 8
MAX_PAGES_PER_FILE = 200


@dataclass(frozen=True)
class Passage:
    filename: str
    page: int  # 1-indexed
    text: str


@dataclass(frozen=True)
class SearchHit:
    passage: Passage
    score: float


def split_text(text: str, chunk_words: int = 170, overlap: int = 35) -> list[str]:
    """Chunk page text while retaining context across adjacent chunks."""
    if chunk_words < 20 or not 0 <= overlap < chunk_words:
        raise ValueError("Chunk size must be >= 20 and overlap must be smaller.")
    words = re.sub(r"\s+", " ", text).strip().split()
    if not words:
        return []
    pieces = []
    step = chunk_words - overlap
    for start in range(0, len(words), step):
        portion = words[start:start + chunk_words]
        if not portion:
            break
        pieces.append(" ".join(portion))
        if start + chunk_words >= len(words):
            break
    return pieces


def parse_pdf(data: bytes, filename: str) -> list[Passage]:
    """Return searchable text chunks from a PDF; never write uploaded bytes to disk."""
    if not data.startswith(b"%PDF-"):
        raise ValueError(f"{filename}: not a valid PDF file.")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"{filename}: file is larger than 15 MB.")
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise ValueError(f"{filename}: password-protected PDFs are not supported.")
        if len(reader.pages) > MAX_PAGES_PER_FILE:
            raise ValueError(f"{filename}: PDF exceeds the 200-page limit.")
        passages: list[Passage] = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text(extraction_mode="plain") or ""
            except Exception:  # Skip damaged or unsupported pages, not entire document
                continue
            for chunk in split_text(text):
                passages.append(Passage(filename=filename, page=page_num, text=chunk))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{filename}: could not read this PDF ({type(exc).__name__}).") from exc
    return passages


def parse_uploads(items: Iterable[tuple[str, bytes]]) -> tuple[list[Passage], list[str]]:
    """Process (filename, PDF bytes); return passages and informative warnings."""
    files = list(items)
    if len(files) > MAX_FILES:
        raise ValueError(f"Upload at most {MAX_FILES} PDFs at a time.")
    passages: list[Passage] = []
    warnings: list[str] = []
    for filename, data in files:
        try:
            chunks = parse_pdf(data, filename)
            if chunks:
                passages.extend(chunks)
            else:
                warnings.append(f"{filename}: no selectable text found (may be a scanned PDF).")
        except ValueError as exc:
            warnings.append(str(exc))
    return passages, warnings


class PdfSearchIndex:
    """TF-IDF retrieval over uploaded PDF passages; works without an API key."""

    def __init__(self, passages: list[Passage]) -> None:
        if not passages:
            raise ValueError("No text was extracted from the PDFs.")
        self.passages = passages
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
            max_features=80_000,
        )
        try:
            self.matrix = self.vectorizer.fit_transform([p.text for p in passages])
        except ValueError as exc:
            raise ValueError("The extracted text contains no searchable words.") from exc

    def search(self, question: str, top_k: int = 4) -> list[SearchHit]:
        if not question.strip():
            return []
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.matrix).ravel()
        best = similarities.argsort()[::-1][:max(1, top_k)]
        return [
            SearchHit(self.passages[i], float(similarities[i]))
            for i in best if float(similarities[i]) > 0.005
        ]


def offline_answer(hits: list[SearchHit]) -> str:
    """Evidence-first fallback when no local language model is running."""
    if not hits:
        return (
            "I couldn't find relevant text in the uploaded PDFs. "
            "Try asking with specific terms that appear in your notes."
        )
    excerpts = []
    for number, hit in enumerate(hits[:3], start=1):
        short = hit.passage.text[:520].rstrip()
        if len(hit.passage.text) > 520:
            short += "…"
        excerpts.append(f"**[{number}] {hit.passage.filename}, page {hit.passage.page}**\n\n> {short}")
    return (
        "I found these relevant passages in your study PDFs:\n\n"
        + "\n\n".join(excerpts)
        + "\n\n*This is evidence-search mode (not an AI-generated explanation). "
          "Choose **Local AI (Ollama)** to get a synthesized answer.*"
    )
