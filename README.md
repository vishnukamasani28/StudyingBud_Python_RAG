# 📚 StudyingBud — AI-powered PDF Study Assistant

A Python reconstruction of the **behavior shown in** [this Studyingbud GitHub reference](https://github.com/vishnukamasani28/Studyingbud-): a dark-themed study chatbot that answers questions about your PDFs using grounded explanations. The reference repository contains a screenshot and minimal README but **no implementation source code**; this is an original implementation, not a copy of the Botpress app.

## Features

- Upload multiple **text-based** study PDFs.
- Extract text and tag it with its original **document name and page number**.
- Split text into overlapping passages to keep context.
- Use **TF-IDF vectorization + cosine similarity (machine-learning based document retrieval)** to find evidence relevant to each question.
- Two answer modes:
  - **Offline evidence search** — works immediately; quotes the most relevant PDF passages with page references (no API key or LLM required).
  - **Local AI via Ollama** — runs a model (e.g. Llama 3.2) on your computer to synthesize a conversational answer from the retrieved PDF passages with citations.
- Chat history; download a study session as Markdown.
- Lightweight dark Streamlit user interface.

## Tech stack

| Area | Tool |
|---|---|
| Language | Python 3.10+ |
| App/UI | Streamlit |
| PDF parsing | pypdf |
| Retrieval | scikit-learn, TF-IDF, cosine similarity |
| Optional local model | Ollama / Llama 3.2 |

## Project layout

```text
StudyingBud_Python_RAG/
├── app.py                  # Streamlit UI and chat session management
├── document_pipeline.py    # PDF extraction, chunking, retrieval and offline answer
├── llm_client.py           # Optional grounded generation using local Ollama
├── requirements.txt
├── tests/
│   └── test_pipeline.py
├── .gitignore
└── README.md
```

## Quick start (Windows / VS Code / PowerShell)

1. Install Python 3.10 or newer.
2. Open the downloaded folder in VS Code and open Terminal.
3. Enter:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks script activation, you can **skip activation** and run these instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

4. Streamlit opens a browser (usually `http://localhost:8501`).
5. Upload PDFs from the sidebar; select **Offline evidence search** to try it immediately.
6. Ask, for example, “Explain photosynthesis based on my notes” or “What is the main conclusion?”

## Enable conversational AI with Ollama (optional)

1. Install **Ollama** from https://ollama.com/download for your operating system.
2. In a *separate terminal*, download a suitable model:

```powershell
ollama pull llama3.2
```

3. Start the Ollama desktop app, or run `ollama serve` if needed.
4. In the StudyingBud sidebar, choose **Local AI (Ollama)**.
5. Ask a question; relevant excerpts will be provided to the local model.

**Hardware:** `llama3.2` requires a model download and several GB of available memory. Use a smaller Ollama model if your computer has limited RAM, and enter its model name in the sidebar.

No paid API key is required. If Ollama is unavailable, the app falls back to evidence search.

## How it works (RAG architecture)

```text
      Student uploads a PDF
               ↓
     Read text + page numbers (pypdf)
               ↓
        Chunk document text
               ↓
    TF-IDF vectorize / build index
               ↓
         Student asks a question
               ↓
        Cosine-similarity search
               ↓
        Top matching PDF chunks
               ↓
    ┌──────────┴─────────────┐
    ↓                        ↓
Offline excerpts      Ollama LLM + context
    ↓                        ↓
  Page citations     Grounded answer + citations
```

## Academic project explanation

**Problem statement:** Students have lengthy course notes and often struggle to find specific concepts quickly.

**Objective:** Create a PDF-based study assistant that retrieves relevant passages and, optionally, generates answers constrained to uploaded sources.

**ML component:** TF-IDF turns text into numerical feature vectors. Cosine similarity scores question-to-passage relevance. The optional language model generates natural-language explanations from top retrieved passages.

**Evaluation:** Create 20–30 questions for one study PDF and manually check (1) whether the correct page appears in top-3 retrieval, (2) whether model answers are supported by citations, and (3) average response time. Record results before reporting any accuracy metrics.

**Further enhancements:** Add semantic embeddings with SentenceTransformers/FAISS, quiz creation, flashcards, persistent user accounts, feedback buttons, multilingual Q&A, and deployment with Azure App Service.

## Important limitations

- Scanned/image-only PDFs contain no extractable text; use OCR to convert them first. This starter does not implement OCR.
- Offline mode is **retrieval**, not a generative chatbot. For explanation-style answers, enable Ollama.
- TF-IDF lexical retrieval can miss semantic paraphrases that do not share wording with the PDF.
- Generated AI responses may still be wrong. Verify all cited statements with the source pages, especially for exams or decisions.
- Files remain in application memory during the session. There is no sign-in, upload storage, or multi-user privacy isolation beyond standard Streamlit session handling, so deploy responsibly.
- The provided project is for development/classroom use, not a production-hardened service.

## Run the tests

Install pytest if needed, then run:

```powershell
python -m pip install pytest
python -m pytest -q
```

## Attribution

Inspired by the visual concept in `vishnukamasani28/Studyingbud-` (screenshot marked "by Botpress"). This source code was written independently using Python, Streamlit, pypdf and scikit-learn. Your academic write-up should distinguish between the referenced screenshot and your own development work.
