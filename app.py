"""StudyingBud: document-grounded learning assistant. Run: streamlit run app.py"""
from __future__ import annotations

from hashlib import sha256

import streamlit as st

from document_pipeline import PdfSearchIndex, offline_answer, parse_uploads
from llm_client import answer_with_ollama


st.set_page_config(
    page_title="StudyingBud · Your AI Study Buddy",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #17243A;
}

/* Sidebar text */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #F8FAFC !important;
}

/* Main headings */
/* MAIN HEADINGS - BLACK */

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {
    color: #111111 !important;
}


/* MAIN PAGE TEXT */

[data-testid="stMain"] {
    color: #111111;
}


/* QUESTION INPUT BOX */

[data-testid="stChatInput"] {
    background-color: #F1F5F9 !important;
    border-radius: 12px;
}


/* QUESTION INPUT TEXT - BLACK */

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input,
[data-testid="stChatInput"] [contenteditable="true"] {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}


/* QUESTION PLACEHOLDER TEXT */

[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] input::placeholder {
    color: #555555 !important;
    opacity: 1;
}


/* PDF UPLOAD BUTTON TEXT - BLACK */

[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button * {
    color: #111111 !important;
}


/* PDF UPLOAD AREA TEXT */

[data-testid="stFileUploaderDropzone"] {
    color: #111111 !important;
}

[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    color: #111111 !important;
}

/* Chat input background */
[data-testid="stChatInput"] {
    background-color: #1E293B !important;
    border-radius: 12px;
}

/* Chat input text */
[data-testid="stChatInput"] textarea {
    color: #F8FAFC !important;
}

/* Chat placeholder */
[data-testid="stChatInput"] textarea::placeholder {
    color: #CBD5E1 !important;
}
/* ===================================== */
/* STUDYINGBUD SIDEBAR TEXT FIX */
/* ===================================== */

/* Sidebar Background */
[data-testid="stSidebar"] {
    background-color: #17243A !important;
    color: #FFFFFF !important;
}

/* Sidebar Headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* Sidebar Labels and Normal Text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* Sidebar Description and Captions */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #CBD5E1 !important;
    opacity: 1 !important;
}

/* Sidebar Radio Options */
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] p {
    color: #FFFFFF !important;
}

/* Slider Text */
[data-testid="stSidebar"] [data-testid="stSlider"] p {
    color: #FFFFFF !important;
}

/* Upload Area Background */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background-color: #FFFFFF !important;
    border-radius: 12px;
}

/* Upload Area Text - Black */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
[data-testid="stSidebar"] [data-testid="stFileUploader"] button,
[data-testid="stSidebar"] [data-testid="stFileUploader"] button * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    opacity: 1 !important;
}

/* Clear Conversation Button */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] *,
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] button * {
    color: #111827 !important;
    opacity: 1 !important;
}

/* Help Icons */
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {
    color: #CBD5E1 !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)
      

if "history" not in st.session_state:
    st.session_state.history = []
if "doc_key" not in st.session_state:
    st.session_state.doc_key = None
if "search_index" not in st.session_state:
    st.session_state.search_index = None
if "warnings" not in st.session_state:
    st.session_state.warnings = []

with st.sidebar:
    st.markdown("### 📚 StudyingBud")
    st.caption("Your personal PDF study assistant")
    files = st.file_uploader(
        "Upload your study notes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Up to 8 PDFs, 15 MB each, 200 pages per file. Scanned PDFs need OCR first.",
    )
    mode = st.radio(
        "Answer mode",
        ["Offline evidence search", "Local AI (Ollama)"],
        help="Offline works immediately. Ollama produces AI explanations but needs a local model.",
    )
    model = "llama3.2"
    if mode == "Local AI (Ollama)":
        model = st.text_input("Ollama model name", value="llama3.2")
        st.caption("Setup: install Ollama, run `ollama pull llama3.2`.")
    top_k = st.slider("Number of source passages", 2, 6, 4)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.caption("🔒 Uploaded PDFs are processed in memory. In AI mode, PDF excerpts are sent only to Ollama on your computer.")

# Rebuild the search index only when uploads change, not on every chat message.
file_items = [(f.name, f.getvalue()) for f in files] if files else []
signature = tuple((name, sha256(content).hexdigest()) for name, content in file_items)
if signature != st.session_state.doc_key:
    st.session_state.doc_key = signature
    st.session_state.search_index = None
    st.session_state.warnings = []
    st.session_state.history = []  # Stop old answers being mistaken for current PDFs.
    if file_items:
        with st.spinner("Reading and indexing your PDFs..."):
            passages, warnings = parse_uploads(file_items)
            st.session_state.warnings = warnings
            if passages:
                try:
                    st.session_state.search_index = PdfSearchIndex(passages)
                except ValueError as exc:
                    st.session_state.warnings.append(str(exc))

index = st.session_state.search_index
with st.sidebar:
    if index:
        st.success(f"Ready: {len(index.passages)} searchable text chunks")
        st.caption("Documents: " + ", ".join(name for name, _ in file_items))
    for warning in st.session_state.warnings:
        st.warning(warning)
    if files and not index:
        st.info("No searchable text yet. Try a normal text-based PDF.")

st.markdown('<div class="brand">◧ &nbsp;Studying<span>Bud</span> ✦</div>', unsafe_allow_html=True)

if not st.session_state.history:
    st.markdown(
        '<div class="hero"><h1>Hello buddy 👋</h1>'
        '<p>Ask a question about your study PDFs for a clear, '
        'document-grounded explanation.</p></div>',
        unsafe_allow_html=True,
    )
    if not index:
        st.info("To get started, upload one or more text-based PDFs using the sidebar.")
    else:
        st.markdown('<p class="hint">Try: “Summarize chapter 1” · “What are the key definitions?” · “Explain this concept with an example.”</p>', unsafe_allow_html=True)

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message.get("sources"):
            with st.expander("📎 Sources from your PDF notes"):
                for number, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**[{number}] {source['filename']} — page {source['page']}**")
                    st.caption(source["excerpt"])

question = st.chat_input(
    "Type your study question...",
    disabled=index is None,
    max_chars=2500,
)

if question and index:
    st.session_state.history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    hits = index.search(question, top_k=top_k)
    with st.chat_message("assistant"):
        with st.spinner("Finding answers in your notes..."):
            if mode == "Local AI (Ollama)" and hits:
                try:
                    answer = answer_with_ollama(question, hits, model=model.strip() or "llama3.2")
                except RuntimeError as exc:
                    answer = f"⚠️ {exc}\n\n" + offline_answer(hits)
            else:
                answer = offline_answer(hits)
        st.markdown(answer)
        sources = [
            {"filename": hit.passage.filename, "page": hit.passage.page,
             "excerpt": hit.passage.text[:700] + ("…" if len(hit.passage.text) > 700 else "")}
            for hit in hits
        ]
        if sources:
            with st.expander("📎 Sources from your PDF notes"):
                for number, source in enumerate(sources, start=1):
                    st.markdown(f"**[{number}] {source['filename']} — page {source['page']}**")
                    st.caption(source["excerpt"])
    st.session_state.history.append({"role": "assistant", "text": answer, "sources": sources})

if st.session_state.history:
    export = "# StudyingBud study session\n\n" + "\n\n".join(
        f"## {m['role'].title()}\n\n{m['text']}" for m in st.session_state.history
    )
    with st.sidebar:
        st.download_button(
            "⬇️ Download chat notes (.md)",
            data=export,
            file_name="studyingbud_chat_notes.md",
            mime="text/markdown",
            use_container_width=True,
        )
