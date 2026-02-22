"""
ui/streamlit_app.py — Career Intelligence Assistant frontend.

Talks to the FastAPI backend over HTTP — no direct imports of app modules.
This keeps the UI fully decoupled from the backend (can run independently).

Layout:
  Sidebar  — document upload, system status, reset button
  Main     — chat interface with source citations
"""
import uuid
import httpx
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"   # Overridden by env var in Docker

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Career Intelligence Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ───────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = {"has_resume": False, "job_ids": []}


# ── API helper functions ───────────────────────────────────────────────────────

def get_health() -> dict:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=5.0)
        return r.json()
    except Exception:
        return {"status": "error", "ollama_reachable": False, "chroma_reachable": False,
                "resume_indexed": False, "jobs_indexed": 0}


def get_docs_status() -> dict:
    try:
        r = httpx.get(f"{API_URL}/docs-status", timeout=5.0)
        return r.json()
    except Exception:
        return {"has_resume": False, "job_ids": []}


def ingest_file(file_bytes: bytes, filename: str, doc_type: str, jd_id: str | None) -> dict:
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    data = {"doc_type": doc_type}
    if jd_id:
        data["jd_id"] = jd_id
    r = httpx.post(f"{API_URL}/ingest", files=files, data=data, timeout=300.0)
    r.raise_for_status()
    return r.json()


def send_chat(query: str, jd_id: str | None = None) -> dict:
    payload = {
        "query": query,
        "session_id": st.session_state.session_id,
        "jd_id": jd_id,
    }
    r = httpx.post(f"{API_URL}/chat", json=payload, timeout=180.0)
    r.raise_for_status()
    return r.json()


def reset_all() -> None:
    httpx.delete(f"{API_URL}/reset", timeout=10.0)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎯 Career Intel")
    st.caption("Powered by Llama 3.2 · ChromaDB · FastAPI")
    st.divider()

    # ── System status ──────────────────────────────────────────────────────────
    st.subheader("System Status")
    health = get_health()

    col1, col2 = st.columns(2)
    with col1:
        ollama_icon = "🟢" if health.get("ollama_reachable") else "🔴"
        st.markdown(f"{ollama_icon} **Ollama**")
    with col2:
        chroma_icon = "🟢" if health.get("chroma_reachable") else "🔴"
        st.markdown(f"{chroma_icon} **ChromaDB**")

    if health.get("status") == "error":
        st.error("⚠️ Backend unreachable. Is the API running?")

    st.divider()

    # ── Document status ────────────────────────────────────────────────────────
    st.subheader("Indexed Documents")
    docs = get_docs_status()
    st.session_state.indexed_docs = docs

    resume_icon = "✅" if docs["has_resume"] else "❌"
    st.markdown(f"{resume_icon} Resume")

    if docs["job_ids"]:
        for jid in docs["job_ids"]:
            st.markdown(f"✅ {jid}")
    else:
        st.markdown("❌ No job descriptions")

    st.divider()

    # ── Resume upload ──────────────────────────────────────────────────────────
    st.subheader("Upload Resume")
    resume_file = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        key="resume_uploader",
        help="Upload your resume. Previous resume will be replaced.",
    )
    if resume_file and st.button("Ingest Resume", use_container_width=True):
        with st.spinner("Processing resume... (may take 20-30s on first run)"):
            try:
                result = ingest_file(
                    resume_file.getvalue(),
                    resume_file.name,
                    "resume",
                    None,
                )
                st.success(f"✅ {result['message']}")
                # Clear conversation memory on new document upload
                st.session_state.messages = []
                st.session_state.session_id = str(uuid.uuid4())
                st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.divider()

    # ── JD upload ──────────────────────────────────────────────────────────────
    st.subheader("Upload Job Description")
    jd_file = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        key="jd_uploader",
    )

    existing_ids = docs["job_ids"]
    next_num = len(existing_ids) + 1
    suggested_id = f"job_{next_num}"
    jd_id_input = st.text_input(
        "Job ID",
        value=suggested_id,
        help="Unique ID for this JD. Use in queries: 'What am I missing for job_1?'",
    )

    if jd_file and st.button("Ingest Job Description", use_container_width=True):
        if not jd_id_input.strip():
            st.error("Please enter a Job ID.")
        else:
            with st.spinner(f"Processing {jd_id_input}... (may take 20-30s)"):
                try:
                    result = ingest_file(
                        jd_file.getvalue(),
                        jd_file.name,
                        "jd",
                        jd_id_input.strip(),
                    )
                    st.success(f"✅ {result['message']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    st.divider()

    # ── Reset ──────────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear All Documents", use_container_width=True, type="secondary"):
        with st.spinner("Clearing..."):
            reset_all()
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")


# ── Main chat area ─────────────────────────────────────────────────────────────

st.title("Career Intelligence Assistant")
st.caption("Ask questions about your resume, job fit, skill gaps, or interview prep.")

# ── Example queries ────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.info(
        "**Get started:**\n"
        "1. Upload your resume in the sidebar\n"
        "2. Upload one or more job descriptions\n"
        "3. Ask questions below"
    )

    with st.expander("💡 Example questions to try"):
        examples = [
            "What are my strongest technical skills?",
            "What skills am I missing for job_1?",
            "How does my Python experience align with job_1?",
            "Which job is the best fit for me overall?",
            "What interview questions should I prepare for job_1?",
            "Give me an overall fit score for job_2.",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=False):
                st.session_state.pending_query = ex
                st.rerun()

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📄 Sources ({len(msg['sources'])} chunks)"):
                for src in msg["sources"]:
                    doc_label = f"Resume — {src['section']}" if src["doc_type"] == "resume" \
                        else f"{src.get('jd_id', 'JD')} — {src['section']}"
                    st.caption(f"**{doc_label}** (page {src['page']})")
                    st.text(src["text"])
                    st.divider()

# ── Query input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask about your resume, job fit, or skill gaps...")

# Handle example query clicks
if hasattr(st.session_state, "pending_query") and st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

if query:
    # Validate documents are uploaded
    if not st.session_state.indexed_docs["has_resume"] and not st.session_state.indexed_docs["job_ids"]:
        st.warning("Please upload at least a resume or job description first.")
        st.stop()

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... (Llama 3.2 3B may take 10-30 seconds)"):
            try:
                response = send_chat(query)
                answer = response["answer"]
                sources = response.get("sources", [])
                intent = response.get("intent", "unknown")
                tokens = response.get("tokens_used", 0)

                st.markdown(answer)

                # Show metadata in a subtle caption
                st.caption(f"Intent: `{intent}` · Tokens: `{tokens}`")

                # Show sources in an expander
                if sources:
                    with st.expander(f"📄 Sources ({len(sources)} chunks retrieved)"):
                        for src in sources:
                            doc_label = f"Resume — {src['section']}" if src["doc_type"] == "resume" \
                                else f"{src.get('jd_id', 'JD')} — {src['section']}"
                            st.caption(f"**{doc_label}** (page {src['page']})")
                            st.text(src["text"])
                            st.divider()

                # Save to session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except httpx.HTTPStatusError as e:
                error_msg = f"API error: {e.response.status_code} — {e.response.text}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except httpx.ConnectError:
                error_msg = "Cannot connect to the API. Make sure the backend is running."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
