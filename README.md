# Career Intelligence Assistant

> **Analyse your resume against job descriptions using a fully local, zero-cost AI stack.**  
> No OpenAI API key. No cloud spend. Runs entirely on your laptop.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)
![Ollama](https://img.shields.io/badge/Ollama-Llama_3.2_3B-black)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)

---

## What It Does

Upload your resume and one or more job descriptions. Ask questions in plain English:

| Query | What happens |
|---|---|
| *"What are my strongest technical skills?"* | Resume retrieval → LLM answer with source citations |
| *"What skills am I missing for job_1?"* | Cross-collection gap analysis → ✅ PRESENT / ⚠️ PARTIAL / ❌ MISSING |
| *"Give me an overall fit score for job_1"* | Weighted 0–100 score with skill breakdown |
| *"Which job fits me best overall?"* | Multi-JD ranking across all uploaded descriptions |
| *"What interview questions should I prepare for job_1?"* | Role-specific prep grounded in the JD and your actual background |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│                   localhost:8501                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────────┐
│                      FastAPI Backend                            │
│                   localhost:8000                                 │
│                                                                  │
│  /ingest  ──►  Loader → Chunker → Embedder → ChromaDB           │
│  /chat    ──►  Retriever → GapAnalyser → PromptBuilder → LLM    │
│  /health  ──►  Status of Ollama + ChromaDB + indexed docs        │
└────────┬──────────────────────────────┬────────────────────────-┘
         │                              │
┌────────▼────────┐          ┌──────────▼────────┐
│    ChromaDB     │          │      Ollama        │
│  (file-based)   │          │  localhost:11434   │
│                 │          │                    │
│  'resume'       │          │  llama3.2:3b       │
│  collection     │          │  (chat + intent)   │
│                 │          │                    │
│  'jobs'         │          │  nomic-embed-text  │
│  collection     │          │  (embeddings)      │
└─────────────────┘          └────────────────────┘
```

### Why a dual-collection design?

Most RAG tutorials dump everything into one index. That breaks for cross-document comparison queries — "What am I missing for Job #2?" can't be answered by naive top-k retrieval across a merged index.

Two separate ChromaDB collections (`resume` + `jobs`) means:
- Independent MMR retrieval per collection
- Filter JD results by `jd_id` with zero false positives  
- Clear one without touching the other (new resume upload)
- Maps directly to Pinecone namespaces for production migration

---

## Key Engineering Decisions

### 1. Three-Pass Hybrid Chunker

Fixed 500-token splitters fail on resumes because sections have wildly different semantic density. The chunker runs three passes:

1. **Section detection** — regex identifies headings (`EXPERIENCE`, `REQUIREMENTS`, `SKILLS` etc.) and creates section-scoped boundaries
2. **Recursive semantic splitting** — splits within sections using `\n\n → \n → . → space` separators, preserving sentence coherence
3. **Min-size enforcement** — drops fragments under 50 tokens, merges tiny adjacent chunks from the same section

Every chunk carries `{section, page, token_count, doc_type, jd_id}` metadata — retrieval can filter by section, not just similarity.

### 2. Pre-computed Gap Analysis (`gap_analyser.py`)

The gap analysis doesn't ask the LLM to figure out what's missing from raw text. Instead:

1. Skills manifests are extracted at **ingest time** (one structured LLM call per document)
2. At query time, `gap_analyser.py` runs a three-tier skill matching algorithm:
   - **Tier 1** — exact match against resume skills list
   - **Tier 2** — Jaccard token overlap ≥ 0.6 (catches "AWS Lambda" vs "AWS")
   - **Tier 3** — keyword presence in full resume text (contextual mention)
3. A weighted fit score is computed: 60% technical skills + 20% soft skills + 20% experience years
4. The structured result (`✅ PRESENT / ⚠️ PARTIAL / ❌ MISSING` with evidence) is injected into the prompt as pre-computed context

This means the LLM focuses on narrative quality — the hard comparison is already done.

### 3. Five-Intent Query Router

Every query is classified into one of five intents before retrieval:

| Intent | Retrieval strategy |
|---|---|
| `resume_lookup` | Resume collection only |
| `jd_lookup` | Jobs collection, filtered by jd_id |
| `gap_analysis` | Both collections + pre-computed GapReport |
| `fit_score` | Both collections + multi-JD ranking |
| `interview_prep` | Both collections, interview format instruction |

This prevents wasting token budget on irrelevant context — a resume question doesn't need JD chunks injected.

### 4. Three-Layer Guardrails

- **Pre-call** — validates resume/JD presence before any LLM call, checks jd_id exists, rejects empty queries
- **In-prompt** — `CONSTRAINT BLOCK` in every system prompt: "Never invent job titles, companies, dates or skills not in the provided context"
- **Post-call** — lightweight grounding verifier checks if the response made claims not traceable to retrieved chunks; adds a `⚠️` caveat if flagged

### 5. Composable Prompt Architecture

The system prompt is assembled from three blocks per request:
- `PERSONA BLOCK` — static, defines role ("Career Intelligence Analyst, reason from evidence")
- `CONTEXT BLOCK` — dynamic, injected retrieved chunks + skills manifests + pre-computed gap report
- `CONSTRAINT BLOCK` — static, grounding rules applied to the specific evidence just provided

Format instructions vary by intent — gap analysis gets `✅/⚠️/❌` structure, fit score gets a numeric breakdown, interview prep gets a bullet list.

---

## Tech Stack

| Layer | Local (this repo) | Production equivalent |
|---|---|---|
| LLM | Ollama + Llama 3.2 3B | OpenAI GPT-4o-mini |
| Embeddings | nomic-embed-text (768-dim) | text-embedding-3-small (1536-dim) |
| Vector DB | ChromaDB persistent (file) | Pinecone Serverless |
| API | FastAPI + Uvicorn | FastAPI on ECS Fargate |
| UI | Streamlit | Streamlit on ECS Fargate |
| Observability | Structured JSON logs | CloudWatch + X-Ray |
| Session memory | In-process sliding window | ElastiCache Redis |

Swapping local → production is **two env var changes**. The application code is identical.

---

## Project Structure

```
career-intelligent-assistant/
├── app/
│   ├── main.py                    # FastAPI entry point — 6 routes
│   ├── config.py                  # Pydantic-settings — all config from env vars
│   ├── models.py                  # Request/response contracts
│   ├── ingestion/
│   │   ├── loader.py              # PyMuPDF PDF + DOCX + TXT loader
│   │   ├── chunker.py             # Three-pass hybrid chunker
│   │   ├── embedder.py            # Ollama nomic-embed-text client
│   │   └── skill_extractor.py     # Structured LLM skill extraction at ingest
│   ├── retrieval/
│   │   ├── vector_store.py        # ChromaDB dual-collection manager + MMR
│   │   ├── retriever.py           # Intent classifier + query router
│   │   └── gap_analyser.py        # Cross-collection comparison engine
│   ├── llm/
│   │   ├── client.py              # Ollama chat client (OpenAI-compatible interface)
│   │   ├── prompt_builder.py      # Composable PERSONA + CONTEXT + CONSTRAINT
│   │   ├── guardrails.py          # Pre/post call grounding checks
│   │   └── memory.py              # Sliding window conversation memory
│   └── observability/
│       └── logger.py              # Structured JSON logging
├── ui/
│   └── streamlit_app.py           # Frontend — upload, chat, source citations
├── tests/
│   ├── test_chunker.py            # 12 unit tests — no LLM required
│   ├── test_guardrails.py         # 12 unit tests — fully mocked
│   └── test_retriever.py          # 11 unit tests — pure regex
├── Dockerfile                     # Multi-stage API image (~180MB)
├── Dockerfile.streamlit           # Multi-stage UI image (~130MB)
├── docker-compose.yml             # One-command startup
├── requirements.txt               # Pinned dependencies
└── .env.example                   # Config template
```

---

## Quick Start

### Prerequisites

- Windows 11 / macOS / Linux
- Python 3.11
- [Ollama](https://ollama.com) installed and running
- Docker Desktop (optional — can run locally without it)

### 1. Pull the models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 2. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/career-intelligent-assistant.git
cd career-intelligent-assistant

cp .env.example .env
# Edit .env if needed — defaults work out of the box
```

### 3. Run with Docker (recommended)

```bash
docker compose up --build
```

### 3. Or run locally

```bash
python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

# Terminal 1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2
streamlit run ui/streamlit_app.py --server.port 8501
```

### 4. Open the app

| URL | What |
|---|---|
| http://localhost:8501 | Streamlit UI |
| http://localhost:8000/health | API health check |
| http://localhost:8000/docs | Swagger interactive docs |

---

## Running Tests

Tests run without Ollama or ChromaDB — pure unit tests with mocked dependencies:

```bash
pytest tests/ -v
```

```
tests/test_chunker.py::TestChunkDocument::test_resume_produces_chunks     PASSED
tests/test_chunker.py::TestChunkDocument::test_section_detection_on_resume PASSED
tests/test_chunker.py::TestChunkDocument::test_no_empty_chunks             PASSED
... (35 tests total)
```

---

## API Reference

### `POST /ingest`
Upload a resume or job description.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@resume.pdf" \
  -F "doc_type=resume"

curl -X POST http://localhost:8000/ingest \
  -F "file=@job_description.pdf" \
  -F "doc_type=jd" \
  -F "jd_id=job_1"
```

### `POST /chat`
Ask a question about your resume and/or job fit.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What skills am I missing for job_1?",
    "session_id": "my-session-123"
  }'
```

Response includes `answer`, `intent`, `sources` (retrieved chunks), and `tokens_used`.

### `GET /health`
```json
{
  "status": "ok",
  "ollama_reachable": true,
  "chroma_reachable": true,
  "resume_indexed": true,
  "jobs_indexed": 1
}
```

### `DELETE /reset`
Clear all indexed documents and start fresh.

---

## Production Path (AWS)

The local stack maps directly to AWS with two env var changes:

```
OLLAMA_BASE_URL    → AWS-hosted inference endpoint (or OpenAI)
CHROMA_PERSIST_DIR → removed; use PINECONE_API_KEY + PINECONE_ENV
```

Full production architecture:
- **ECS Fargate** — stateless API + UI containers, auto-scaling
- **Pinecone Serverless** — managed vector DB, same namespace pattern as ChromaDB collections  
- **ElastiCache Redis** — session memory (swap `memory.py` backend)
- **S3** — document storage with pre-signed upload URLs
- **CloudWatch + X-Ray** — structured logs already emitted in JSON
- **WAF** — rate limiting (100 req/min per IP)

Estimated cost at 100 active users/month: ~$130–220 USD.

---

## What I'd Do With More Time

In priority order:

1. **Evaluation harness** — synthetic dataset of 50 resume+JD pairs with ground-truth gap labels; measure retrieval precision@5 and answer grounding rate on every PR. Should have been built before retrieval code, not after.

2. **Hybrid retrieval** — BM25 + dense vectors. Keyword-heavy queries like "AWS Lambda 3 years" are poorly served by pure dense retrieval. `rank_bm25` + ChromaDB with score fusion would improve recall meaningfully.

3. **Structured citations** — replace the post-call grounding verifier with OpenAI JSON mode + `cite_chunk_id` fields. Lower latency, lower cost, more auditable than a second LLM call.

4. **Streaming responses** — FastAPI SSE + Streamlit `st.write_stream`. Long gap analysis answers feel slow at 800 tokens; streaming cuts perceived latency to near-zero.

5. **Multi-user isolation** — Auth (Clerk/Cognito) + per-user ChromaDB namespaces. Currently all users share one resume collection.

---

## What I'd Do Differently

- **Evaluation before retrieval code** — retrofitting evals is harder than building them in from the start
- **No LangChain** — even minimal usage adds abstraction. Direct Ollama SDK + ChromaDB client gives full visibility and simpler debugging
- **Hybrid retrieval from day one** — pure dense retrieval misses keyword-heavy skill queries
- **OpenAI Structured Outputs** for skill extraction from day one — eliminates JSON parsing edge cases entirely

---

## How I Used AI Tools

I used Claude as a pair programmer, not as an author. The split:

| What I wrote myself | What AI assisted with |
|---|---|
| Architecture decisions | Pydantic model skeletons |
| Chunking strategy and rationale | FastAPI route scaffolding |
| Prompt design and constraints | ChromaDB collection init pattern |
| Gap analysis algorithm | Error handling boilerplate |
| This README | Dockerfile boilerplate |
| Trade-off analysis | pytest fixture scaffolding |

Every AI suggestion was reviewed as if a junior wrote it — checked for edge cases, verified imports, confirmed it actually solved the problem. I never accepted AI-generated architecture without whiteboarding first, and I never let AI write the system prompt (requires domain judgment).

---

## License

MIT
