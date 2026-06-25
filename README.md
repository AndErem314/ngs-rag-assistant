# NGS‑RAG Assistant

A **local, privacy‑friendly RAG (Retrieval‑Augmented Generation) assistant** for lab technicians working with NGS sample preparation protocols.  
It ingests PDF user manuals (e.g., Illumina TruSight Oncology 500), answers natural language questions with source citations, and generates structured Markdown reports.

Everything runs **fully locally** via Ollama — no API keys, no internet connection required, no data leaves your machine.

Built with:

- **Ollama** — local embeddings (`haybu/mxbai-embed-large-latest:latest`) and generation (`llama3.1:8b`)
- **Streamlit** — interactive web UI
- **ChromaDB** — persistent vector storage with content-hash deduplication
- **PDFPlumber** — page-by-page text extraction from PDFs

---

## ✨ Features

- Upload up to 5 PDF manuals per session
- Ask natural language questions and receive answers with **source citations** (filename + page number + relevance score)
- Tune retrieval quality with **Top-K** and **Max distance** sliders directly in the UI
- Generate a **structured Markdown report** covering key protocol topics (input amounts, shearing settings, reagent storage, PCR cycles, QC criteria, safety notes)
- Download the report as a `.md` file
- **Re-ingest safely** — uploading the same PDF twice updates records in place; no duplicates are created
- Ollama **connectivity check** built into the sidebar with a live status indicator
- **4 Chunking Strategies** — basic, table-aware, semantic, keyword-anchored (NGS-optimized)
- **Hybrid Search** — combines vector similarity (60%) + BM25 keyword matching (40%)
- **Dual Backend** — ChromaDB (primary) + pgvector (PostgreSQL, enterprise-ready)
- **Expanded Testing Framework** — 45+ unit tests + adversarial/bias/drift tests
- **Observability Dashboard** — Streamlit dashboard with KPIs, time-series charts, latency tracking

---

## 🗂 Project Structure

```
ngs-rag-assistant/
├── README.md
├── requirements.txt
├── .env.example
├── docker-compose.yml          # optional (pgvector + optional Prometheus/Grafana)
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── pdf_parser.py       # page-by-page text extraction (pdfplumber)
│   │   ├── chunker.py          # 4 strategies: basic, table_aware, semantic, keyword
│   │   └── chunking_strategy.py # ChunkingStrategy enum
│   ├── embeddings/
│   │   └── embedder.py         # OllamaEmbedder — host-aware ollama.Client
│   ├── retrieval/
│   │   ├── vector_store.py     # ChromaDB persistence, hash IDs, upsert, hybrid search
│   │   ├── pgvector_store.py   # PostgreSQL/pgvector backend (enterprise)
│   │   └── query_processor.py  # embed question → search → return context + metadata
│   ├── generation/
│   │   └── llm_client.py       # OllamaGenerator — chat() with system/user roles
│   ├── report/
│   │   └── report_builder.py   # run predefined questions → compile Markdown report
│   ├── ui/
│   │   └── streamlit_app.py    # main Streamlit application
│   └── observability/          # Phase 7: Metrics & observability (NEW)
│       ├── __init__.py
│       ├── metrics.py           # MetricsCollector (SQLite), LatencyTimer
│       └── dashboard.py         # Streamlit observability dashboard
├── tests/
│   ├── test_ingestion.py       # Chunker tests
│   ├── test_retrieval.py       # VectorStore.search + retrieve_context tests
│   ├── test_chunker.py        # 4 chunking strategies, edge cases
│   ├── test_embedder.py       # OllamaEmbedder tests
│   ├── test_pgvector_store.py # pgvector integration tests (auto-skip without DB)
│   ├── test_adversarial.py     # Phase 6.2.A: 17 adversarial/malformed query tests (NEW)
│   └── test_bias.py           # Phase 6.2.C: 11 bias/fairness tests (NEW)
├── scripts/
│   ├── generate_questions.py   # GPT-4o-mini / Gemini question generation from PDF manuals
│   ├── test_retrieval_accuracy.py # Retrieval accuracy test (page-level, drift-integrated)
│   └── drift_monitor.py       # Phase 6.2.B: embedding + retrieval drift tracking (NEW)
├── validation/
│   └── questions/              # pre-generated Q&A sets for retrieval evaluation
│       ├── TruSight-Oncology-500-v2_questions.json
│       ├── TruSeq-DNA-PCR-Free_questions.json
│       ├── TruSeq-Nano-DNA_questions.json
│       ├── TruSeq-Stranded-Total-RNA_questions.json
│       └── Nextera-XT-DNA_questions.json
├── data/                       # place your PDFs here (gitignored)
├── observability/               # Phase 7: Metrics database (gitignored)
│   └── metrics.db             # SQLite database for historical metrics
└── notebooks/                  # experiment notebooks
```

---

## 🔁 Pipeline — how it works

Understanding the full pipeline helps you tune the system and diagnose issues.

### Ingestion (one-time per document)

```
PDF file
   │
   ▼
pdf_parser.py       — pdfplumber extracts text page by page;
                       each page becomes {"text": ..., "page": N}
   │
   ▼
chunker.py          — text is split into overlapping chunks
                       (default: 500 tokens, 50-token overlap)
                       metadata {"source": filename, "page": N} is attached
   │
   ▼
embedder.py         — each chunk is sent to Ollama's embeddings endpoint
                       using the haybu/mxbai-embed-large-latest:latest model;
                       returns a float vector per chunk
   │
   ▼
vector_store.py     — chunks + vectors are upserted into ChromaDB
                       using content-hash IDs (SHA-256 of source|page|text[:128]);
                       re-ingesting the same file updates in place — no duplicates
```

### Query (on every question)

```
User question
   │
   ▼
embedder.py         — question is embedded with the same model
   │
   ▼
vector_store.py     — cosine similarity search across stored chunks;
   │                   optional source_filter restricts to selected PDFs;
   │                   optional max_distance drops low-relevance chunks
   ▼
query_processor.py  — assembles context string + metadata list
                       (source, page, distance per chunk)
   │
   ▼
llm_client.py       — system prompt + context + question sent to Ollama
                       via ollama.chat() with explicit system/user roles;
                       model answers strictly from context and cites sources
   │
   ▼
Answer displayed in UI with expandable sources & relevance scores
```

### Report generation

```
For each of 7 predefined questions (see report_builder.py):
   → retrieve_context()  (same pipeline as above, top_k=5)
   → generator.answer_question()
   → append "## Question\n\nAnswer\n" to report

Final Markdown report displayed + available for download
```

The 7 predefined report questions are:

1. What is the minimum DNA/RNA input amount?
2. What are the recommended shearing settings for the Covaris instruments?
3. List all reagents and their storage temperatures from the kit boxes.
4. How many cycles are used in the index PCR?
5. What are the steps in the library preparation workflow?
6. What are the quality control criteria for DNA and RNA samples?
7. What are the important safety precautions or handling notes?

---

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Platform  | Any OS with Python 3.10+ | Mac with Apple Silicon (M1–M4) |
| RAM       | 16 GB   | 32 GB |
| Storage   | 10 GB free | 20 GB free |
| GPU       | CPU-only (slow) | Apple Metal / NVIDIA CUDA |

Both models run locally via Ollama. On Apple Silicon, Ollama automatically offloads computation to the GPU via Metal — no configuration needed. On NVIDIA machines, CUDA is used automatically if drivers are present.

---

## 🚀 Setup & Installation

### 1. Install Ollama

Download from [ollama.com](https://ollama.com/) and install. Then pull both required models:

```bash
# Embedding model
ollama pull haybu/mxbai-embed-large-latest:latest

# Generation model
ollama pull llama3.1:8b

# Optional: smaller/faster generation model
ollama pull phi3:mini
```

Start the Ollama server (it may already be running as a background service):

```bash
ollama serve
```

### 2. Clone the repository

```bash
git clone git@github.com:AndErem314/ngs-rag-assistant.git
cd ngs-rag-assistant
```

### 3. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# or:  .\venv\Scripts\activate  # Windows
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` if needed (all values have sensible defaults):

```env
# Ollama server (local by default)
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=haybu/mxbai-embed-large-latest:latest
LLM_MODEL=llama3.1:8b

# Only needed for the validation question-generation script (not the main app)
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

If Ollama runs on a different machine or port, update `OLLAMA_HOST` here. The UI also has an Ollama host text field that overrides this at runtime.

### 6. Run the app

```bash
streamlit run src/ui/streamlit_app.py
```

The app opens in your browser at **http://localhost:8501**.

---

## 📖 Usage Guide — step by step

### Step 1 — Verify Ollama connection

Open the sidebar. The **Ollama host** field shows the configured URL (default `http://localhost:11434`).  
Click **🔌 Check connection**. You should see a green *"Ollama is reachable"* message and a **● Ollama connected** status indicator.  
If you see an error, make sure `ollama serve` is running and the host URL is correct.

### Step 2 — Tune retrieval settings (optional)

Two sliders in the **Retrieval settings** section control retrieval quality:

| Setting | Default | What it does |
|---------|---------|--------------|
| **Top-K chunks** | 5 | How many candidate chunks are fetched from ChromaDB per query |
| **Max distance threshold** | 1.0 | Cosine distance upper bound — chunks above this are discarded as irrelevant |

**Tuning guidance:**
- Start with the defaults. If answers feel vague, lower Max distance to `0.7` to force higher-relevance chunks only.
- If answers come back empty ("No relevant chunks found"), raise Max distance toward `1.5` or increase Top-K.
- A distance of `0.0` = identical vector (perfect match); `1.0` = orthogonal (no similarity); `2.0` = opposite.

### Step 3 — Ingest PDFs

In the **Document ingestion** section:
1. Click the file uploader and select up to 5 PDF manuals.
2. Click **📥 Ingest PDFs**.

The app will show a progress bar and a per-file summary:
- ✅ `manual.pdf — 142 chunk(s) ingested.` — success
- ⚠️ Warning — some chunks failed to embed (Ollama may be slow or overloaded)
- ❌ Error — file could not be parsed or all embeddings failed

**Re-ingesting the same PDF is safe.** Content-hash IDs mean existing chunks are updated in place; no duplicates are created.

After ingestion, the **Active protocols** section appears in the sidebar listing each ingested filename with a checkbox. Uncheck a protocol to exclude it from all queries and report generation.

### Step 4 — Ask a question

Type a question in the **Ask a question** field and click **🔍 Ask**.

The answer appears below. Expand **📚 Sources & relevance scores** to see exactly which chunks were used, their page numbers, and their cosine distance scores (lower = more relevant).

If you see *"No relevant chunks found"*, try:
- Raising the Max distance threshold in the sidebar
- Rephrasing your question
- Confirming the correct protocols are checked

### Step 5 — Generate a report

Click **📊 Generate report** to run all 7 predefined questions against the selected protocols and compile a structured Markdown report.

This takes longer than a single question (one LLM call per question). When complete, the report is displayed inline and a **📥 Download as Markdown** button appears.

### Step 6 — Clear data

Click **🗑️ Clear collection** in the sidebar to delete all stored chunks from ChromaDB and reset the source list. Use this when switching to a different set of protocols or starting fresh.

---

## 💡 Example questions

```
What is the minimum DNA input recommended for TruSight Oncology 500?
How many cycles are used in the index PCR step?
What reagents are in box 2 and where should they be stored?
What Covaris shearing settings are recommended for the E220 evolution?
What genes can be detected by TSO500?
What is the storage temperature for the Library Normalization Beads?
What are the quality control thresholds for library quantification?
What are the safety precautions when handling the enzymatic reagents?
```

---

## 🧠 Model reference

| Model | Purpose | RAM (Q4) | Notes |
|-------|---------|----------|-------|
| `haybu/mxbai-embed-large-latest:latest` | Embeddings | ~1 GB | Top BEIR benchmark performance; 100+ languages; supports dimension cropping to 256/512 for memory efficiency |
| `llama3.1:8b` | Generation | ~6–8 GB | Best balance of speed and accuracy; strong instruction-following; recommended default |
| `phi3:mini` | Generation (alt) | ~3–4 GB | 3.8B parameters; use if RAM is limited; slightly lower answer quality |

To switch the generation model, update `LLM_MODEL` in `.env` before starting the app, or pass `model=` to `OllamaGenerator` directly.

---

## ⚙️ Key technical decisions

**Content-hash chunk IDs**  
Each chunk's ChromaDB ID is a 16-character SHA-256 hex digest of `source|page|text[:128]`. This makes IDs stable across sessions and safe for re-ingestion: the same chunk always gets the same ID, so upsert updates it in place rather than creating a duplicate or raising a key collision error.

**`upsert` instead of `add`**  
`VectorStore.add_chunks()` uses ChromaDB's `upsert` operation. This means ingesting the same PDF twice (or after editing it) is always safe — existing records are updated, new chunks are inserted, nothing is silently overwritten with a wrong index.

**`ollama.Client(host=host)` — explicit host binding**  
Both `OllamaEmbedder` and `OllamaGenerator` now instantiate a bound `ollama.Client` object rather than calling the module-level functions (`ollama.embeddings`, `ollama.generate`). This means the `OLLAMA_HOST` environment variable and the sidebar host field actually take effect. Previously the host parameter was stored but silently ignored.

**`ollama.chat()` with system/user roles**  
The generator uses `client.chat()` with a `system` role message instead of `client.generate()` with a concatenated prompt string. Chat-fine-tuned models like `llama3.1:8b` follow system instructions significantly more reliably with this approach — the model is more likely to stay within the provided context and format citations correctly.

**`max_distance` threshold**  
Retrieved chunks are filtered by cosine distance before being sent to the LLM. This prevents low-relevance chunks (noise) from degrading answer quality. The threshold is exposed as a sidebar slider so users can tune it without editing code.

---

## 🧪 Testing

This project uses **local testing only** (no CI/CD pipeline). Run the full test suite with:

```bash
pytest tests/ -v
```

**Unit tests** use `chromadb.EphemeralClient` (in-memory, no disk I/O) and mock `OllamaEmbedder` via `unittest.mock.MagicMock`, so **no Ollama server is required**.

**Integration tests** (pgvector) require a running PostgreSQL + pgvector instance. They auto-skip if the database is unavailable. To run them locally:

```bash
docker compose up -d pgvector
export NGS_PGVECTOR_URL=postgresql://ngs_user:***@localhost:5432/ngs_rag
pytest tests/test_pgvector_store.py -v
```

| Test file | What it covers |
|-----------|---------------|
| `test_ingestion.py` | `_make_chunk_id` stability and collision resistance; `add_chunks` happy path and ID determinism; re-ingestion deduplication (upsert regression); `clear_collection` |
| `test_retrieval.py` | `VectorStore.search` — top-K, source filter, max_distance, empty collection; `retrieve_context` — happy path, empty embedding (Ollama down), max_distance filtering, metadata shape |
| `test_chunker.py` | Page number assignment, chunk size boundaries, overlap behavior, **4 chunking strategies** |
| `test_embedder.py` | OllamaEmbedder initialization, embedding generation, error handling |
| `test_pgvector_store.py` | pgvector backend: init, add chunks, search, clear (auto-skips without DB) |
| `test_adversarial.py` | **Phase 6.2.A**: 17 tests for malformed queries, SQL injection, XSS, unicode, edge cases |
| `test_bias.py` | **Phase 6.2.C**: 11 tests for cross-protocol bias, dataset parity, version detection |

### Phase 6: Expanded Testing & Automation (✅ COMPLETE)

Phase 6 added AI-specific testing aligned with the **IDBS Principal AI Test Engineer** role requirements:

#### 6.2.A: Adversarial Testing (✅)
Tests system robustness against malicious/malformed inputs:
- Malformed queries (empty, whitespace, incomplete terms)
- Security attacks (SQL injection, XSS, null bytes, control characters)
- Edge cases (10k char queries, unicode/emoji, conflicting terms)
- **File**: `tests/test_adversarial.py` (17 test cases)

#### 6.2.B: Drift Monitoring (✅)
Tracks embedding and retrieval quality over time:
- Embedding drift detection (cosine similarity <0.95 triggers alert)
- Retrieval accuracy drift (page-level metrics across runs)
- Reference NGS texts for stability checks
- **File**: `scripts/drift_monitor.py` (logs to `drift_metrics.json`)

#### 6.2.C: Bias/Fairness Checks (✅)
Ensures no bias in retrieval across NGS datasets:
- Cross-protocol comparison (TruSight, TruSeq, Nextera)
- Dataset parity checks (underrepresented protocols)
- Version bias detection (v1 vs v2)
- **File**: `tests/test_bias.py` (11 test cases)

---

## 📊 Phase 7: Observability & Monitoring (🔶 IN PROGRESS)

Real-time metrics and dashboards for RAG pipeline health monitoring.

### 7.1 Observability Module

**Metrics Tracking:**
- **Retrieval metrics**: Precision@k, Recall@k, MRR, NDCG
- **Latency tracking**: Embedding time, search time, end-to-end query latency
- **Chunk quality**: Average chunk size, overlap percentage, table vs. text ratio

**Components:**
- ✅ **`src/observability/metrics.py`** — SQLite-based metrics collector
  - `MetricsCollector` class for logging retrieval, embedding, system metrics
  - `LatencyTimer` context manager for timing operations
  - Query methods: `get_retrieval_summary(days)`, `get_strategy_comparison()`
- ✅ **`src/observability/dashboard.py`** — Streamlit dashboard
  - KPI cards (total queries, accuracy, latency, distance)
  - Time-series charts for accuracy trends
  - Latency distribution histogram
  - Strategy comparison bar charts
  - Raw data explorer
- ✅ **Integration**: Metrics collection integrated into:
  - `scripts/drift_monitor.py` — logs retrieval + embedding metrics
  - `scripts/test_retrieval_accuracy.py` — logs retrieval metrics + latency

**Launch the dashboard:**
```bash
streamlit run src/observability/dashboard.py
```

**Query metrics database directly:**
```bash
sqlite3 observability/metrics.db "SELECT * FROM retrieval_metrics LIMIT 5;"
```

### 7.2 Bias/Drift Detection Module (⏳ PENDING)
- Automated weekly drift checks (`scripts/weekly_check.py`)
- Stakeholder dashboard ("RAG Health Score")
- Markdown reports with drift alerts and bias findings

---

## 📜 Phase 3: Enhanced Multi-Modal Chunking

The pipeline supports multiple chunking strategies for NGS-specific content:

| Strategy | Function | Best For |
|----------|----------|----------|
| `basic` | `chunk_document()` | Default; general text with `RecursiveCharacterTextSplitter` |
| `table_aware` | `chunk_document_table_aware()` | PDFs with tables; extracts tables as structured Markdown |
| `semantic` | `chunk_document_semantic()` | Topic-based splitting using embedding similarity |
| `keyword` | `chunk_document_keyword_anchored()` | NGS protocols; anchors on terms like DNA, RNA, PCR |

Switch strategy in your pipeline:
```python
from src.ingestion import chunk_document_with_strategy, ChunkingStrategy

chunks = chunk_document_with_strategy(
    pages=pages,
    source_filename="manual.pdf",
    pdf_path="data/manual.pdf",  # Required for table_aware
    strategy=ChunkingStrategy.TABLE_AWARE,
)
```

### Hybrid Search (Phase 4 Enhancement)

ChromDB now supports hybrid search combining vector similarity with BM25 keyword matching:

```python
from src.retrieval.vector_store import VectorStore

store = VectorStore()
results = store.search(
    query_embedding=embedding,
    top_k=5,
    hybrid=True,           # Enable hybrid search
    query_text="DNA input PCR",  # Required for BM25
)
```

Hybrid search gives 60% weight to vector similarity and 40% to keyword relevance — ideal for NGS queries where exact terms matter.

### Retrieval Accuracy Test (Phase 2)

Test how well the RAG pipeline retrieves the correct pages for NGS questions:

```bash
# Basic test (vector search only)
python scripts/test_retrieval_accuracy.py

# Test with table-aware chunking
python scripts/test_retrieval_accuracy.py --strategy table_aware

# Test with hybrid search
python scripts/test_retrieval_accuracy.py --hybrid

# Custom PDF and questions
python scripts/test_retrieval_accuracy.py \
    --pdf data/manual.pdf \
    --questions validation/questions/custom.json
```

The test reports:
- **Exact page match**: Expected page is in top-k results
- **Within tolerance**: Expected page within ±2 pages
- **Average distance**: Lower is better (cosine distance)

---

## 📜 Validation Script — `scripts/generate_questions.py`

This script generates a structured JSON question-and-answer set directly from one or more PDF manuals, using either **GPT-4o-mini** (OpenAI) or **Gemini Flash** (Google). Its primary purpose is to produce an evaluation dataset you can use to measure and tune the retrieval quality of the RAG pipeline.

> **Note:** This script calls an external API and requires either `OPENAI_API_KEY` or `GEMINI_API_KEY` in your `.env` file. It is only needed for evaluation — the main assistant runs fully locally without it.

### What it does

For each PDF you provide, the script:

1. Extracts the full text from the PDF (up to 120,000 characters to stay within LLM context limits).
2. Sends the text to the chosen LLM with a prompt asking it to generate 20–25 realistic questions a lab technician might ask.
3. For each question, the LLM also provides an expected short answer and the approximate page number in the manual where the answer can be found.
4. Saves the output as a JSON file in `validation/questions/`.

### Output format

Each JSON file is a list of objects with three fields:

```json
[
  {
    "question": "What is the minimum input amount for DNA and RNA samples in the TSO500v2 assay?",
    "expected_answer": "30 ng DNA and 40 ng RNA.",
    "source_page": 1
  },
  {
    "question": "What DV200 value is recommended for RNA sample quality assessment?",
    "expected_answer": "DV200 value of ≥20%.",
    "source_page": 1
  }
]
```

- `question` — the natural language question to ask the RAG assistant
- `expected_answer` — a short reference answer derived from the manual (used as ground truth for manual review)
- `source_page` — the approximate page in the PDF where the answer can be verified

A ready-made example for the TruSight Oncology 500 v2 manual is already included in the repository at `validation/questions/TruSight-Oncology-500-v2_questions.json`, so you can run a validation immediately without needing an API key.

### Usage

```bash
# Generate questions from a single PDF using OpenAI (default)
python scripts/generate_questions.py data/manual.pdf

# Generate from multiple PDFs
python scripts/generate_questions.py data/manual1.pdf data/manual2.pdf

# Use Gemini instead of OpenAI
python scripts/generate_questions.py --model gemini data/manual.pdf
```

Output files are saved to `validation/questions/<pdf_stem>_questions.json`.

### How to use it for retrieval evaluation

Once you have a question set, use it to systematically test whether the RAG pipeline retrieves the right chunks and generates accurate answers:

1. **Ingest the PDF** into the app as usual (Steps 1–3 in the Usage Guide above).
2. **Open the question set** (e.g., `TruSight-Oncology-500-v2_questions.json`).
3. For each entry, **paste the `question`** into the app's Q&A field and click **🔍 Ask**.
4. **Compare the answer** returned by the assistant against the `expected_answer` field. Also note which page(s) appear in the *Sources & relevance scores* expander — they should match or be close to `source_page`.
5. If the answers are incomplete or the wrong pages are cited, **adjust the retrieval settings**:
   - Increase **Top-K** to give the model more candidate chunks.
   - Lower **Max distance** to filter out low-relevance noise.
   - Or raise **Max distance** if relevant chunks are being discarded.
6. Re-run the same questions after adjusting settings to confirm improvement.

This process lets you dial in the `top_k` and `max_distance` values for a specific protocol rather than relying on the defaults.

### Scoring tips

There is no automated scoring built in — evaluation is currently manual. When reviewing answers, a simple rubric works well:

| Rating | Meaning |
|--------|---------|
| ✅ Correct | Answer matches expected answer; cited page is within ±2 pages |
| ⚠️ Partial | Answer is on the right track but incomplete or imprecise |
| ❌ Wrong | Answer is factually incorrect, hallucinated, or not found |

Track your results in a spreadsheet to compare performance across different `top_k` / `max_distance` combinations or after switching embedding/generation models.

---

## 🔧 Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| "Cannot reach Ollama" | `ollama serve` not running | Run `ollama serve` in a terminal |
| "Cannot reach Ollama" | Wrong host | Update `OLLAMA_HOST` in `.env` or the sidebar field |
| All embeddings failed | Model not pulled | Run `ollama pull haybu/mxbai-embed-large-latest:latest` |
| Empty answers / "cannot find information" | Max distance too strict | Raise the Max distance slider in the sidebar |
| Slow ingestion | Large PDF or CPU-only Ollama | Normal — embedding 500-chunk PDFs takes ~1–2 min on CPU |
| Duplicate sources in sidebar | — | Fixed in current version via set-based deduplication on re-ingest |
| `KeyError: selected_sources` | Session state not initialised | Fixed in current version; clear browser cache if persisting |
| `generate_questions.py` fails | Missing API key | Set `OPENAI_API_KEY` or `GEMINI_API_KEY` in `.env` |

---

## 🤝 Contributing

This is a personal portfolio project. Feel free to fork and adapt for your own use.

---

## 📄 License

This project is for educational and research purposes.  
Models used (Ollama, llama3.1, nomic-embed) are subject to their respective licenses.

---

## 🙏 Acknowledgements

- [Ollama](https://ollama.com/)
- [Streamlit](https://streamlit.io/)
- [ChromaDB](https://www.trychroma.com/)
- [PDFPlumber](https://github.com/jsvine/pdfplumber)
- Illumina — TruSight Oncology 500.pdf manual used as example input
- Illumina — TruSeq-DNA-PCR-Free.pdf manual used as example input
- Illumina — Nextera-XT-DNA.pdf manual used as example input
- Illumina — TruSeq-Nano-DNA.pdf manual used as example input
- Illumina — TruSeq-Stranded-Total-RNA.pdf manual used as example input