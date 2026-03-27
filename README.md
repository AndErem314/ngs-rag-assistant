# NGS‑RAG Assistant

A **local, privacy‑friendly RAG (Retrieval‑Augmented Generation) assistant** for lab technicians working with NGS sample preparation protocols.  
It ingests PDF user manuals (e.g., Illumina TruSight Oncology 500), answers natural language questions with source citations, and generates structured Markdown reports.

Everything runs **fully locally** via Ollama — no API keys, no internet connection required, no data leaves your machine.

Built with:

- **Ollama** — local embeddings (`nomic-embed-text-v2-moe`) and generation (`llama3.1:8b`)
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

---

## 🗂 Project Structure

```
ngs-rag-assistant/
├── README.md
├── requirements.txt
├── .env.example
├── docker-compose.yml          # optional
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── pdf_parser.py       # page-by-page text extraction (pdfplumber)
│   │   └── chunker.py          # overlapping token-based chunking
│   ├── embeddings/
│   │   └── embedder.py         # OllamaEmbedder — host-aware ollama.Client
│   ├── retrieval/
│   │   ├── vector_store.py     # ChromaDB persistence, hash IDs, upsert, distance filter
│   │   └── query_processor.py  # embed question → search → return context + metadata
│   ├── generation/
│   │   └── llm_client.py       # OllamaGenerator — chat() with system/user roles
│   ├── report/
│   │   └── report_builder.py   # run predefined questions → compile Markdown report
│   └── ui/
│       └── streamlit_app.py    # main Streamlit application
├── tests/
│   ├── test_ingestion.py       # VectorStore: IDs, upsert, re-ingestion, clear
│   └── test_retrieval.py       # VectorStore.search + retrieve_context smoke tests
├── scripts/
│   └── generate_questions.py   # GPT-4o-mini question generation from PDF manuals
├── data/                       # place your PDFs here (gitignored)
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
                       using the nomic-embed-text-v2-moe model;
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

Both models run locally via Ollama and use your GPU automatically (Metal on Apple Silicon, CUDA on NVIDIA).

---

## 🚀 Setup & Installation

### 1. Install Ollama

Download from [ollama.com](https://ollama.com/) and install. Then pull both required models:

```bash
# Embedding model
ollama pull nomic-embed-text-v2-moe

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
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text-v2-moe
LLM_MODEL=llama3.1:8b
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
| `nomic-embed-text-v2-moe` | Embeddings | ~1 GB | Top BEIR benchmark performance; 100+ languages; supports dimension cropping to 256/512 for memory efficiency |
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

Run the full test suite with:

```bash
pytest tests/ -v
```

Tests use `chromadb.EphemeralClient` (in-memory, no disk I/O) and mock `OllamaEmbedder` via `unittest.mock.MagicMock`, so **no Ollama server is required** to run the tests.

| Test file | What it covers |
|-----------|---------------|
| `test_ingestion.py` | `_make_chunk_id` stability and collision resistance; `add_chunks` happy path and ID determinism; re-ingestion deduplication (the critical upsert regression test); `clear_collection` |
| `test_retrieval.py` | `VectorStore.search` — top-K, source filter, max_distance, empty collection; `retrieve_context` — happy path, empty embedding (Ollama down), max_distance filtering, metadata shape including distance field |

---

## 🔧 Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| "Cannot reach Ollama" | `ollama serve` not running | Run `ollama serve` in a terminal |
| "Cannot reach Ollama" | Wrong host | Update `OLLAMA_HOST` in `.env` or the sidebar field |
| All embeddings failed | Model not pulled | Run `ollama pull nomic-embed-text-v2-moe` |
| Empty answers / "cannot find information" | Max distance too strict | Raise the Max distance slider in the sidebar |
| Slow ingestion | Large PDF or CPU-only Ollama | Normal — embedding 500-chunk PDFs takes ~1–2 min on CPU |
| Duplicate sources in sidebar | — | Fixed in current version via set-based deduplication on re-ingest |
| `KeyError: selected_sources` | Session state not initialised | Fixed in current version; clear browser cache if persisting |

---

## 📜 Scripts

### `scripts/generate_questions.py`

Generates a JSON test-question set from one or more PDF manuals using **GPT-4o-mini** (requires `OPENAI_API_KEY` in `.env`). Useful for building an evaluation set to measure retrieval quality.

```bash
python scripts/generate_questions.py data/manual.pdf
python scripts/generate_questions.py data/manual1.pdf data/manual2.pdf
```

Output is saved to `data/questions/<stem>_questions.json` with fields `question`, `expected_answer`, `source_page`.

> **Note:** This script uses the OpenAI API and is only needed for evaluation. The main assistant runs fully locally without it.

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
- Illumina — TruSight Oncology 500 manual used as example input