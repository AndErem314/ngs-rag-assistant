# NGS‑RAG Assistant

A **local, privacy‑friendly RAG (Retrieval‑Augmented Generation) assistant** for lab technicians working with NGS sample preparation protocols.  
It ingests PDF user manuals (e.g., Illumina TruSight Oncology 500), answers natural language questions with citations, and generates structured reports.

Built with:
- **Ollama** – for local embeddings (`nomic‑embed‑text‑v2‑moe`) and generation (`llama3.1:8b`)
- **Streamlit** – interactive web UI
- **ChromaDB** – persistent vector storage
- **PDFPlumber** – text extraction from PDFs

## ✨ Features

- Upload up to 5 PDF manuals
- Ask questions like:
  - “What is the minimum DNA input?”
  - “How many cycles for index PCR?”
  - “What genes are covered by TSO500?”
  - “What reagents are in box 2 and where should they be stored?”
- Get answers with **source citations** (file name and page number)
- Generate a **structured report** for a selected protocol (e.g., reagents, steps, conditions)
- Everything runs **locally** – no API keys, no data leaves your computer

## 🗂 Project Structure
ngs-rag-assistant/
├── README.md
├── requirements.txt
├── .env.example
├── docker-compose.yml # optional
├── src/
│ ├── init.py
│ ├── ingestion/
│ │ ├── pdf_parser.py # extract text from PDFs
│ │ └── chunker.py # split into overlapping chunks
│ ├── embeddings/
│ │ └── embedder.py # Ollama embedding client
│ ├── retrieval/
│ │ └── vector_store.py # ChromaDB persistence & search
│ ├── generation/
│ │ └── llm_client.py # Ollama generation client
│ ├── report/
│ │ └── report_builder.py # structured report generation
│ └── ui/
│ └── streamlit_app.py # main Streamlit application
├── tests/ # unit tests
├── data/ # place your PDFs here (gitignored)
├── notebooks/ # experiment notebooks
└── scripts/ # utility scripts

text

## 🔁 Workflow

1. **Ingestion** – PDFs are parsed page‑by‑page with `pdfplumber`.
2. **Chunking** – Text is split into overlapping chunks (500 tokens, 50 overlap). Metadata (source file, page number) is attached.
3. **Embedding** – Each chunk is converted to a vector using **Ollama** with the model `nomic‑embed‑text‑v2‑moe`.
4. **Storage** – Chunks and embeddings are stored in **ChromaDB** (persistent on disk).
5. **Query** – User’s question is embedded with the same model, then used to retrieve the top‑k most relevant chunks.
6. **Generation** – The retrieved chunks are fed to a **local LLM** (via Ollama) together with the question. The LLM answers strictly from the context and cites the source.
7. **Report** – For a selected protocol, the system runs a set of pre‑defined questions against the same RAG pipeline and compiles the answers into a Markdown report.

## 🖥️ Hardware Requirements

- **Mac with Apple Silicon** (M1/M2/M3/M4) – recommended for best performance.
- **Minimum 16 GB RAM**
- **Ollama** running locally.

## 🚀 Setup & Installation

### 1. Install Ollama

Download from [ollama.com](https://ollama.com/) and install.  
After installation, pull the required models:

```bash
# Embedding model (multilingual, supports dimension cropping)
ollama pull nomic-embed-text-v2-moe

# Generation model (8B parameters, efficient and accurate)
ollama pull llama3.1:8b
Optionally, you can use phi3:mini if you prefer a smaller generation model.

2. Clone the repository

bash
git clone git@github.com:AndErem314/ngs-rag-assistant.git
cd ngs-rag-assistant
3. Create a virtual environment (recommended)

bash
python -m venv venv
source venv/bin/activate   # on macOS/Linux
# or .\venv\Scripts\activate on Windows
4. Install Python dependencies

bash
pip install -r requirements.txt
5. Prepare your PDFs

Place your user manuals (up to 5) into the data/ folder.

6. Run the Streamlit app

bash
streamlit run src/ui/streamlit_app.py
The app will open in your browser at http://localhost:8501.

🔧 Configuration

Environment variables can be set in a .env file (copy .env.example to .env).
Currently supported:

text
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text-v2-moe
LLM_MODEL=llama3.1:8b
If you run Ollama on a different host/port, adjust OLLAMA_HOST.

📖 Usage Guide

Ingest PDFs

Use the sidebar to upload PDFs (multiple files allowed).
Click “Ingest PDFs” – the app will parse, chunk, embed, and store them.
Progress messages will appear.
Ask a Question

Type your question in the main area.
Click “Ask” – the answer will appear with citations (file name, page number).
Generate a Report

Click “Generate Report” – the app will run a set of pre‑defined questions against the ingested documents and produce a Markdown report.
You can download the report as .md file.
Clear Data

Use the “Clear Collection” button in the sidebar to start fresh.
💡 Example Questions to Try

“What is the minimum DNA input recommended for TruSight Oncology 500?”
“How many cycles are used in the index PCR step?”
“What reagents are in box 2 and where should they be stored?”
“What Covaris shearing settings are recommended for the E220 evolution?”
“What genes can be detected by TSO500?”
“What is the storage temperature for the Library Normalization Beads?”
🧠 Model Recommendations

Model	Purpose	Why
nomic-embed-text-v2-moe	Embedding	Excellent retrieval performance (BEIR benchmark), supports 100+ languages, allows dimension cropping to 256/512 dims for memory efficiency.
llama3.1:8b	Generation	Good balance of speed and accuracy. Quantized version (Q4) uses ~6‑8 GB RAM. Works well with instruction‑following.
phi3:mini (optional)	Generation	Smaller (3.8B) but very capable, uses less memory (~3‑4 GB).
Both models run locally via Ollama and utilise your Mac’s GPU (Metal) for acceleration.

🧪 Testing

Run unit tests with:

bash
pytest tests/
🤝 Contributing

This is a personal portfolio project. Feel free to fork and adapt for your own use.

📄 License

This project is for educational and research purposes.
Models used (Ollama, llama3.1, nomic‑embed) are subject to their respective licenses.

🙏 Acknowledgements

Ollama
Streamlit
ChromaDB
PDFPlumber
Illumina for the TruSight Oncology 500 manual used as example
text

This Markdown file is ready to be copied into your project. All formatting is consistent and will render correctly on GitHub and other Markdown viewers.