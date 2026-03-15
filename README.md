# Multimodal ArXiv GraphRAG System

> Automatically download research papers from arXiv, extract structured knowledge
> (text, tables, equations, figures), store them in a Neo4j knowledge graph, and
> answer research questions using Graph RAG + GPT-4o.

---

## Architecture

```
User Query
    │
    ▼
arXiv Search API  ──► PDF Downloader
                           │
                           ▼
                  Document Layout Detection
                  (Unstructured / DocLayNet)
                           │
                           ▼
                    Element Router
                  ┌────┬──────┬──────┐
                  │    │      │      │
                 Text Table Equation Figure
                  │    │      │      │
                  └────┴──┬───┴──────┘
                          │
                          ▼
                  Neo4j Knowledge Graph
                  (Paper·Author·Method·Dataset·
                   Equation·Table·Figure·Concept)
                          │
                          ▼
                  Graph RAG (LlamaIndex / LangChain)
                          │
                          ▼
                  GPT-4o Generated Answer
```

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | arXiv search + PDF download | ✅ Done |
| 2 | Layout detection + element routing | 🔜 Next |
| 3 | Neo4j graph builder | 🔜 |
| 4 | Graph RAG + Streamlit UI | 🔜 |

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Docker (for Neo4j)
- An OpenAI API key

### 2. Install

```bash
git clone <your-repo>
cd arxiv_graphrag

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_trf
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env: add your OPENAI_API_KEY and set NEO4J_PASSWORD
```

### 4. Start Neo4j

```bash
docker-compose up -d
# Neo4j Browser available at http://localhost:7474
```

### 5. Run Milestone 1

```bash
# Search + download (5 papers on transformers)
python pipeline/09_run_pipeline.py --query "attention transformer" --max 5 --stages ingest

# Or fetch a specific paper
python pipeline/09_run_pipeline.py --id 1706.03762 --stages ingest
```

### 6. Use individual stages

```bash
# Stage 1 only: search
python pipeline/01_arxiv_search.py --query "graph neural networks" --max 10

# Stage 2 only: download from a saved search
python pipeline/02_pdf_downloader.py --input data/extracted/search_xyz.json

# Inspect downloaded PDFs
python pipeline/02_pdf_downloader.py --report
```

---

## Project Structure

```
arxiv_graphrag/
├── config/settings.py          # Central config (reads .env)
├── pipeline/
│   ├── 01_arxiv_search.py      # arXiv API → PaperMetadata
│   ├── 02_pdf_downloader.py    # Download + cache PDFs
│   ├── 03_layout_detector.py   # [Milestone 2]
│   ├── 04_element_router.py    # [Milestone 2]
│   ├── 05_text_extractor.py    # [Milestone 2]
│   ├── 06_table_extractor.py   # [Milestone 2]
│   ├── 07_equation_extractor.py# [Milestone 2]
│   ├── 08_figure_extractor.py  # [Milestone 2]
│   └── 09_run_pipeline.py      # Full orchestrator
├── graph/                      # [Milestone 3] Neo4j client + builder
├── rag/                        # [Milestone 4] Graph RAG engine
├── ui/                         # [Milestone 4] Streamlit app
├── tests/
├── notebooks/
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Example Queries (once fully built)

```
Which datasets are used to evaluate the transformer model?
What equation defines the attention mechanism?
Which papers use the ImageNet dataset?
Show all benchmark tables from paper 1706.03762.
What methods does author "Yann LeCun" propose?
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| arXiv API | `arxiv` Python SDK |
| PDF parsing | `PyMuPDF` (fitz) |
| Layout detection | `unstructured` + detectron2 |
| Table extraction | `camelot-py`, `pdfplumber` |
| Equation OCR | `pix2tex` (LaTeX-OCR) |
| NER | `spaCy` + `scispacy` |
| Graph DB | Neo4j 5 (Docker) |
| Graph RAG | LlamaIndex + Neo4jGraphStore |
| LLM | OpenAI GPT-4o |
| UI | Streamlit + streamlit-agraph |