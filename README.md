# 🔬 Multimodal ArXiv GraphRAG System

> **An AI-powered research assistant that reads academic papers, builds a knowledge graph, and answers questions that traditional RAG cannot.**

---

## 📌 What is this?

Most RAG systems treat a research PDF like a plain text document.
This system doesn't.

It detects **every element type** in a document and routes it to a **specialized extraction model**:

| Element | Model Used | Output |
|---------|-----------|--------|
| 📝 Text sections | spaCy NER + 60+ regex patterns | Entities: methods, datasets, metrics |
| 📊 Tables | pdfplumber + camelot | Structured rows, columns, benchmark flag |
| 🔢 Equations | pix2tex LaTeX-OCR | LaTeX string (e.g. `\frac{QK^T}{\sqrt{d_k}}`) |
| 🖼️ Figures | PyMuPDF xref extraction | PNG image + caption text |

Everything is stored in a **Neo4j knowledge graph** with explicit relationships — enabling multi-hop queries that vector RAG simply cannot do.

---

## 🧠 Graph RAG vs Traditional RAG

```
Traditional RAG:  PDF → chunks → embeddings → cosine similarity → LLM
Graph RAG:        PDF → typed nodes → Neo4j graph → Cypher traversal → LLM
```

**Question traditional RAG cannot answer:**
> *"Which datasets were used to evaluate transformer methods across all papers?"*

This requires traversing: `Topic → Paper → Method → EVALUATED_ON → Dataset`
— a graph path, not a text similarity search.

---

## 🏗️ System Architecture

```
User types a topic
        │
        ▼
① arXiv Search API          → finds related papers
        │
        ▼
② PDF Downloader             → downloads + caches PDFs
        │
        ▼
③ Layout Detector            → labels every region (text/table/equation/figure)
        │
        ▼
④ Element Router             → dispatches to specialized extractors
   ┌────┼──────┬──────┐
   ▼    ▼      ▼      ▼
 Text Table Equation Figure
   └────┴──────┴──────┘
        │
        ▼
⑤ Neo4j Knowledge Graph      → 9 node types, 10 relationship types
        │
        ▼
⑥ Graph RAG Engine           → Cypher traversal + GPT-4o-mini
        │
        ▼
⑦ Streamlit UI               → chat, LaTeX, tables, figures, live logs
```

---

## 🕸️ Knowledge Graph Schema

**Node types:** `Paper` · `Author` · `Method` · `Dataset` · `Equation` · `Table` · `Figure` · `Concept` · `Topic`

**Relationships:**
```
(Author)  -[:WROTE]-----------> (Paper)
(Topic)   -[:INCLUDES]--------> (Paper)
(Paper)   -[:PROPOSES]--------> (Method)
(Paper)   -[:USES_DATASET]----> (Dataset)
(Paper)   -[:HAS_TABLE]-------> (Table)
(Paper)   -[:HAS_EQUATION]----> (Equation)
(Paper)   -[:HAS_FIGURE]------> (Figure)
(Paper)   -[:MENTIONS]--------> (Concept)
(Method)  -[:EVALUATED_ON]----> (Dataset)
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Ingestion | `arxiv` SDK | Search + metadata |
| Download | `requests` + `tenacity` | PDF download with retry |
| Layout detection | `unstructured[pdf]` | Element classification |
| Text extraction | `spaCy en_core_web_sm` | Named entity recognition |
| Table extraction | `pdfplumber` + `camelot` | Structured table rows/cols |
| Equation extraction | `pix2tex` (LaTeX-OCR) | Image → LaTeX string |
| Figure extraction | `PyMuPDF` | Lossless image extraction |
| Graph database | `Neo4j AuraDB` | Knowledge graph storage |
| RAG engine | Custom Cypher + `openai` | Graph-aware retrieval |
| LLM | `GPT-4o-mini` | Answer generation |
| UI | `Streamlit` | Web interface |

> ✅ Everything except GPT-4o-mini is **completely free**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Neo4j AuraDB account (free) → [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura)
- OpenAI API key → [platform.openai.com](https://platform.openai.com)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/arxiv-graphrag.git
cd arxiv-graphrag

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your keys:
# OPENAI_API_KEY=sk-...
# NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=your-password
# OPENAI_MODEL=gpt-4o-mini
```

### 3. Run the UI

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501` → type a topic → click **Run Full Pipeline** → ask questions.

### 4. Or run via CLI

```bash
# Search + download + extract + graph ingest
python pipeline/run_pipeline.py --query "transformer attention" --max 5 --stages graph

# Check graph stats
python graph/cypher_queries.py
```

---

## 💬 Example Questions

Once the pipeline runs, ask these in the chat:

```
Summarise the key contributions of each paper
Which datasets are used for evaluation across all papers?
What methods are proposed in these papers?
Show me the benchmark results
What mathematical equations are used?
Show me figures and architecture diagrams
Which authors wrote these papers?
Which paper proposes the most novel approach?
```

---

## 📁 Project Structure

```
arxiv_graphrag/
├── config/
│   └── settings.py              # Pydantic settings — reads .env
├── pipeline/
│   ├── models.py                # ExtractedElement dataclass
│   ├── arxiv_search.py          # arXiv API + MD5 caching
│   ├── pdf_downloader.py        # Download with retry + validation
│   ├── layout_detector.py       # unstructured + pdfplumber fallback
│   ├── element_router.py        # Registry + Strategy dispatch
│   ├── text_extractor.py        # spaCy NER + domain regex
│   ├── table_extractor.py       # pdfplumber + camelot
│   ├── equation_extractor.py    # pix2tex LaTeX-OCR
│   ├── figure_extractor.py      # PyMuPDF image extraction
│   └── run_pipeline.py          # Full pipeline orchestrator
├── graph/
│   ├── schema.py                # Node labels + relationship types
│   ├── neo4j_client.py          # AuraDB connection + MERGE helpers
│   ├── graph_builder.py         # ExtractedElement → Neo4j
│   └── cypher_queries.py        # All named Cypher queries
├── rag/
│   ├── graph_rag.py             # Graph RAG engine
│   └── prompt_templates.py      # System prompt + context builder
├── ui/
│   └── app.py                   # Streamlit application
├── .env.example                 # Environment variable template
├── requirements.txt             # All dependencies
└── README.md
```

---

## 🔍 How the Graph RAG Retrieval Works

```python
# 1. Detect query type
query_type = detect_query_type("Which datasets are used?")
# → "dataset"

# 2. Run targeted Cypher query
results = neo4j.run("""
    MATCH (p:Paper)
    OPTIONAL MATCH (p)-[:USES_DATASET]->(d:Dataset)
    RETURN p.title, collect(DISTINCT d.name) AS datasets
""")

# 3. Build structured context
context = build_context_string(results)

# 4. Generate answer with GPT-4o-mini
answer = openai.chat(system_prompt + context + question)
```

---

## ⚠️ Known Limitations

- Equation extraction requires Tesseract OCR for best results (`strategy=hi_res`)
  - Without it, falls back to text-based equation detection (still works for inline math)
- Table extraction quality varies across PDF generators
- Entity patterns tuned for CV/NLP papers — extend `text_extractor.py` for other domains

---

## 🔮 Future Improvements

- [ ] Install Tesseract for full `hi_res` layout detection
- [ ] Add Neo4j vector index for semantic similarity search
- [ ] Deploy to Streamlit Cloud for public access
- [ ] Support Semantic Scholar and PubMed as additional sources
- [ ] Add paper citation graph (`CITES` relationship)
- [ ] Async pipeline with Celery for 100+ paper processing

---

## 📖 Blog Post

Read the full technical breakdown here:
👉 **[Your Blog Link]**

Covers: Graph RAG vs RAG, knowledge graph design, model choices, and everything A to Z.

---

## 🤝 Contributing

Pull requests welcome. For major changes, open an issue first.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

<p align="center">
  Built with ❤️ using Python · Neo4j · OpenAI · Streamlit
</p>
