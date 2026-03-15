"""
ui/app.py - ArXiv GraphRAG System
Full UI with LaTeX rendering, image display, and table rendering.
"""

import sys
import re
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(
    page_title="ArXiv GraphRAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 12px;
    margin-bottom: 1.5rem; color: white;
}
.main-header h1 { color: white; margin: 0; font-size: 2rem; }
.main-header p  { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0; font-size: 1rem; }
.stat-card {
    background: #f8f9fa; border: 1px solid #e9ecef;
    border-radius: 10px; padding: 1rem;
    text-align: center; margin-bottom: 0.5rem;
}
.stat-number { font-size: 1.8rem; font-weight: 700; color: #667eea; line-height: 1; }
.stat-label  { font-size: 0.75rem; color: #6c757d; margin-top: 0.2rem;
               text-transform: uppercase; letter-spacing: 0.05em; }
.log-box {
    background: #1e1e1e; color: #d4d4d4;
    font-family: 'Courier New', monospace; font-size: 0.75rem;
    padding: 0.75rem; border-radius: 8px;
    max-height: 300px; overflow-y: auto; line-height: 1.5;
}
.log-success { color: #4ec9b0; }
.log-info    { color: #9cdcfe; }
.log-warning { color: #dcdcaa; }
.log-error   { color: #f44747; }
.chat-user {
    background: #667eea; color: white;
    padding: 0.75rem 1rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0 0.5rem 20%; font-size: 0.95rem;
}
.paper-card {
    border: 1px solid #e9ecef; border-radius: 10px;
    padding: 1rem; margin-bottom: 0.75rem;
    border-left: 4px solid #667eea;
}
.paper-title    { font-weight: 600; font-size: 0.95rem; color: #1a1a1a; margin-bottom: 0.3rem; }
.paper-meta     { font-size: 0.8rem; color: #6c757d; }
.paper-abstract { font-size: 0.85rem; color: #4a4a4a; margin-top: 0.4rem; line-height: 1.5; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "pipeline_logs":  [],
        "pipeline_done":  False,
        "graph_stats":    {},
        "papers":         [],
        "chat_history":   [],
        "current_topic":  "",
        "rag_engine":     None,
        "stage_status": {
            "Search":   "pending",
            "Download": "pending",
            "Extract":  "pending",
            "Graph":    "pending",
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────
def add_log(msg: str, level: str = "info"):
    ts = time.strftime("%H:%M:%S")
    st.session_state.pipeline_logs.append({"ts": ts, "msg": msg, "level": level})

def set_stage(stage: str, status: str):
    st.session_state.stage_status[stage] = status


# ── LaTeX + rich answer renderer ──────────────────────────────────────────────
def render_answer(text: str):
    """
    Render answer with LaTeX equations, tables, and markdown.
    Detects [ ... ] or \\[ ... \\] blocks and renders them with st.latex().
    Everything else renders as normal markdown.
    """
    parts = re.split(r'(\\\[.*?\\\]|\[[^\]]{10,}\])', text, flags=re.DOTALL)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        is_latex = (
            (part.startswith('\\[') and part.endswith('\\]')) or
            (part.startswith('[') and part.endswith(']') and
             any(c in part for c in ['\\', '^', '_', '{', '}']))
        )
        if is_latex:
            latex = re.sub(r'^[\[\\]+|[\]\\]+$', '', part).strip()
            try:
                st.latex(latex)
            except Exception:
                st.markdown(f"`{latex}`")
        else:
            st.markdown(part)


# ── Query type detector ───────────────────────────────────────────────────────
def detect_rich_query(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["figure", "image", "diagram", "architecture", "picture", "visual", "illustration"]):
        return "figure"
    elif any(w in q for w in ["table", "benchmark", "results", "performance", "accuracy", "comparison"]):
        return "table"
    elif any(w in q for w in ["equation", "formula", "math", "latex", "expression"]):
        return "equation"
    return "text"


# ── Rich content fetcher ──────────────────────────────────────────────────────
def fetch_rich_content(question: str, client) -> dict:
    """
    Fetch figures, tables, or equations directly from Neo4j
    based on what the user is asking for.
    Returns dict with type and data.
    """
    qtype = detect_rich_query(question)

    if qtype == "figure":
        rows = client.run("""
            MATCH (p:Paper)-[:HAS_FIGURE]->(f:Figure)
            WHERE f.image_path IS NOT NULL AND f.image_path <> ''
            RETURN p.title AS paper, p.arxiv_id AS arxiv_id,
                   f.fig_number AS fig_number, f.caption AS caption,
                   f.image_path AS image_path, f.page_number AS page
            ORDER BY p.published DESC, f.page_number
            LIMIT 12
        """, {})
        return {"type": "figure", "data": rows}

    elif qtype == "table":
        rows = client.run("""
            MATCH (p:Paper)-[:HAS_TABLE]->(t:Table)
            RETURN p.title AS paper, p.arxiv_id AS arxiv_id,
                   t.headers AS headers, t.content AS content,
                   t.page_number AS page, t.is_benchmark AS is_benchmark,
                   t.row_count AS rows, t.col_count AS cols
            ORDER BY t.is_benchmark DESC, p.published DESC
            LIMIT 10
        """, {})
        return {"type": "table", "data": rows}

    elif qtype == "equation":
        rows = client.run("""
            MATCH (p:Paper)-[:HAS_EQUATION]->(e:Equation)
            RETURN p.title AS paper, p.arxiv_id AS arxiv_id,
                   e.latex AS latex, e.page_number AS page,
                   e.source AS source
            ORDER BY p.published DESC, e.page_number
            LIMIT 15
        """, {})
        return {"type": "equation", "data": rows}

    return {"type": "text", "data": []}


# ── Rich content renderer ─────────────────────────────────────────────────────
def render_rich_content(rich: dict):
    """
    Render figures, tables, or equations visually in Streamlit.
    Called in addition to the text answer from GPT.
    """
    qtype = rich.get("type")
    data  = rich.get("data", [])

    if not data:
        return

    # ── Figures ───────────────────────────────────────────────────────────────
    if qtype == "figure":
        st.markdown("---")
        st.markdown("#### 🖼️ Figures from Papers")

        # Group by paper
        papers_seen = {}
        for row in data:
            pid = row.get("arxiv_id", "")
            if pid not in papers_seen:
                papers_seen[pid] = {"title": row.get("paper", ""), "figures": []}
            papers_seen[pid]["figures"].append(row)

        for pid, info in papers_seen.items():
            st.markdown(f"**📄 {info['title'][:80]}**")
            figs = info["figures"]
            cols = st.columns(min(len(figs), 3))
            for i, fig in enumerate(figs):
                with cols[i % 3]:
                    img_path = fig.get("image_path", "")
                    caption  = fig.get("caption", "")
                    fig_num  = fig.get("fig_number", f"Figure {i+1}")

                    if img_path and Path(img_path).exists():
                        try:
                            st.image(
                                img_path,
                                caption=f"{fig_num} (p.{fig.get('page','')})",
                                use_column_width=True,
                            )
                        except Exception:
                            st.info(f"📷 {fig_num} — image file found but could not render")
                    else:
                        # Show caption as card if image not available
                        st.markdown(
                            f"<div style='border:1px solid #e9ecef;border-radius:8px;"
                            f"padding:0.75rem;background:#f8f9fa;min-height:80px'>"
                            f"<div style='font-size:0.75rem;color:#667eea;font-weight:600'>{fig_num}</div>"
                            f"<div style='font-size:0.8rem;color:#4a4a4a;margin-top:4px'>{caption[:120]}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
            st.divider()

    # ── Tables ────────────────────────────────────────────────────────────────
    elif qtype == "table":
        st.markdown("---")
        st.markdown("#### 📊 Tables from Papers")

        import pandas as pd
        import json

        for i, row in enumerate(data, 1):
            paper_title  = row.get("paper", "Unknown paper")
            is_benchmark = row.get("is_benchmark", False)
            badge = "🏆 Benchmark" if is_benchmark else "📋 Table"
            page  = row.get("page", "?")

            with st.expander(f"{badge} — {paper_title[:70]} (page {page})", expanded=(i == 1)):
                # Try to render as a proper dataframe
                try:
                    headers_raw = row.get("headers", "[]")
                    headers = json.loads(headers_raw) if isinstance(headers_raw, str) else headers_raw

                    content = row.get("content", "")
                    if content and headers:
                        lines = [l for l in content.split("\n") if "|" in l and "---" not in l]
                        table_rows = []
                        for line in lines:
                            cells = [c.strip() for c in line.split("|") if c.strip()]
                            if cells:
                                table_rows.append(cells)

                        if table_rows and headers:
                            # Align columns
                            max_cols = max(len(headers), max(len(r) for r in table_rows))
                            padded_headers = headers + [""] * (max_cols - len(headers))
                            padded_rows = [r + [""] * (max_cols - len(r)) for r in table_rows]
                            df = pd.DataFrame(padded_rows, columns=padded_headers[:max_cols])
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        else:
                            st.text(content[:800])
                    else:
                        st.text(row.get("content", "No content")[:800])

                except Exception:
                    st.text(row.get("content", "No content")[:800])

    # ── Equations ─────────────────────────────────────────────────────────────
    elif qtype == "equation":
        st.markdown("---")
        st.markdown("#### 🔢 Equations from Papers")

        papers_seen = {}
        for row in data:
            pid = row.get("arxiv_id", "unknown")
            if pid not in papers_seen:
                papers_seen[pid] = {"title": row.get("paper", ""), "eqs": []}
            papers_seen[pid]["eqs"].append(row)

        for pid, info in papers_seen.items():
            st.markdown(f"**📄 {info['title'][:80]}**")
            for j, eq in enumerate(info["eqs"], 1):
                latex  = eq.get("latex", "")
                source = eq.get("source", "")
                page   = eq.get("page", "?")
                if latex:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        try:
                            st.latex(latex)
                        except Exception:
                            st.code(latex, language=None)
                    with col2:
                        st.caption(f"p.{page}\n{source}")
            st.divider()


# ── Pipeline runner ───────────────────────────────────────────────────────────
def run_full_pipeline(topic: str, max_papers: int):
    st.session_state.pipeline_logs = []
    st.session_state.pipeline_done = False
    st.session_state.papers = []
    st.session_state.stage_status = {s: "pending" for s in st.session_state.stage_status}

    add_log(f"Starting pipeline for topic: '{topic}'", "info")

    set_stage("Search", "running")
    add_log("Stage 1 · Searching arXiv…", "info")
    yield

    try:
        from pipeline.arxiv_search import search_arxiv, save_metadata_batch
        from config.settings import settings
        papers = search_arxiv(topic, max_results=max_papers, use_cache=True)
        st.session_state.papers = [p.to_dict() for p in papers]
        add_log(f"Found {len(papers)} papers for '{topic}'", "success")
        set_stage("Search", "done")
        yield
        meta_path = settings.extracted_dir / f"stage1_{int(time.time())}.json"
        save_metadata_batch(papers, meta_path)
        add_log(f"Metadata saved → {meta_path.name}", "info")
    except Exception as e:
        add_log(f"Search failed: {e}", "error")
        set_stage("Search", "error")
        return
    yield

    set_stage("Download", "running")
    add_log("Stage 2 · Downloading PDFs…", "info")
    yield
    try:
        from pipeline.pdf_downloader import download_papers
        papers = download_papers(papers, delay=2.0)
        downloaded = sum(1 for p in papers if p.download_status == "downloaded")
        failed     = sum(1 for p in papers if p.download_status == "failed")
        add_log(f"Downloaded {downloaded} PDFs · {failed} failed",
                "success" if not failed else "warning")
        set_stage("Download", "done")
    except Exception as e:
        add_log(f"Download failed: {e}", "error")
        set_stage("Download", "error")
        return
    yield

    set_stage("Extract", "running")
    add_log("Stage 3–8 · Extracting elements from PDFs…", "info")
    yield
    try:
        from pipeline.element_router import route_paper_batch
        paper_dicts = [
            {"paper_id": p.arxiv_id, "pdf_path": p.local_pdf_path}
            for p in papers if p.download_status == "downloaded"
        ]
        for pd_item in paper_dicts:
            add_log(f"Processing [{pd_item['paper_id']}]…", "info")
            yield
        extraction_results = route_paper_batch(paper_dicts, prefer_ml=False)
        total_elements = sum(len(r.elements) for r in extraction_results)
        add_log(f"Extracted {total_elements} elements from {len(extraction_results)} papers", "success")
        set_stage("Extract", "done")
    except Exception as e:
        add_log(f"Extraction failed: {e}", "error")
        set_stage("Extract", "error")
        return
    yield

    set_stage("Graph", "running")
    add_log("Stage 7 · Ingesting into Neo4j knowledge graph…", "info")
    yield
    try:
        from graph.graph_builder import GraphBuilder, load_extraction_results
        from config.settings import settings
        extraction_results = load_extraction_results(settings.extracted_dir)
        paper_dicts_all = [p.to_dict() if hasattr(p, "to_dict") else p for p in papers]
        with GraphBuilder() as builder:
            add_log("Creating indexes…", "info")
            yield
            builder.ingest_topic(topic=topic, papers=paper_dicts_all, results=extraction_results)
        add_log("Graph populated successfully!", "success")
        set_stage("Graph", "done")
    except Exception as e:
        add_log(f"Graph ingest failed: {e}", "error")
        set_stage("Graph", "error")
        return
    yield

    try:
        from rag.graph_rag import GraphRAG
        rag = GraphRAG(topic=topic)
        st.session_state.graph_stats   = rag.get_graph_stats()
        st.session_state.rag_engine    = rag
        st.session_state.current_topic = topic
    except Exception as e:
        add_log(f"RAG engine init failed: {e}", "error")

    st.session_state.pipeline_done = True
    add_log("Pipeline complete! You can now ask questions.", "success")
    yield


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 ArXiv GraphRAG")
    st.caption("Research Paper Knowledge Graph + AI Q&A")
    st.divider()

    st.markdown("### Pipeline Status")
    stage_icons = {"pending": "⬜", "running": "🔄", "done": "✅", "error": "❌"}
    for stage, status in st.session_state.stage_status.items():
        icon  = stage_icons.get(status, "⬜")
        color = {"done": "#155724", "running": "#004085",
                 "error": "#721c24", "pending": "#6c757d"}[status]
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;padding:4px 0'>"
            f"<span style='font-size:16px'>{icon}</span>"
            f"<span style='font-size:0.85rem;color:{color};font-weight:500'>{stage}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.divider()

    st.markdown("### Graph Stats")
    stats = st.session_state.graph_stats
    if stats:
        nodes = stats.get("nodes", {})
        cols  = st.columns(2)
        items = [
            ("Papers",    nodes.get("Paper",    0), "📄"),
            ("Authors",   nodes.get("Author",   0), "👤"),
            ("Methods",   nodes.get("Method",   0), "⚙️"),
            ("Datasets",  nodes.get("Dataset",  0), "📊"),
            ("Tables",    nodes.get("Table",    0), "📋"),
            ("Equations", nodes.get("Equation", 0), "🔢"),
            ("Figures",   nodes.get("Figure",   0), "🖼️"),
            ("Concepts",  nodes.get("Concept",  0), "💡"),
        ]
        for i, (label, count, icon) in enumerate(items):
            with cols[i % 2]:
                st.markdown(
                    f"<div class='stat-card'>"
                    f"<div class='stat-number'>{count}</div>"
                    f"<div class='stat-label'>{icon} {label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.markdown(
            f"<div style='text-align:center;font-size:0.8rem;color:#6c757d;margin-top:0.5rem'>"
            f"Total: <b>{stats.get('total_nodes',0)}</b> nodes · "
            f"<b>{stats.get('total_rels',0)}</b> relationships"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("Run the pipeline to see stats here.")
    st.divider()

    st.markdown("### Backend Logs")
    logs = st.session_state.pipeline_logs
    if logs:
        log_html = ""
        for entry in logs[-30:]:
            css = f"log-{entry['level']}"
            sym = {"success": "✓", "info": "→", "warning": "⚠", "error": "✗"}.get(entry["level"], "·")
            log_html += (
                f"<div class='{css}'>"
                f"<span style='opacity:0.5'>[{entry['ts']}]</span> "
                f"{sym} {entry['msg']}"
                f"</div>"
            )
        st.markdown(f"<div class='log-box'>{log_html}</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='log-box'><span style='color:#6c757d'>Waiting for pipeline…</span></div>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("Built with arXiv API · Neo4j AuraDB · GPT-4o-mini · Streamlit")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='main-header'>
    <h1>🔬 ArXiv GraphRAG System</h1>
    <p>Search research papers · Build a knowledge graph · Ask AI questions</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🚀 Run Pipeline", "💬 Ask Questions", "📄 Papers & Graph"])


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Search & Build Knowledge Graph")
    st.markdown("Enter a research topic and the system will automatically download papers, extract knowledge, and build a graph.")

    col1, col2 = st.columns([3, 1])
    with col1:
        topic_input = st.text_input(
            "Research Topic",
            placeholder="e.g. transformer attention, graph neural networks, diffusion models…",
            value=st.session_state.current_topic or "",
            label_visibility="collapsed",
        )
    with col2:
        max_papers = st.number_input("Max papers", min_value=1, max_value=20, value=3)

    run_btn = st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)

    if run_btn and topic_input.strip():
        topic = topic_input.strip()
        log_placeholder  = st.empty()
        prog_placeholder = st.empty()
        prog_bar = prog_placeholder.progress(0, text="Starting pipeline…")

        for step_idx, _ in enumerate(run_full_pipeline(topic, max_papers), 1):
            pct = min(int((step_idx / 20) * 100), 99)
            prog_bar.progress(pct, text=f"Running pipeline… step {step_idx}")
            logs = st.session_state.pipeline_logs
            if logs:
                log_html = ""
                for entry in logs[-15:]:
                    css = f"log-{entry['level']}"
                    sym = {"success": "✓", "info": "→", "warning": "⚠", "error": "✗"}.get(entry["level"], "·")
                    log_html += (
                        f"<div class='{css}'>"
                        f"<span style='opacity:0.5'>[{entry['ts']}]</span> {sym} {entry['msg']}"
                        f"</div>"
                    )
                log_placeholder.markdown(
                    f"<div class='log-box'>{log_html}</div>", unsafe_allow_html=True
                )

        if st.session_state.pipeline_done:
            prog_bar.progress(100, text="Pipeline complete!")
            st.success(f"✅ Knowledge graph built for **'{topic}'** — switch to the **Ask Questions** tab!")
            st.balloons()
        else:
            prog_bar.progress(100, text="Pipeline finished with errors — check logs")

    elif run_btn:
        st.warning("Please enter a research topic first.")

    st.divider()
    st.markdown("#### How it works")
    cols = st.columns(4)
    steps = [
        ("🔍", "Search",   "Queries arXiv API for research papers on your topic"),
        ("📥", "Download", "Downloads PDFs and caches them locally"),
        ("🧩", "Extract",  "Detects layout and extracts text, tables, equations, figures"),
        ("🕸️", "Graph",    "Stores everything in Neo4j as a knowledge graph"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"<div style='text-align:center;padding:1rem;border:1px solid #e9ecef;"
                f"border-radius:10px;height:120px'>"
                f"<div style='font-size:2rem'>{icon}</div>"
                f"<div style='font-weight:700;font-size:0.9rem;margin:0.3rem 0'>{title}</div>"
                f"<div style='font-size:0.8rem;color:#6c757d'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Chat
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.pipeline_done:
        st.info("👈 Run the pipeline first from the **Run Pipeline** tab.")
    else:
        st.markdown(f"### Ask about **'{st.session_state.current_topic}'**")

        # ── Suggested questions ────────────────────────────────────────────────
        st.markdown("**Suggested questions:**")
        suggestions = [
            "What methods are proposed in these papers?",
            "Which datasets are used for evaluation?",
            "Summarise the key contributions of each paper",
            "What are the benchmark results?",
            "Show me figures and diagrams from these papers",
            "What mathematical equations are used?",
        ]
        sug_cols = st.columns(3)
        for i, sug in enumerate(suggestions):
            with sug_cols[i % 3]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": sug})
                    st.rerun()

        st.divider()

        # ── Chat history ───────────────────────────────────────────────────────
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='chat-user'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                with st.chat_message("assistant", avatar="🔬"):
                    render_answer(msg["content"])

                    # Show rich content (figures/tables/equations) stored in message
                    if msg.get("rich"):
                        render_rich_content(msg["rich"])

                    if msg.get("sources"):
                        with st.expander("📚 Sources used"):
                            for src in msg["sources"]:
                                st.caption(f"• {src}")
                    if msg.get("tokens"):
                        st.caption(f"🔢 {msg['tokens']} tokens used")

        # ── Process pending user message ───────────────────────────────────────
        last = st.session_state.chat_history[-1] if st.session_state.chat_history else None
        if last and last["role"] == "user" and (
            len(st.session_state.chat_history) == 1
            or st.session_state.chat_history[-2]["role"] != "user"
        ):
            question = last["content"]
            rag = st.session_state.rag_engine

            if rag:
                with st.chat_message("assistant", avatar="🔬"):
                    with st.spinner("Querying knowledge graph…"):
                        result = rag.ask(question)

                    render_answer(result["answer"])

                    # Fetch and render rich content
                    rich = fetch_rich_content(question, rag.client)
                    render_rich_content(rich)

                    if result.get("sources"):
                        with st.expander("📚 Sources used"):
                            for src in result["sources"]:
                                st.caption(f"• {src}")
                    if result.get("tokens_used"):
                        st.caption(f"🔢 {result['tokens_used']} tokens used")

                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": result["answer"],
                        "rich":    rich,
                        "sources": result.get("sources", []),
                        "tokens":  result.get("tokens_used", 0),
                    })
                    st.rerun()

        # ── Input ──────────────────────────────────────────────────────────────
        user_question = st.chat_input("Ask a question about the research papers…")
        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat"):
                st.session_state.chat_history = []
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Papers & Graph
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    if not st.session_state.papers:
        st.info("👈 Run the pipeline first to see papers and graph data here.")
    else:
        st.markdown("### Downloaded Papers")
        for paper in st.session_state.papers:
            authors    = paper.get("authors", [])
            author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            abstract   = paper.get("abstract", "")[:250]
            st.markdown(
                f"<div class='paper-card'>"
                f"<div class='paper-title'>📄 {paper.get('title','Untitled')}</div>"
                f"<div class='paper-meta'>"
                f"🆔 {paper.get('arxiv_id','')} &nbsp;·&nbsp; "
                f"👤 {author_str} &nbsp;·&nbsp; "
                f"📅 {str(paper.get('published',''))[:10]} &nbsp;·&nbsp; "
                f"🏷️ {paper.get('primary_category','')}"
                f"</div>"
                f"<div class='paper-abstract'>{abstract}…</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("### Knowledge Graph Breakdown")
        stats = st.session_state.graph_stats
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Node types**")
                nodes = stats.get("nodes", {})
                if nodes:
                    import pandas as pd
                    df_nodes = pd.DataFrame(list(nodes.items()), columns=["Label", "Count"])
                    df_nodes = df_nodes.sort_values("Count", ascending=False)
                    st.bar_chart(df_nodes.set_index("Label"))
            with col2:
                st.markdown("**Relationship types**")
                rels = stats.get("rels", {})
                if rels:
                    import pandas as pd
                    df_rels = pd.DataFrame(list(rels.items()), columns=["Type", "Count"])
                    df_rels = df_rels.sort_values("Count", ascending=False)
                    st.bar_chart(df_rels.set_index("Type"))

            st.divider()
            st.markdown("**Papers in graph**")
            papers_in_graph = stats.get("papers", [])
            if papers_in_graph:
                import pandas as pd
                df_p = pd.DataFrame(papers_in_graph)
                if "title" in df_p.columns and "arxiv_id" in df_p.columns:
                    df_p = df_p[["title", "arxiv_id", "published"]].rename(columns={
                        "title": "Title", "arxiv_id": "ArXiv ID", "published": "Published"
                    })
                    df_p["Published"] = df_p["Published"].str[:10]
                    st.dataframe(df_p, use_container_width=True, hide_index=True)