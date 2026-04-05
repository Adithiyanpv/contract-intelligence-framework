import streamlit as st
st.set_page_config(page_title="ContractIQ", page_icon="⚖️", layout="wide")
import os, sys, requests, tempfile
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path: sys.path.insert(0, _APP_DIR)
os.chdir(_APP_DIR)
if not os.path.exists(os.path.join(_APP_DIR, "resources", "deberta-clause-final")):
    with st.spinner("Downloading model artifacts (first run only)..."):
        try:
            from download_models import ensure_models
            ensure_models()
        except Exception as e:
            st.error(f"Model download failed: {e}")
            st.stop()
from llm.llm_client import get_llm_client, build_safe_prompt
from pipeline import (analyze_document, ask_document, build_contract_summary,
                      narrate_contract_summary, export_results_csv, export_results_json)
from summarizer.contract_summarizer import summarize_contract, evaluate_summary
from summarizer.hrs_engine import hierarchical_summarize, CLAUSE_CATEGORIES
from rag.contract_rag import crag_answer
from negotiation.simulator import simulate_negotiation, STANCES
from obligation_graph.extractor import extract_obligations, build_obligation_graph

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("analyzed", False), ("_active_tab", "overview"), ("last_answer", None),
             ("_force_ask_tab", False), ("neg_results", None), ("neg_clause", None),
             ("ob_graph", None), ("contract_doc_summary", None),
             ("multi_doc_results", None), ("analysis_mode", "single")]:
    if k not in st.session_state: st.session_state[k] = v

llm, llm_source = get_llm_client()

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0f1117; }
.stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%); }

/* Hero header */
.hero { background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 50%, #1a1d2e 100%);
  border: 1px solid rgba(99,179,237,0.2); border-radius: 16px;
  padding: 2.5rem 2rem 2rem; margin-bottom: 1.5rem;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
.hero h1 { font-size: 2.2rem; font-weight: 700; color: #e2e8f0; margin: 0 0 0.4rem; letter-spacing: -0.5px; }
.hero p { color: #94a3b8; font-size: 0.95rem; margin: 0; }
.hero .badge { display:inline-block; background:rgba(99,179,237,0.15);
  color:#63b3ed; border:1px solid rgba(99,179,237,0.3);
  border-radius:20px; padding:3px 12px; font-size:0.75rem; font-weight:600;
  margin-right:8px; margin-top:10px; }

/* Metric cards */
.metric-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px; padding: 1.2rem 1rem; text-align: center; }
.metric-card .val { font-size: 2rem; font-weight: 700; color: #e2e8f0; }
.metric-card .lbl { font-size: 0.75rem; color: #64748b; text-transform: uppercase;
  letter-spacing: 0.08em; margin-top: 4px; }

/* Risk card */
.risk-card { background: rgba(255,255,255,0.03); border-left: 3px solid;
  border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0; }
.risk-high { border-color: #fc8181; background: rgba(252,129,129,0.06); }
.risk-medium { border-color: #f6ad55; background: rgba(246,173,85,0.06); }
.risk-low { border-color: #68d391; background: rgba(104,211,145,0.06); }

/* Severity pill */
.pill { display:inline-block; padding:2px 10px; border-radius:20px;
  font-size:0.72rem; font-weight:700; letter-spacing:0.05em; }
.pill-high { background:#7f1d1d; color:#fca5a5; }
.pill-medium { background:#78350f; color:#fcd34d; }
.pill-low { background:#14532d; color:#86efac; }

/* Section headers */
.section-header { font-size:0.7rem; font-weight:700; color:#475569;
  text-transform:uppercase; letter-spacing:0.12em; margin:1.5rem 0 0.8rem; }

/* Answer box */
.answer-box { background: rgba(99,179,237,0.06); border: 1px solid rgba(99,179,237,0.2);
  border-radius: 12px; padding: 1.2rem 1.4rem; margin: 1rem 0; }

/* Privacy notice */
.privacy-notice { background: rgba(104,211,145,0.06); border: 1px solid rgba(104,211,145,0.2);
  border-radius: 8px; padding: 0.6rem 1rem; font-size: 0.78rem; color: #86efac; margin-bottom: 1rem; }

/* Footer */
.footer { text-align:center; padding:2rem 0 1rem; color:#334155;
  font-size:0.78rem; border-top:1px solid rgba(255,255,255,0.06); margin-top:3rem; }
.footer a { color:#475569; text-decoration:none; }
.footer a:hover { color:#94a3b8; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid rgba(255,255,255,0.06); }
section[data-testid="stSidebar"] .stButton>button { width:100%; background:linear-gradient(135deg,#1e40af,#1d4ed8);
  color:white; border:none; border-radius:8px; padding:0.6rem; font-weight:600;
  font-size:0.9rem; transition:all 0.2s; }
section[data-testid="stSidebar"] .stButton>button:hover { background:linear-gradient(135deg,#1d4ed8,#2563eb); transform:translateY(-1px); }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background:rgba(255,255,255,0.03); border-radius:10px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] { border-radius:8px; color:#64748b; font-weight:500; font-size:0.85rem; padding:0.5rem 1rem; }
.stTabs [aria-selected="true"] { background:rgba(99,179,237,0.15) !important; color:#63b3ed !important; }

/* Expander */
.streamlit-expanderHeader { background:rgba(255,255,255,0.03) !important; border-radius:8px !important; }

/* Input */
.stTextInput>div>div>input { background:rgba(255,255,255,0.05) !important;
  border:1px solid rgba(255,255,255,0.1) !important; border-radius:8px !important; color:#e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
SEV_COLORS = {"High": ("pill-high","risk-high"), "Medium": ("pill-medium","risk-medium"), "Low": ("pill-low","risk-low")}

def pill(level):
    pc = SEV_COLORS.get(level, ("pill-low","risk-low"))[0]
    return f'<span class="pill {pc}">{level}</span>'

def risk_card(clause, reasons, severity):
    rc = SEV_COLORS.get(severity, ("","risk-medium"))[1]
    reasons_html = " &nbsp;·&nbsp; ".join(reasons)
    return f'<div class="risk-card {rc}"><b style="color:#e2e8f0">{clause}</b> &nbsp; {pill(severity)}<br><span style="color:#94a3b8;font-size:0.82rem">{reasons_html}</span></div>'

# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>ContractIQ</h1>
  <p>AI-powered contract clause detection, deviation analysis &amp; explainable risk scoring</p>
  <span class="badge">DeBERTa-v3</span>
  <span class="badge">33 Clause Types</span>
  <span class="badge">Privacy-Safe LLM</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">CRAG Engine Status</p>', unsafe_allow_html=True)
    if llm_source == "groq":
        st.success("🟢 CRAG + Groq active")
        st.caption("Retrieval: local ML · Generation: Groq llama-3.1-8b")
    elif llm_source == "ollama":
        st.success("🟢 CRAG + Ollama active")
        st.caption("Fully local · No data leaves device")
    else:
        st.info("🔍 CRAG deterministic mode")
        st.caption("Add GROQ_API_KEY for LLM generation")
    mode = st.radio("Mode", ["Single Document", "Multi-Document"], label_visibility="collapsed", horizontal=True)
    st.session_state.analysis_mode = "multi" if mode == "Multi-Document" else "single"

    st.markdown('<p class="section-header">Upload Contract(s)</p>', unsafe_allow_html=True)
    if st.session_state.analysis_mode == "single":
        uploaded_pdfs = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed", accept_multiple_files=False)
        uploaded_pdfs = [uploaded_pdfs] if uploaded_pdfs else []
    else:
        st.caption("Upload 2–5 contracts to compare")
        uploaded_pdfs = st.file_uploader("Upload PDFs", type=["pdf"], label_visibility="collapsed", accept_multiple_files=True)

    analyze_clicked = st.button("▶  Analyze Contract(s)", use_container_width=True)

    if st.session_state.analyzed:
        st.markdown('<p class="section-header">Export</p>', unsafe_allow_html=True)
        clause_df = st.session_state.clause_df
        spans = st.session_state.spans
        summary = st.session_state.contract_summary
        st.download_button("⬇ CSV Report", data=export_results_csv(clause_df, spans),
                           file_name="contractiq_analysis.csv", mime="text/csv", use_container_width=True)
        st.download_button("⬇ JSON Report", data=export_results_json(clause_df, spans, summary),
                           file_name="contractiq_analysis.json", mime="application/json", use_container_width=True)
    st.markdown("---")
    st.markdown('<p style="color:#334155;font-size:0.72rem;text-align:center">ContractIQ v2.0<br>© 2026 Team 2022AIE01<br>Not legal advice</p>', unsafe_allow_html=True)

# ── Analyze ────────────────────────────────────────────────────────────────────
if analyze_clicked:
    if not uploaded_pdfs:
        st.error("Please upload at least one PDF file.")
    else:
        is_multi = st.session_state.analysis_mode == "multi" and len(uploaded_pdfs) > 1
        progress_bar = st.progress(0, text="Initializing analysis pipeline...")

        def update_progress(step, total, msg):
            pct = int((step / max(total, 1)) * 100)
            progress_bar.progress(pct, text=msg)

        # Always analyze the first (or only) doc as the primary
        # Read all file bytes upfront — file objects become unreadable after rerun
        all_pdf_bytes = [(f.name, f.read()) for f in uploaded_pdfs]
        primary_name, primary_bytes = all_pdf_bytes[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(primary_bytes)
            pdf_path = tmp.name

        clause_df, spans, embeddings, embedder = analyze_document(pdf_path, progress_callback=update_progress)
        progress_bar.progress(100, text="Finalizing primary document...")
        clause_df["_known"] = clause_df["final_clause"] != "Unknown"
        clause_df = clause_df.sort_values(by=["_known","span_id"], ascending=[False,True]).drop(columns="_known").reset_index(drop=True)
        contract_summary = build_contract_summary(clause_df, spans)
        summary_narration = narrate_contract_summary(contract_summary, llm_client=llm)

        st.session_state.clause_df = clause_df
        st.session_state.spans = spans
        st.session_state.embeddings = embeddings
        st.session_state.embedder = embedder
        st.session_state.contract_summary = contract_summary
        st.session_state.summary_narration = summary_narration
        st.session_state.last_answer = None
        st.session_state.analyzed = True
        st.session_state.contract_doc_summary = None  # clear stale summary on new analysis
        st.session_state.multi_doc_results = None

        # Multi-doc: analyze remaining documents
        if is_multi:
            doc_results = [{
                "name": primary_name,
                "clause_df": clause_df,
                "spans": spans,
                "summary": contract_summary,
            }]
            for i, (extra_name, extra_bytes) in enumerate(all_pdf_bytes[1:], 2):
                progress_bar.progress(0, text=f"Analyzing document {i}/{len(all_pdf_bytes)}: {extra_name}...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(extra_bytes)
                    extra_path = tmp.name
                extra_df, extra_spans, _, _ = analyze_document(extra_path)
                extra_df["_known"] = extra_df["final_clause"] != "Unknown"
                extra_df = extra_df.sort_values(by=["_known","span_id"], ascending=[False,True]).drop(columns="_known").reset_index(drop=True)
                extra_summary = build_contract_summary(extra_df, extra_spans)
                doc_results.append({
                    "name": extra_name,
                    "clause_df": extra_df,
                    "spans": extra_spans,
                    "summary": extra_summary,
                })
            st.session_state.multi_doc_results = aggregate_documents(doc_results)
            st.session_state.multi_doc_raw = doc_results

        st.session_state["_active_tab"] = "overview"
        progress_bar.empty()
        st.rerun()
if st.session_state.analyzed:
    clause_df = st.session_state.clause_df
    spans = st.session_state.spans

    # ── Document selector for multi-doc mode ──────────────────────────────
    _raw_docs = st.session_state.get("multi_doc_raw", [])
    if _raw_docs and len(_raw_docs) > 1:
        _doc_names = [d["name"] for d in _raw_docs]
        _selected_doc = st.selectbox("📄 Viewing document:", _doc_names, key="doc_selector")
        _sel_idx = _doc_names.index(_selected_doc)
        _sel = _raw_docs[_sel_idx]
        clause_df = _sel["clause_df"]
        spans = _sel["spans"]
        summary = _sel["summary"]
        # Rebuild narration for selected doc
        if _sel_idx == 0:
            _narration = st.session_state.summary_narration
        else:
            _narration = (f"Viewing: {_selected_doc}. " +
                f"{summary['overview']['deviating_spans']} deviating clause(s) detected " +
                f"out of {summary['overview']['recognized_clauses']} recognized.")
    else:
        _narration = st.session_state.summary_narration

    summary = st.session_state.contract_summary
    _active = st.session_state.get("_active_tab", st.query_params.get("tab", "overview"))
    _has_multidoc = bool(st.session_state.get("multi_doc_results"))
    _tab_labels = ["  Overview  ","  Deviating Clauses  ","  Risk Analysis  ","  Analytics  ","  Summary  ","  Ask the Contract  ","  Negotiate  ","  Obligation Graph  "]
    if _has_multidoc: _tab_labels.append("  Multi-Doc  ")
    _TAB_IDX = {k:i for i,k in enumerate(["overview","deviations","risk","analytics","summary","ask","negotiate","obgraph"] + (["multidoc"] if _has_multidoc else []))}
    _idx = _TAB_IDX.get(_active, 0)
    # If answer was just generated, force Ask tab regardless of _active_tab
    if st.session_state.get('_force_ask_tab'):
        _idx = _TAB_IDX.get('ask', 5)
        st.session_state['_force_ask_tab'] = False
    _tabs = st.tabs(_tab_labels)
    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = _tabs[:8]
    tab9 = _tabs[8] if _has_multidoc else None

    if _idx > 0:
        st.components.v1.html(f"""<script>
function clickTab(){{
    var t=window.parent.document.querySelectorAll('[data-baseweb="tab"]');
    if(t.length>{_idx}){{t[{_idx}].click();}}else{{setTimeout(clickTab,150);}}
}}
setTimeout(clickTab,300);
</script>""", height=0)

    # ── TAB 1: OVERVIEW ──────────────────────────────────────────────────────
    with tab1:
        st.markdown('<p class="section-header">Executive Summary</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box"><p style="color:#cbd5e1;line-height:1.7">{_narration}</p></div>', unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        for col, val, lbl in [
            (c1, summary["overview"]["total_spans"], "Total Sections"),
            (c2, summary["overview"]["recognized_clauses"], "Recognized"),
            (c3, summary["overview"]["unknown_spans"], "Unclassified"),
            (c4, summary["overview"]["deviating_spans"], "Deviating"),
        ]:
            col.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("")

        st.markdown('<p class="section-header">Clause Coverage</p>', unsafe_allow_html=True)
        counts = summary["coverage"].get("clause_counts", {})
        coverage_html = " &nbsp; ".join(
            f'<span style="background:rgba(99,179,237,0.1);color:#63b3ed;padding:3px 10px;border-radius:20px;font-size:0.78rem">{c} <b>{counts.get(c,"?")}</b></span>'
            for c in summary["coverage"]["detected_clauses"]
        )
        st.markdown(f'<div style="line-height:2.2">{coverage_html}</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#334155;font-size:0.75rem;margin-top:1.5rem">Deviation detection uses semantic similarity, polarity analysis, and clause-specific rules. Not legal advice.</p>', unsafe_allow_html=True)
    # ── TAB 2: DEVIATING CLAUSES ─────────────────────────────────────────────
    with tab2:
        deviating = clause_df[clause_df["final_deviation"]]
        if deviating.empty:
            st.success("✅ No deviating clauses detected in this contract.")
        else:
            high = int((deviating["severity"]=="High").sum())
            med  = int((deviating["severity"]=="Medium").sum())
            low  = int((deviating["severity"]=="Low").sum())
            c1,c2,c3,c4 = st.columns(4)
            c1.markdown(f'<div class="metric-card"><div class="val">{len(deviating)}</div><div class="lbl">Total Deviations</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="val" style="color:#fc8181">{high}</div><div class="lbl">High Severity</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="val" style="color:#f6ad55">{med}</div><div class="lbl">Medium Severity</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric-card"><div class="val" style="color:#68d391">{low}</div><div class="lbl">Low Severity</div></div>', unsafe_allow_html=True)
            st.markdown("")

            from pipeline import explain_deviation_reasons
            sev_order = {"High":0,"Medium":1,"Low":2,None:3}
            sorted_dev = deviating.copy()
            sorted_dev["_so"] = sorted_dev["severity"].map(sev_order)
            sorted_dev = sorted_dev.sort_values("_so").drop(columns="_so")

            for _, row in sorted_dev.iterrows():
                sid = int(row["span_id"])
                sev = row.get("severity","Medium")
                score = row.get("deviation_score", 0.0)
                rc = SEV_COLORS.get(sev,("","risk-medium"))[1]
                with st.expander(f"{row['final_clause']}  ·  Span {sid}  ·  {sev} severity  ·  score {score:.2f}"):
                    st.markdown(f'<div class="risk-card {rc}">', unsafe_allow_html=True)
                    st.markdown(f"**Severity:** {pill(sev)}", unsafe_allow_html=True)
                    st.markdown("**Deviation signals:**")
                    for r in row["deviation_reasons"]:
                        st.markdown(f'<span style="color:#94a3b8;font-size:0.85rem">→ {r}</span>', unsafe_allow_html=True)
                    st.markdown("**Why this matters:**")
                    for exp in explain_deviation_reasons(row["deviation_reasons"]):
                        st.markdown(f'<div style="background:rgba(255,255,255,0.03);border-radius:6px;padding:0.6rem 0.8rem;color:#94a3b8;font-size:0.83rem;margin:0.3rem 0">{exp}</div>', unsafe_allow_html=True)
                    st.markdown("**Clause text:**")
                    st.markdown(f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.8rem;color:#cbd5e1;font-size:0.85rem;line-height:1.6;border-left:3px solid rgba(99,179,237,0.3)">{spans[sid]}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    # ── TAB 3: ANALYTICS ─────────────────────────────────────────────────────

    # ── TAB 3: RISK ANALYSIS ──────────────────────────────────────────────
    with tab3:
        if summary["deviations"]:
            st.markdown('<p class="section-header">Risk Snapshot</p>', unsafe_allow_html=True)
            for d in summary["deviations"]:
                st.markdown(risk_card(d["clause"], d["reasons"], d.get("severity","Medium")), unsafe_allow_html=True)
        else:
            st.success("✅ No non-standard clause patterns detected.")


        st.markdown('<p class="section-header">Clause Text Mapping</p>', unsafe_allow_html=True)
        for clause_name in sorted(clause_df["final_clause"].unique()):
            if clause_name == "Unknown":
                continue
            rows = clause_df[clause_df["final_clause"] == clause_name]
            has_dev = rows["final_deviation"].any()
            icon = "⚠️ " if has_dev else "✅ "
            with st.expander(f"{icon}{clause_name}  ({len(rows)} span{'s' if len(rows)>1 else ''})"):
                for _, row in rows.iterrows():
                    sid = int(row["span_id"])
                    conf = row["confidence"]
                    sev = row.get("severity")
                    dev_badge = f' &nbsp; {pill(sev)}' if row["final_deviation"] and sev else ""
                    st.markdown(f'<span style="color:#64748b;font-size:0.8rem">Span {sid} &nbsp;·&nbsp; confidence <b style="color:#63b3ed">{conf:.0%}</b>{dev_badge}</span>', unsafe_allow_html=True)
                    st.markdown(f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.8rem;margin:0.4rem 0;color:#cbd5e1;font-size:0.88rem;line-height:1.6">{spans[sid]}</div>', unsafe_allow_html=True)

        st.markdown('<p style="color:#334155;font-size:0.75rem;margin-top:2rem">Deviation detection uses semantic similarity, polarity analysis, and clause-specific rules. Not legal advice.</p>', unsafe_allow_html=True)


    with tab4:
        import pandas as pd
        st.markdown('<p class="section-header">Confidence Distribution</p>', unsafe_allow_html=True)
        known_df = clause_df[clause_df["final_clause"] != "Unknown"]
        if not known_df.empty:
            conf_chart = (known_df.groupby("final_clause")["confidence"].mean()
                          .reset_index()
                          .rename(columns={"final_clause":"Clause","confidence":"Avg Confidence"})
                          .sort_values("Avg Confidence", ascending=False))
            st.bar_chart(conf_chart.set_index("Clause"))

        st.markdown('<p class="section-header">Clause Frequency</p>', unsafe_allow_html=True)
        freq = (clause_df["final_clause"].value_counts()
                .reset_index()
                .rename(columns={"final_clause":"Clause","count":"Count"}))
        st.bar_chart(freq.set_index("Clause"))

        st.markdown('<p class="section-header">Full Deviation Table</p>', unsafe_allow_html=True)
        rows_t = []
        for c in sorted(clause_df["final_clause"].unique()):
            if c == "Unknown": continue
            s = clause_df[clause_df["final_clause"]==c]
            rows_t.append({"Clause":c,"Spans":len(s),"Deviating":int(s["final_deviation"].sum()),
                           "Avg Confidence":f"{s['confidence'].mean():.0%}",
                           "Max Dev Score":f"{s['deviation_score'].max():.2f}" if "deviation_score" in s else "—"})
        if rows_t:
            st.dataframe(pd.DataFrame(rows_t), use_container_width=True, hide_index=True)


    with tab5:
        st.markdown('<p class="section-header">Document Summary</p>', unsafe_allow_html=True)
        st.markdown("""<div style="background:rgba(104,211,145,0.04);border:1px solid rgba(104,211,145,0.15);border-radius:8px;padding:0.7rem 1rem;font-size:0.8rem;color:#68d391;margin-bottom:1rem">
🔒 Fully local — uses <b>DistilBART (sshleifer/distilbart-cnn-12-6)</b> for abstractive summarization. No document content sent externally.<br>
<span style="color:#94a3b8">Workflow: spans grouped by clause type → DistilBART generates 2-4 sentence abstract per group → regex extracts template fields → ROUGE evaluation</span>
</div>""", unsafe_allow_html=True)

        if st.button("Generate Summary", use_container_width=False, key="gen_summary"):
            with st.spinner("Generating summary..."):
                doc_summary = summarize_contract(spans, clause_df, st.session_state.embedder, summary)
                metrics = evaluate_summary(doc_summary, spans)
                st.session_state.contract_doc_summary = {"summary": doc_summary, "metrics": metrics}
                st.session_state["_active_tab"] = "summary"

        if "contract_doc_summary" in st.session_state and st.session_state.contract_doc_summary and "overall_summary" in st.session_state.contract_doc_summary.get("summary", {}):
            ds = st.session_state.contract_doc_summary["summary"]
            mt = st.session_state.contract_doc_summary["metrics"]

            # Overall summary
            st.markdown('<p class="section-header">Overall Summary</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box"><p style="color:#cbd5e1;line-height:1.8">{ds["overall_summary"]}</p></div>', unsafe_allow_html=True)
            st.caption(f"Model: {ds['model']}")

            # Template fields
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p class="section-header">Parties Involved</p>', unsafe_allow_html=True)
                if ds["parties"]:
                    for p in ds["parties"]:
                        st.markdown(f'<span style="background:rgba(99,179,237,0.1);color:#63b3ed;padding:3px 10px;border-radius:20px;font-size:0.82rem;display:inline-block;margin:2px">{p}</span>', unsafe_allow_html=True)
                else:
                    st.caption("No named parties detected")

                st.markdown('<p class="section-header">Key Dates</p>', unsafe_allow_html=True)
                if ds["effective_date"]:
                    st.markdown(f'📅 **Effective:** {ds["effective_date"]}')
                if ds["expiry_date"] and ds["expiry_date"] != ds["effective_date"]:
                    st.markdown(f'📅 **Expiry/End:** {ds["expiry_date"]}')
                if not ds["effective_date"]:
                    st.caption("No dates detected")

                st.markdown('<p class="section-header">Governing Law</p>', unsafe_allow_html=True)
                st.write(ds["governing_law"] or "Not explicitly stated")

                st.markdown('<p class="section-header">Defined Terms</p>', unsafe_allow_html=True)
                if ds["key_terms"]:
                    terms_html = " ".join(f'<span style="background:rgba(255,255,255,0.05);color:#94a3b8;padding:2px 8px;border-radius:4px;font-size:0.78rem;margin:2px;display:inline-block">{t}</span>' for t in ds["key_terms"])
                    st.markdown(terms_html, unsafe_allow_html=True)
                else:
                    st.caption("No defined terms detected")

            with c2:
                st.markdown('<p class="section-header">Key Obligations</p>', unsafe_allow_html=True)
                for ob in (ds["obligations"] or ["None detected"]):
                    st.markdown(f'<div style="background:rgba(255,255,255,0.03);border-left:2px solid #63b3ed;padding:0.5rem 0.8rem;border-radius:0 6px 6px 0;color:#94a3b8;font-size:0.82rem;margin:0.3rem 0">{ob}</div>', unsafe_allow_html=True)

                st.markdown('<p class="section-header">Key Rights</p>', unsafe_allow_html=True)
                for r in (ds["rights"] or ["None detected"]):
                    st.markdown(f'<div style="background:rgba(255,255,255,0.03);border-left:2px solid #68d391;padding:0.5rem 0.8rem;border-radius:0 6px 6px 0;color:#94a3b8;font-size:0.82rem;margin:0.3rem 0">{r}</div>', unsafe_allow_html=True)

                st.markdown('<p class="section-header">Payment Terms</p>', unsafe_allow_html=True)
                for pt in (ds["payment_terms"] or ["None detected"]):
                    st.markdown(f'<div style="background:rgba(255,255,255,0.03);border-left:2px solid #f6ad55;padding:0.5rem 0.8rem;border-radius:0 6px 6px 0;color:#94a3b8;font-size:0.82rem;margin:0.3rem 0">{pt}</div>', unsafe_allow_html=True)

            st.markdown('<p class="section-header">Termination Conditions</p>', unsafe_allow_html=True)
            for tc in (ds["termination"] or ["None detected"]):
                st.markdown(f'<div style="background:rgba(255,255,255,0.03);border-left:2px solid #fc8181;padding:0.5rem 0.8rem;border-radius:0 6px 6px 0;color:#94a3b8;font-size:0.82rem;margin:0.3rem 0">{tc}</div>', unsafe_allow_html=True)

            # Per-clause abstractive summaries
            # HRS Tree — Category → Clause hierarchy
            if ds.get("clause_summaries"):
                st.markdown('<p class="section-header">HRS Summary Tree — Hierarchical Recursive Summarization</p>', unsafe_allow_html=True)
                st.caption(f"Model: {ds.get('model', 'N/A')}")

                # Show category-level summaries with clause drill-down
                hrs_tree = ds.get("hrs_tree", {})
                level2 = hrs_tree.get("level_2_categories", {})

                if level2:
                    for cat, cat_data in level2.items():
                        cat_sum = cat_data.get("summary", "")
                        level1 = cat_data.get("level_1_clauses", {})
                        has_dev = any(
                            any(d["clause"] == c for d in ds["risk_flags"])
                            for c in level1.keys()
                        )
                        icon = "⚠️ " if has_dev else "📁 "
                        with st.expander(f"{icon}{cat}  ({len(level1)} clause{'s' if len(level1)>1 else ''})"):
                            st.markdown(f'<div style="background:rgba(99,179,237,0.06);border-left:3px solid #63b3ed;padding:0.6rem 1rem;border-radius:0 8px 8px 0;color:#cbd5e1;font-size:0.88rem;line-height:1.7;margin-bottom:0.8rem">{cat_sum}</div>', unsafe_allow_html=True)
                            for clause, clause_sum in level1.items():
                                dev = any(d["clause"] == clause for d in ds["risk_flags"])
                                c_icon = "⚠️" if dev else "✅"
                                st.markdown(f'<div style="margin-left:1rem;margin-bottom:0.4rem"><span style="color:#64748b;font-size:0.78rem">{c_icon} {clause}</span><div style="background:rgba(255,255,255,0.02);border-radius:6px;padding:0.5rem 0.8rem;color:#94a3b8;font-size:0.82rem;line-height:1.6">{clause_sum}</div></div>', unsafe_allow_html=True)
                else:
                    # Flat view fallback
                    for clause, clause_sum in ds["clause_summaries"].items():
                        dev = any(d["clause"] == clause for d in ds["risk_flags"])
                        icon = "⚠️ " if dev else "✅ "
                        with st.expander(f"{icon}{clause}"):
                            st.markdown(f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.8rem 1rem;color:#cbd5e1;font-size:0.88rem;line-height:1.7">{clause_sum}</div>', unsafe_allow_html=True)
                st.markdown('<p class="section-header">Risk Flags</p>', unsafe_allow_html=True)
                for rf in ds["risk_flags"]:
                    sev = rf.get("severity","Medium")
                    rc = {"High":"risk-high","Medium":"risk-medium","Low":"risk-low"}.get(sev,"risk-medium")
                    st.markdown(f'<div class="risk-card {rc}">{pill(sev)} <b style="color:#e2e8f0">{rf["clause"]}</b><br><span style="color:#94a3b8;font-size:0.8rem">{"; ".join(rf["reasons"])}</span></div>', unsafe_allow_html=True)

            # Metrics
            st.markdown('<p class="section-header">Summary Quality Metrics</p>', unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.4rem">{mt["rouge_1"]["f1"]:.3f}</div><div class="lbl">ROUGE-1 F1</div></div>', unsafe_allow_html=True)
            mc2.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.4rem">{mt["rouge_2"]["f1"]:.3f}</div><div class="lbl">ROUGE-2 F1</div></div>', unsafe_allow_html=True)
            mc3.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.4rem">{mt["coverage"]:.3f}</div><div class="lbl">Coverage</div></div>', unsafe_allow_html=True)
            mc4.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.4rem">{mt["compression_ratio"]:.3f}</div><div class="lbl">Compression</div></div>', unsafe_allow_html=True)
            st.caption(f"Summarized {mt['summary_clauses']} clause groups from {mt['reference_spans']} total spans · ROUGE measures n-gram overlap with original text")
    with tab6:
        st.markdown('<p class="section-header">Ask a Question</p>', unsafe_allow_html=True)

        # CRAG status banner
        if llm_source in ("groq", "ollama"):
            st.markdown(f"""<div style="background:rgba(104,211,145,0.06);border:1px solid rgba(104,211,145,0.2);border-radius:8px;padding:0.7rem 1rem;font-size:0.8rem;color:#68d391;margin-bottom:1rem">
🔒 <b>CRAG active</b> — Constrained Retrieval-Augmented Generation. Every answer is grounded in verified contract spans with hallucination scoring.
Powered by <b>{"Groq · llama-3.1-8b-instant" if llm_source=="groq" else "Ollama (local)"}</b>.
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background:rgba(99,179,237,0.06);border:1px solid rgba(99,179,237,0.2);border-radius:8px;padding:0.7rem 1rem;font-size:0.8rem;color:#63b3ed;margin-bottom:1rem">
🔍 <b>CRAG deterministic mode</b> — answers derived from verified contract evidence without LLM. Add GROQ_API_KEY for natural language generation.
</div>""", unsafe_allow_html=True)

        question = st.text_input("Question", placeholder="e.g. What are the risks? What does the liability clause say? Who are the parties?", label_visibility="collapsed")

        if st.button("Ask ContractIQ", use_container_width=False):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                st.session_state["_active_tab"] = "ask"
                st.session_state["_force_ask_tab"] = True
                with st.spinner("Running CRAG pipeline..."):
                    result = crag_answer(
                        question, clause_df, spans,
                        st.session_state.embeddings,
                        st.session_state.embedder,
                        llm_fn=llm
                    )
                    st.session_state.last_answer = result

        if st.session_state.last_answer:
            ans = st.session_state.last_answer

            # Grounding score banner
            gs = ans.get("grounding_score", 0)
            hr = ans.get("hallucination_risk", "Unknown")
            hr_color = {"Low": "#68d391", "Medium": "#f6ad55", "High": "#fc8181"}.get(hr, "#94a3b8")
            intent = ans.get("intent", "").replace("_", " ").title()
            st.markdown(f"""<div style="display:flex;gap:1rem;margin-bottom:0.8rem;flex-wrap:wrap">
<span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:3px 10px;font-size:0.78rem;color:#94a3b8">Intent: <b style="color:#e2e8f0">{intent}</b></span>
<span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:3px 10px;font-size:0.78rem;color:#94a3b8">Grounding: <b style="color:#63b3ed">{gs:.2f}</b></span>
<span style="background:rgba(255,255,255,0.04);border:1px solid {hr_color}33;border-radius:6px;padding:3px 10px;font-size:0.78rem;color:{hr_color}">Hallucination Risk: <b>{hr}</b></span>
<span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:3px 10px;font-size:0.78rem;color:#94a3b8">Evidence: <b style="color:#e2e8f0">{len(ans.get("evidence", []))} spans</b></span>
</div>""", unsafe_allow_html=True)

            if not ans.get("is_answerable", True):
                st.warning("⚠️ Insufficient evidence to answer this question reliably from the contract.")

            st.markdown('<p class="section-header">Answer</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box"><p style="color:#cbd5e1;line-height:1.8;white-space:pre-wrap">{ans["answer"]}</p></div>', unsafe_allow_html=True)

            if ans.get("evidence"):
                st.markdown('<p class="section-header">Verified Evidence (Citations)</p>', unsafe_allow_html=True)
                for ev in ans["evidence"]:
                    sev = ev.get("severity")
                    sim = ev.get("similarity", 0)
                    dev_tag = f" ⚠️ {sev}" if ev["deviating"] else ""
                    with st.expander(f"[SPAN {ev['span_id']}] {ev['clause']}{dev_tag}  ·  relevance {sim:.0%}"):
                        if ev["deviating"] and ev["deviation_reasons"]:
                            for r in ev["deviation_reasons"]:
                                st.markdown(f'<span style="color:#f6ad55;font-size:0.83rem">⚡ {r}</span>', unsafe_allow_html=True)
                        st.markdown(f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.8rem;color:#94a3b8;font-size:0.85rem;line-height:1.6;margin-top:0.5rem">{ev["text"]}</div>', unsafe_allow_html=True)

            st.markdown("---")
            for note in ans.get("confidence_notes", []):
                st.caption(note)


    # ── TAB 7: NEGOTIATE ─────────────────────────────────────────────────────
    with tab7:
        st.markdown('<p class="section-header">Clause Negotiation Simulator</p>', unsafe_allow_html=True)
        st.markdown("""<div style="background:rgba(99,179,237,0.06);border:1px solid rgba(99,179,237,0.2);border-radius:8px;padding:0.8rem 1rem;font-size:0.82rem;color:#94a3b8;margin-bottom:1rem">
⚖️ <b style="color:#e2e8f0">Novel feature</b> — Select a deviating clause and generate alternative phrasings at three negotiation stances.
Each alternative is scored for how much it moves the clause toward standard language (similarity improvement).
<b style="color:#68d391">This feature does not exist in any commercial contract analysis tool.</b>
</div>""", unsafe_allow_html=True)

        deviating = clause_df[clause_df["final_deviation"]]
        if deviating.empty:
            st.success("No deviating clauses to negotiate — this contract has no flagged deviations.")
        else:
            # Clause selector
            dev_options = []
            for _, row in deviating.iterrows():
                sid = int(row["span_id"])
                sev = row.get("severity", "Medium")
                score = row.get("deviation_score", 0.0)
                dev_options.append(f"{row['final_clause']} | Span {sid} | {sev} | score {score:.2f}")

            selected = st.selectbox("Select a deviating clause to negotiate:", dev_options, key="neg_selector")
            sel_idx = dev_options.index(selected)
            sel_row = deviating.iloc[sel_idx]
            sel_sid = int(sel_row["span_id"])
            sel_clause = sel_row["final_clause"]
            sel_reasons = sel_row.get("deviation_reasons", [])
            sel_score = float(sel_row.get("deviation_score", 0.0))
            sel_text = spans[sel_sid]

            # Show original clause
            st.markdown('<p class="section-header">Original Clause</p>', unsafe_allow_html=True)
            sev = sel_row.get("severity", "Medium")
            rc = {"High":"risk-high","Medium":"risk-medium","Low":"risk-low"}.get(sev,"risk-medium")
            st.markdown(f'<div class="risk-card {rc}">{pill(sev)} <b style="color:#e2e8f0">{sel_clause}</b> · deviation score {sel_score:.2f}<br><span style="color:#94a3b8;font-size:0.8rem">{"; ".join(sel_reasons)}</span></div>', unsafe_allow_html=True)
            with st.expander("View original text"):
                st.markdown(f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.8rem;color:#94a3b8;font-size:0.85rem;line-height:1.6">{sel_text}</div>', unsafe_allow_html=True)

            if st.button("⚖️ Generate Negotiation Alternatives", use_container_width=False, key="neg_btn"):
                with st.spinner("Generating alternatives at 3 negotiation stances..."):
                    from pipeline import load_baselines
                    _centroids, _, _, _, _ = load_baselines()
                    neg_results = simulate_negotiation(
                        sel_clause, sel_text, sel_reasons, sel_score,
                        st.session_state.embedder, _centroids, llm_fn=llm
                    )
                    st.session_state["neg_results"] = neg_results
                    st.session_state["neg_clause"] = sel_clause

            if st.session_state.get("neg_results"):
                st.markdown('<p class="section-header">Negotiation Alternatives</p>', unsafe_allow_html=True)
                orig_sim = st.session_state["neg_results"][0].get("original_similarity")
                if orig_sim is not None:
                    st.caption(f"Original clause similarity to standard '{st.session_state['neg_clause']}' language: {orig_sim:.3f}")

                for res in st.session_state["neg_results"]:
                    imp = res.get("similarity_improvement")
                    imp_str = f" · similarity +{imp:.3f}" if imp and imp > 0 else (f" · similarity {imp:.3f}" if imp else "")
                    with st.expander(f"{res['icon']} {res['stance']} stance{imp_str}"):
                        st.markdown(f'<div style="color:{res["color"]};font-size:0.8rem;margin-bottom:0.5rem">{res["description"]}</div>', unsafe_allow_html=True)
                        st.markdown("**Rewritten clause:**")
                        st.markdown(f'<div style="background:rgba(255,255,255,0.04);border-left:3px solid {res["color"]};border-radius:0 8px 8px 0;padding:0.8rem 1rem;color:#cbd5e1;font-size:0.87rem;line-height:1.7;margin:0.5rem 0">{res["rewritten"]}</div>', unsafe_allow_html=True)
                        if res.get("explanation"):
                            st.markdown("**What changed:**")
                            st.markdown(f'<div style="background:rgba(255,255,255,0.02);border-radius:6px;padding:0.6rem 0.8rem;color:#94a3b8;font-size:0.83rem;line-height:1.6">{res["explanation"]}</div>', unsafe_allow_html=True)
                        if imp is not None:
                            direction = "↑ closer to standard" if imp > 0 else "↓ further from standard" if imp < 0 else "→ no change"
                            st.caption(f"Similarity to standard language: {res.get('new_similarity', 0):.3f} ({direction})")

                st.caption("Note: These are AI-generated suggestions for negotiation purposes only. Not legal advice.")


    # ── TAB 8: OBLIGATION GRAPH ───────────────────────────────────────────────
    with tab8:
        st.markdown('<p class="section-header">Obligation Graph</p>', unsafe_allow_html=True)
        st.markdown("""<div style="background:rgba(99,179,237,0.06);border:1px solid rgba(99,179,237,0.2);border-radius:8px;padding:0.8rem 1rem;font-size:0.82rem;color:#94a3b8;margin-bottom:1rem">
🔗 <b style="color:#e2e8f0">Novel feature</b> — Visualizes the structural balance of obligations in the contract.
Extracts who owes what to whom, detects one-sided contracts, and flags missing reciprocal duties.
</div>""", unsafe_allow_html=True)

        if st.button("🔗 Build Obligation Graph", use_container_width=False, key="ob_btn"):
            with st.spinner("Extracting obligation relationships..."):
                ob_list = extract_obligations(spans, clause_df)
                ob_graph = build_obligation_graph(ob_list)
                st.session_state["ob_graph"] = ob_graph
                st.session_state["ob_list"] = ob_list

        if st.session_state.get("ob_graph"):
            import pandas as pd
            g = st.session_state["ob_graph"]
            ob_list = st.session_state.get("ob_list", [])

            if not g["nodes"]:
                st.warning("No obligation relationships could be extracted from this contract.")
            else:
                # ── Balance score ──────────────────────────────────────────
                bs = g["balance_score"]
                bs_color = "#68d391" if bs >= 0.6 else "#f6ad55" if bs >= 0.3 else "#fc8181"
                bs_label = "Balanced" if bs >= 0.6 else "Moderately One-Sided" if bs >= 0.3 else "Highly One-Sided"

                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f'<div class="metric-card"><div class="val" style="color:{bs_color}">{bs:.2f}</div><div class="lbl">Balance Score</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><div class="val">{len(g["nodes"])}</div><div class="lbl">Parties</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><div class="val">{len([o for o in ob_list if o["verb_type"]=="obligation"])}</div><div class="lbl">Obligations</div></div>', unsafe_allow_html=True)
                c4.markdown(f'<div class="metric-card"><div class="val">{len(g["missing_reciprocals"])}</div><div class="lbl">Missing Reciprocals</div></div>', unsafe_allow_html=True)

                st.markdown(f'<div style="text-align:center;color:{bs_color};font-size:0.85rem;margin:0.5rem 0">Contract obligation balance: <b>{bs_label}</b></div>', unsafe_allow_html=True)

                # ── Obligation matrix ──────────────────────────────────────
                st.markdown('<p class="section-header">Obligation Matrix (Party × Party)</p>', unsafe_allow_html=True)
                st.caption("Each cell shows how many obligations flow from the row party to the column party")

                parties = g["nodes"]
                if len(parties) > 1:
                    matrix_rows = []
                    for p_from in parties:
                        row = {"Party": p_from}
                        for p_to in parties:
                            if p_from == p_to:
                                row[p_to] = "—"
                            else:
                                count = g["adjacency"].get(p_from, {}).get(p_to, 0)
                                row[p_to] = count if count > 0 else "·"
                        matrix_rows.append(row)
                    st.dataframe(pd.DataFrame(matrix_rows).set_index("Party"), use_container_width=True)

                # ── Per-party breakdown ────────────────────────────────────
                st.markdown('<p class="section-header">Per-Party Obligation Breakdown</p>', unsafe_allow_html=True)
                party_rows = []
                for party, counts in g["obligation_counts"].items():
                    total = counts["obligation"] + counts["permission"] + counts["prohibition"]
                    party_rows.append({
                        "Party": party,
                        "Obligations (shall/must)": counts["obligation"],
                        "Permissions (may)": counts["permission"],
                        "Prohibitions (shall not)": counts["prohibition"],
                        "Total": total,
                    })
                party_rows.sort(key=lambda x: x["Total"], reverse=True)
                st.dataframe(pd.DataFrame(party_rows), use_container_width=True, hide_index=True)

                if g["dominant_party"]:
                    dom = g["dominant_party"]
                    dom_count = g["obligation_counts"][dom]["obligation"]
                    st.markdown(f'<div style="background:rgba(252,129,129,0.06);border-left:3px solid #fc8181;border-radius:0 8px 8px 0;padding:0.6rem 1rem;color:#94a3b8;font-size:0.85rem;margin:0.5rem 0">⚠️ <b style="color:#e2e8f0">{dom}</b> carries the most obligations ({dom_count} obligation statements)</div>', unsafe_allow_html=True)

                # ── Missing reciprocals ────────────────────────────────────
                if g["missing_reciprocals"]:
                    st.markdown('<p class="section-header">Missing Reciprocal Obligations</p>', unsafe_allow_html=True)
                    st.caption("These obligations exist for one party but have no corresponding duty for the other party")
                    for mr in g["missing_reciprocals"][:6]:
                        st.markdown(f'<div style="background:rgba(246,173,85,0.06);border-left:3px solid #f6ad55;border-radius:0 8px 8px 0;padding:0.6rem 1rem;margin:0.3rem 0"><span style="color:#f6ad55;font-size:0.82rem"><b>{mr["party_a"]}</b> has a "{mr["obligation"]}" obligation but <b>{mr["party_b"]}</b> does not</span><br><span style="color:#64748b;font-size:0.78rem">{mr["example"]}</span></div>', unsafe_allow_html=True)

                # ── Clause density ─────────────────────────────────────────
                if g["clause_density"]:
                    st.markdown('<p class="section-header">Obligation Density by Clause Type</p>', unsafe_allow_html=True)
                    density_df = pd.DataFrame([
                        {"Clause Type": k, "Obligation Count": v}
                        for k, v in list(g["clause_density"].items())[:10]
                    ])
                    st.bar_chart(density_df.set_index("Clause Type"))

                # ── Raw obligations table ──────────────────────────────────
                with st.expander(f"View all {len(ob_list)} extracted obligation statements"):
                    ob_df = pd.DataFrame([{
                        "Party": o["subject"],
                        "Type": o["verb_type"],
                        "Action": o["action"][:80],
                        "Clause": o["clause_type"],
                        "Span": o["span_id"],
                    } for o in ob_list])
                    st.dataframe(ob_df, use_container_width=True, hide_index=True)

                st.caption("Note: Party extraction uses pattern matching and may not capture all parties. Not legal advice.")

    # ── TAB 7: MULTI-DOC ─────────────────────────────────────────────────────
    # ── TAB 7: MULTI-DOC ─────────────────────────────────────────────────────
    if tab9 is not None:
        with tab9:
            st.markdown('<p class="section-header">Multi-Document Analysis</p>', unsafe_allow_html=True)
            if not st.session_state.get("multi_doc_results"):
                if st.session_state.analysis_mode == "multi":
                    st.info("Upload 2 or more contracts and click **Analyze Contract(s)** to compare them.")
                else:
                    st.markdown("""<div style="background:rgba(99,179,237,0.06);border:1px solid rgba(99,179,237,0.15);border-radius:10px;padding:1.2rem;color:#94a3b8;font-size:0.88rem">
Switch to <b style="color:#63b3ed">Multi-Document</b> mode in the sidebar to upload and compare multiple contracts simultaneously.<br><br>
Use cases: contract version comparison &middot; master + amendments &middot; vendor portfolio risk &middot; counterparty benchmarking.
</div>""", unsafe_allow_html=True)
            else:
                import pandas as pd
                agg = st.session_state.multi_doc_results
                raw = st.session_state.get("multi_doc_raw", [])

                st.markdown(f'<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:1rem">Comparing <b style="color:#63b3ed">{agg["n_docs"]} documents</b></div>', unsafe_allow_html=True)

                cols = st.columns(len(agg["doc_overviews"]))
                for col, ov in zip(cols, agg["doc_overviews"]):
                    dev_color = "#fc8181" if ov["deviation_rate"] > 0.3 else "#f6ad55" if ov["deviation_rate"] > 0.1 else "#68d391"
                    col.markdown(f"""<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:1rem;text-align:center">
<div style="font-size:0.75rem;color:#64748b;margin-bottom:0.4rem;word-break:break-all">{ov["document"][:30]}</div>
<div style="font-size:1.6rem;font-weight:700;color:{dev_color}">{ov["deviation_rate"]:.0%}</div>
<div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.06em">Deviation Rate</div>
<div style="font-size:0.78rem;color:#94a3b8;margin-top:0.4rem">{ov["deviating"]} / {ov["recognized"]} clauses</div>
</div>""", unsafe_allow_html=True)

                st.markdown('<p class="section-header" style="margin-top:1.5rem">Clause x Document Heatmap</p>', unsafe_allow_html=True)
                st.caption("⚠️ Deviating  ·  ✅ Present  ·  — Absent")
                heatmap_df = build_heatmap_dataframe(agg)
                st.dataframe(heatmap_df, use_container_width=True)

                if agg["systemic_risks"]:
                    st.markdown('<p class="section-header">🔴 Systemic Risks — Deviating Across All Documents</p>', unsafe_allow_html=True)
                    for sr in agg["systemic_risks"]:
                        st.markdown(f'<div style="background:rgba(252,129,129,0.08);border-left:3px solid #fc8181;border-radius:0 8px 8px 0;padding:0.7rem 1rem;margin:0.3rem 0;color:#e2e8f0;font-size:0.88rem"><b>{sr["clause"]}</b> <span style="color:#fc8181;font-size:0.78rem">— deviating in all {len(sr["affected_docs"])} documents</span></div>', unsafe_allow_html=True)
                else:
                    st.success("No systemic risks detected across all documents.")

                if agg["isolated_risks"]:
                    st.markdown('<p class="section-header">🟡 Isolated Risks — Deviating in Only One Document</p>', unsafe_allow_html=True)
                    for ir in agg["isolated_risks"]:
                        doc = ir["affected_docs"][0] if ir["affected_docs"] else "unknown"
                        st.markdown(f'<div style="background:rgba(246,173,85,0.06);border-left:3px solid #f6ad55;border-radius:0 8px 8px 0;padding:0.7rem 1rem;margin:0.3rem 0;color:#e2e8f0;font-size:0.88rem"><b>{ir["clause"]}</b> <span style="color:#f6ad55;font-size:0.78rem">— only in: {doc}</span></div>', unsafe_allow_html=True)

                if agg["risk_ranking"]:
                    st.markdown('<p class="section-header">Clause Risk Ranking</p>', unsafe_allow_html=True)
                    rank_rows = [{"Clause": c, "Deviation Rate": f"{r:.0%}", "Deviating In": f"{d}/{agg['clause_deviation_rates'][c]['present_in']} docs"} for c, r, d in agg["risk_ranking"]]
                    st.dataframe(pd.DataFrame(rank_rows), use_container_width=True, hide_index=True)

                if agg["top_reasons"]:
                    st.markdown('<p class="section-header">Most Common Deviation Signals</p>', unsafe_allow_html=True)
                    total_r = sum(c for _, c in agg["top_reasons"])
                    for reason, count in agg["top_reasons"][:6]:
                        pct = count / max(total_r, 1)
                        st.markdown(f'<div style="display:flex;align-items:center;gap:0.8rem;margin:0.3rem 0"><span style="color:#94a3b8;font-size:0.82rem;min-width:280px">{reason}</span><div style="flex:1;background:rgba(255,255,255,0.05);border-radius:4px;height:6px"><div style="background:#63b3ed;width:{pct:.0%};height:6px;border-radius:4px"></div></div><span style="color:#63b3ed;font-size:0.78rem;min-width:30px">{count}x</span></div>', unsafe_allow_html=True)

                if raw:
                    st.markdown('<p class="section-header">Per-Document Breakdown</p>', unsafe_allow_html=True)
                    for doc in raw:
                        dev_count = int(doc["clause_df"]["final_deviation"].sum())
                        total = len(doc["clause_df"][doc["clause_df"]["final_clause"] != "Unknown"])
                        with st.expander(f"{doc['name']}  ·  {dev_count} deviation(s) / {total} clauses"):
                            dev_rows = doc["clause_df"][doc["clause_df"]["final_deviation"]]
                            if dev_rows.empty:
                                st.success("No deviations detected.")
                            else:
                                for _, row in dev_rows.iterrows():
                                    sid = int(row["span_id"])
                                    sev = row.get("severity", "Medium")
                                    color = "#fc8181" if sev == "High" else "#f6ad55"
                                    st.markdown(f'<div style="background:rgba(255,255,255,0.03);border-left:2px solid {color};padding:0.5rem 0.8rem;border-radius:0 6px 6px 0;margin:0.3rem 0;color:#94a3b8;font-size:0.82rem"><b style="color:#e2e8f0">{row["final_clause"]}</b> · {sev} · {"; ".join(row.get("deviation_reasons", []))}</div>', unsafe_allow_html=True)


else:
    st.markdown("""
<div style="text-align:center;padding:4rem 2rem;color:#334155">
  <div style="font-size:3rem;margin-bottom:1rem"></div>
  <h3 style="color:#475569;font-weight:500">Upload a contract PDF to begin</h3>
  <p style="color:#334155;font-size:0.9rem">Use the sidebar to upload your document and click Analyze Contract</p>
</div>
""", unsafe_allow_html=True)



    st.markdown("---")
    # ── ABOUT CONTRACTIQ ──────────────────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg,#0d1b2a 0%,#1a2744 100%);border:1px solid rgba(99,179,237,0.2);border-radius:16px;padding:2.5rem 3rem;margin:2rem 0">
  <div style="margin-bottom:1rem">
    <div style="font-size:1.5rem;font-weight:700;color:#e2e8f0;margin-bottom:0.3rem">⚖️ About ContractIQ</div>
    <div style="color:#63b3ed;font-size:0.82rem;font-weight:600;letter-spacing:0.06em">THE WORLD'S FIRST MULTI-SIGNAL SEMANTIC CONTRACT DEVIATION ENGINE</div>
  </div>
  <p style="color:#94a3b8;line-height:1.8;font-size:0.93rem;margin:0">
    ContractIQ is not a keyword scanner. It is not a template matcher. It is not a rules engine.
    It is a <b style="color:#e2e8f0">purpose-built AI system</b> that understands contract language
    the way a trained legal analyst does — by learning what standard clauses <i>should</i> look like,
    and precisely identifying where a given contract deviates from that standard.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div style="font-size:1.05rem;font-weight:700;color:#e2e8f0;margin:1.5rem 0 1rem">🧠 The CSDA Algorithm — Composite Semantic Deviation Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;font-size:0.88rem;line-height:1.8;margin-bottom:1.2rem">At the core of ContractIQ is our proprietary <b style="color:#63b3ed">CSDA (Composite Semantic Deviation Analysis)</b> algorithm — a novel, multi-signal framework designed and built from the ground up. No existing commercial or academic tool implements this exact combination of signals.</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div style="background:rgba(99,179,237,0.06);border:1px solid rgba(99,179,237,0.2);border-radius:10px;padding:1.2rem;height:100%">
<div style="color:#63b3ed;font-weight:700;font-size:0.88rem;margin-bottom:0.6rem">① DeBERTa Clause Classifier</div>
<div style="color:#94a3b8;font-size:0.82rem;line-height:1.7">A fine-tuned <b style="color:#cbd5e1">DeBERTa-v3</b> transformer classifies every paragraph into one of <b style="color:#cbd5e1">33 legal clause types</b> — trained on 3,688 real commercial contract spans. We use a dual-threshold system: minimum confidence 0.45 AND minimum gap 0.10 between top-2 predictions, ensuring only high-certainty classifications proceed to deviation analysis.</div>
</div>""", unsafe_allow_html=True)
        st.markdown("""<div style="background:rgba(246,173,85,0.06);border:1px solid rgba(246,173,85,0.2);border-radius:10px;padding:1.2rem;margin-top:1rem;height:100%">
<div style="color:#f6ad55;font-weight:700;font-size:0.88rem;margin-bottom:0.6rem">③ Polarity Profile Violation</div>
<div style="color:#94a3b8;font-size:0.82rem;line-height:1.7">Every clause type has a learned <b style="color:#cbd5e1">polarity profile</b> — the statistical frequency of obligation signals ("shall", "must", "not") vs. permission signals ("freely", "without restriction"). If a normally obligation-heavy clause contains permissive language, it is flagged as a <b style="color:#cbd5e1">polarity mismatch</b> — a signal that the power balance has been shifted.</div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div style="background:rgba(104,211,145,0.06);border:1px solid rgba(104,211,145,0.2);border-radius:10px;padding:1.2rem;height:100%">
<div style="color:#68d391;font-weight:700;font-size:0.88rem;margin-bottom:0.6rem">② Centroid-Based Semantic Distance</div>
<div style="color:#94a3b8;font-size:0.82rem;line-height:1.7">Each clause is embedded using <b style="color:#cbd5e1">all-mpnet-base-v2</b> and compared against a <b style="color:#cbd5e1">per-clause centroid</b> — the mean embedding of all training examples for that clause type. Cosine distance is compared against a statistically derived threshold (90th percentile). Clauses outside this boundary are semantically anomalous — their language pattern is statistically unusual for their type.</div>
</div>""", unsafe_allow_html=True)
        st.markdown("""<div style="background:rgba(252,129,129,0.06);border:1px solid rgba(252,129,129,0.2);border-radius:10px;padding:1.2rem;margin-top:1rem;height:100%">
<div style="color:#fc8181;font-weight:700;font-size:0.88rem;margin-bottom:0.6rem">④ Per-Clause Invariant Rules</div>
<div style="color:#94a3b8;font-size:0.82rem;line-height:1.7">For each of the 12 highest-risk clause types, we define <b style="color:#cbd5e1">hard invariants</b>: forbidden patterns (e.g. "unlimited liability" in Cap On Liability), negation detection (e.g. "no right to audit" in Audit Rights), missing required keywords, and unilateral rights signals. These are non-negotiable legal red lines no standard contract should cross.</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""<div style="background:rgba(99,179,237,0.08);border:1px solid rgba(99,179,237,0.25);border-radius:10px;padding:1.2rem;margin-top:1rem">
<div style="color:#63b3ed;font-weight:700;font-size:0.88rem;margin-bottom:0.6rem">⑤ Composite Deviation Score (0.0 – 1.0)</div>
<div style="color:#94a3b8;font-size:0.82rem;line-height:1.7">The four signals are fused into a single score. Semantic distance contributes up to <b style="color:#cbd5e1">0.5</b> (proportional to how far beyond the threshold the clause falls), and each additional signal adds <b style="color:#cbd5e1">0.15</b>. This drives severity: <b style="color:#fc8181">High</b> (invariant violations, uncapped liability, negated obligations) · <b style="color:#f6ad55">Medium</b> (polarity mismatch, semantic deviation) · <b style="color:#68d391">Low</b> (minor keyword gaps). No other publicly available tool uses this composite scoring approach.</div>
</div>""", unsafe_allow_html=True)

    st.markdown("")
    m1, m2, m3 = st.columns(3)
    m1.markdown('<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:1.2rem;text-align:center"><div style="font-size:2rem;font-weight:700;color:#63b3ed">33</div><div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;margin-top:4px">Clause Types</div><div style="color:#94a3b8;font-size:0.78rem;margin-top:0.4rem">Full spectrum of commercial contract clauses</div></div>', unsafe_allow_html=True)
    m2.markdown('<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:1.2rem;text-align:center"><div style="font-size:2rem;font-weight:700;color:#68d391">4</div><div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;margin-top:4px">Detection Signals</div><div style="color:#94a3b8;font-size:0.78rem;margin-top:0.4rem">Semantic · Polarity · Invariant · Composite</div></div>', unsafe_allow_html=True)
    m3.markdown('<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:1.2rem;text-align:center"><div style="font-size:2rem;font-weight:700;color:#f6ad55">3,688</div><div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;margin-top:4px">Training Samples</div><div style="color:#94a3b8;font-size:0.78rem;margin-top:0.4rem">Real commercial contract spans</div></div>', unsafe_allow_html=True)

    st.markdown("")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("""<div style="background:rgba(104,211,145,0.04);border:1px solid rgba(104,211,145,0.15);border-radius:12px;padding:1.2rem">
<div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;margin-bottom:0.8rem">🔒 Privacy-First Architecture</div>
<div style="color:#94a3b8;font-size:0.82rem;line-height:1.8">
<b style="color:#68d391">✓ Raw contract text never leaves your session.</b><br>
All ML inference runs on the server in-memory.<br>
The LLM receives only structured metadata — clause types, deviation flags, severity scores.<br>
No contract content is stored, logged, or transmitted to any third party.
</div></div>""", unsafe_allow_html=True)
    with p2:
        st.markdown("""<div style="background:rgba(99,179,237,0.04);border:1px solid rgba(99,179,237,0.15);border-radius:12px;padding:1.2rem">
<div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;margin-bottom:0.8rem">🔍 Fully Explainable Outputs</div>
<div style="color:#94a3b8;font-size:0.82rem;line-height:1.8">
<b style="color:#63b3ed">✓ Every flag is traceable and explained.</b><br>
Every deviation comes with a plain-English reason.<br>
Every confidence score is shown. Every signal is visible.<br>
ContractIQ augments human review — it does not replace it.
</div></div>""", unsafe_allow_html=True)

    st.markdown("""<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:1.5rem;margin-top:1rem">
<div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem">🆚 How ContractIQ Compares</div>
<table style="width:100%;border-collapse:collapse;font-size:0.82rem">
<tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
  <th style="text-align:left;color:#64748b;padding:0.5rem 0.8rem;font-weight:600">Capability</th>
  <th style="text-align:center;color:#63b3ed;padding:0.5rem;font-weight:600">ContractIQ</th>
  <th style="text-align:center;color:#64748b;padding:0.5rem;font-weight:600">Keyword Tools</th>
  <th style="text-align:center;color:#64748b;padding:0.5rem;font-weight:600">Generic LLM</th>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.04)"><td style="color:#94a3b8;padding:0.5rem 0.8rem">Understands clause context</td><td style="text-align:center;color:#68d391">✓ DeBERTa</td><td style="text-align:center;color:#fc8181">✗</td><td style="text-align:center;color:#f6ad55">Partial</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.04)"><td style="color:#94a3b8;padding:0.5rem 0.8rem">Semantic deviation scoring</td><td style="text-align:center;color:#68d391">✓ CSDA</td><td style="text-align:center;color:#fc8181">✗</td><td style="text-align:center;color:#fc8181">✗</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.04)"><td style="color:#94a3b8;padding:0.5rem 0.8rem">Polarity profile analysis</td><td style="text-align:center;color:#68d391">✓ Learned</td><td style="text-align:center;color:#fc8181">✗</td><td style="text-align:center;color:#fc8181">✗</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.04)"><td style="color:#94a3b8;padding:0.5rem 0.8rem">Privacy-safe (no raw text to LLM)</td><td style="text-align:center;color:#68d391">✓</td><td style="text-align:center;color:#68d391">✓</td><td style="text-align:center;color:#fc8181">✗</td></tr>
<tr><td style="color:#94a3b8;padding:0.5rem 0.8rem">Composite deviation score</td><td style="text-align:center;color:#68d391">✓ 0.0–1.0</td><td style="text-align:center;color:#fc8181">✗</td><td style="text-align:center;color:#fc8181">✗</td></tr>
</table>
</div>""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <p>
    <b style="color:#475569">ContractIQ</b> &nbsp;·&nbsp; AI-Powered Contract Analysis &nbsp;·&nbsp; v2.0
    <br>
    © 2026 Team 2022AIE01. All rights reserved.
    &nbsp;·&nbsp;
    <a href="https://github.com/Adithiyanpv/contract-intelligence-framework">GitHub</a>
    &nbsp;·&nbsp;
    This tool does not provide legal advice. Always consult a qualified legal professional.
  </p>
</div>
""", unsafe_allow_html=True)
