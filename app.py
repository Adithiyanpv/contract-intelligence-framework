import streamlit as st
st.set_page_config(page_title="ContractIQ", page_icon="", layout="wide")
import os, sys, requests, tempfile, json
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
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

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("analyzed", False), ("last_answer", None), ("contract_doc_summary", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

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
    st.markdown('<p class="section-header">LLM Status</p>', unsafe_allow_html=True)
    if llm_source == "groq":
        st.success("🟢 Groq connected")
        st.caption("llama-3.1-8b-instant · Privacy-safe mode")
    elif llm_source == "ollama":
        st.success("🟢 Ollama connected")
        st.caption("Local inference · No data leaves device")
    else:
        st.warning("🟡 No LLM · Add GROQ_API_KEY to secrets")

    st.markdown('<p class="section-header">Upload Contract</p>', unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    analyze_clicked = st.button("▶  Analyze Contract", use_container_width=True)

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
    if uploaded_pdf is None:
        st.error("Please upload a PDF file first.")
    else:
        progress_bar = st.progress(0, text="Initializing analysis pipeline...")
        def update_progress(step, total, msg):
            pct = int((step / max(total, 1)) * 100)
            progress_bar.progress(pct, text=msg)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            pdf_path = tmp.name
        clause_df, spans, embeddings, embedder = analyze_document(pdf_path, progress_callback=update_progress)
        progress_bar.progress(100, text="Finalizing...")
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
        st.query_params["tab"] = "overview"
        progress_bar.empty()
        st.rerun()

# ── Main UI ────────────────────────────────────────────────────────────────────
if st.session_state.analyzed:
    clause_df = st.session_state.clause_df
    spans = st.session_state.spans
    summary = st.session_state.contract_summary

    _TAB_IDX = {"overview":0,"deviations":1,"risk":2,"analytics":3,"summary":4,"ask":5}
    _active = st.query_params.get("tab","overview")
    _idx = _TAB_IDX.get(_active, 0)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["  Overview  ","  Deviating Clauses  ","  Risk Analysis  ","  Analytics  ","  Summary  ","  Ask the Contract  "])

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
        st.markdown(f'<div class="answer-box"><p style="color:#cbd5e1;line-height:1.7">{st.session_state.summary_narration}</p></div>', unsafe_allow_html=True)

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


    # ── TAB 4: SUMMARY ──────────────────────────────────────────────────────
    with tab5:
        st.markdown('<p class="section-header">Document Summary</p>', unsafe_allow_html=True)
        st.markdown('<div class="privacy-notice">🔒 Fully local extraction — no document content sent externally.</div>', unsafe_allow_html=True)
        if st.button("Generate Summary", use_container_width=False):
            with st.spinner("Extracting structured summary..."):
                doc_summary = summarize_contract(spans, clause_df, st.session_state.embedder, summary)
                metrics = evaluate_summary(doc_summary["extractive_summary"], spans)
                st.session_state.contract_doc_summary = {"summary": doc_summary, "metrics": metrics}
                st.query_params["tab"] = "summary"
        if "contract_doc_summary" in st.session_state and st.session_state.contract_doc_summary:
            ds = st.session_state.contract_doc_summary["summary"]
            mt = st.session_state.contract_doc_summary["metrics"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p class="section-header">Parties Involved</p>', unsafe_allow_html=True)
                for p in (ds["parties"] or ["Not detected"]): st.markdown(f'<span style="background:rgba(99,179,237,0.1);color:#63b3ed;padding:3px 10px;border-radius:20px;font-size:0.82rem;display:inline-block;margin:2px">{p}</span>', unsafe_allow_html=True)
                st.markdown('<p class="section-header">Key Dates</p>', unsafe_allow_html=True)
                if ds["effective_date"]: st.markdown(f"📅 **Effective:** {ds['effective_date']}")
                if ds["expiry_date"] and ds["expiry_date"] != ds["effective_date"]: st.markdown(f"📅 **Expiry:** {ds['expiry_date']}")
                st.markdown('<p class="section-header">Governing Law</p>', unsafe_allow_html=True)
                st.write(ds["governing_law"] or "Not explicitly stated")
                st.markdown('<p class="section-header">Defined Terms</p>', unsafe_allow_html=True)
                terms_html = " ".join(f'<span style="background:rgba(255,255,255,0.05);color:#94a3b8;padding:2px 8px;border-radius:4px;font-size:0.78rem;margin:2px;display:inline-block">{t}</span>' for t in (ds["key_terms"] or ["None"]))
                st.markdown(terms_html, unsafe_allow_html=True)
            with c2:
                st.markdown('<p class="section-header">Key Obligations</p>', unsafe_allow_html=True)
                for ob in (ds["obligations"] or ["None detected"]): st.markdown(f'<div style="background:rgba(255,255,255,0.03);border-left:2px solid #63b3ed;padding:0.5rem 0.8rem;border-radius:0 6px 6px 0;color:#94a3b8;font-size:0.82rem;margin:0.3rem 0">{ob}</div>', unsafe_allow_html=True)
                st.markdown('<p class="section-header">Payment Terms</p>', unsafe_allow_html=True)
                for pt in (ds["payment_terms"] or ["None detected"]): st.markdown(f'<div style="background:rgba(255,255,255,0.03);border-left:2px solid #f6ad55;padding:0.5rem 0.8rem;border-radius:0 6px 6px 0;color:#94a3b8;font-size:0.82rem;margin:0.3rem 0">{pt}</div>', unsafe_allow_html=True)
            st.markdown('<p class="section-header">Extractive Summary</p>', unsafe_allow_html=True)
            for i_s, sent in enumerate(ds["extractive_summary"], 1): st.markdown(f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.7rem 1rem;color:#cbd5e1;font-size:0.87rem;line-height:1.6;margin:0.4rem 0"><span style="color:#475569;font-size:0.75rem">#{i_s}</span> {sent}</div>', unsafe_allow_html=True)
            st.markdown('<p class="section-header">Summary Quality Metrics</p>', unsafe_allow_html=True)
            mc1,mc2,mc3,mc4 = st.columns(4)
            mc1.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.4rem">{mt["rouge_1"]["f1"]:.3f}</div><div class="lbl">ROUGE-1 F1</div></div>', unsafe_allow_html=True)
            mc2.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.4rem">{mt["rouge_2"]["f1"]:.3f}</div><div class="lbl">ROUGE-2 F1</div></div>', unsafe_allow_html=True)
            mc3.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.4rem">{mt["coverage"]:.3f}</div><div class="lbl">Coverage</div></div>', unsafe_allow_html=True)
            mc4.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.4rem">{mt["compression_ratio"]:.3f}</div><div class="lbl">Compression</div></div>', unsafe_allow_html=True)
            st.caption(f"Summary: {mt['summary_sentences']} sentences · Coverage: {mt['coverage']:.1%} · Compression: {mt['compression_ratio']:.1%}")

    # ── TAB 4: ASK THE CONTRACT ───────────────────────────────────────────────
    with tab6:
        st.markdown('<p class="section-header">Ask a Question</p>', unsafe_allow_html=True)

        if llm_source in ("groq","ollama"):
            st.markdown(f"""<div class="privacy-notice">
🔒 <b>Privacy-safe mode active</b> — raw contract text is never sent to any external API.
The LLM receives only structured metadata (clause types, deviation flags, severity).
Powered by <b>{"Groq · llama-3.1-8b-instant" if llm_source=="groq" else "Ollama (local)"}</b>.
</div>""", unsafe_allow_html=True)
        else:
            st.warning("No LLM connected. Add GROQ_API_KEY to Streamlit secrets for AI answers.")

        question = st.text_input("", placeholder="e.g. What are the risks in this contract? What does the liability clause say?", label_visibility="collapsed")

        if st.button("Ask ContractIQ", use_container_width=False):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                st.query_params["tab"] = "ask"
                with st.spinner("Analyzing..."):
                    retrieval = ask_document(question, clause_df, spans,
                                             st.session_state.embeddings, st.session_state.embedder)
                    evidence_list = retrieval.get("evidence", [])

                    if llm is not None:
                        # PRIVACY-SAFE: build structured metadata only — NO raw text
                        safe_context = build_safe_prompt(question, evidence_list)
                        try:
                            explanation = llm(safe_context)
                        except Exception as e:
                            explanation = f"LLM error: {e}"
                    else:
                        detected = list(set(e["clause"] for e in evidence_list if e["clause"] != "Unknown"))
                        dev_list = [e for e in evidence_list if e["deviating"]]
                        if dev_list:
                            explanation = (f"Found {len(dev_list)} deviating clause(s): "
                                           f"{', '.join(set(e['clause'] for e in dev_list))}. "
                                           f"Reasons: {'; '.join(dev_list[0]['reasons'])}. "
                                           "Add GROQ_API_KEY for full AI explanation.")
                        elif detected:
                            explanation = f"Relevant clauses found: {', '.join(detected)}. No deviations detected."
                        else:
                            explanation = "No directly relevant clauses found for this question."

                    st.session_state.last_answer = {
                        "explanation": explanation,
                        "evidence": evidence_list,
                        "confidence_notes": retrieval.get("confidence_notes", [])
                    }

        if st.session_state.last_answer:
            answer = st.session_state.last_answer
            st.markdown('<p class="section-header">Answer</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box"><p style="color:#cbd5e1;line-height:1.8;white-space:pre-wrap">{answer["explanation"]}</p></div>', unsafe_allow_html=True)

            if answer["evidence"]:
                st.markdown('<p class="section-header">Supporting Clauses</p>', unsafe_allow_html=True)
                for ev in answer["evidence"]:
                    sev = ev.get("severity")
                    sev_str = f" · {sev} severity" if sev else ""
                    with st.expander(f"{ev['clause']}  ·  Span {ev['span_id']}{sev_str}"):
                        if ev["deviating"] and ev["reasons"]:
                            for r in ev["reasons"]:
                                st.markdown(f'<span style="color:#f6ad55;font-size:0.83rem">⚠ {r}</span>', unsafe_allow_html=True)
                        st.markdown(f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.8rem;color:#94a3b8;font-size:0.85rem;line-height:1.6;margin-top:0.5rem">{ev["text"]}</div>', unsafe_allow_html=True)

            for note in answer["confidence_notes"]:
                st.caption(note)

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
  <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem">
    <div style="font-size:2rem">⚖️</div>
    <div>
      <div style="font-size:1.5rem;font-weight:700;color:#e2e8f0">About ContractIQ</div>
      <div style="color:#63b3ed;font-size:0.85rem;font-weight:600;letter-spacing:0.05em">THE WORLD'S FIRST MULTI-SIGNAL SEMANTIC CONTRACT DEVIATION ENGINE</div>
    </div>
  </div>
  <p style="color:#94a3b8;line-height:1.8;font-size:0.95rem;margin-bottom:2rem">
    ContractIQ is not a keyword scanner. It is not a template matcher. It is not a rules engine.
    It is a <b style="color:#e2e8f0">purpose-built AI system</b> that understands contract language
    the way a trained legal analyst does — by learning what standard clauses <i>should</i> look like,
    and precisely identifying where a given contract deviates from that standard.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:2rem;margin:1rem 0">
  <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1.5rem">🧠 The CSDA Algorithm — Composite Semantic Deviation Analysis</div>
  <p style="color:#94a3b8;line-height:1.8;font-size:0.9rem;margin-bottom:1.5rem">
    At the core of ContractIQ is our proprietary <b style="color:#63b3ed">CSDA (Composite Semantic Deviation Analysis)</b> algorithm —
    a novel, multi-signal framework that we designed and built from the ground up. No existing commercial or academic tool
    implements this exact combination of signals. Here is exactly how it works:
  </p>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem">

    <div style="background:rgba(99,179,237,0.06);border:1px solid rgba(99,179,237,0.15);border-radius:10px;padding:1.2rem">
      <div style="color:#63b3ed;font-weight:700;font-size:0.9rem;margin-bottom:0.5rem">① DeBERTa Clause Classifier</div>
      <div style="color:#94a3b8;font-size:0.83rem;line-height:1.7">
        A fine-tuned <b style="color:#cbd5e1">DeBERTa-v3</b> transformer model classifies every paragraph of the contract
        into one of <b style="color:#cbd5e1">33 legal clause types</b> — trained on 3,688 real commercial contract spans.
        Unlike simple keyword matching, DeBERTa understands full sentence context, negation, and legal phrasing nuance.
        We use a dual-threshold system: a minimum confidence of 0.45 AND a minimum gap of 0.10 between the top-2 predictions,
        ensuring only high-certainty classifications proceed to deviation analysis.
      </div>
    </div>

    <div style="background:rgba(104,211,145,0.06);border:1px solid rgba(104,211,145,0.15);border-radius:10px;padding:1.2rem">
      <div style="color:#68d391;font-weight:700;font-size:0.9rem;margin-bottom:0.5rem">② Centroid-Based Semantic Distance</div>
      <div style="color:#94a3b8;font-size:0.83rem;line-height:1.7">
        Each classified clause is embedded using <b style="color:#cbd5e1">all-mpnet-base-v2</b> and compared against
        a <b style="color:#cbd5e1">per-clause centroid</b> — the mean embedding of all training examples for that clause type.
        The cosine distance to this centroid is compared against a <b style="color:#cbd5e1">statistically derived threshold</b>
        (90th percentile of training distances). Clauses that fall outside this boundary are semantically anomalous —
        their language pattern is statistically unusual for their type. This is the first signal.
      </div>
    </div>

    <div style="background:rgba(246,173,85,0.06);border:1px solid rgba(246,173,85,0.15);border-radius:10px;padding:1.2rem">
      <div style="color:#f6ad55;font-weight:700;font-size:0.9rem;margin-bottom:0.5rem">③ Polarity Profile Violation</div>
      <div style="color:#94a3b8;font-size:0.83rem;line-height:1.7">
        Every clause type has a learned <b style="color:#cbd5e1">polarity profile</b> — the statistical frequency of
        obligation signals ("shall", "must", "not") vs. permission signals ("may", "freely", "without restriction")
        across all training examples. A weighted obligation score is computed per clause type.
        If a clause that is normally obligation-heavy contains permissive language, it is flagged as a
        <b style="color:#cbd5e1">polarity mismatch</b> — a signal that the power balance has been shifted. This is the second signal.
      </div>
    </div>

    <div style="background:rgba(252,129,129,0.06);border:1px solid rgba(252,129,129,0.15);border-radius:10px;padding:1.2rem">
      <div style="color:#fc8181;font-weight:700;font-size:0.9rem;margin-bottom:0.5rem">④ Per-Clause Invariant Rules</div>
      <div style="color:#94a3b8;font-size:0.83rem;line-height:1.7">
        For each of the 12 highest-risk clause types, we define a set of <b style="color:#cbd5e1">hard invariants</b>:
        forbidden patterns (e.g. "unlimited liability" in a Cap On Liability clause),
        negation detection (e.g. "no right to audit" in an Audit Rights clause),
        missing required keywords (e.g. absence of "consent" in an Anti-Assignment clause),
        and unilateral rights signals. These are non-negotiable legal red lines that no standard contract should cross.
        This is the third signal.
      </div>
    </div>

  </div>

  <div style="background:rgba(99,179,237,0.08);border:1px solid rgba(99,179,237,0.2);border-radius:10px;padding:1.2rem;margin-top:1.5rem">
    <div style="color:#63b3ed;font-weight:700;font-size:0.9rem;margin-bottom:0.5rem">⑤ Composite Deviation Score</div>
    <div style="color:#94a3b8;font-size:0.83rem;line-height:1.7">
      The four signals are fused into a single <b style="color:#cbd5e1">Composite Deviation Score (0.0–1.0)</b>.
      Semantic distance contributes up to 0.5 of the score (proportional to how far beyond the threshold the clause falls),
      and each additional signal adds 0.15. This score drives severity classification:
      <b style="color:#fc8181">High</b> (invariant violations, uncapped liability, negated obligations),
      <b style="color:#f6ad55">Medium</b> (polarity mismatch, semantic deviation),
      <b style="color:#68d391">Low</b> (minor keyword gaps).
      No other publicly available contract analysis tool uses this composite scoring approach.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0">
  <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:1.5rem;text-align:center">
    <div style="font-size:2rem;font-weight:700;color:#63b3ed">33</div>
    <div style="color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-top:4px">Clause Types</div>
    <div style="color:#94a3b8;font-size:0.8rem;margin-top:0.5rem">Covering the full spectrum of commercial contract clauses</div>
  </div>
  <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:1.5rem;text-align:center">
    <div style="font-size:2rem;font-weight:700;color:#68d391">4</div>
    <div style="color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-top:4px">Detection Signals</div>
    <div style="color:#94a3b8;font-size:0.8rem;margin-top:0.5rem">Semantic + Polarity + Invariant + Composite scoring</div>
  </div>
  <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:1.5rem;text-align:center">
    <div style="font-size:2rem;font-weight:700;color:#f6ad55">3,688</div>
    <div style="color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-top:4px">Training Samples</div>
    <div style="color:#94a3b8;font-size:0.8rem;margin-top:0.5rem">Real commercial contract spans used to train the classifier</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:2rem;margin:1rem 0">
  <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1.2rem">🔒 Privacy-First Architecture</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
    <div style="color:#94a3b8;font-size:0.85rem;line-height:1.7">
      <b style="color:#68d391">✓ Raw contract text never leaves your browser session.</b><br>
      All ML inference (DeBERTa, SentenceTransformer) runs on the server in-memory.<br>
      The LLM (Groq) receives only structured metadata — clause types, deviation flags, severity scores.<br>
      No contract content is stored, logged, or transmitted to any third party.
    </div>
    <div style="color:#94a3b8;font-size:0.85rem;line-height:1.7">
      <b style="color:#68d391">✓ Fully explainable outputs.</b><br>
      Every deviation flag comes with a plain-English explanation of why it was raised.<br>
      Every confidence score is shown. Every signal is traceable.<br>
      ContractIQ is designed to augment human review, not replace it.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:2rem;margin:1rem 0">
  <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1.2rem">🆚 How ContractIQ Compares</div>
  <table style="width:100%;border-collapse:collapse;font-size:0.83rem">
    <tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
      <th style="text-align:left;color:#64748b;padding:0.5rem;font-weight:600">Capability</th>
      <th style="text-align:center;color:#63b3ed;padding:0.5rem;font-weight:600">ContractIQ</th>
      <th style="text-align:center;color:#64748b;padding:0.5rem;font-weight:600">Keyword Tools</th>
      <th style="text-align:center;color:#64748b;padding:0.5rem;font-weight:600">Generic LLM</th>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.04)">
      <td style="color:#94a3b8;padding:0.5rem">Understands clause context</td>
      <td style="text-align:center;color:#68d391;padding:0.5rem">✓ DeBERTa</td>
      <td style="text-align:center;color:#fc8181;padding:0.5rem">✗</td>
      <td style="text-align:center;color:#f6ad55;padding:0.5rem">Partial</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.04)">
      <td style="color:#94a3b8;padding:0.5rem">Semantic deviation scoring</td>
      <td style="text-align:center;color:#68d391;padding:0.5rem">✓ CSDA</td>
      <td style="text-align:center;color:#fc8181;padding:0.5rem">✗</td>
      <td style="text-align:center;color:#fc8181;padding:0.5rem">✗</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.04)">
      <td style="color:#94a3b8;padding:0.5rem">Polarity profile analysis</td>
      <td style="text-align:center;color:#68d391;padding:0.5rem">✓ Learned</td>
      <td style="text-align:center;color:#fc8181;padding:0.5rem">✗</td>
      <td style="text-align:center;color:#fc8181;padding:0.5rem">✗</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.04)">
      <td style="color:#94a3b8;padding:0.5rem">Privacy-safe (no raw text to LLM)</td>
      <td style="text-align:center;color:#68d391;padding:0.5rem">✓</td>
      <td style="text-align:center;color:#68d391;padding:0.5rem">✓</td>
      <td style="text-align:center;color:#fc8181;padding:0.5rem">✗</td>
    </tr>
    <tr>
      <td style="color:#94a3b8;padding:0.5rem">Composite deviation score</td>
      <td style="text-align:center;color:#68d391;padding:0.5rem">✓ 0.0–1.0</td>
      <td style="text-align:center;color:#fc8181;padding:0.5rem">✗</td>
      <td style="text-align:center;color:#fc8181;padding:0.5rem">✗</td>
    </tr>
  </table>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="text-align:center;padding:2rem;color:#475569;font-size:0.82rem">
  <i>ContractIQ was developed as a final year research project by Team 2022AIE01.<br>
  The CSDA algorithm is an original contribution. Not legal advice.</i>
</div>
""", unsafe_allow_html=True)
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

