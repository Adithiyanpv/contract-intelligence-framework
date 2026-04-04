import streamlit as st
st.set_page_config(page_title="Contract Clause Deviation Detector", layout="wide")
import os, sys, requests, tempfile
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
os.chdir(_APP_DIR)
if not os.path.exists(os.path.join(_APP_DIR, "resources", "deberta-clause-final")):
    with st.spinner("Downloading model artifacts..."):
        try:
            from download_models import ensure_models
            ensure_models()
        except Exception as e:
            st.error(f"Model download failed: {e}")
            st.stop()
from llm.llm_client import get_llm_client
from pipeline import (analyze_document, ask_document, build_contract_summary, narrate_contract_summary, export_results_csv, export_results_json)
for key, default in [("analyzed", False), ("last_answer", None)]:
    if key not in st.session_state:
        st.session_state[key] = default
llm, llm_source = get_llm_client()
SEVERITY_COLORS = {"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#2ecc71"}
def severity_badge(level):
    color = SEVERITY_COLORS.get(level, "#888")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.8em;font-weight:bold">{level}</span>'
st.title("Contract Clause Deviation Detector")
st.caption("AI-assisted contract analysis - not legal advice.")
if llm_source == "groq":
    st.sidebar.success("Groq LLM connected (cloud)")
elif llm_source == "ollama":
    st.sidebar.success("Ollama LLM connected (local)")
else:
    st.sidebar.warning("No LLM. Add GROQ_API_KEY to Streamlit secrets.")
st.sidebar.header("Upload Contract")
uploaded_pdf = st.sidebar.file_uploader("Upload a contract PDF", type=["pdf"])
if st.sidebar.button("Analyze Contract"):
    if uploaded_pdf is None:
        st.sidebar.error("Please upload a PDF file first.")
    else:
        progress_bar = st.progress(0, text="Starting analysis...")
        def update_progress(step, total, msg):
            pct = int((step / max(total, 1)) * 100)
            progress_bar.progress(pct, text=msg)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            pdf_path = tmp.name
        clause_df, spans, embeddings, embedder = analyze_document(pdf_path, progress_callback=update_progress)
        progress_bar.progress(100, text="Building summary...")
        clause_df["_known"] = clause_df["final_clause"] != "Unknown"
        clause_df = clause_df.sort_values(by=["_known", "span_id"], ascending=[False, True]).drop(columns="_known").reset_index(drop=True)
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
        progress_bar.empty()
        st.sidebar.success("Analysis complete")
if st.session_state.analyzed:
    clause_df = st.session_state.clause_df
    spans = st.session_state.spans
    summary = st.session_state.contract_summary
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Results")
    st.sidebar.download_button("Download CSV", data=export_results_csv(clause_df, spans), file_name="contract_analysis.csv", mime="text/csv")
    st.sidebar.download_button("Download JSON", data=export_results_json(clause_df, spans, summary), file_name="contract_analysis.json", mime="application/json")
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Deviating Clauses", "Analytics", "Ask the Contract"])
    with tab1:
        st.subheader("Contract Summary")
        st.write(st.session_state.summary_narration)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Spans", summary["overview"]["total_spans"])
        col2.metric("Recognized", summary["overview"]["recognized_clauses"])
        col3.metric("Unknown", summary["overview"]["unknown_spans"])
        col4.metric("Deviating", summary["overview"]["deviating_spans"])
        if summary["deviations"]:
            st.markdown("### Risk Snapshot")
            for d in summary["deviations"]:
                sev = d.get("severity", "Medium")
                st.markdown(f"{severity_badge(sev)} **{d['clause']}** - {', '.join(d['reasons'])}", unsafe_allow_html=True)
        else:
            st.success("No non-standard clause patterns detected.")
        st.markdown("### Clause Coverage")
        counts = summary["coverage"].get("clause_counts", {})
        st.write(", ".join(f"{c} ({counts.get(c,'?')})" for c in summary["coverage"]["detected_clauses"]))
        st.markdown("---")
        st.subheader("Clause to Text Mapping")
        for clause_name in sorted(clause_df["final_clause"].unique()):
            if clause_name == "Unknown":
                continue
            rows = clause_df[clause_df["final_clause"] == clause_name]
            has_dev = rows["final_deviation"].any()
            with st.expander(f"{'[DEV] ' if has_dev else ''}{clause_name} ({len(rows)} spans)"):
                for _, row in rows.iterrows():
                    sid = int(row["span_id"])
                    st.markdown(f"**Span {sid} | conf={row['confidence']:.2f}{'  DEVIATION' if row['final_deviation'] else ''}**")
                    st.write(spans[sid])
                    st.markdown("---")
    with tab2:
        deviating = clause_df[clause_df["final_deviation"]]
        if deviating.empty:
            st.success("No deviating clauses detected.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total", len(deviating))
            col2.metric("High", int((deviating["severity"] == "High").sum()))
            col3.metric("Medium", int((deviating["severity"] == "Medium").sum()))
            from pipeline import explain_deviation_reasons
            for _, row in deviating.sort_values("deviation_score", ascending=False).iterrows():
                sid = int(row["span_id"])
                sev = row.get("severity", "Medium")
                with st.expander(f"{row['final_clause']} | span {sid} | {sev}"):
                    st.markdown(f"**Severity:** {severity_badge(sev)}", unsafe_allow_html=True)
                    for r in row["deviation_reasons"]:
                        st.write(f"- {r}")
                    for exp in explain_deviation_reasons(row["deviation_reasons"]):
                        st.info(exp)
                    st.write(spans[sid])
    with tab3:
        import pandas as pd
        st.subheader("Analytics")
        known_df = clause_df[clause_df["final_clause"] != "Unknown"]
        if not known_df.empty:
            conf_chart = known_df.groupby("final_clause")["confidence"].mean().reset_index().rename(columns={"final_clause":"Clause","confidence":"Avg Confidence"}).sort_values("Avg Confidence", ascending=False)
            st.bar_chart(conf_chart.set_index("Clause"))
        freq = clause_df["final_clause"].value_counts().reset_index().rename(columns={"final_clause":"Clause","count":"Count"})
        st.bar_chart(freq.set_index("Clause"))
        rows_t = [{"Clause":c,"Spans":len(s:=clause_df[clause_df["final_clause"]==c]),"Deviating":int(s["final_deviation"].sum()),"Avg Conf":f"{s['confidence'].mean():.2f}"} for c in sorted(clause_df["final_clause"].unique()) if c != "Unknown"]
        if rows_t:
            st.dataframe(pd.DataFrame(rows_t), use_container_width=True)
    with tab4:
        st.subheader("Ask a question about the contract")
        if llm_source == "groq":
            st.caption("Powered by Groq - llama-3.1-8b-instant")
        elif llm_source != "ollama":
            st.warning("Add GROQ_API_KEY to Streamlit secrets for AI answers.")
        question = st.text_input("Example: What does the cap on liability mean for me?")
        if st.button("Ask"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                retrieval = ask_document(question, clause_df, spans, st.session_state.embeddings, st.session_state.embedder)
                evidence_list = retrieval.get("evidence", [])
                if llm is not None:
                    top_evidence = evidence_list[:3]
                    context_blocks = []
                    for e in top_evidence:
                        snippet = e["text"][:500] + ("..." if len(e["text"]) > 500 else "")
                        block = f"CLAUSE: {e['clause']}"
                        if e["deviating"] and e["reasons"]:
                            block += f"\nDEVIATION: {'; '.join(e['reasons'])}"
                        block += f"\nTEXT: {snippet}"
                        context_blocks.append(block)
                    context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant clauses found."
                    prompt = f"You are a contract analyst. Answer using ONLY the clause text below. Be specific and clear. Use bullet points. End with: Note: This is not legal advice.\n\nCLAUSES:\n{context_str}\n\nQUESTION: {question}\n\nANSWER:"
                    try:
                        explanation = llm(prompt)
                    except Exception as e:
                        explanation = f"LLM error: {e}"
                else:
                    detected = list(set(e["clause"] for e in evidence_list if e["clause"] != "Unknown"))
                    explanation = f"Relevant clauses: {', '.join(detected) or 'None'}. Add GROQ_API_KEY for full AI explanation."
                st.session_state.last_answer = {"explanation": explanation, "evidence": evidence_list, "confidence_notes": retrieval.get("confidence_notes", [])}
        if st.session_state.last_answer:
            answer = st.session_state.last_answer
            st.markdown("### Answer")
            st.write(answer["explanation"])
            if answer["evidence"]:
                st.markdown("---")
                for ev in answer["evidence"]:
                    with st.expander(f"{ev['clause']} | span {ev['span_id']}"):
                        if ev["deviating"] and ev["reasons"]:
                            for r in ev["reasons"]:
                                st.write(f"- {r}")
                        st.write(ev["text"])
            for note in answer["confidence_notes"]:
                st.caption(note)
else:
    st.info("Upload a PDF and click Analyze Contract to begin.")