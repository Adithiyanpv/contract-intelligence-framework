import streamlit as st

st.set_page_config(
    page_title="Contract Clause Deviation Detector",
    layout="wide"
)

import os
import json
import requests

if not os.path.exists("resources/deberta-clause-final"):
    from download_models import ensure_models
    ensure_models()

import tempfile
from llm.llm_client import ollama_client
from pipeline import (
    analyze_document,
    ask_document,
    build_contract_summary,
    narrate_contract_summary,
    export_results_csv,
    export_results_json,
)

# ============================================================
# SESSION STATE INIT
# ============================================================

for key, default in [
    ("analyzed", False),
    ("last_answer", None),
    ("active_tab_index", 0),
    ("ollama_ok", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# OLLAMA HEALTH CHECK
# ============================================================

def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# ============================================================
# SEVERITY BADGE HELPER
# ============================================================

SEVERITY_COLORS = {
    "High":   "#ff4b4b",
    "Medium": "#ffa500",
    "Low":    "#2ecc71",
}

def severity_badge(level):
    color = SEVERITY_COLORS.get(level, "#888")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.8em;font-weight:bold">{level}</span>'

# ============================================================
# HEADER
# ============================================================

st.title("📄 Contract Clause Deviation Detector")
st.caption(
    "AI-assisted contract analysis using clause detection, deviation analysis, "
    "and explainable reasoning — not legal advice."
)

# Ollama status indicator
if st.session_state.ollama_ok is None:
    st.session_state.ollama_ok = check_ollama()

if st.session_state.ollama_ok:
    st.sidebar.success("🟢 Ollama LLM connected")
else:
    st.sidebar.warning(
        "🟡 Ollama not detected — LLM narration and Q&A explanations will use "
        "deterministic fallbacks. Run `ollama serve` to enable."
    )

llm = ollama_client if st.session_state.ollama_ok else None

# ============================================================
# SIDEBAR — UPLOAD & ANALYZE
# ============================================================

st.sidebar.header("📂 Upload Contract")
uploaded_pdf = st.sidebar.file_uploader("Upload a contract PDF", type=["pdf"])

if st.sidebar.button("▶ Analyze Contract"):
    if uploaded_pdf is None:
        st.sidebar.error("Please upload a PDF file first.")
    else:
        progress_bar = st.progress(0, text="Starting analysis…")

        def update_progress(step, total, msg):
            pct = int((step / max(total, 1)) * 100)
            progress_bar.progress(pct, text=msg)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            pdf_path = tmp.name

        clause_df, spans, embeddings, embedder = analyze_document(
            pdf_path, progress_callback=update_progress
        )

        progress_bar.progress(100, text="Building summary…")

        # Sort: known clauses first, then unknown
        clause_df["_known"] = clause_df["final_clause"] != "Unknown"
        clause_df = clause_df.sort_values(
            by=["_known", "span_id"], ascending=[False, True]
        ).drop(columns="_known").reset_index(drop=True)

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
        st.session_state.active_tab_index = 0

        progress_bar.empty()
        st.sidebar.success("Analysis complete ✅")

# ============================================================
# MAIN UI
# ============================================================

if st.session_state.analyzed:
    clause_df = st.session_state.clause_df
    spans = st.session_state.spans
    summary = st.session_state.contract_summary

    # ---- Export buttons in sidebar ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export Results")

    csv_data = export_results_csv(clause_df, spans)
    st.sidebar.download_button(
        label="⬇ Download CSV",
        data=csv_data,
        file_name="contract_analysis.csv",
        mime="text/csv"
    )

    json_data = export_results_json(clause_df, spans, summary)
    st.sidebar.download_button(
        label="⬇ Download JSON",
        data=json_data,
        file_name="contract_analysis.json",
        mime="application/json"
    )

    # Tab persistence via query params
    TAB_NAMES = ["overview", "deviations", "analytics", "ask"]
    TAB_LABELS = ["📘 Overview", "⚠️ Deviating Clauses", "📊 Analytics", "❓ Ask the Contract"]

    # Read active tab from query params
    params = st.query_params
    current_tab = params.get("tab", "overview")
    if current_tab not in TAB_NAMES:
        current_tab = "overview"
    active_idx = TAB_NAMES.index(current_tab)

    tab1, tab2, tab3, tab4 = st.tabs(TAB_LABELS)
    # ========================================================
    # TAB 1 — OVERVIEW
    # ========================================================
    with tab1:
        st.subheader("🧾 Contract Summary")
        st.caption("Auto-generated overview based on detected clauses and deviations")

        st.markdown("### 📝 Executive Summary")
        st.write(st.session_state.summary_narration)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Spans", summary["overview"]["total_spans"])
        col2.metric("Recognized Clauses", summary["overview"]["recognized_clauses"])
        col3.metric("Unknown Sections", summary["overview"]["unknown_spans"])
        col4.metric("Deviating Clauses", summary["overview"]["deviating_spans"])

        # Risk snapshot with severity badges
        if summary["deviations"]:
            st.markdown("### ⚠️ Risk Snapshot")
            for d in summary["deviations"]:
                sev = d.get("severity", "Medium")
                badge = severity_badge(sev)
                st.markdown(
                    f"{badge} &nbsp; **{d['clause']}** — {', '.join(d['reasons'])}",
                    unsafe_allow_html=True
                )
        else:
            st.success("✅ No non-standard clause patterns detected.")

        # Clause coverage with counts
        st.markdown("### 📌 Clause Coverage")
        counts = summary["coverage"].get("clause_counts", {})
        coverage_parts = [
            f"{c} ({counts.get(c, '?')})"
            for c in summary["coverage"]["detected_clauses"]
        ]
        st.write(", ".join(coverage_parts))
        st.caption(summary["coverage"]["undetected_note"])

        # Clause → Span Mapping
        st.markdown("---")
        st.subheader("📄 Clause to Text Mapping")

        for clause_name in sorted(clause_df["final_clause"].unique()):
            if clause_name == "Unknown":
                continue

            rows = clause_df[clause_df["final_clause"] == clause_name]
            has_deviation = rows["final_deviation"].any()
            label = f"{'⚠️ ' if has_deviation else ''}{clause_name} ({len(rows)} span{'s' if len(rows) > 1 else ''})"

            with st.expander(label):
                for _, row in rows.iterrows():
                    sid = int(row["span_id"])
                    conf = row["confidence"]
                    flag = " ⚠️" if row["final_deviation"] else ""
                    sev = row.get("severity")

                    header = f"**Span {sid} | confidence={conf:.2f}{flag}**"
                    if sev:
                        header += f" &nbsp; {severity_badge(sev)}"
                        st.markdown(header, unsafe_allow_html=True)
                    else:
                        st.markdown(header)

                    st.write(spans[sid])
                    st.markdown("---")

        st.markdown("---")
        for note in summary["confidence_notes"]:
            st.caption(note)

    # ========================================================
    # TAB 2 — DEVIATING CLAUSES
    # ========================================================
    with tab2:
        deviating = clause_df[clause_df["final_deviation"]]

        if deviating.empty:
            st.success("No non-standard / deviating clauses detected.")
        else:
            # Sort by severity: High first
            sev_order = {"High": 0, "Medium": 1, "Low": 2, None: 3}
            deviating = deviating.copy()
            deviating["_sev_order"] = deviating["severity"].map(sev_order)
            deviating = deviating.sort_values("_sev_order").drop(columns="_sev_order")

            high = (deviating["severity"] == "High").sum()
            med = (deviating["severity"] == "Medium").sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Deviations", len(deviating))
            col2.metric("High Severity", int(high))
            col3.metric("Medium Severity", int(med))

            st.markdown("---")

            for _, row in deviating.iterrows():
                sid = int(row["span_id"])
                sev = row.get("severity", "Medium")
                badge = severity_badge(sev)

                with st.expander(
                    f"{row['final_clause']} | span {sid} | {sev} severity"
                ):
                    st.markdown(f"**Severity:** {badge}", unsafe_allow_html=True)

                    st.markdown("**Deviation reasons:**")
                    for r in row["deviation_reasons"]:
                        st.write(f"- {r}")

                    from pipeline import explain_deviation_reasons
                    explanations = explain_deviation_reasons(row["deviation_reasons"])
                    if explanations:
                        st.markdown("**Explanations:**")
                        for exp in explanations:
                            st.info(exp)

                    st.markdown("**Clause text:**")
                    st.write(spans[sid])

    # ========================================================
    # TAB 3 — ANALYTICS
    # ========================================================
    with tab3:
        st.subheader("📊 Clause Analytics")

        import pandas as pd

        # Confidence distribution
        st.markdown("#### Confidence Score Distribution")
        known_df = clause_df[clause_df["final_clause"] != "Unknown"]
        if not known_df.empty:
            conf_chart = (
                known_df.groupby("final_clause")["confidence"]
                .mean()
                .reset_index()
                .rename(columns={"final_clause": "Clause", "confidence": "Avg Confidence"})
                .sort_values("Avg Confidence", ascending=False)
            )
            st.bar_chart(conf_chart.set_index("Clause"))

        # Clause frequency
        st.markdown("#### Clause Frequency")
        freq = (
            clause_df["final_clause"]
            .value_counts()
            .reset_index()
            .rename(columns={"final_clause": "Clause", "count": "Count"})
        )
        st.bar_chart(freq.set_index("Clause"))

        # Deviation breakdown table
        st.markdown("#### Deviation Summary Table")
        table_rows = []
        for clause in sorted(clause_df["final_clause"].unique()):
            if clause == "Unknown":
                continue
            subset = clause_df[clause_df["final_clause"] == clause]
            dev_count = subset["final_deviation"].sum()
            avg_conf = subset["confidence"].mean()
            table_rows.append({
                "Clause": clause,
                "Spans": len(subset),
                "Deviating": int(dev_count),
                "Avg Confidence": f"{avg_conf:.2f}",
            })
        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # ========================================================
    # TAB 4 — ASK THE CONTRACT
    # ========================================================
    with tab4:
        st.subheader("Ask a question about the contract")

        if not st.session_state.ollama_ok:
            st.warning(
                "Ollama is not running. Answers will show retrieved clause text "
                "without LLM explanation. Run `ollama serve` for full Q&A."
            )

        question = st.text_input(
            "Example: What does the cap on liability mean for me?"
        )

        if st.button("Ask"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                st.query_params["tab"] = "ask"
                retrieval = ask_document(
                    question,
                    clause_df,
                    spans,
                    st.session_state.embeddings,
                    st.session_state.embedder
                )

                evidence_list = retrieval.get("evidence", [])

                if llm is not None:
                    # Limit to top 2 evidence blocks to reduce token count and inference time
                    top_evidence = evidence_list[:2]
                    context_blocks = []
                    for e in top_evidence:
                        # Truncate long clause texts to 400 chars to keep prompt tight
                        text_snippet = e["text"][:400] + ("…" if len(e["text"]) > 400 else "")
                        block = f"CLAUSE: {e['clause']}"
                        if e["deviating"] and e["reasons"]:
                            block += f"\nDEVIATION: {'; '.join(e['reasons'])}"
                        block += f"\nTEXT: {text_snippet}"
                        context_blocks.append(block)

                    context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant clauses found."

                    prompt = f"""You are a contract analyst. Answer the question below using only the clause text provided. Be concise and use plain English. Use bullet points if helpful. End with: "Note: This is not legal advice."

CLAUSES:
{context_str}

QUESTION: {question}

ANSWER:"""
                    explanation = ollama_client(prompt)
                else:
                    # Deterministic fallback — summarise from evidence directly
                    detected = list(set(e["clause"] for e in evidence_list))
                    deviating = [e for e in evidence_list if e["deviating"]]
                    if deviating:
                        explanation = (
                            f"Relevant clauses found: {', '.join(detected)}. "
                            f"{len(deviating)} deviating clause(s) detected. "
                            f"Reasons: {'; '.join(deviating[0]['reasons'])}. "
                            "See the clause text below for details. (Ollama not running — enable for full explanation.)"
                        )
                    elif detected:
                        explanation = (
                            f"Relevant clauses found: {', '.join(detected)}. "
                            "No deviations detected. See the clause text below for details. "
                            "(Ollama not running — enable for full explanation.)"
                        )
                    else:
                        explanation = (
                            "No directly relevant clauses were found. "
                            "Please review the contract text shown below."
                        )

                st.session_state.last_answer = {
                    "explanation": explanation,
                    "evidence": retrieval.get("evidence", []),
                    "confidence_notes": retrieval.get("confidence_notes", [])
                }

        # Render answer
        if st.session_state.last_answer:
            answer = st.session_state.last_answer

            st.markdown("### 📌 Answer")
            st.write(answer["explanation"])

            if answer["evidence"]:
                st.markdown("---")
                st.caption("Supporting sections from the contract:")

                for ev in answer["evidence"]:
                    sev = ev.get("severity")
                    label = f"{ev['clause']} | span {ev['span_id']}"
                    if sev:
                        label += f" | {sev} severity"

                    with st.expander(label):
                        if ev["deviating"] and ev["reasons"]:
                            st.markdown("**Deviation reasons:**")
                            for r in ev["reasons"]:
                                st.write(f"- {r}")
                        st.write(ev["text"])

            st.markdown("---")
            for note in answer["confidence_notes"]:
                st.caption(note)

else:
    st.info("⬅️ Upload a PDF and click **Analyze Contract** to begin.")
