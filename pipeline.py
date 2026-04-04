# ============================================================
# CONTRACT CLAUSE DEVIATION — PIPELINE
# ============================================================

import pdfplumber
import re
import torch
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================

MAX_SEQ_LENGTH = 128
CONFIDENCE_THRESHOLD = 0.5
device = torch.device("cpu")

CLAUSE_MODEL_PATH = "resources/deberta-clause-final"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ============================================================
# EXPLANATION ENGINE
# ============================================================

EXPLANATION_TEMPLATES = {
    "Uncapped liability detected": (
        "This clause is considered risky because it does not place a clear limit on liability. "
        "In standard contracts, liability is typically capped to control financial exposure. "
        "Without a cap, potential losses may be unlimited."
    ),
    "Permission / obligation polarity mismatch": (
        "This clause alters the usual balance between obligations and permissions. "
        "Such deviations can create ambiguity in enforcement or responsibility."
    ),
    "Semantic deviation from standard clause language": (
        "The language in this clause differs from commonly observed contract patterns. "
        "Unusual phrasing may increase interpretation risk."
    ),
    "Violation of non-negotiable license ownership invariant": (
        "This clause appears to affect ownership or transfer of rights, which is usually treated "
        "as non-negotiable in standard agreements."
    ),
}

CLAUSE_ALIASES = {
    "Warranty Duration": ["warranty", "warranty period"],
    "Cap On Liability": ["liability", "liability cap", "cap on liability"],
    "License Grant": ["license", "licensing"],
    "Termination For Convenience": ["termination", "terminate"],
    "Anti-Assignment": ["assignment", "assign"],
    "Audit Rights": ["audit", "inspection"],
    "Non-Compete": ["non-compete", "noncompete", "competition"],
    "Exclusivity": ["exclusive", "exclusivity"],
    "Insurance": ["insurance", "indemnity"],
    "Ip Ownership Assignment": ["intellectual property", "ip ownership", "ip assignment"],
}

SEVERITY_MAP = {
    "Violation of non-negotiable license ownership invariant": "High",
    "Uncapped liability detected": "High",
    "Permission / obligation polarity mismatch": "Medium",
    "Semantic deviation from standard clause language": "Medium",
}


def explain_deviation_reasons(reasons):
    return [
        EXPLANATION_TEMPLATES.get(
            r,
            "This clause deviates from standard contractual patterns and may warrant review."
        )
        for r in reasons
    ]


def get_severity(reasons):
    """Returns the highest severity level across all deviation reasons."""
    levels = {"High": 3, "Medium": 2, "Low": 1}
    max_level = 1
    for r in reasons:
        s = SEVERITY_MAP.get(r, "Low")
        if levels[s] > max_level:
            max_level = levels[s]
    return {3: "High", 2: "Medium", 1: "Low"}[max_level]


# ============================================================
# CACHED LOADERS
# ============================================================

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(
        CLAUSE_MODEL_PATH,
        use_fast=False,
        local_files_only=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(CLAUSE_MODEL_PATH)
    model.eval()
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return tokenizer, model, embedder


@st.cache_resource
def load_baselines():
    return (
        np.load("resources/clause_centroids.npy", allow_pickle=True).item(),
        np.load("resources/clause_thresholds.npy", allow_pickle=True).item(),
        np.load("resources/clause_applicability.npy", allow_pickle=True).item(),
        np.load("resources/clause_polarity.npy", allow_pickle=True).item()
    )


clause_centroids, clause_thresholds, clause_applicability_thresholds, clause_polarity_profiles = load_baselines()

# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def normalize_span(text):
    text = re.sub(r"\b(company|licensor|licensee|producer|party)\b", "party", text)
    text = re.sub(r"\b\d+(\.\d+)?\b", "num", text)
    text = re.sub(r"\b(day|days|month|months|year|years)\b", "time", text)
    return text


def generate_spans(text, min_len=80):
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= min_len]
    # Further split very long paragraphs on numbered subsections
    refined = []
    for p in paragraphs:
        if len(p) > 1000:
            parts = re.split(r'(?=\n?\d+\.\d+\s)', p)
            refined.extend([x.strip() for x in parts if len(x.strip()) >= min_len])
        else:
            refined.append(p)
    return refined


# ============================================================
# INVARIANT RULES
# ============================================================

def license_grant_invariant(text):
    return any(t in text for t in [
        "ownership", "transfer ownership", "full ownership", "full rights"
    ])


def cap_liability_invariant(text):
    return any(t in text for t in [
        "unlimited liability", "without limitation", "no limitation"
    ])


def polarity_violation(text, profile):
    if profile.get("not", 0) > 0.6:
        return any(p in text for p in [
            "freely", "without restriction", "without approval"
        ])
    return False


# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

def analyze_document(pdf_path, progress_callback=None):
    """
    Full pipeline: PDF → spans → clause classification → deviation detection.

    Args:
        pdf_path: path to the uploaded PDF
        progress_callback: optional callable(step: int, total: int, message: str)

    Returns:
        clause_df, spans, embeddings, embedder
    """
    tokenizer, model, embedder = load_models()
    id_to_clause = {int(k): v for k, v in model.config.id2label.items()}

    # ---- PDF TO TEXT ----
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            if p.extract_text():
                pages.append(p.extract_text())

    spans = generate_spans("\n\n".join(pages))
    total_steps = len(spans) + 1  # +1 for embedding step

    # ---- CLAUSE CLASSIFICATION ----
    records = []
    for i, span in enumerate(spans):
        if progress_callback:
            progress_callback(i, total_steps, f"Classifying span {i+1}/{len(spans)}…")

        encoded = tokenizer(
            span,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )

        with torch.no_grad():
            logits = model(**encoded).logits

        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_id = int(np.argmax(probs))
        pred_clause = id_to_clause[pred_id]
        confidence = float(probs[pred_id])

        final_clause = pred_clause if confidence >= CONFIDENCE_THRESHOLD else "Unknown"

        records.append({
            "span_id": i,
            "final_clause": final_clause,
            "predicted_clause": pred_clause,
            "confidence": confidence
        })

    clause_df = pd.DataFrame(records)

    # ---- EMBEDDINGS ----
    if progress_callback:
        progress_callback(total_steps - 1, total_steps, "Computing embeddings…")

    norm_spans = [normalize_span(clean_text(s)) for s in spans]
    embeddings = embedder.encode(norm_spans, batch_size=16)

    # ---- DEVIATION DETECTION ----
    deviation_rows = []
    for idx, row in clause_df.iterrows():
        clause = row["final_clause"]
        raw_text = spans[idx].lower()
        reasons = []

        if clause == "Unknown" or clause not in clause_centroids:
            deviation_rows.append({
                "final_deviation": False,
                "deviation_reasons": [],
                "semantic_distance": None,
                "severity": None,
            })
            continue

        dist = cosine_distances(
            embeddings[idx].reshape(1, -1),
            clause_centroids[clause].reshape(1, -1)
        )[0][0]

        if dist > clause_thresholds[clause]:
            reasons.append("Semantic deviation from standard clause language")

        if polarity_violation(raw_text, clause_polarity_profiles[clause]):
            reasons.append("Permission / obligation polarity mismatch")

        if clause == "License Grant" and license_grant_invariant(raw_text):
            reasons.append("Violation of non-negotiable license ownership invariant")

        if clause == "Cap On Liability" and cap_liability_invariant(raw_text):
            reasons.append("Uncapped liability detected")

        deviation_rows.append({
            "final_deviation": len(reasons) > 0,
            "deviation_reasons": reasons,
            "semantic_distance": float(dist),
            "severity": get_severity(reasons) if reasons else None,
        })

    deviation_df = pd.DataFrame(deviation_rows)
    clause_df = pd.concat(
        [clause_df.reset_index(drop=True), deviation_df.reset_index(drop=True)],
        axis=1
    )

    return clause_df, spans, embeddings, embedder


# ============================================================
# QUESTION ANSWERING
# ============================================================

def ask_document(
    question,
    clause_df,
    spans,
    embeddings,
    embedder,
    top_k=5,
    similarity_threshold=0.35
):
    q = question.lower()

    # --- Risk overview ---
    if any(k in q for k in ["risk", "deviation", "non-standard", "red flag", "concern", "problem"]):
        deviating = clause_df[clause_df["final_deviation"]]
        evidence = []
        for _, row in deviating.iterrows():
            reasons = row.get("deviation_reasons", [])
            evidence.append({
                "span_id": row["span_id"],
                "clause": row["final_clause"],
                "deviating": True,
                "reasons": reasons,
                "explanations": explain_deviation_reasons(reasons),
                "severity": row.get("severity"),
                "text": spans[row["span_id"]]
            })
        return {
            "intent": "RISK_EXPLANATION",
            "evidence": evidence,
            "confidence_notes": [
                "Explanations are derived from detected deviations.",
                "This analysis is informational and not legal advice."
            ]
        }

    # --- Clause lookup ---
    for clause_name in clause_df["final_clause"].unique():
        aliases = CLAUSE_ALIASES.get(clause_name, [])
        if clause_name != "Unknown" and (
            clause_name.lower() in q or any(a in q for a in aliases)
        ):
            matched = clause_df[clause_df["final_clause"] == clause_name]
            evidence = []
            for _, row in matched.iterrows():
                reasons = row.get("deviation_reasons", [])
                evidence.append({
                    "span_id": row["span_id"],
                    "clause": clause_name,
                    "deviating": row["final_deviation"],
                    "reasons": reasons,
                    "explanations": explain_deviation_reasons(reasons),
                    "severity": row.get("severity"),
                    "text": spans[row["span_id"]]
                })
            return {
                "intent": "CLAUSE_EXPLANATION",
                "evidence": evidence,
                "confidence_notes": [
                    f"Answer based on detected '{clause_name}' clauses.",
                    "This is not legal advice."
                ]
            }

    # --- Semantic retrieval ---
    q_emb = embedder.encode([question])[0]
    sims = cosine_similarity(q_emb.reshape(1, -1), embeddings)[0]

    candidates = sorted(
        [(i, sims[i]) for i in range(len(sims)) if sims[i] >= similarity_threshold],
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    evidence = []
    for idx, score in candidates:
        row = clause_df.iloc[idx]
        reasons = row.get("deviation_reasons", [])
        evidence.append({
            "span_id": idx,
            "clause": row["final_clause"],
            "deviating": row["final_deviation"],
            "reasons": reasons,
            "explanations": explain_deviation_reasons(reasons),
            "severity": row.get("severity"),
            "similarity_score": float(score),
            "text": spans[idx]
        })

    return {
        "intent": "EVIDENCE_LOOKUP",
        "evidence": evidence,
        "confidence_notes": [
            "Answer derived from semantic similarity and clause analysis.",
            "This is not legal advice."
        ]
    }


# ============================================================
# DETERMINISTIC SUMMARY BUILDER (NO LLM)
# ============================================================

def build_contract_summary(clause_df, spans):
    recognized = clause_df[clause_df["final_clause"] != "Unknown"]
    unknown = clause_df[clause_df["final_clause"] == "Unknown"]
    deviating = clause_df[clause_df["final_deviation"] == True]

    # Per-clause span counts
    clause_counts = (
        recognized.groupby("final_clause")
        .size()
        .reset_index(name="count")
        .set_index("final_clause")["count"]
        .to_dict()
    )

    return {
        "overview": {
            "total_spans": int(len(clause_df)),
            "recognized_clauses": int(len(recognized)),
            "unknown_spans": int(len(unknown)),
            "deviating_spans": int(len(deviating)),
        },
        "coverage": {
            "detected_clauses": sorted(recognized["final_clause"].unique().tolist()),
            "clause_counts": clause_counts,
            "undetected_note": "Some sections could not be confidently mapped to known clause categories."
        },
        "deviations": [
            {
                "clause": row["final_clause"],
                "span_id": int(row["span_id"]),
                "reasons": row["deviation_reasons"],
                "severity": row.get("severity", "Medium"),
                "excerpt": spans[int(row["span_id"])][:300] + (
                    "…" if len(spans[int(row["span_id"])]) > 300 else ""
                ),
            }
            for _, row in deviating.iterrows()
        ],
        "confidence_notes": [
            "Deviation detection is based on learned reference patterns.",
            "Absence of deviation does not imply low legal risk.",
            "This system does not provide legal advice."
        ]
    }


# ============================================================
# GUARDED NARRATION (OPTIONAL LLM)
# ============================================================

def narrate_contract_summary(summary, llm_client=None):
    if llm_client is None:
        return (
            "This contract has been analyzed for standard clause structure and deviations. "
            "Highlighted sections may require closer review."
        )

    prompt = f"""
You are summarizing contract analysis findings.

Rules:
- Use only the provided information
- Do not add legal advice
- Be neutral and concise

Detected clauses:
{", ".join(summary["coverage"]["detected_clauses"])}

Number of deviating sections:
{summary["overview"]["deviating_spans"]}

Write a short executive summary (3–5 sentences).
"""

    try:
        return llm_client(prompt).strip()
    except Exception:
        return "Executive summary could not be generated at this time."


# ============================================================
# EXPORT HELPERS
# ============================================================

def export_results_csv(clause_df, spans):
    """Returns a CSV string of the full analysis results."""
    export_df = clause_df.copy()
    export_df["text"] = export_df["span_id"].apply(
        lambda i: spans[i] if i < len(spans) else ""
    )
    export_df["deviation_reasons"] = export_df["deviation_reasons"].apply(
        lambda r: "; ".join(r) if isinstance(r, list) else ""
    )
    cols = ["span_id", "final_clause", "confidence", "final_deviation",
            "severity", "deviation_reasons", "semantic_distance", "text"]
    available = [c for c in cols if c in export_df.columns]
    return export_df[available].to_csv(index=False)


def export_results_json(clause_df, spans, contract_summary):
    """Returns a JSON string of the full analysis results."""
    import json
    rows = []
    for _, row in clause_df.iterrows():
        sid = int(row["span_id"])
        rows.append({
            "span_id": sid,
            "final_clause": row["final_clause"],
            "confidence": round(float(row["confidence"]), 4),
            "final_deviation": bool(row["final_deviation"]),
            "severity": row.get("severity"),
            "deviation_reasons": row.get("deviation_reasons", []),
            "text": spans[sid] if sid < len(spans) else ""
        })
    return json.dumps({
        "summary": contract_summary["overview"],
        "clauses": rows
    }, indent=2)
