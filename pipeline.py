# ============================================================
# CONTRACT CLAUSE DEVIATION — ENHANCED PIPELINE
# ============================================================

import re
import torch
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st

from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================

MAX_SEQ_LENGTH = 512          # increased from 128 — captures full clause context
CONFIDENCE_THRESHOLD = 0.45   # slightly lower — adaptive per-clause thresholds handle precision
CONFIDENCE_GAP_MIN = 0.10     # top-1 must beat top-2 by at least this margin
SEMANTIC_DEVIATION_PERCENTILE = 90  # was 95 — catches more deviations
device = torch.device("cpu")

import os as _os
_BASE = _os.path.dirname(_os.path.abspath(__file__))
CLAUSE_MODEL_PATH = _os.path.join(_BASE, "resources", "deberta-clause-final")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ============================================================
# EXPLANATION TEMPLATES
# ============================================================

EXPLANATION_TEMPLATES = {
    "Uncapped liability detected": (
        "This clause does not place a clear limit on liability. Standard contracts cap liability "
        "to control financial exposure. Without a cap, potential losses may be unlimited."
    ),
    "Permission / obligation polarity mismatch": (
        "This clause uses permissive language ('freely', 'without restriction') where standard "
        "language is typically restrictive. This may create ambiguity in enforcement."
    ),
    "Semantic deviation from standard clause language": (
        "The language in this clause differs significantly from commonly observed contract patterns. "
        "Unusual phrasing may increase interpretation risk."
    ),
    "Violation of non-negotiable license ownership invariant": (
        "This clause appears to affect ownership or transfer of rights, which is typically "
        "non-negotiable in standard agreements."
    ),
    "Missing standard protective keyword": (
        "This clause is missing language that typically appears in standard versions of this clause type. "
        "The absence may reduce legal protection."
    ),
    "Negation of standard obligation": (
        "This clause negates or removes an obligation that is normally present in standard contracts. "
        "This may reduce enforceability or protection."
    ),
    "Unusual unilateral right detected": (
        "This clause grants one-sided rights that are atypical for this clause type. "
        "This may create an imbalance in contractual obligations."
    ),
}

SEVERITY_MAP = {
    "Violation of non-negotiable license ownership invariant": "High",
    "Uncapped liability detected": "High",
    "Negation of standard obligation": "High",
    "Unusual unilateral right detected": "High",
    "Permission / obligation polarity mismatch": "Medium",
    "Missing standard protective keyword": "Medium",
    "Semantic deviation from standard clause language": "Medium",
}

CLAUSE_ALIASES = {
    "Warranty Duration": ["warranty", "warranty period", "warrants"],
    "Cap On Liability": ["liability", "liability cap", "cap on liability", "limitation of liability"],
    "License Grant": ["license", "licensing", "licence", "licensed"],
    "Termination For Convenience": ["termination", "terminate", "terminating"],
    "Anti-Assignment": ["assignment", "assign", "assignable", "transfer"],
    "Audit Rights": ["audit", "inspection", "examine", "records"],
    "Non-Compete": ["non-compete", "noncompete", "competition", "compete"],
    "Exclusivity": ["exclusive", "exclusivity", "sole", "only supplier"],
    "Insurance": ["insurance", "insured", "indemnity", "indemnification"],
    "Ip Ownership Assignment": ["intellectual property", "ip ownership", "ip assignment", "work for hire"],
    "Confidentiality": ["confidential", "nda", "non-disclosure", "proprietary"],
    "Governing Law": ["governing law", "jurisdiction", "applicable law"],
    "Dispute Resolution": ["dispute", "arbitration", "mediation", "litigation"],
}


def explain_deviation_reasons(reasons):
    return [
        EXPLANATION_TEMPLATES.get(r, "This clause deviates from standard patterns and may warrant review.")
        for r in reasons
    ]


def get_severity(reasons):
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
        CLAUSE_MODEL_PATH, use_fast=False, local_files_only=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(CLAUSE_MODEL_PATH)
    model.eval()
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return tokenizer, model, embedder


@st.cache_resource
def load_baselines():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(base, "resources")
    centroids     = np.load(os.path.join(res, "clause_centroids.npy"),      allow_pickle=True).item()
    thresholds    = np.load(os.path.join(res, "clause_thresholds.npy"),     allow_pickle=True).item()
    applicability = np.load(os.path.join(res, "clause_applicability.npy"),  allow_pickle=True).item()
    polarity      = np.load(os.path.join(res, "clause_polarity.npy"),       allow_pickle=True).item()
    kw_path = os.path.join(res, "clause_keywords.npy")
    keywords = np.load(kw_path, allow_pickle=True).item() if os.path.exists(kw_path) else {}
    return centroids, thresholds, applicability, polarity, keywords


def _get_baselines():
    """Lazy loader — called inside analyze_document, not at import time."""
    return load_baselines()



# ============================================================
# ENHANCED TEXT PROCESSING
# ============================================================

# Heading patterns that signal a new clause section
HEADING_PATTERN = re.compile(
    r'^(\d+[\.\)]\s*[A-Z][^\n]{3,60}|'       # numbered: "1. Confidentiality"
    r'[A-Z][A-Z\s]{4,50}[A-Z]|'              # ALL CAPS heading
    r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}:)', # Title Case heading with colon
    re.MULTILINE
)

SECTION_SPLIT_PATTERN = re.compile(
    r'\n(?=\s*(?:\d+[\.\)]\s+[A-Z]|[A-Z]{4,}|\b(?:WHEREAS|NOW THEREFORE|IN WITNESS)\b))'
)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\,\;\:\-]", "", text)
    return text.strip()


def normalize_span(text):
    """Normalize for embedding — replace variable tokens with canonical forms."""
    text = re.sub(r"\b(company|licensor|licensee|producer|vendor|client|customer|partner)\b", "party", text)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", "num", text)   # numbers incl. currency
    text = re.sub(r"\$[\d,\.]+", "num", text)                           # dollar amounts
    text = re.sub(r"\b(day|days|month|months|year|years|week|weeks)\b", "time", text)
    text = re.sub(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", "month_name", text)
    return text


def _merge_short_spans(spans, min_len=80, max_merge_len=1200):
    """Merge consecutive short spans that are likely continuation of the same clause."""
    merged = []
    buffer = ""
    for span in spans:
        if len(buffer) + len(span) < max_merge_len and len(buffer) < min_len:
            buffer = (buffer + " " + span).strip()
        else:
            if buffer:
                merged.append(buffer)
            buffer = span
    if buffer:
        merged.append(buffer)
    return [s for s in merged if len(s) >= min_len]


def generate_spans(text, min_len=80, max_len=1500):
    """
    Enhanced span extraction:
    1. Split on section headings (numbered, ALL CAPS, Title Case:)
    2. Split on double newlines within sections
    3. Split long spans on sentence boundaries
    4. Merge orphaned short spans
    5. Enforce min/max length
    """
    # Step 1: Split on major section boundaries
    sections = SECTION_SPLIT_PATTERN.split(text)
    if len(sections) <= 1:
        # Fallback: split on double newlines
        sections = text.split("\n\n")

    raw_spans = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        # Step 2: Within each section, split on double newlines
        sub = [p.strip() for p in section.split("\n\n") if p.strip()]
        raw_spans.extend(sub)

    # Step 3: Split spans that are too long on sentence boundaries
    refined = []
    for span in raw_spans:
        if len(span) > max_len:
            # Split on sentence endings followed by capital letter
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', span)
            chunk = ""
            for sent in sentences:
                if len(chunk) + len(sent) <= max_len:
                    chunk = (chunk + " " + sent).strip()
                else:
                    if len(chunk) >= min_len:
                        refined.append(chunk)
                    chunk = sent
            if len(chunk) >= min_len:
                refined.append(chunk)
        elif len(span) >= min_len:
            refined.append(span)

    # Step 4: Merge orphaned short spans
    refined = _merge_short_spans(refined, min_len=min_len)

    # Step 5: Deduplicate while preserving order
    seen = set()
    deduped = []
    for s in refined:
        key = s[:100]
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    return deduped


# ============================================================
# ENHANCED DEVIATION RULES
# ============================================================

# Per-clause keyword profiles: (required_keywords, forbidden_keywords, unilateral_signals)
CLAUSE_RULES = {
    "License Grant": {
        "required": ["license", "grant", "right"],
        "forbidden": ["ownership", "transfer ownership", "full ownership", "assign all rights"],
        "unilateral": ["irrevocable", "perpetual", "unconditional"],
        "negation_targets": ["sublicense", "transfer", "assign"],
    },
    "Cap On Liability": {
        "required": [],
        "forbidden": ["unlimited liability", "without limitation", "no cap", "no limitation", "liable for all"],
        "unilateral": ["sole discretion", "unilaterally"],
        "negation_targets": ["cap", "limit", "ceiling"],
    },
    "Termination For Convenience": {
        "required": ["terminat"],
        "forbidden": ["irrevocable", "cannot be terminated", "no right to terminate"],
        "unilateral": ["sole discretion", "without cause", "at will"],
        "negation_targets": ["notice", "cure period", "remedy"],
    },
    "Anti-Assignment": {
        "required": ["assign"],
        "forbidden": ["freely assign", "assign without consent", "assign at will"],
        "unilateral": ["sole discretion"],
        "negation_targets": ["consent", "approval", "prior written"],
    },
    "Audit Rights": {
        "required": ["audit", "inspect", "examin"],
        "forbidden": ["no right to audit", "waive.*audit", "cannot inspect"],
        "unilateral": [],
        "negation_targets": ["records", "books", "access"],
    },
    "Non-Compete": {
        "required": ["compet"],
        "forbidden": ["freely compete", "no restriction on competition"],
        "unilateral": ["sole discretion"],
        "negation_targets": ["restrict", "prohibit", "shall not"],
    },
    "Exclusivity": {
        "required": ["exclusiv"],
        "forbidden": ["non-exclusive", "not exclusive"],
        "unilateral": ["sole", "only"],
        "negation_targets": ["exclusive", "sole"],
    },
    "Warranty Duration": {
        "required": ["warrant"],
        "forbidden": ["no warranty", "as is", "disclaim", "without warranty"],
        "unilateral": [],
        "negation_targets": ["warrant", "guarantee", "represent"],
    },
    "Insurance": {
        "required": ["insur"],
        "forbidden": ["no insurance", "waive insurance"],
        "unilateral": [],
        "negation_targets": ["maintain", "procure", "obtain"],
    },
    "Ip Ownership Assignment": {
        "required": ["intellectual property", "ip", "invention", "work product"],
        "forbidden": ["retain ownership", "does not assign", "no assignment of ip"],
        "unilateral": ["automatically assign", "hereby assign"],
        "negation_targets": ["assign", "transfer", "vest"],
    },
    "Confidentiality": {
        "required": ["confidential"],
        "forbidden": ["no confidentiality", "freely disclose", "no obligation of confidentiality"],
        "unilateral": [],
        "negation_targets": ["shall not disclose", "keep confidential", "protect"],
    },
    "Uncapped Liability": {
        "required": ["liab"],
        "forbidden": [],
        "unilateral": [],
        "negation_targets": [],
    },
}


def _has_negation(text, targets):
    """Check if any target word is negated (preceded by not/no/never/without)."""
    negation_pattern = re.compile(
        r'\b(not|no|never|without|waive|waiver|disclaim)\b.{0,40}(' + '|'.join(targets) + r')',
        re.IGNORECASE
    )
    return bool(negation_pattern.search(text))


def _keyword_density_score(text, keywords):
    """Returns fraction of keywords present in text (0.0–1.0)."""
    if not keywords:
        return 1.0
    hits = sum(1 for kw in keywords if kw.lower() in text.lower())
    return hits / len(keywords)


def check_clause_rules(clause, raw_text):
    """
    Apply per-clause rule checks. Returns list of deviation reasons.
    """
    reasons = []
    rules = CLAUSE_RULES.get(clause, {})
    text_lower = raw_text.lower()

    # 1. Forbidden patterns (hard violations)
    for forbidden in rules.get("forbidden", []):
        if re.search(forbidden, text_lower):
            if clause == "Cap On Liability":
                reasons.append("Uncapped liability detected")
            elif clause == "License Grant":
                reasons.append("Violation of non-negotiable license ownership invariant")
            else:
                reasons.append("Unusual unilateral right detected")
            break

    # 2. Negation of standard obligations
    negation_targets = rules.get("negation_targets", [])
    if negation_targets and _has_negation(text_lower, negation_targets):
        reasons.append("Negation of standard obligation")

    # 3. Missing required keywords (only flag if clause has required keywords defined)
    required = rules.get("required", [])
    if required:
        density = _keyword_density_score(text_lower, required)
        if density < 0.5:
            reasons.append("Missing standard protective keyword")

    # 4. Unilateral rights
    unilateral = rules.get("unilateral", [])
    if unilateral and any(u in text_lower for u in unilateral):
        # Only flag if combined with other signals (avoid false positives)
        if len(reasons) > 0 or _keyword_density_score(text_lower, required) < 0.7:
            reasons.append("Unusual unilateral right detected")

    return list(dict.fromkeys(reasons))  # deduplicate preserving order


def polarity_violation(text, profile):
    """Enhanced polarity check with weighted signals."""
    text_lower = text.lower()
    permission_terms = [
        "freely", "without restriction", "without approval",
        "at its sole discretion", "unconditionally", "without limitation"
    ]
    obligation_score = profile.get("not", 0) * 0.5 + profile.get("shall", 0) * 0.3
    if obligation_score > 0.5:
        return any(p in text_lower for p in permission_terms)
    return False


# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

def analyze_document(pdf_path, progress_callback=None):
    """
    Enhanced pipeline: PDF → spans → clause classification → multi-signal deviation detection.
    """
    tokenizer, model, embedder = load_models()
    id_to_clause = {int(k): v for k, v in model.config.id2label.items()}

    # Load baselines lazily (after models/files are downloaded)
    clause_centroids, clause_thresholds, clause_applicability_thresholds, \
        clause_polarity_profiles, clause_keyword_profiles = _get_baselines()
    num_labels = len(id_to_clause)

    # ---- PDF TO TEXT ----
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            page_text = p.extract_text()
            if page_text:
                pages.append(page_text)

    full_text = "\n\n".join(pages)
    spans = generate_spans(full_text)
    total_steps = len(spans) + 1

    # ---- CLAUSE CLASSIFICATION ----
    records = []
    for i, span in enumerate(spans):
        if progress_callback:
            progress_callback(i, total_steps, f"Classifying span {i+1}/{len(spans)}…")

        encoded = tokenizer(
            span,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=True,
        )

        with torch.no_grad():
            logits = model(**encoded).logits

        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # Top-2 predictions
        top2_idx = np.argsort(probs)[::-1][:2]
        pred_id = int(top2_idx[0])
        second_id = int(top2_idx[1])
        pred_clause = id_to_clause[pred_id]
        confidence = float(probs[pred_id])
        confidence_gap = float(probs[pred_id] - probs[second_id])
        second_clause = id_to_clause[second_id]
        second_confidence = float(probs[second_id])

        # Adaptive threshold: require both minimum confidence AND minimum gap
        if confidence >= CONFIDENCE_THRESHOLD and confidence_gap >= CONFIDENCE_GAP_MIN:
            final_clause = pred_clause
        elif confidence >= 0.65:
            # High confidence overrides gap requirement
            final_clause = pred_clause
        else:
            final_clause = "Unknown"

        records.append({
            "span_id": i,
            "final_clause": final_clause,
            "predicted_clause": pred_clause,
            "confidence": confidence,
            "confidence_gap": confidence_gap,
            "second_clause": second_clause,
            "second_confidence": second_confidence,
        })

    clause_df = pd.DataFrame(records)

    # ---- EMBEDDINGS ----
    if progress_callback:
        progress_callback(total_steps - 1, total_steps, "Computing semantic embeddings…")

    norm_spans = [normalize_span(clean_text(s)) for s in spans]
    embeddings = embedder.encode(norm_spans, batch_size=16, show_progress_bar=False)

    # ---- MULTI-SIGNAL DEVIATION DETECTION ----
    deviation_rows = []
    for idx, row in clause_df.iterrows():
        clause = row["final_clause"]
        raw_text = spans[idx]
        reasons = []
        semantic_dist = None

        if clause == "Unknown" or clause not in clause_centroids:
            deviation_rows.append({
                "final_deviation": False,
                "deviation_reasons": [],
                "semantic_distance": None,
                "severity": None,
                "deviation_score": 0.0,
            })
            continue

        # Signal 1: Semantic distance from centroid
        semantic_dist = float(cosine_distances(
            embeddings[idx].reshape(1, -1),
            clause_centroids[clause].reshape(1, -1)
        )[0][0])

        # Use adaptive threshold (90th percentile instead of 95th)
        threshold = clause_thresholds[clause]
        adaptive_threshold = threshold * 0.95  # slightly more sensitive
        if semantic_dist > adaptive_threshold:
            reasons.append("Semantic deviation from standard clause language")

        # Signal 2: Enhanced polarity check
        if clause in clause_polarity_profiles:
            if polarity_violation(raw_text, clause_polarity_profiles[clause]):
                reasons.append("Permission / obligation polarity mismatch")

        # Signal 3: Per-clause rule checks (expanded invariants)
        rule_reasons = check_clause_rules(clause, raw_text)
        for r in rule_reasons:
            if r not in reasons:
                reasons.append(r)

        # Compute composite deviation score (0.0–1.0)
        score = 0.0
        if semantic_dist is not None and threshold > 0:
            score += min(semantic_dist / (threshold * 1.5), 0.5)  # up to 0.5 from semantic
        score += len(reasons) * 0.15  # 0.15 per additional signal
        score = min(score, 1.0)

        deviation_rows.append({
            "final_deviation": len(reasons) > 0,
            "deviation_reasons": reasons,
            "semantic_distance": semantic_dist,
            "severity": get_severity(reasons) if reasons else None,
            "deviation_score": round(score, 3),
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

def ask_document(question, clause_df, spans, embeddings, embedder, top_k=5, similarity_threshold=0.30):
    q = question.lower()

    # --- Risk overview ---
    if any(k in q for k in ["risk", "deviation", "non-standard", "red flag", "concern", "problem", "issue", "flag"]):
        deviating = clause_df[clause_df["final_deviation"]]
        evidence = []
        for _, row in deviating.iterrows():
            reasons = row.get("deviation_reasons", [])
            evidence.append({
                "span_id": int(row["span_id"]),
                "clause": row["final_clause"],
                "deviating": True,
                "reasons": reasons,
                "explanations": explain_deviation_reasons(reasons),
                "severity": row.get("severity"),
                "deviation_score": row.get("deviation_score", 0.0),
                "text": spans[int(row["span_id"])]
            })
        # Sort by deviation score descending
        evidence.sort(key=lambda x: x.get("deviation_score", 0), reverse=True)
        return {
            "intent": "RISK_EXPLANATION",
            "evidence": evidence,
            "confidence_notes": [
                "Deviations ranked by composite severity score.",
                "This analysis is informational and not legal advice."
            ]
        }

    # --- Clause lookup by name or alias ---
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
                    "span_id": int(row["span_id"]),
                    "clause": clause_name,
                    "deviating": bool(row["final_deviation"]),
                    "reasons": reasons,
                    "explanations": explain_deviation_reasons(reasons),
                    "severity": row.get("severity"),
                    "deviation_score": row.get("deviation_score", 0.0),
                    "text": spans[int(row["span_id"])]
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
        [(i, float(sims[i])) for i in range(len(sims)) if sims[i] >= similarity_threshold],
        key=lambda x: x[1], reverse=True
    )[:top_k]

    evidence = []
    for idx, score in candidates:
        row = clause_df.iloc[idx]
        reasons = row.get("deviation_reasons", [])
        evidence.append({
            "span_id": idx,
            "clause": row["final_clause"],
            "deviating": bool(row["final_deviation"]),
            "reasons": reasons,
            "explanations": explain_deviation_reasons(reasons),
            "severity": row.get("severity"),
            "similarity_score": score,
            "deviation_score": row.get("deviation_score", 0.0),
            "text": spans[idx]
        })

    return {
        "intent": "EVIDENCE_LOOKUP",
        "evidence": evidence,
        "confidence_notes": [
            "Answer derived from semantic similarity search.",
            "This is not legal advice."
        ]
    }


# ============================================================
# SUMMARY BUILDER
# ============================================================

def build_contract_summary(clause_df, spans):
    recognized = clause_df[clause_df["final_clause"] != "Unknown"]
    unknown = clause_df[clause_df["final_clause"] == "Unknown"]
    deviating = clause_df[clause_df["final_deviation"] == True]

    clause_counts = (
        recognized.groupby("final_clause").size()
        .reset_index(name="count")
        .set_index("final_clause")["count"].to_dict()
    )

    # Confidence quality breakdown
    high_conf = recognized[recognized["confidence"] >= 0.75]
    med_conf = recognized[(recognized["confidence"] >= 0.5) & (recognized["confidence"] < 0.75)]
    low_conf = recognized[recognized["confidence"] < 0.5]

    return {
        "overview": {
            "total_spans": int(len(clause_df)),
            "recognized_clauses": int(len(recognized)),
            "unknown_spans": int(len(unknown)),
            "deviating_spans": int(len(deviating)),
            "high_confidence_spans": int(len(high_conf)),
            "medium_confidence_spans": int(len(med_conf)),
            "low_confidence_spans": int(len(low_conf)),
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
                "deviation_score": row.get("deviation_score", 0.0),
                "excerpt": spans[int(row["span_id"])][:300] + (
                    "…" if len(spans[int(row["span_id"])]) > 300 else ""
                ),
            }
            for _, row in deviating.sort_values("deviation_score", ascending=False).iterrows()
        ],
        "confidence_notes": [
            "Deviation detection uses semantic similarity, polarity analysis, and clause-specific rules.",
            "Absence of deviation does not imply low legal risk.",
            "This system does not provide legal advice."
        ]
    }


def narrate_contract_summary(summary, llm_client=None):
    if llm_client is None:
        n_dev = summary["overview"]["deviating_spans"]
        clauses = ", ".join(summary["coverage"]["detected_clauses"][:6])
        if n_dev > 0:
            return (
                f"This contract contains {n_dev} clause(s) that deviate from standard patterns. "
                f"Detected clause types include: {clauses}. "
                "Highlighted sections are recommended for closer review."
            )
        return (
            f"No non-standard clause patterns were detected. "
            f"Detected clause types include: {clauses}."
        )

    prompt = (
        "Summarize these contract analysis findings in 3-5 neutral sentences. "
        "Do not add legal advice. Use only the data provided.\n\n"
        f"Clauses detected: {', '.join(summary['coverage']['detected_clauses'])}\n"
        f"Total sections: {summary['overview']['total_spans']}\n"
        f"Deviating sections: {summary['overview']['deviating_spans']}\n"
        f"High confidence detections: {summary['overview']['high_confidence_spans']}\n"
    )
    try:
        return llm_client(prompt).strip()
    except Exception:
        return "Executive summary could not be generated at this time."


# ============================================================
# EXPORT HELPERS
# ============================================================

def export_results_csv(clause_df, spans):
    export_df = clause_df.copy()
    export_df["text"] = export_df["span_id"].apply(lambda i: spans[i] if i < len(spans) else "")
    export_df["deviation_reasons"] = export_df["deviation_reasons"].apply(
        lambda r: "; ".join(r) if isinstance(r, list) else ""
    )
    cols = ["span_id", "final_clause", "confidence", "confidence_gap",
            "final_deviation", "severity", "deviation_score", "deviation_reasons", "text"]
    available = [c for c in cols if c in export_df.columns]
    return export_df[available].to_csv(index=False)


def export_results_json(clause_df, spans, contract_summary):
    import json
    rows = []
    for _, row in clause_df.iterrows():
        sid = int(row["span_id"])
        rows.append({
            "span_id": sid,
            "final_clause": row["final_clause"],
            "confidence": round(float(row["confidence"]), 4),
            "confidence_gap": round(float(row.get("confidence_gap", 0)), 4),
            "final_deviation": bool(row["final_deviation"]),
            "severity": row.get("severity"),
            "deviation_score": round(float(row.get("deviation_score", 0)), 3),
            "deviation_reasons": row.get("deviation_reasons", []),
            "text": spans[sid] if sid < len(spans) else ""
        })
    return json.dumps({"summary": contract_summary["overview"], "clauses": rows}, indent=2)
