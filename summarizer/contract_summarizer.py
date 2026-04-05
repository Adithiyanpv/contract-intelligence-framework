"""
Contract Summarizer Module
==========================
Generates structured contract summaries using:
  - Groq LLM (if connected) for per-clause abstractive summaries
  - Regex-based template field extraction (parties, dates, obligations, etc.)
  - ROUGE + coverage evaluation metrics

Summarization approach:
  1. Spans are grouped by detected clause type
  2. Each clause group is summarized independently (max 600 chars sent to LLM)
  3. Top clause summaries are combined into an overall executive summary
  4. Regex extracts structured fields from the full document text

PRIVACY: Only individual clause group text (~600 chars) is sent to the LLM.
         The full document is never transmitted to any external API.
         If no LLM is available, extractive fallback (first 2 sentences) is used.
"""

import re
from collections import defaultdict


# ── Regex patterns ─────────────────────────────────────────────────────────────
DATE_PATTERN = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\s+\d{1,2},?\s+\d{4}\b'
    r'|\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b'
    r'|\b\d{4}[\/\-]\d{2}[\/\-]\d{2}\b',
    re.IGNORECASE
)

PARTY_PATTERN = re.compile(
    r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s+'
    r'(?:Inc|LLC|Ltd|Corp|Co|Company|Group|Holdings|Technologies|Solutions|'
    r'Services|Pvt|Limited|Bank|Trust|Partners|Associates)\.?)\b',
    re.IGNORECASE
)

OBLIGATION_PATTERN = re.compile(
    r'[^.!?]*\b(?:shall|must|is required to|agrees to|undertakes to)\b[^.!?]{15,200}[.!?]',
    re.IGNORECASE
)

RIGHTS_PATTERN = re.compile(
    r'[^.!?]*\b(?:may|is entitled to|has the right to|reserves the right)\b[^.!?]{15,200}[.!?]',
    re.IGNORECASE
)

PAYMENT_PATTERN = re.compile(
    r'[^.!?]*\b(?:payment|fee|fees|compensation|royalt|revenue|invoice|'
    r'pay\b|amount|price|cost)\b[^.!?]{10,200}[.!?]',
    re.IGNORECASE
)

TERMINATION_PATTERN = re.compile(
    r'[^.!?]*\b(?:terminat|cancel|expir|end of term|notice period|'
    r'upon termination)\b[^.!?]{10,200}[.!?]',
    re.IGNORECASE
)

GOVERNING_LAW_PATTERN = re.compile(
    r'[^.!?]*\b(?:governed by|governing law|jurisdiction|laws of|courts of)\b[^.!?]{5,150}[.!?]',
    re.IGNORECASE
)

DEFINED_TERM_PATTERN = re.compile(r'"([A-Z][a-zA-Z\s]{2,40})"')

PRIORITY_CLAUSES = [
    "Cap On Liability", "License Grant", "Termination For Convenience",
    "Anti-Assignment", "Confidentiality", "Ip Ownership Assignment",
    "Audit Rights", "Non-Compete", "Exclusivity", "Warranty Duration",
    "Insurance", "Governing Law",
]


# ── LLM client ────────────────────────────────────────────────────────────────
def _get_llm():
    try:
        from llm.llm_client import get_llm_client
        fn, source = get_llm_client()
        return fn, source
    except Exception:
        return None, "none"


# ── Summarization ──────────────────────────────────────────────────────────────
def _summarize_clause(clause_name, clause_texts, llm_fn, max_input=600):
    """
    Summarize a single clause group.
    Sends only this clause's text to the LLM (max 600 chars).
    Falls back to first 2 sentences if no LLM.
    """
    if not clause_texts:
        return None

    combined = " ".join(clause_texts)[:max_input]

    if llm_fn is not None:
        prompt = (
            f"Summarize this '{clause_name}' contract clause in 2-3 clear sentences. "
            f"Be specific about what it says. Plain English only. No legal advice.\n\n"
            f"CLAUSE TEXT:\n{combined}\n\nSUMMARY:"
        )
        try:
            result = llm_fn(prompt)
            return result.strip() if result else None
        except Exception:
            pass

    # Extractive fallback
    sentences = re.split(r'(?<=[.!?])\s+', combined)
    return " ".join(sentences[:2]).strip() or combined[:200]


# ── Template field extractors ──────────────────────────────────────────────────
def _clean(text):
    return re.sub(r'\s+', ' ', text).strip()


def _deduplicate(items, max_items=5):
    seen, result = set(), []
    for item in items:
        key = item[:60].lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
        if len(result) >= max_items:
            break
    return result


def _extract_parties(full_text):
    matches = PARTY_PATTERN.findall(full_text)
    return _deduplicate([m.strip() for m in matches if len(m.strip()) > 3], max_items=6)


def _extract_dates(full_text):
    return list(dict.fromkeys(DATE_PATTERN.findall(full_text)))[:8]


def _extract_defined_terms(full_text):
    terms = DEFINED_TERM_PATTERN.findall(full_text)
    return _deduplicate([t.strip() for t in terms if len(t) > 2], max_items=10)


def _extract_pattern_sentences(full_text, pattern, max_items=3):
    matches = pattern.findall(full_text)
    return _deduplicate([_clean(m) for m in matches if len(m.strip()) > 20], max_items=max_items)


def _extract_governing_law(full_text):
    matches = GOVERNING_LAW_PATTERN.findall(full_text)
    return _clean(matches[0]) if matches else None


# ── Main summarizer ────────────────────────────────────────────────────────────
def summarize_contract(spans, clause_df, embedder, contract_summary):
    """
    Generate a structured contract summary.

    Args:
        spans: list of text spans from the document
        clause_df: DataFrame with clause classifications
        embedder: SentenceTransformer (kept for API compatibility, unused here)
        contract_summary: dict from build_contract_summary()

    Returns:
        dict with structured summary fields
    """
    full_text = "\n\n".join(spans)
    llm_fn, llm_source = _get_llm()

    # Template field extraction
    parties = _extract_parties(full_text)
    all_dates = _extract_dates(full_text)
    effective_date = all_dates[0] if all_dates else None
    expiry_date = all_dates[-1] if len(all_dates) > 1 else None
    key_terms = _extract_defined_terms(full_text)
    obligations = _extract_pattern_sentences(full_text, OBLIGATION_PATTERN, max_items=4)
    rights = _extract_pattern_sentences(full_text, RIGHTS_PATTERN, max_items=3)
    payment_terms = _extract_pattern_sentences(full_text, PAYMENT_PATTERN, max_items=3)
    termination = _extract_pattern_sentences(full_text, TERMINATION_PATTERN, max_items=3)
    governing_law = _extract_governing_law(full_text)

    # Group spans by clause type
    clause_groups = defaultdict(list)
    for _, row in clause_df.iterrows():
        clause = row["final_clause"]
        if clause != "Unknown":
            sid = int(row["span_id"])
            if sid < len(spans):
                clause_groups[clause].append(spans[sid])

    # Summarize each clause group
    ordered_clauses = [c for c in PRIORITY_CLAUSES if c in clause_groups]
    ordered_clauses += [c for c in clause_groups if c not in PRIORITY_CLAUSES]

    clause_summaries = {}
    for clause in ordered_clauses:
        s = _summarize_clause(clause, clause_groups[clause], llm_fn)
        if s:
            clause_summaries[clause] = s

    # Overall executive summary
    top = [clause_summaries[c] for c in ordered_clauses[:5] if c in clause_summaries]
    if top and llm_fn is not None:
        combined = " ".join(top)[:800]
        try:
            overall_summary = llm_fn(
                "Write a 3-4 sentence executive summary of this contract based on the "
                "clause summaries below. Be neutral and specific. No legal advice.\n\n"
                f"CLAUSE SUMMARIES:\n{combined}\n\nEXECUTIVE SUMMARY:"
            ).strip()
        except Exception:
            overall_summary = " ".join(top[:2])
    elif top:
        overall_summary = " ".join(top[:2])
    else:
        overall_summary = "No clauses were detected with sufficient confidence to generate a summary."

    risk_flags = [
        {"clause": d["clause"], "severity": d.get("severity", "Medium"), "reasons": d["reasons"]}
        for d in contract_summary["deviations"]
    ]

    model_desc = (
        f"Groq (llama-3.1-8b-instant) + regex extraction"
        if llm_source == "groq"
        else f"Ollama (local) + regex extraction"
        if llm_source == "ollama"
        else "Regex extraction only (no LLM connected — add GROQ_API_KEY for full summaries)"
    )

    return {
        "model": model_desc,
        "overall_summary": overall_summary,
        "clause_summaries": clause_summaries,
        "parties": parties,
        "effective_date": effective_date,
        "expiry_date": expiry_date,
        "key_terms": key_terms,
        "obligations": obligations,
        "rights": rights,
        "payment_terms": payment_terms,
        "termination": termination,
        "governing_law": governing_law,
        "key_clauses": contract_summary["coverage"]["detected_clauses"],
        "risk_flags": risk_flags,
    }


# ── Evaluation metrics ─────────────────────────────────────────────────────────
def evaluate_summary(summary_dict, reference_spans):
    """ROUGE-1, ROUGE-2, coverage, compression ratio."""
    def tokenize(text):
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def rouge_n(hyp_tokens, ref_tokens, n):
        hyp_ng = set(ngrams(hyp_tokens, n))
        ref_ng = set(ngrams(ref_tokens, n))
        if not ref_ng:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        overlap = hyp_ng & ref_ng
        p = len(overlap) / max(len(hyp_ng), 1)
        r = len(overlap) / max(len(ref_ng), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    hyp_parts = list(summary_dict.get("clause_summaries", {}).values())
    if summary_dict.get("overall_summary"):
        hyp_parts.insert(0, summary_dict["overall_summary"])
    hyp_text = " ".join(hyp_parts)
    ref_text = " ".join(reference_spans)

    hyp_tokens = tokenize(hyp_text)
    ref_tokens = tokenize(ref_text)

    return {
        "rouge_1": rouge_n(hyp_tokens, ref_tokens, 1),
        "rouge_2": rouge_n(hyp_tokens, ref_tokens, 2),
        "coverage": round(len(set(hyp_tokens) & set(ref_tokens)) / max(len(set(ref_tokens)), 1), 4),
        "compression_ratio": round(len(hyp_tokens) / max(len(ref_tokens), 1), 4),
        "summary_clauses": len(summary_dict.get("clause_summaries", {})),
        "reference_spans": len(reference_spans),
    }
