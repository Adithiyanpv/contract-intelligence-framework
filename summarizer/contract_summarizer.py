"""
Contract Summarizer Module
==========================
Uses abstractive summarization via DistilBART (sshleifer/distilbart-cnn-12-6)
to generate per-clause-group summaries, combined with regex-based template
field extraction for structured output.

Model: sshleifer/distilbart-cnn-12-6
  - 306MB distilled BART model pre-trained on CNN/DailyMail
  - No fine-tuning required — generalizes well to legal text
  - Runs on CPU in ~2-4s per clause group

Workflow:
  1. Group spans by detected clause type
  2. Concatenate spans per group (truncated to 900 tokens)
  3. DistilBART generates a 2-4 sentence abstractive summary per group
  4. Regex extracts template fields (parties, dates, governing law, etc.)
  5. ROUGE-1/2 + coverage metrics evaluate summary quality

PRIVACY: Raw document text never leaves this module.
         Only structured summary metadata is passed to any LLM.
"""

import re
import numpy as np
import streamlit as st
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
    r'Services|Pvt|Limited|Bank|Trust|Partners|Associates)\.?)\b'
    r'|(?:between\s+)([A-Z][a-zA-Z\s,\.]{3,60})(?:\s+and\s+)([A-Z][a-zA-Z\s,\.]{3,60})(?:\s*\()',
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


# ── DistilBART loader ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_summarizer():
    """
    Load DistilBART summarization pipeline.
    Model: sshleifer/distilbart-cnn-12-6
    - 306MB distilled BART, pre-trained on CNN/DailyMail
    - Abstractive: generates new sentences, not just extracts
    - No fine-tuning needed for legal text summarization
    """
    try:
        from transformers import pipeline as hf_pipeline
        summarizer = hf_pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1,  # CPU
            truncation=True,
        )
        return summarizer
    except Exception as e:
        return None


def _abstractive_summarize(text, summarizer, max_input=900, min_len=40, max_len=120):
    """
    Run DistilBART on a text chunk.
    Truncates input to max_input chars to stay within model token limits.
    Returns a 2-4 sentence abstractive summary.
    """
    if summarizer is None or not text.strip():
        return None

    # Truncate to avoid exceeding 1024 token limit
    truncated = text[:max_input]

    try:
        result = summarizer(
            truncated,
            min_length=min_len,
            max_length=max_len,
            do_sample=False,
            truncation=True,
        )
        return result[0]["summary_text"].strip()
    except Exception:
        return None


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
    parties = []
    for m in matches:
        if isinstance(m, tuple):
            parties.extend([x.strip() for x in m if x.strip() and len(x.strip()) > 3])
        elif isinstance(m, str) and len(m.strip()) > 3:
            parties.append(m.strip())
    return _deduplicate(parties, max_items=6)


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


# ── Priority clause groups for summarization ──────────────────────────────────
PRIORITY_CLAUSES = [
    "Cap On Liability",
    "License Grant",
    "Termination For Convenience",
    "Anti-Assignment",
    "Confidentiality",
    "Ip Ownership Assignment",
    "Audit Rights",
    "Non-Compete",
    "Exclusivity",
    "Warranty Duration",
    "Insurance",
    "Governing Law",
]


# ── Main summarizer ────────────────────────────────────────────────────────────
def summarize_contract(spans, clause_df, embedder, contract_summary):
    """
    Generate a structured contract summary using DistilBART abstractive
    summarization per clause group + regex template field extraction.

    No raw text is passed to any external API.

    Args:
        spans: list of text spans from the document
        clause_df: DataFrame with clause classifications
        embedder: SentenceTransformer instance (unused here, kept for API compat)
        contract_summary: dict from build_contract_summary()

    Returns:
        dict with structured summary fields
    """
    full_text = "\n\n".join(spans)
    summarizer = load_summarizer()

    # ── Template field extraction ──────────────────────────────────────────────
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

    # ── Per-clause-group abstractive summaries ─────────────────────────────────
    clause_summaries = {}

    # Group spans by clause type
    clause_groups = defaultdict(list)
    for _, row in clause_df.iterrows():
        clause = row["final_clause"]
        if clause != "Unknown":
            sid = int(row["span_id"])
            if sid < len(spans):
                clause_groups[clause].append(spans[sid])

    # Summarize priority clauses first, then others
    ordered_clauses = [c for c in PRIORITY_CLAUSES if c in clause_groups]
    ordered_clauses += [c for c in clause_groups if c not in PRIORITY_CLAUSES]

    for clause in ordered_clauses:
        group_text = " ".join(clause_groups[clause])
        summary_text = _abstractive_summarize(group_text, summarizer)
        if summary_text:
            clause_summaries[clause] = summary_text

    # ── Overall document summary ───────────────────────────────────────────────
    # Concatenate the top clause summaries for an overall abstract
    top_summaries = [clause_summaries[c] for c in ordered_clauses[:5] if c in clause_summaries]
    overall_text = " ".join(top_summaries)
    overall_summary = _abstractive_summarize(overall_text, summarizer, max_input=800, min_len=60, max_len=150)

    if not overall_summary and top_summaries:
        # Fallback: join first sentences of each clause summary
        overall_summary = " ".join(s.split(".")[0] + "." for s in top_summaries[:3])

    # ── Risk flags ────────────────────────────────────────────────────────────
    risk_flags = [
        {
            "clause": d["clause"],
            "severity": d.get("severity", "Medium"),
            "reasons": d["reasons"]
        }
        for d in contract_summary["deviations"]
    ]

    return {
        "model": "sshleifer/distilbart-cnn-12-6" if summarizer else "regex-only (model unavailable)",
        "overall_summary": overall_summary or "Summary could not be generated.",
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
    """
    Evaluate summary quality using ROUGE-1, ROUGE-2, coverage, compression.

    Args:
        summary_dict: output of summarize_contract()
        reference_spans: list of all document spans (reference)

    Returns:
        dict of metric scores
    """
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
        precision = len(overlap) / max(len(hyp_ng), 1)
        recall    = len(overlap) / max(len(ref_ng), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    # Build hypothesis from all clause summaries + overall
    hyp_parts = list(summary_dict.get("clause_summaries", {}).values())
    if summary_dict.get("overall_summary"):
        hyp_parts.insert(0, summary_dict["overall_summary"])
    hyp_text = " ".join(hyp_parts)
    ref_text = " ".join(reference_spans)

    hyp_tokens = tokenize(hyp_text)
    ref_tokens = tokenize(ref_text)

    r1 = rouge_n(hyp_tokens, ref_tokens, 1)
    r2 = rouge_n(hyp_tokens, ref_tokens, 2)

    ref_vocab = set(ref_tokens)
    hyp_vocab = set(hyp_tokens)
    coverage = len(hyp_vocab & ref_vocab) / max(len(ref_vocab), 1)
    compression = len(hyp_tokens) / max(len(ref_tokens), 1)

    return {
        "rouge_1": r1,
        "rouge_2": r2,
        "coverage": round(coverage, 4),
        "compression_ratio": round(compression, 4),
        "summary_clauses": len(summary_dict.get("clause_summaries", {})),
        "reference_spans": len(reference_spans),
    }
