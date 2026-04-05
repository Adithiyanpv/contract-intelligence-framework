"""
Contract Summarizer Module
==========================
Uses extractive summarization (no fine-tuning needed) via:
  - KeyBERT for keyword/keyphrase extraction
  - sentence-transformers for semantic sentence ranking
  - Template-based structured output (dates, parties, terms, obligations)
  - ROUGE + BLEU evaluation metrics

PRIVACY: Raw document text never leaves this module.
         Only structured summary metadata is passed to any LLM.
"""

import re
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


# ── Template fields ────────────────────────────────────────────────────────────
SUMMARY_TEMPLATE = {
    "parties":        [],   # organisations / parties involved
    "effective_date": None, # contract start date
    "expiry_date":    None, # contract end / expiry date
    "key_terms":      [],   # important defined terms
    "obligations":    [],   # key obligations (shall / must)
    "rights":         [],   # key rights (may / entitled)
    "payment_terms":  [],   # financial terms
    "termination":    [],   # termination conditions
    "governing_law":  None, # jurisdiction
    "key_clauses":    [],   # top clause types detected
    "risk_flags":     [],   # deviating clauses summary
    "extractive_summary": [],  # top-ranked sentences
}

# ── Regex patterns ─────────────────────────────────────────────────────────────
DATE_PATTERN = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\s+\d{1,2},?\s+\d{4}\b'
    r'|\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b'
    r'|\b\d{4}[\/\-]\d{2}[\/\-]\d{2}\b',
    re.IGNORECASE
)

PARTY_PATTERN = re.compile(
    r'\b(?:(?:[A-Z][a-z]+\s+){1,4}(?:Inc|LLC|Ltd|Corp|Co|Company|Group|Holdings|Technologies|Solutions|Services|Pvt|Limited)\.?)\b'
    r'|(?:"([A-Z][a-zA-Z\s]+)")',
)

OBLIGATION_PATTERN = re.compile(
    r'[^.]*\b(?:shall|must|is required to|agrees to|will|undertakes to)\b[^.]{10,150}\.', re.IGNORECASE
)

RIGHTS_PATTERN = re.compile(
    r'[^.]*\b(?:may|is entitled to|has the right to|reserves the right|at its discretion)\b[^.]{10,150}\.', re.IGNORECASE
)

PAYMENT_PATTERN = re.compile(
    r'[^.]*\b(?:payment|fee|fees|compensation|royalt|revenue|invoice|pay|amount|price|cost)\b[^.]{5,150}\.', re.IGNORECASE
)

TERMINATION_PATTERN = re.compile(
    r'[^.]*\b(?:terminat|cancel|expir|end of term|notice period|upon termination)\b[^.]{5,150}\.', re.IGNORECASE
)

GOVERNING_LAW_PATTERN = re.compile(
    r'[^.]*\b(?:governed by|governing law|jurisdiction|laws of|courts of)\b[^.]{5,100}\.', re.IGNORECASE
)

DEFINED_TERM_PATTERN = re.compile(r'"([A-Z][a-zA-Z\s]{2,40})"')


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


# ── Extractive sentence ranking ────────────────────────────────────────────────
def _rank_sentences(spans, embedder, top_k=8):
    """
    Rank spans by centrality (similarity to document centroid).
    Returns top_k most representative spans.
    """
    if not spans or embedder is None:
        return spans[:top_k]

    norm_spans = [s[:512] for s in spans]  # truncate for embedding
    embeddings = embedder.encode(norm_spans, batch_size=16, show_progress_bar=False)
    centroid = embeddings.mean(axis=0, keepdims=True)
    scores = cosine_similarity(embeddings, centroid).flatten()

    # Boost spans that contain key signals
    boost_terms = ["shall", "must", "agree", "terminat", "payment", "licens", "confidential", "govern"]
    for i, span in enumerate(spans):
        sl = span.lower()
        boost = sum(0.05 for t in boost_terms if t in sl)
        scores[i] += boost

    top_idx = np.argsort(scores)[::-1][:top_k]
    top_idx_sorted = sorted(top_idx)  # preserve document order
    return [_clean(spans[i]) for i in top_idx_sorted]


# ── Field extractors ───────────────────────────────────────────────────────────
def _extract_parties(full_text):
    matches = PARTY_PATTERN.findall(full_text)
    parties = []
    for m in matches:
        name = m if isinstance(m, str) else m
        name = name.strip().strip('"')
        if len(name) > 3:
            parties.append(name)
    return _deduplicate(parties, max_items=6)


def _extract_dates(full_text):
    return list(dict.fromkeys(DATE_PATTERN.findall(full_text)))[:8]


def _extract_defined_terms(full_text):
    terms = DEFINED_TERM_PATTERN.findall(full_text)
    return _deduplicate([t.strip() for t in terms if len(t) > 2], max_items=10)


def _extract_pattern_sentences(full_text, pattern, max_items=4):
    matches = pattern.findall(full_text)
    return _deduplicate([_clean(m) for m in matches if len(m.strip()) > 20], max_items=max_items)


def _extract_governing_law(full_text):
    matches = GOVERNING_LAW_PATTERN.findall(full_text)
    if matches:
        return _clean(matches[0])
    return None


# ── Main summarizer ────────────────────────────────────────────────────────────
def summarize_contract(spans, clause_df, embedder, contract_summary):
    """
    Generate a structured contract summary using extractive methods only.
    No raw text is passed to any external API.

    Args:
        spans: list of text spans from the document
        clause_df: DataFrame with clause classifications
        embedder: SentenceTransformer instance
        contract_summary: dict from build_contract_summary()

    Returns:
        dict following SUMMARY_TEMPLATE
    """
    full_text = "\n\n".join(spans)
    result = {k: (v.copy() if isinstance(v, list) else v) for k, v in SUMMARY_TEMPLATE.items()}

    # 1. Parties
    result["parties"] = _extract_parties(full_text)

    # 2. Dates
    all_dates = _extract_dates(full_text)
    result["effective_date"] = all_dates[0] if all_dates else None
    result["expiry_date"] = all_dates[-1] if len(all_dates) > 1 else None

    # 3. Defined terms
    result["key_terms"] = _extract_defined_terms(full_text)

    # 4. Obligations
    result["obligations"] = _extract_pattern_sentences(full_text, OBLIGATION_PATTERN, max_items=5)

    # 5. Rights
    result["rights"] = _extract_pattern_sentences(full_text, RIGHTS_PATTERN, max_items=4)

    # 6. Payment terms
    result["payment_terms"] = _extract_pattern_sentences(full_text, PAYMENT_PATTERN, max_items=3)

    # 7. Termination
    result["termination"] = _extract_pattern_sentences(full_text, TERMINATION_PATTERN, max_items=3)

    # 8. Governing law
    result["governing_law"] = _extract_governing_law(full_text)

    # 9. Key clauses from ML classification
    result["key_clauses"] = contract_summary["coverage"]["detected_clauses"]

    # 10. Risk flags from deviation analysis
    result["risk_flags"] = [
        {
            "clause": d["clause"],
            "severity": d.get("severity", "Medium"),
            "reasons": d["reasons"]
        }
        for d in contract_summary["deviations"]
    ]

    # 11. Extractive summary — top representative spans
    result["extractive_summary"] = _rank_sentences(spans, embedder, top_k=6)

    return result


# ── Evaluation metrics ─────────────────────────────────────────────────────────
def evaluate_summary(summary_sentences, reference_sentences):
    """
    Evaluate summary quality using:
    - ROUGE-1 (unigram recall/precision/F1)
    - ROUGE-2 (bigram recall/precision/F1)
    - Coverage score (fraction of key terms retained)
    - Compression ratio

    Args:
        summary_sentences: list of extracted summary sentences
        reference_sentences: list of all document sentences (reference)

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

    hyp_text = " ".join(summary_sentences)
    ref_text = " ".join(reference_sentences)
    hyp_tokens = tokenize(hyp_text)
    ref_tokens = tokenize(ref_text)

    r1 = rouge_n(hyp_tokens, ref_tokens, 1)
    r2 = rouge_n(hyp_tokens, ref_tokens, 2)

    # Coverage: fraction of reference unigrams present in summary
    ref_vocab = set(ref_tokens)
    hyp_vocab = set(hyp_tokens)
    coverage = len(hyp_vocab & ref_vocab) / max(len(ref_vocab), 1)

    # Compression ratio
    compression = len(hyp_tokens) / max(len(ref_tokens), 1)

    # Density: average length of extractive fragments
    density = len(hyp_tokens) / max(len(summary_sentences), 1)

    return {
        "rouge_1": r1,
        "rouge_2": r2,
        "coverage": round(coverage, 4),
        "compression_ratio": round(compression, 4),
        "avg_sentence_length": round(density, 1),
        "summary_sentences": len(summary_sentences),
        "reference_sentences": len(reference_sentences),
    }
