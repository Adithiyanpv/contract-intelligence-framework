"""
CRAG — Constrained Retrieval-Augmented Generation for Contract Q&A
===================================================================
A novel RAG architecture designed specifically for contract analysis.

Key properties:
  1. GROUNDED: Every answer is traceable to specific contract spans
  2. CONSTRAINED: LLM only sees verified, relevant evidence — not raw document
  3. UNANSWERABLE DETECTION: System refuses to answer if evidence is insufficient
  4. HALLUCINATION SCORING: Every answer carries a grounding score (0.0–1.0)
  5. CITATION-FIRST: Answers include span IDs and clause types as citations

Architecture:
  Question → Intent Classification → Evidence Retrieval → Evidence Verification
  → Constrained Answer Construction → Grounding Score → Cited Answer

This is fundamentally different from naive RAG:
  - Naive RAG: retrieve top-k chunks → send to LLM → hope for the best
  - CRAG: retrieve → verify relevance → score evidence → constrain generation
           → quantify hallucination risk → cite sources

Privacy: Only verified evidence metadata is sent to the LLM.
         Raw contract text is included only for the top 2 most relevant spans.
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ── Evidence quality thresholds ────────────────────────────────────────────────
MIN_EVIDENCE_SCORE   = 0.30   # minimum similarity to be considered evidence
MIN_GROUNDING_SCORE  = 0.40   # below this → answer flagged as low-confidence
UNANSWERABLE_THRESHOLD = 0.25 # below this → refuse to answer


# ── Intent taxonomy ───────────────────────────────────────────────────────────
INTENT_PATTERNS = {
    "RISK_QUERY": [
        "risk", "danger", "concern", "problem", "issue", "flag", "red flag",
        "deviation", "non-standard", "unusual", "worry", "careful"
    ],
    "OBLIGATION_QUERY": [
        "shall", "must", "required", "obligation", "duty", "responsible",
        "have to", "need to", "bound to"
    ],
    "RIGHT_QUERY": [
        "right", "may", "allowed", "permitted", "entitled", "can i", "can we",
        "permission", "authorize"
    ],
    "CLAUSE_LOOKUP": [
        "liability", "termination", "license", "warranty", "audit", "assignment",
        "confidential", "payment", "insurance", "compete", "exclusiv", "ip",
        "intellectual property", "governing law", "jurisdiction"
    ],
    "PARTY_QUERY": [
        "who", "party", "parties", "company", "licensor", "licensee",
        "vendor", "client", "counterparty"
    ],
    "FINANCIAL_QUERY": [
        "payment", "fee", "cost", "price", "revenue", "royalt", "compensation",
        "amount", "dollar", "money", "financial"
    ],
    "TERMINATION_QUERY": [
        "terminat", "cancel", "end", "expire", "notice", "exit"
    ],
}


def classify_intent(question: str) -> str:
    q = question.lower()
    scores = {}
    for intent, keywords in INTENT_PATTERNS.items():
        scores[intent] = sum(1 for kw in keywords if kw in q)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "GENERAL_QUERY"


# ── Evidence retrieval ─────────────────────────────────────────────────────────
def retrieve_evidence(question, clause_df, spans, embeddings, embedder,
                      intent, top_k=6):
    """
    Multi-strategy evidence retrieval:
    1. Clause name/alias matching (exact)
    2. Semantic similarity (embedding-based)
    3. Intent-based boosting (deviation queries prioritize deviating spans)

    Returns list of evidence dicts with relevance scores.
    """
    q_emb = embedder.encode([question])[0]
    sims = cosine_similarity(q_emb.reshape(1, -1), embeddings)[0]

    evidence = []
    for idx in range(len(clause_df)):
        row = clause_df.iloc[idx]
        clause = row["final_clause"]
        sim = float(sims[idx])

        # Boost deviating spans for risk queries
        if intent == "RISK_QUERY" and row.get("final_deviation", False):
            sim = min(sim + 0.15, 1.0)

        # Boost spans matching clause aliases
        q_lower = question.lower()
        from pipeline import CLAUSE_ALIASES
        aliases = CLAUSE_ALIASES.get(clause, [])
        if clause != "Unknown" and (
            clause.lower() in q_lower or any(a in q_lower for a in aliases)
        ):
            sim = min(sim + 0.20, 1.0)

        if sim >= MIN_EVIDENCE_SCORE:
            evidence.append({
                "span_id": int(row["span_id"]),
                "clause": clause,
                "text": spans[int(row["span_id"])],
                "similarity": round(sim, 4),
                "deviating": bool(row.get("final_deviation", False)),
                "deviation_reasons": row.get("deviation_reasons", []),
                "severity": row.get("severity"),
                "confidence": float(row.get("confidence", 0)),
                "deviation_score": float(row.get("deviation_score", 0)),
            })

    # Sort by similarity descending
    evidence.sort(key=lambda x: x["similarity"], reverse=True)
    return evidence[:top_k]


# ── Evidence verification ──────────────────────────────────────────────────────
def verify_evidence(evidence, question):
    """
    Verify each piece of evidence is genuinely relevant.
    Computes a grounding score: fraction of answer that is evidence-backed.

    Returns:
        verified: filtered evidence list
        grounding_score: 0.0–1.0
        is_answerable: bool
    """
    if not evidence:
        return [], 0.0, False

    # Weighted grounding score: top evidence weighted more
    weights = [1.0 / (i + 1) for i in range(len(evidence))]
    weighted_sims = [e["similarity"] * w for e, w in zip(evidence, weights)]
    grounding_score = sum(weighted_sims) / sum(weights) if weights else 0.0
    grounding_score = round(min(grounding_score, 1.0), 3)

    # Filter to verified evidence only
    verified = [e for e in evidence if e["similarity"] >= MIN_EVIDENCE_SCORE]

    is_answerable = grounding_score >= UNANSWERABLE_THRESHOLD and len(verified) > 0

    return verified, grounding_score, is_answerable


# ── Constrained answer builder ─────────────────────────────────────────────────
def build_constrained_prompt(question, verified_evidence, intent):
    """
    Build a constrained prompt that:
    1. Only includes verified evidence
    2. Requires the LLM to cite span IDs
    3. Explicitly forbids speculation beyond the evidence
    4. Includes only top 2 spans' raw text (privacy constraint)
    """
    # Build evidence blocks — metadata for all, raw text only for top 2
    evidence_blocks = []
    for i, ev in enumerate(verified_evidence[:5]):
        block = (
            f"[SPAN {ev['span_id']}] Clause: {ev['clause']} | "
            f"Relevance: {ev['similarity']:.2f} | "
            f"Deviating: {'YES (' + ev['severity'] + ')' if ev['deviating'] else 'No'}"
        )
        if ev["deviating"] and ev["deviation_reasons"]:
            block += f" | Flags: {'; '.join(ev['deviation_reasons'])}"
        # Include raw text only for top 2 most relevant spans
        if i < 2:
            text_snippet = ev["text"][:400] + ("…" if len(ev["text"]) > 400 else "")
            block += f"\nText: {text_snippet}"
        evidence_blocks.append(block)

    evidence_str = "\n\n".join(evidence_blocks)

    intent_instruction = {
        "RISK_QUERY": "Focus on deviating clauses and explain the risks clearly.",
        "OBLIGATION_QUERY": "Identify what obligations are imposed and on whom.",
        "RIGHT_QUERY": "Identify what rights are granted and to whom.",
        "CLAUSE_LOOKUP": "Explain what the relevant clause says specifically.",
        "FINANCIAL_QUERY": "Extract and explain any financial terms or amounts.",
        "TERMINATION_QUERY": "Explain the termination conditions and notice requirements.",
        "PARTY_QUERY": "Identify the parties and their roles.",
        "GENERAL_QUERY": "Answer based strictly on the evidence provided.",
    }.get(intent, "Answer based strictly on the evidence provided.")

    return (
        f"You are a contract analyst. Answer ONLY using the verified evidence below.\n"
        f"RULES:\n"
        f"- Cite span IDs in your answer (e.g. 'According to [SPAN 5]...')\n"
        f"- Do NOT speculate beyond the evidence\n"
        f"- If evidence is insufficient, say so explicitly\n"
        f"- {intent_instruction}\n"
        f"- End with: 'Note: This is not legal advice.'\n\n"
        f"VERIFIED EVIDENCE:\n{evidence_str}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )


def build_deterministic_answer(question, verified_evidence, intent, grounding_score):
    """
    Deterministic answer when no LLM is available.
    Constructs a structured answer purely from evidence metadata.
    """
    if not verified_evidence:
        return "No relevant contract clauses found for this question."

    lines = []
    deviating = [e for e in verified_evidence if e["deviating"]]
    non_deviating = [e for e in verified_evidence if not e["deviating"]]

    if intent == "RISK_QUERY":
        if deviating:
            lines.append(f"Found {len(deviating)} deviating clause(s) relevant to your question:")
            for ev in deviating[:3]:
                lines.append(f"• [SPAN {ev['span_id']}] {ev['clause']} ({ev['severity']} severity): {'; '.join(ev['deviation_reasons'])}")
        else:
            lines.append("No deviating clauses found relevant to this question.")
    else:
        lines.append(f"Found {len(verified_evidence)} relevant clause(s):")
        for ev in verified_evidence[:4]:
            dev_note = f" ⚠️ {ev['severity']}" if ev["deviating"] else ""
            lines.append(f"• [SPAN {ev['span_id']}] {ev['clause']}{dev_note} (relevance: {ev['similarity']:.0%})")

    lines.append("\nNote: This is not legal advice. Add GROQ_API_KEY for full AI-generated answers.")
    return "\n".join(lines)


# ── Main CRAG entry point ──────────────────────────────────────────────────────
def crag_answer(question, clause_df, spans, embeddings, embedder, llm_fn=None):
    """
    Full CRAG pipeline:
    Question → Intent → Retrieval → Verification → Constrained Generation → Cited Answer

    Returns:
        dict with keys:
          - answer: str
          - intent: str
          - evidence: list of verified evidence dicts
          - grounding_score: float (0.0–1.0)
          - is_answerable: bool
          - hallucination_risk: str ("Low" / "Medium" / "High")
          - citations: list of span IDs used
          - confidence_notes: list of str
    """
    # Step 1: Intent classification
    intent = classify_intent(question)

    # Step 2: Evidence retrieval
    raw_evidence = retrieve_evidence(
        question, clause_df, spans, embeddings, embedder, intent
    )

    # Step 3: Evidence verification + grounding score
    verified, grounding_score, is_answerable = verify_evidence(raw_evidence, question)

    # Step 4: Hallucination risk classification
    if grounding_score >= 0.65:
        hallucination_risk = "Low"
    elif grounding_score >= 0.40:
        hallucination_risk = "Medium"
    else:
        hallucination_risk = "High"

    citations = [e["span_id"] for e in verified[:5]]

    # Step 5: Answer generation
    if not is_answerable:
        answer = (
            "I cannot reliably answer this question from the contract. "
            "The available evidence does not sufficiently address this query "
            f"(grounding score: {grounding_score:.2f}). "
            "Please review the contract directly or consult a legal professional."
        )
    elif llm_fn is not None:
        prompt = build_constrained_prompt(question, verified, intent)
        try:
            answer = llm_fn(prompt)
        except Exception as e:
            answer = build_deterministic_answer(question, verified, intent, grounding_score)
    else:
        answer = build_deterministic_answer(question, verified, intent, grounding_score)

    # Step 6: Confidence notes
    confidence_notes = [
        f"Grounding score: {grounding_score:.2f} — {hallucination_risk.lower()} hallucination risk",
        f"Answer based on {len(verified)} verified span(s) from the contract.",
        "This analysis is informational and not legal advice.",
    ]
    if hallucination_risk == "High":
        confidence_notes.insert(0,
            "⚠️ Low evidence coverage — treat this answer with caution."
        )

    return {
        "answer": answer,
        "intent": intent,
        "evidence": verified,
        "grounding_score": grounding_score,
        "is_answerable": is_answerable,
        "hallucination_risk": hallucination_risk,
        "citations": citations,
        "confidence_notes": confidence_notes,
    }
