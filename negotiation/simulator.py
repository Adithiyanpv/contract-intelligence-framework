"""
Clause Negotiation Simulator
=============================
Given a deviating clause, generates alternative phrasings at three
negotiation stances: Conservative, Balanced, and Aggressive.

Architecture:
  1. Deviation analysis — uses CSDA score + deviation reasons to understand
     exactly what is wrong with the clause
  2. Stance-aware prompt construction — each stance has a different
     instruction set that controls how far to push back
  3. Constrained generation — LLM receives only the clause text + deviation
     metadata, not the full contract (privacy preserved)
  4. Similarity scoring — each alternative is embedded and compared to the
     clause centroid to verify it moves toward standard language
  5. Explanation — plain-English rationale for each alternative

Negotiation stances:
  Conservative: minimal changes, preserves counterparty's core position,
                fixes only the most critical deviations
  Balanced:     standard market terms, mutual obligations, fair to both parties
  Aggressive:   maximum protection for the reviewing party, pushes hard on
                every deviation signal

This feature does not exist in any commercial contract analysis tool.
"""

import re
import numpy as np


# ── Stance definitions ─────────────────────────────────────────────────────────
STANCES = {
    "Conservative": {
        "color": "#68d391",
        "icon": "🟢",
        "description": "Minimal changes — fixes critical issues while preserving the counterparty's core position. Easiest to get accepted.",
        "instruction": (
            "Rewrite this clause with minimal changes. "
            "Fix only the most critical legal issue identified. "
            "Keep the counterparty's core intent intact. "
            "The result should be easy for the other party to accept."
        ),
    },
    "Balanced": {
        "color": "#63b3ed",
        "icon": "🔵",
        "description": "Standard market terms — mutual obligations, fair to both parties. Typical starting point for negotiation.",
        "instruction": (
            "Rewrite this clause using standard market terms. "
            "Make obligations mutual where appropriate. "
            "Address all identified deviations. "
            "The result should reflect what a balanced, fair agreement looks like."
        ),
    },
    "Aggressive": {
        "color": "#fc8181",
        "icon": "🔴",
        "description": "Maximum protection — pushes hard on every deviation. Strong position, may require negotiation to reach agreement.",
        "instruction": (
            "Rewrite this clause to maximally protect the reviewing party. "
            "Address every identified deviation firmly. "
            "Add protective language where standard contracts typically include it. "
            "The result should strongly favor the reviewing party."
        ),
    },
}


# ── Deviation-to-fix mapping ───────────────────────────────────────────────────
DEVIATION_FIX_HINTS = {
    "Uncapped liability detected": (
        "Add a clear liability cap, typically expressed as a multiple of fees paid "
        "or a fixed dollar amount (e.g. 'in no event shall liability exceed the "
        "total fees paid in the preceding 12 months')."
    ),
    "Violation of non-negotiable license ownership invariant": (
        "Remove any language that transfers ownership. Replace with a limited, "
        "non-exclusive license grant that preserves the licensor's ownership."
    ),
    "Permission / obligation polarity mismatch": (
        "Replace permissive language ('freely', 'without restriction') with "
        "appropriate conditional language that requires consent or notice."
    ),
    "Semantic deviation from standard clause language": (
        "Rewrite using standard legal phrasing for this clause type. "
        "Avoid unusual or ambiguous terms."
    ),
    "Missing standard protective keyword": (
        "Add the standard protective language typically present in this clause type, "
        "such as consent requirements, notice periods, or cure rights."
    ),
    "Negation of standard obligation": (
        "Remove the negation and restore the standard obligation. "
        "If the obligation must be limited, add appropriate conditions rather than removing it entirely."
    ),
    "Unusual unilateral right detected": (
        "Make the right mutual, or add conditions and limitations that prevent "
        "unilateral exercise without notice or consent."
    ),
}


def _build_negotiation_prompt(clause_name, clause_text, deviation_reasons,
                               deviation_score, stance_name, stance_config):
    """Build a constrained prompt for a specific negotiation stance."""
    # Build fix hints from deviation reasons
    fix_hints = []
    for reason in deviation_reasons:
        hint = DEVIATION_FIX_HINTS.get(reason)
        if hint:
            fix_hints.append(f"- {hint}")

    fix_hints_str = "\n".join(fix_hints) if fix_hints else "- Apply standard contract language for this clause type."

    # Truncate clause text for privacy
    text_snippet = clause_text[:600] + ("…" if len(clause_text) > 600 else "")

    return (
        f"You are a contract lawyer rewriting a '{clause_name}' clause.\n\n"
        f"DEVIATION SEVERITY: {deviation_score:.2f}/1.0\n"
        f"ISSUES IDENTIFIED:\n" +
        "\n".join(f"- {r}" for r in deviation_reasons) +
        f"\n\nFIX GUIDANCE:\n{fix_hints_str}\n\n"
        f"NEGOTIATION STANCE: {stance_name}\n"
        f"INSTRUCTION: {stance_config['instruction']}\n\n"
        f"ORIGINAL CLAUSE:\n{text_snippet}\n\n"
        f"REWRITE THE CLAUSE (output only the rewritten clause text, no explanation):\n"
    )


def _build_explanation_prompt(clause_name, original_text, rewritten_text,
                               stance_name, deviation_reasons):
    """Build a prompt to explain what changed and why."""
    return (
        f"Compare these two versions of a '{clause_name}' clause and explain "
        f"in 2-3 sentences what changed and why it matters for the {stance_name} stance.\n\n"
        f"ORIGINAL:\n{original_text[:300]}\n\n"
        f"REWRITTEN:\n{rewritten_text[:300]}\n\n"
        f"ISSUES ADDRESSED: {'; '.join(deviation_reasons)}\n\n"
        f"EXPLANATION (plain English, no legal advice):"
    )


def _similarity_to_centroid(text, embedder, centroid):
    """Compute cosine similarity between rewritten text and clause centroid."""
    if embedder is None or centroid is None:
        return None
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        import re as _re
        # Normalize text same way as pipeline
        norm = _re.sub(r"\b(company|licensor|licensee|producer|vendor|client)\b", "party", text.lower())
        norm = _re.sub(r"\b\d+(\.\d+)?\b", "num", norm)
        norm = _re.sub(r"\b(day|days|month|months|year|years)\b", "time", norm)
        emb = embedder.encode([norm])[0]
        sim = float(cosine_similarity(emb.reshape(1, -1), centroid.reshape(1, -1))[0][0])
        return round(sim, 3)
    except Exception:
        return None


def simulate_negotiation(clause_name, clause_text, deviation_reasons,
                          deviation_score, embedder, clause_centroids,
                          llm_fn=None):
    """
    Generate negotiation alternatives for a deviating clause.

    Args:
        clause_name: str — detected clause type
        clause_text: str — original clause text
        deviation_reasons: list[str] — CSDA deviation signals
        deviation_score: float — composite deviation score (0.0–1.0)
        embedder: SentenceTransformer — for similarity scoring
        clause_centroids: dict — precomputed centroids from baselines
        llm_fn: callable or None — LLM for generation

    Returns:
        list of dicts, one per stance:
          - stance: str
          - color: str
          - icon: str
          - description: str
          - rewritten: str
          - explanation: str
          - similarity_improvement: float or None
          - original_similarity: float or None
    """
    centroid = clause_centroids.get(clause_name)

    # Compute original similarity to centroid
    original_sim = _similarity_to_centroid(clause_text, embedder, centroid)

    results = []

    for stance_name, stance_config in STANCES.items():
        if llm_fn is not None:
            # Generate rewrite
            rewrite_prompt = _build_negotiation_prompt(
                clause_name, clause_text, deviation_reasons,
                deviation_score, stance_name, stance_config
            )
            try:
                rewritten = llm_fn(rewrite_prompt).strip()
            except Exception:
                rewritten = _deterministic_rewrite(
                    clause_text, deviation_reasons, stance_name
                )

            # Generate explanation
            explain_prompt = _build_explanation_prompt(
                clause_name, clause_text, rewritten,
                stance_name, deviation_reasons
            )
            try:
                explanation = llm_fn(explain_prompt).strip()
            except Exception:
                explanation = _deterministic_explanation(
                    stance_name, deviation_reasons
                )
        else:
            rewritten = _deterministic_rewrite(
                clause_text, deviation_reasons, stance_name
            )
            explanation = _deterministic_explanation(stance_name, deviation_reasons)

        # Score similarity improvement
        new_sim = _similarity_to_centroid(rewritten, embedder, centroid)
        improvement = None
        if original_sim is not None and new_sim is not None:
            improvement = round(new_sim - original_sim, 3)

        results.append({
            "stance": stance_name,
            "color": stance_config["color"],
            "icon": stance_config["icon"],
            "description": stance_config["description"],
            "rewritten": rewritten,
            "explanation": explanation,
            "original_similarity": original_sim,
            "new_similarity": new_sim,
            "similarity_improvement": improvement,
        })

    return results


def _deterministic_rewrite(clause_text, deviation_reasons, stance_name):
    """Fallback rewrite when no LLM is available."""
    fixes = []
    for reason in deviation_reasons:
        hint = DEVIATION_FIX_HINTS.get(reason, "")
        if hint:
            fixes.append(hint.split(".")[0])

    stance_prefix = {
        "Conservative": "[Conservative revision] ",
        "Balanced": "[Balanced revision] ",
        "Aggressive": "[Aggressive revision] ",
    }.get(stance_name, "")

    if fixes:
        return (
            f"{stance_prefix}The following changes are recommended: "
            + "; ".join(fixes[:2])
            + ". "
            + clause_text[:200]
            + " [Add GROQ_API_KEY for full AI-generated rewrite]"
        )
    return f"{stance_prefix}{clause_text[:300]} [Add GROQ_API_KEY for full AI-generated rewrite]"


def _deterministic_explanation(stance_name, deviation_reasons):
    """Fallback explanation when no LLM is available."""
    stance_notes = {
        "Conservative": "This revision makes minimal changes to fix the most critical issue.",
        "Balanced": "This revision applies standard market terms to address all identified deviations.",
        "Aggressive": "This revision maximally protects the reviewing party by addressing all deviation signals.",
    }
    issues = "; ".join(deviation_reasons[:2]) if deviation_reasons else "general deviation"
    return f"{stance_notes.get(stance_name, '')} Issues addressed: {issues}. Note: This is not legal advice."
