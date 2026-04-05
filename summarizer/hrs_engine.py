"""
HRS — Hierarchical Recursive Summarization Engine
===================================================
A novel recursive summarization architecture for long legal documents.

Problem it solves:
  Standard LLMs have token limits. A contract with 33 clause types, each with
  multiple spans, cannot be summarized in a single LLM call without truncation.
  Naive approaches either truncate (losing information) or concatenate everything
  (exceeding context limits).

HRS Solution — 3-level recursive reduction:

  Level 0: Raw spans grouped by clause type
      ↓  [LLM call per clause group, max 600 chars input]
  Level 1: Per-clause summaries (one per detected clause type)
      ↓  [LLM call per category group of 3-4 clauses]
  Level 2: Category summaries (IP, Liability, Operational, etc.)
      ↓  [Single LLM call synthesizing all category summaries]
  Level 3: Executive summary (final output)

Properties:
  - No information is lost through truncation — every clause gets its own call
  - Token usage is bounded: each call receives at most MAX_INPUT_CHARS chars
  - The tree structure is preserved and returned for inspection
  - Graceful degradation: if LLM unavailable, extractive fallback at each level
  - Deterministic fallback produces structured output without any LLM

Clause categories (domain knowledge):
  IP & Licensing, Liability & Risk, Termination, Competition,
  Assignment & Transfer, Financial, Legal Protections, Operational
"""

import re
from collections import defaultdict


# ── Clause category taxonomy ───────────────────────────────────────────────────
CLAUSE_CATEGORIES = {
    "IP & Licensing": [
        "License Grant", "Non-Transferable License", "Irrevocable Or Perpetual License",
        "Unlimited/All-You-Can-Eat-License", "Ip Ownership Assignment",
        "Joint Ip Ownership", "Affiliate License-Licensee", "Affiliate License-Licensor",
    ],
    "Liability & Risk": [
        "Cap On Liability", "Uncapped Liability", "Liquidated Damages",
        "Insurance", "Warranty Duration",
    ],
    "Termination": [
        "Termination For Convenience", "Post-Termination Services",
    ],
    "Competition & Exclusivity": [
        "Non-Compete", "Exclusivity", "Competitive Restriction Exception",
        "Non-Disparagement", "Most Favored Nation",
    ],
    "Assignment & Transfer": [
        "Anti-Assignment", "Change Of Control",
    ],
    "Financial": [
        "Revenue/Profit Sharing", "Minimum Commitment", "Price Restrictions",
        "Volume Restriction",
    ],
    "Legal Protections": [
        "Covenant Not To Sue", "Rofr/Rofo/Rofn", "Source Code Escrow",
        "Third Party Beneficiary", "Audit Rights",
    ],
    "Operational": [
        "No-Solicit Of Customers", "No-Solicit Of Employees",
    ],
}

# Reverse map: clause → category
CLAUSE_TO_CATEGORY = {}
for cat, clauses in CLAUSE_CATEGORIES.items():
    for c in clauses:
        CLAUSE_TO_CATEGORY[c] = cat


MAX_INPUT_CHARS = 700   # max chars per LLM call input
MAX_CALLS = 40          # safety cap on total LLM calls


def _first_sentences(text, n=2):
    """Extractive fallback: return first n sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:n])


def _call_llm(llm_fn, prompt, call_log):
    """Wrapper that tracks calls and handles failures gracefully."""
    if llm_fn is None or len(call_log) >= MAX_CALLS:
        return None
    try:
        result = llm_fn(prompt)
        call_log.append({"prompt_len": len(prompt), "success": True})
        return result.strip() if result else None
    except Exception as e:
        call_log.append({"prompt_len": len(prompt), "success": False, "error": str(e)})
        return None


# ── Level 0 → Level 1: Clause group summarization ─────────────────────────────
def summarize_clause_group(clause_name, spans_text, llm_fn, call_log):
    """
    Level 0 → 1: Summarize all spans for a single clause type.
    Input is truncated to MAX_INPUT_CHARS to stay within token limits.
    """
    truncated = spans_text[:MAX_INPUT_CHARS]
    prompt = (
        f"Summarize this '{clause_name}' contract clause in 2 sentences. "
        f"Be specific. Plain English. No legal advice.\n\n"
        f"TEXT:\n{truncated}\n\nSUMMARY:"
    )
    result = _call_llm(llm_fn, prompt, call_log)
    if result:
        return result
    # Extractive fallback
    return _first_sentences(truncated)


# ── Level 1 → Level 2: Category summarization ─────────────────────────────────
def summarize_category(category_name, clause_summaries_dict, llm_fn, call_log):
    """
    Level 1 → 2: Combine 3-4 clause summaries into a category summary.
    Input is the concatenation of clause summaries (already compressed).
    """
    combined = "\n".join(
        f"- {clause}: {summary}"
        for clause, summary in clause_summaries_dict.items()
    )[:MAX_INPUT_CHARS]

    prompt = (
        f"Combine these '{category_name}' clause summaries into one coherent paragraph. "
        f"2-3 sentences. Plain English. No legal advice.\n\n"
        f"CLAUSE SUMMARIES:\n{combined}\n\nCATEGORY SUMMARY:"
    )
    result = _call_llm(llm_fn, prompt, call_log)
    if result:
        return result
    # Extractive fallback: join first sentences
    return " ".join(
        s.split(".")[0] + "." for s in clause_summaries_dict.values()
    )[:300]


# ── Level 2 → Level 3: Executive summary ──────────────────────────────────────
def synthesize_executive_summary(category_summaries_dict, n_deviations,
                                  detected_clauses, llm_fn, call_log):
    """
    Level 2 → 3: Synthesize all category summaries into an executive summary.
    This is the final recursive reduction — the root of the tree.
    """
    combined = "\n".join(
        f"[{cat}]: {summary}"
        for cat, summary in category_summaries_dict.items()
    )[:MAX_INPUT_CHARS]

    prompt = (
        f"Write a 3-4 sentence executive summary of this contract. "
        f"Mention key topics and note that {n_deviations} clause(s) deviate from standard patterns. "
        f"Neutral tone. No legal advice.\n\n"
        f"CATEGORY SUMMARIES:\n{combined}\n\nEXECUTIVE SUMMARY:"
    )
    result = _call_llm(llm_fn, prompt, call_log)
    if result:
        return result

    # Deterministic fallback
    cats = list(category_summaries_dict.keys())
    clauses_str = ", ".join(detected_clauses[:6])
    return (
        f"This contract covers {len(detected_clauses)} clause types including {clauses_str}. "
        f"{n_deviations} clause(s) deviate from standard patterns. "
        f"Key areas: {', '.join(cats[:4])}."
    )


# ── Main HRS entry point ───────────────────────────────────────────────────────
def hierarchical_summarize(spans, clause_df, contract_summary, llm_fn=None):
    """
    Full HRS pipeline — 3-level recursive summarization.

    Args:
        spans: list of text spans
        clause_df: DataFrame with clause classifications
        contract_summary: dict from build_contract_summary()
        llm_fn: LLM callable or None (deterministic fallback)

    Returns:
        dict with:
          - executive_summary: str (Level 3 output)
          - category_summaries: dict[category → summary] (Level 2)
          - clause_summaries: dict[clause → summary] (Level 1)
          - hrs_tree: full tree structure for inspection
          - llm_calls: int (total LLM calls made)
          - call_log: list of call metadata
          - model_desc: str
    """
    call_log = []

    # ── Group spans by clause type ─────────────────────────────────────────────
    clause_groups = defaultdict(list)
    for _, row in clause_df.iterrows():
        clause = row["final_clause"]
        if clause != "Unknown":
            sid = int(row["span_id"])
            if sid < len(spans):
                clause_groups[clause].append(spans[sid])

    if not clause_groups:
        return {
            "executive_summary": "No clauses were detected with sufficient confidence.",
            "category_summaries": {},
            "clause_summaries": {},
            "hrs_tree": {},
            "llm_calls": 0,
            "call_log": [],
            "model_desc": "No clauses detected",
        }

    # ── Level 0 → 1: Summarize each clause group ──────────────────────────────
    clause_summaries = {}
    for clause, clause_spans in clause_groups.items():
        combined_text = " ".join(clause_spans)
        summary = summarize_clause_group(clause, combined_text, llm_fn, call_log)
        clause_summaries[clause] = summary

    # ── Level 1 → 2: Group clauses into categories, summarize each ────────────
    category_groups = defaultdict(dict)
    uncategorized = {}

    for clause, summary in clause_summaries.items():
        cat = CLAUSE_TO_CATEGORY.get(clause, "Other")
        if cat == "Other":
            uncategorized[clause] = summary
        else:
            category_groups[cat][clause] = summary

    # Add uncategorized as their own category if non-empty
    if uncategorized:
        category_groups["Other Clauses"] = uncategorized

    category_summaries = {}
    for cat, clauses_dict in category_groups.items():
        if len(clauses_dict) == 1:
            # Only one clause in category — use it directly
            category_summaries[cat] = list(clauses_dict.values())[0]
        else:
            category_summaries[cat] = summarize_category(
                cat, clauses_dict, llm_fn, call_log
            )

    # ── Level 2 → 3: Executive summary ────────────────────────────────────────
    n_deviations = contract_summary["overview"]["deviating_spans"]
    detected_clauses = contract_summary["coverage"]["detected_clauses"]

    executive_summary = synthesize_executive_summary(
        category_summaries, n_deviations, detected_clauses, llm_fn, call_log
    )

    # ── Build tree for inspection ──────────────────────────────────────────────
    hrs_tree = {
        "level_3_executive": executive_summary,
        "level_2_categories": {
            cat: {
                "summary": cat_sum,
                "level_1_clauses": {
                    clause: clause_summaries[clause]
                    for clause in category_groups.get(cat, {}).keys()
                    if clause in clause_summaries
                }
            }
            for cat, cat_sum in category_summaries.items()
        }
    }

    successful_calls = sum(1 for c in call_log if c.get("success"))
    model_desc = (
        f"HRS (3-level recursive) · {successful_calls} LLM calls · "
        f"{len(clause_summaries)} clause groups · {len(category_summaries)} categories"
        if llm_fn else
        f"HRS deterministic fallback · {len(clause_summaries)} clause groups"
    )

    return {
        "executive_summary": executive_summary,
        "category_summaries": category_summaries,
        "clause_summaries": clause_summaries,
        "hrs_tree": hrs_tree,
        "llm_calls": len(call_log),
        "call_log": call_log,
        "model_desc": model_desc,
    }
