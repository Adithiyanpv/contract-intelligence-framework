"""
Multi-Document Aggregation Module
==================================
Aggregates analysis results from multiple contracts into:
  - Cross-document deviation heatmap (clause × document)
  - Systemic vs isolated risk classification
  - Per-document summary cards
  - Aggregate deviation rate per clause type
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def aggregate_documents(doc_results):
    """
    Aggregate analysis results from multiple documents.

    Args:
        doc_results: list of dicts, each with keys:
            - name: str (filename)
            - clause_df: pd.DataFrame
            - spans: list[str]
            - summary: dict (from build_contract_summary)

    Returns:
        dict with aggregated metrics
    """
    if not doc_results:
        return {}

    n_docs = len(doc_results)
    doc_names = [d["name"] for d in doc_results]

    # ── Per-document overview ──────────────────────────────────────────────────
    doc_overviews = []
    for d in doc_results:
        ov = d["summary"]["overview"]
        doc_overviews.append({
            "document": d["name"],
            "total_spans": ov["total_spans"],
            "recognized": ov["recognized_clauses"],
            "unknown": ov["unknown_spans"],
            "deviating": ov["deviating_spans"],
            "deviation_rate": round(ov["deviating_spans"] / max(ov["recognized_clauses"], 1), 3),
        })

    # ── Clause × Document deviation heatmap ───────────────────────────────────
    # Collect all clause types across all docs
    all_clauses = set()
    for d in doc_results:
        all_clauses.update(
            d["clause_df"][d["clause_df"]["final_clause"] != "Unknown"]["final_clause"].unique()
        )
    all_clauses = sorted(all_clauses)

    # Build heatmap: rows=clauses, cols=documents
    # Value: 0=not present, 1=present no deviation, 2=deviating
    heatmap = {}
    for clause in all_clauses:
        heatmap[clause] = {}
        for d in doc_results:
            df = d["clause_df"]
            clause_rows = df[df["final_clause"] == clause]
            if clause_rows.empty:
                heatmap[clause][d["name"]] = "absent"
            elif clause_rows["final_deviation"].any():
                heatmap[clause][d["name"]] = "deviating"
            else:
                heatmap[clause][d["name"]] = "present"

    # ── Systemic vs isolated risk ──────────────────────────────────────────────
    systemic = []   # deviating in ALL docs that contain it
    isolated = []   # deviating in only 1 doc
    common   = []   # present in all docs, no deviation

    for clause in all_clauses:
        present_docs = [d for d in doc_names if heatmap[clause][d] != "absent"]
        deviating_docs = [d for d in doc_names if heatmap[clause][d] == "deviating"]

        if not present_docs:
            continue

        dev_rate = len(deviating_docs) / len(present_docs)

        if len(deviating_docs) == len(present_docs) and len(present_docs) > 1:
            systemic.append({"clause": clause, "affected_docs": deviating_docs, "rate": 1.0})
        elif len(deviating_docs) == 1:
            isolated.append({"clause": clause, "affected_docs": deviating_docs, "rate": dev_rate})
        elif len(deviating_docs) == 0 and len(present_docs) == n_docs:
            common.append(clause)

    # Sort systemic by number of affected docs
    systemic.sort(key=lambda x: len(x["affected_docs"]), reverse=True)

    # ── Deviation rate per clause type (across all docs) ──────────────────────
    clause_deviation_rates = {}
    for clause in all_clauses:
        present = sum(1 for d in doc_names if heatmap[clause][d] != "absent")
        deviating = sum(1 for d in doc_names if heatmap[clause][d] == "deviating")
        if present > 0:
            clause_deviation_rates[clause] = {
                "present_in": present,
                "deviating_in": deviating,
                "rate": round(deviating / present, 3),
            }

    # ── Highest risk clauses overall ──────────────────────────────────────────
    risk_ranking = sorted(
        [(c, v["rate"], v["deviating_in"]) for c, v in clause_deviation_rates.items() if v["deviating_in"] > 0],
        key=lambda x: (x[1], x[2]),
        reverse=True
    )

    # ── Cross-doc deviation reasons aggregation ───────────────────────────────
    reason_counts = defaultdict(int)
    for d in doc_results:
        for _, row in d["clause_df"][d["clause_df"]["final_deviation"]].iterrows():
            for r in row.get("deviation_reasons", []):
                reason_counts[r] += 1

    top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "n_docs": n_docs,
        "doc_names": doc_names,
        "doc_overviews": doc_overviews,
        "all_clauses": all_clauses,
        "heatmap": heatmap,
        "systemic_risks": systemic,
        "isolated_risks": isolated,
        "common_clauses": common,
        "clause_deviation_rates": clause_deviation_rates,
        "risk_ranking": risk_ranking,
        "top_reasons": top_reasons,
    }


def build_heatmap_dataframe(agg_result):
    """Convert heatmap dict to a styled DataFrame for display."""
    heatmap = agg_result["heatmap"]
    doc_names = agg_result["doc_names"]
    clauses = agg_result["all_clauses"]

    # Value mapping for display
    symbol = {"deviating": "⚠️", "present": "✅", "absent": "—"}

    rows = []
    for clause in clauses:
        row = {"Clause": clause}
        for doc in doc_names:
            row[doc] = symbol.get(heatmap[clause][doc], "—")
        rows.append(row)

    return pd.DataFrame(rows).set_index("Clause")
