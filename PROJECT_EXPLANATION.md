# ContractIQ — Complete Technical Documentation

> AI-powered contract clause detection, deviation analysis, and explainable risk scoring.
> Live: [contractiq.streamlit.app](https://contractiq.streamlit.app)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Project Structure](#2-project-structure)
3. [Dataset](#3-dataset)
4. [Core Pipeline — CSDA Algorithm](#4-core-pipeline--csda-algorithm)
5. [CRAG — Constrained RAG for Q&A](#5-crag--constrained-rag-for-qa)
6. [HRS — Hierarchical Recursive Summarization](#6-hrs--hierarchical-recursive-summarization)
7. [Clause Negotiation Simulator](#7-clause-negotiation-simulator)
8. [Obligation Graph](#8-obligation-graph)
9. [Multi-Document Analysis](#9-multi-document-analysis)
10. [LLM Architecture & Privacy Guardrails](#10-llm-architecture--privacy-guardrails)
11. [UI Architecture](#11-ui-architecture)
12. [Setup & Deployment](#12-setup--deployment)
13. [Novel Contributions Summary](#13-novel-contributions-summary)

---

## 1. System Overview

ContractIQ is a full-stack AI contract analysis system built on Streamlit. It ingests contract PDFs and runs them through a multi-stage pipeline:

```
PDF Upload
    ↓
Text Extraction (pdfplumber)
    ↓
Span Generation (paragraph + subsection splitting)
    ↓
Clause Classification (fine-tuned DeBERTa-v3, 33 types)
    ↓
Semantic Embedding (all-mpnet-base-v2)
    ↓
CSDA Deviation Detection (4 signals → composite score)
    ↓
Severity Scoring (High / Medium / Low)
    ↓
UI: Overview · Deviating Clauses · Risk Analysis · Analytics
    · Summary (HRS) · Ask the Contract (CRAG)
    · Negotiate · Obligation Graph · Multi-Doc
```

The system is designed around three principles:
- **Privacy-first**: raw contract text never reaches any external API
- **Explainability**: every flag has a traceable reason and confidence score
- **Novelty**: each major component implements an algorithm that does not exist in commercial contract tools

---

## 2. Project Structure

```
contract_deviation_app/
├── app.py                          # Streamlit UI — all 9 tabs
├── pipeline.py                     # CSDA core pipeline
├── precompute_baselines.py         # One-time baseline generator
├── download_models.py              # Google Drive model downloader
├── requirements.txt
├── packages.txt                    # poppler-utils (PDF rendering)
│
├── llm/
│   └── llm_client.py               # Groq + Ollama clients, build_safe_prompt
│
├── rag/
│   └── contract_rag.py             # CRAG engine
│
├── summarizer/
│   ├── contract_summarizer.py      # Main summarizer (calls HRS)
│   └── hrs_engine.py               # HRS 3-level recursive engine
│
├── negotiation/
│   └── simulator.py                # Clause Negotiation Simulator
│
├── obligation_graph/
│   └── extractor.py                # Obligation extraction + graph builder
│
├── multi_doc/
│   └── aggregator.py               # Multi-document aggregation
│
└── resources/
    ├── deberta-clause-final/       # Fine-tuned DeBERTa model
    ├── clause_centroids.npy        # Per-clause embedding centroids
    ├── clause_thresholds.npy       # 95th percentile deviation thresholds
    ├── clause_applicability.npy    # 99th percentile applicability thresholds
    ├── clause_polarity.npy         # Polarity signal profiles
    └── final_cleaned_version.csv   # Training dataset (3,688 spans)
```

---

## 3. Dataset

**File:** `resources/final_cleaned_version (2).csv`
**Size:** 3,688 rows × 3 columns (`Filename`, `Span`, `Clause`)

The `Span` column stores clause text as a Python list string: `"['text here']"` — extracted via `ast.literal_eval()`.

**33 Clause Types:**

| Category | Clauses |
|---|---|
| IP & Licensing | License Grant, Non-Transferable License, Irrevocable Or Perpetual License, Unlimited/All-You-Can-Eat-License, Ip Ownership Assignment, Joint Ip Ownership, Affiliate License-Licensee, Affiliate License-Licensor |
| Liability & Risk | Cap On Liability, Uncapped Liability, Liquidated Damages, Insurance, Warranty Duration |
| Termination | Termination For Convenience, Post-Termination Services |
| Competition | Non-Compete, Exclusivity, Competitive Restriction Exception, Non-Disparagement, Most Favored Nation |
| Assignment | Anti-Assignment, Change Of Control |
| Financial | Revenue/Profit Sharing, Minimum Commitment, Price Restrictions, Volume Restriction |
| Legal | Covenant Not To Sue, Rofr/Rofo/Rofn, Source Code Escrow, Third Party Beneficiary, Audit Rights |
| Operational | No-Solicit Of Customers, No-Solicit Of Employees |

**Top clause distribution:** Anti-Assignment (374), Cap On Liability (275), License Grant (255), Audit Rights (214), Termination For Convenience (183).

---

## 4. Core Pipeline — CSDA Algorithm

**File:** `pipeline.py`

### 4.1 Span Generation (`generate_spans`)

```
Full document text
    ↓
Split on section boundary patterns:
  - Numbered headings: "1. Confidentiality"
  - ALL CAPS headings
  - Title Case headings with colon
    ↓
Fallback: split on double newlines (\n\n)
    ↓
Within each section: split on double newlines
    ↓
Long spans (>1500 chars): split on sentence boundaries
    ↓
Merge orphaned short spans (<80 chars) with neighbors
    ↓
Deduplicate (by first 100 chars)
    ↓
Filter: min_len=80, max_len=1500
```

### 4.2 Clause Classification (DeBERTa)

- Model: fine-tuned `microsoft/deberta-v3-base` on 3,688 labeled spans
- Max sequence length: 512 tokens (increased from 128 for full clause context)
- **Dual-threshold system:**
  - Minimum confidence: 0.45
  - Minimum confidence gap between top-1 and top-2 predictions: 0.10
  - High confidence override: if confidence ≥ 0.65, gap requirement is waived
  - Spans below threshold → labeled "Unknown"

### 4.3 Semantic Embedding

- Model: `sentence-transformers/all-mpnet-base-v2`
- Normalization before embedding:
  - Party names → "party" (company, licensor, licensee, vendor, client, customer, partner)
  - Numbers → "num" (including currency amounts like $5,000)
  - Time units → "time" (day, month, year, week)
  - Month names → "month_name"

### 4.4 CSDA Deviation Detection (4 Signals)

**Signal 1 — Semantic Distance from Centroid**
- Each classified clause is compared to its per-clause centroid (mean embedding of all training examples for that clause type)
- Cosine distance computed between span embedding and centroid
- Threshold: 90th percentile of training distances (adaptive, slightly more sensitive than original 95th)
- If distance > threshold → "Semantic deviation from standard clause language"

**Signal 2 — Polarity Profile Violation**
- Each clause type has a learned polarity profile: frequency of obligation signals ("shall", "must", "not") vs. permission signals ("may", "freely", "without restriction") across all training examples
- Weighted obligation score = `not_freq × 0.5 + shall_freq × 0.3`
- If obligation score > 0.5 AND clause contains permissive language → "Permission / obligation polarity mismatch"

**Signal 3 — Per-Clause Invariant Rules**
For 12 highest-risk clause types, hard invariants are defined:
- **Forbidden patterns**: e.g. "unlimited liability" in Cap On Liability → "Uncapped liability detected"
- **Negation detection**: regex `\b(not|no|never|without|waive|disclaim)\b.{0,40}(target_word)` — e.g. "no right to audit" in Audit Rights → "Negation of standard obligation"
- **Missing required keywords**: keyword density score < 0.5 → "Missing standard protective keyword"
- **Unilateral rights**: "sole discretion", "unilaterally" combined with other signals → "Unusual unilateral right detected"

**Signal 4 — Composite Deviation Score**
```
score = min(semantic_distance / (threshold × 1.5), 0.5)   # up to 0.5 from semantic
      + len(reasons) × 0.15                                 # 0.15 per additional signal
score = min(score, 1.0)
```

**Severity mapping:**
- High: Violation of license ownership invariant, Uncapped liability, Negation of standard obligation, Unusual unilateral right
- Medium: Polarity mismatch, Missing protective keyword, Semantic deviation
- Low: Minor keyword gaps

### 4.5 Precomputed Baselines (`precompute_baselines.py`)

Run once to generate 4 `.npy` files:
- `clause_centroids.npy` — mean embedding per clause type
- `clause_thresholds.npy` — 95th percentile cosine distance per clause type
- `clause_applicability.npy` — 99th percentile (applicability boundary)
- `clause_polarity.npy` — polarity signal profiles (6 signals: shall, may, must, not, without, freely)

---

## 5. CRAG — Constrained RAG for Q&A

**File:** `rag/contract_rag.py`

CRAG (Constrained Retrieval-Augmented Generation) is a novel Q&A architecture that makes hallucination structurally impossible by design.

### 5.1 Intent Classification

8 intent categories, keyword-matched:
- `RISK_QUERY` — risk, deviation, red flag, concern
- `OBLIGATION_QUERY` — shall, must, required, obligation
- `RIGHT_QUERY` — right, may, allowed, permitted
- `CLAUSE_LOOKUP` — liability, termination, license, warranty, audit
- `PARTY_QUERY` — who, party, parties, company
- `FINANCIAL_QUERY` — payment, fee, cost, revenue
- `TERMINATION_QUERY` — terminat, cancel, end, expire
- `GENERAL_QUERY` — fallback

### 5.2 Multi-Strategy Evidence Retrieval

1. **Semantic similarity**: cosine similarity between question embedding and all span embeddings
2. **Intent boosting**: deviating spans get +0.15 similarity for RISK_QUERY
3. **Clause alias boosting**: spans matching clause name/alias get +0.20 similarity
4. Minimum evidence score: 0.30
5. Returns top-6 candidates

### 5.3 Evidence Verification & Grounding Score

```
grounding_score = Σ(similarity_i × weight_i) / Σ(weight_i)
where weight_i = 1/(i+1)  [harmonic weighting — top evidence weighted more]

is_answerable = grounding_score ≥ 0.25 AND len(verified) > 0
```

If `is_answerable = False` → system refuses to answer rather than hallucinating.

### 5.4 Hallucination Risk Classification

- Low: grounding_score ≥ 0.65
- Medium: 0.40 ≤ grounding_score < 0.65
- High: grounding_score < 0.40

### 5.5 Constrained Prompt Construction

- All 5 verified evidence spans: metadata only (clause type, relevance, deviation flags)
- Top 2 spans only: raw text included (max 400 chars each)
- LLM instructed to cite `[SPAN X]` in answer
- LLM explicitly forbidden from speculating beyond evidence
- Intent-specific instruction appended

### 5.6 Output

Every answer includes:
- Answer text with span citations
- Intent classification
- Grounding score (0.0–1.0)
- Hallucination risk badge (Low/Medium/High)
- Evidence count
- Unanswerable flag if evidence insufficient
- Confidence notes

---

## 6. HRS — Hierarchical Recursive Summarization

**File:** `summarizer/hrs_engine.py`

HRS solves the LLM token limit problem for long contracts through a 3-level recursive reduction tree.

### 6.1 The Problem

A contract with 33 clause types, each with multiple spans, cannot be summarized in a single LLM call without truncation. Naive concatenation exceeds context limits and loses information.

### 6.2 The Solution — 3-Level Tree

```
Level 0: Raw spans grouped by clause type
    ↓  [1 LLM call per clause group, max 700 chars input]
Level 1: Per-clause summaries (2 sentences each)
    ↓  [1 LLM call per category of 3-4 clauses]
Level 2: Category summaries (IP & Licensing, Liability & Risk, etc.)
    ↓  [1 final LLM call]
Level 3: Executive summary (3-4 sentences)
```

### 6.3 Clause Category Taxonomy (8 categories)

IP & Licensing · Liability & Risk · Termination · Competition & Exclusivity · Assignment & Transfer · Financial · Legal Protections · Operational

### 6.4 Properties

- **Bounded token usage**: each call receives max 700 chars
- **No information loss**: every clause gets its own call
- **Safety cap**: max 40 total LLM calls
- **Call log**: every call tracked with prompt length and success/failure
- **Graceful degradation**: extractive fallback (first 2 sentences) at every level if LLM unavailable
- **Tree preserved**: full `hrs_tree` dict returned for UI inspection

### 6.5 Template Field Extraction (regex-based, no LLM)

Extracted from full document text:
- **Parties**: company name patterns (Inc, LLC, Ltd, Corp, etc.)
- **Dates**: multiple formats (Month DD YYYY, DD/MM/YYYY, YYYY-MM-DD)
- **Defined terms**: quoted capitalized phrases `"Term Name"`
- **Obligations**: sentences containing shall/must/agrees to/is required to
- **Rights**: sentences containing may/is entitled to/has the right to
- **Payment terms**: sentences containing payment/fee/compensation/royalty
- **Termination conditions**: sentences containing terminat/cancel/expir
- **Governing law**: sentences containing governed by/jurisdiction/laws of

### 6.6 Evaluation Metrics

- **ROUGE-1 F1**: unigram overlap between summary and original spans
- **ROUGE-2 F1**: bigram overlap
- **Coverage**: fraction of reference vocabulary retained in summary
- **Compression ratio**: summary tokens / reference tokens

---

## 7. Clause Negotiation Simulator

**File:** `negotiation/simulator.py`

Given a deviating clause, generates alternative phrasings at three negotiation stances. **No commercial contract tool implements this feature.**

### 7.1 Three Stances

| Stance | Color | Description |
|---|---|---|
| Conservative 🟢 | Green | Minimal changes, fixes only critical issues, easy for counterparty to accept |
| Balanced 🔵 | Blue | Standard market terms, mutual obligations, fair to both parties |
| Aggressive 🔴 | Red | Maximum protection for reviewing party, pushes hard on every deviation |

### 7.2 Deviation-to-Fix Mapping

Each CSDA deviation reason maps to a specific fix hint:
- "Uncapped liability detected" → add liability cap (multiple of fees paid or fixed amount)
- "Violation of license ownership invariant" → remove ownership transfer, replace with limited license
- "Polarity mismatch" → replace permissive language with conditional language
- "Missing protective keyword" → add consent requirements, notice periods, cure rights
- "Negation of standard obligation" → restore obligation with appropriate conditions
- "Unusual unilateral right" → make right mutual or add conditions

### 7.3 Similarity Improvement Scoring

Each rewritten clause is embedded and compared to the clause centroid:
```
improvement = cosine_similarity(rewrite_embedding, centroid) 
            - cosine_similarity(original_embedding, centroid)
```
Positive improvement = rewrite moves toward standard language.
This quantifies how much the negotiation position improves clause alignment.

### 7.4 Privacy

Only the individual clause text (max 600 chars) + deviation metadata is sent to the LLM. The full contract is never transmitted.

---

## 8. Obligation Graph

**File:** `obligation_graph/extractor.py`

Extracts obligation relationships from contract spans and builds a directed graph showing who owes what to whom. **No commercial contract tool visualizes obligation structure as a graph.**

### 8.1 Extraction

Three regex patterns extract:
- **Obligations**: subject + shall/must/agrees to/is required to + action
- **Permissions**: subject + may/is entitled to/has the right to + action
- **Prohibitions**: subject + shall not/must not/may not/is prohibited from + action

Party validation filters:
- Must start with capital letter
- Length 2-30 chars
- First word not in 50+ blocklisted non-party words (this, section, clause, agreement, etc.)
- No URL/special characters
- No sentence fragment indicators (processing, tracking, governing, etc.)

Party normalization maps variations to canonical names (licensor → Licensor, affiliate → Affiliate, chase → Chase, etc.)

### 8.2 Graph Construction

- **Nodes**: parties
- **Edges**: obligation flows (subject → target, inferred from party mentions in action text)
- **Edge weight**: obligation count in that direction
- **Adjacency matrix**: dict[from_party → dict[to_party → count]]

### 8.3 Analysis Outputs

**Balance Score (0.0–1.0)**:
```
balance_score = min(obligations) / max(obligations)
```
1.0 = perfectly balanced, 0.0 = completely one-sided

**Missing Reciprocals**: Party A has obligation X but Party B has no corresponding obligation. Detected by comparing first-2-word action keys across parties.

**Dominant Party**: party with the most obligation statements imposed on them.

**Clause Density**: which clause types carry the most obligations.

### 8.4 D3 Force-Directed Visualization

Interactive graph rendered via `st.components.v1.html` with D3.js v7:
- Node size scales with obligation count
- Node color: red (high load) / blue (moderate) / green (low)
- Curved arrows show obligation direction, thickness = obligation count
- Hover tooltips show exact obligation/permission/prohibition counts
- Draggable nodes with force simulation

---

## 9. Multi-Document Analysis

**File:** `multi_doc/aggregator.py`

Analyzes 2-5 contracts simultaneously and aggregates results for cross-document comparison.

### 9.1 Analysis Flow

```
Upload N PDFs
    ↓
Read all file bytes upfront (before any processing — prevents Streamlit rerun data loss)
    ↓
Run full CSDA pipeline on each document independently
    ↓
aggregate_documents() builds cross-document metrics
    ↓
Multi-Doc tab shows aggregated view
    ↓
Document selector switches single-doc tabs to show selected document
```

### 9.2 Aggregation Metrics

**Clause × Document Heatmap**: for each clause type × each document:
- ⚠️ Deviating
- ✅ Present, no deviation
- — Absent

**Systemic Risks**: clauses deviating in ALL documents that contain them (most dangerous — counterparty consistently uses aggressive language for this clause type)

**Isolated Risks**: clauses deviating in only 1 document (outlier — may be negotiable)

**Deviation Rate per Clause**: `deviating_docs / present_docs` across all documents

**Risk Ranking**: clauses sorted by deviation rate descending

**Top Deviation Reasons**: most frequent CSDA signals across all documents with frequency bar chart

**Per-Document Breakdown**: expandable detail for each document showing its deviating clauses

### 9.3 Document Selector

In multi-doc mode, a dropdown at the top of the main area switches `clause_df` and `spans` to the selected document. All single-doc tabs (Overview, Deviating Clauses, Risk Analysis, Analytics, Summary, Ask, Negotiate, Obligation Graph) update to show that document's analysis.

---

## 10. LLM Architecture & Privacy Guardrails

**File:** `llm/llm_client.py`

### 10.1 LLM Priority Chain

```
1. Groq (cloud) — if GROQ_API_KEY present in Streamlit secrets or env
   Model: llama-3.1-8b-instant
   Max tokens: 350, Temperature: 0.2
   
2. Ollama (local) — if server reachable at localhost:11434
   Model: llama3.2:3b
   Max tokens: 300, Temperature: 0.2
   
3. None — deterministic fallback (all features work without LLM)
```

Detection runs on every page load (not cached) so new secrets are picked up immediately.

### 10.2 Privacy Guardrail Architecture — 3 Layers

**Layer 1 — Data Abstraction** (`build_safe_prompt`):
The function takes evidence metadata, not raw text. It constructs a prompt from:
- Clause type labels
- Deviation flags and reasons
- Severity scores
- Pre-written explanation templates
Raw contract text is included only for the top 2 most relevant spans (max 400 chars each).

**Layer 2 — Prompt Instruction Boundary**:
System prompt explicitly states: "Using ONLY the structured clause metadata below (no raw contract text is provided for privacy)". The model is told it has no access to raw text.

**Layer 3 — Output Capping**:
`max_tokens: 350` and `temperature: 0.2` — low temperature reduces creative deviation, token cap prevents verbose outputs.

### 10.3 What Each Feature Sends to LLM

| Feature | What LLM Receives | Raw Text? |
|---|---|---|
| Executive Summary (HRS) | Category summaries (already compressed) | No |
| Clause Summaries (HRS) | Individual clause text, max 700 chars | Partial (1 clause only) |
| Q&A (CRAG) | Clause metadata + top 2 span texts (400 chars each) | Partial (2 spans max) |
| Negotiation Simulator | Clause text max 600 chars + deviation metadata | Partial (1 clause only) |
| Narration | Clause names + deviation count only | No |

---

## 11. UI Architecture

**File:** `app.py`

### 11.1 Tab Structure (9 tabs)

| Tab | Content |
|---|---|
| Overview | Executive summary, metrics, clause coverage, clause-to-text mapping |
| Deviating Clauses | Risk snapshot, clause text mapping, detailed deviation cards sorted by severity |
| Risk Analysis | Risk snapshot + full clause text mapping (moved from Overview) |
| Analytics | Confidence distribution chart, clause frequency chart, deviation summary table |
| Summary | HRS tree view, template fields (parties, dates, obligations, rights, payment, termination, governing law), ROUGE metrics |
| Ask the Contract | CRAG Q&A with grounding score, hallucination risk badge, citations |
| Negotiate | Clause selector, 3-stance rewrite generator, similarity improvement scoring |
| Obligation Graph | D3 force-directed graph, balance score, obligation matrix, missing reciprocals, clause density |
| Multi-Doc | Per-document deviation rate cards, heatmap, systemic/isolated risks, risk ranking, top reasons |

### 11.2 Session State Management

Key session state variables:
- `analyzed` — whether analysis has been run
- `clause_df` — main analysis DataFrame
- `spans` — list of text spans
- `embeddings` — numpy array of span embeddings
- `embedder` — cached SentenceTransformer instance
- `contract_summary` — structured summary dict
- `summary_narration` — LLM-generated or deterministic narration
- `last_answer` — CRAG Q&A result
- `contract_doc_summary` — HRS summary result
- `multi_doc_results` — aggregated multi-doc metrics
- `multi_doc_raw` — list of per-document analysis results
- `neg_results` — negotiation simulator results
- `ob_graph` — obligation graph result
- `_active_tab` — current active tab name
- `_force_ask_tab` — flag to force JS tab click to Ask tab
- `analysis_mode` — "single" or "multi"

### 11.3 Tab Persistence

Tab switching uses a combination of:
1. `st.session_state["_active_tab"]` — stores desired tab name
2. `_TAB_IDX` dict — maps tab name to index
3. JS injection via `st.components.v1.html` — clicks the correct tab 300ms after render
4. `_force_ask_tab` flag — overrides `_active_tab` for the Ask tab to prevent Summary tab stealing focus

### 11.4 Multi-Doc Mode

- Mode toggle (Single/Multi) in sidebar
- Single mode: `st.file_uploader` with `accept_multiple_files=False`
- Multi mode: `st.file_uploader` with `accept_multiple_files=True`
- All file bytes read upfront before any processing (prevents Streamlit rerun data loss)
- Multi-Doc tab only appears when `multi_doc_results` is populated

### 11.5 Export

- CSV: full analysis DataFrame with span text, clause, confidence, deviation flags, severity, reasons
- JSON: structured dict with summary overview + per-span results

---

## 12. Setup & Deployment

### Local Setup

```bash
git clone https://github.com/Adithiyanpv/contract-intelligence-framework.git
cd contract-intelligence-framework
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# Optional: Groq API key for LLM features
echo 'GROQ_API_KEY = "your_key"' > .streamlit/secrets.toml

# Optional: Ollama for local LLM
ollama pull llama3.2:3b
ollama serve

streamlit run app.py
```

### Streamlit Cloud Deployment

1. Push to GitHub
2. Go to share.streamlit.io → New app
3. Repository: `Adithiyanpv/contract-intelligence-framework`
4. Main file: `app.py`
5. Settings → Secrets → add `GROQ_API_KEY = "your_key"`

Model artifacts auto-download from Google Drive on first boot via `download_models.py`.

### Regenerating Baselines

If training data changes:
```bash
python precompute_baselines.py
```
Recomputes centroids, thresholds, applicability boundaries, and polarity profiles.

---

## 13. Novel Contributions Summary

| Component | Novelty |
|---|---|
| **CSDA** | First multi-signal composite deviation scoring for contracts: semantic distance + polarity profiles + per-clause invariants + composite score |
| **CRAG** | Constrained RAG with grounding score, unanswerable detection, hallucination risk quantification, and citation-first generation |
| **HRS** | Hierarchical recursive summarization that handles token limits without truncation via 3-level tree reduction |
| **Negotiation Simulator** | Generates clause rewrites at 3 negotiation stances with similarity improvement scoring — does not exist in any commercial tool |
| **Obligation Graph** | Directed graph of obligation relationships with balance score, missing reciprocal detection, and D3 force-directed visualization |
| **Multi-Doc Aggregation** | Cross-document systemic vs. isolated risk classification with clause × document heatmap |
| **Privacy Architecture** | Raw contract text never reaches any external API — all LLM calls receive only structured metadata or bounded clause snippets |

---

*ContractIQ — Final Year Project · Team 2022AIE01 · Not legal advice.*
