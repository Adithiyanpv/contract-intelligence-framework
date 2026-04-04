# Contract Clause Deviation Detector — Complete Project Explanation

## Overview

This is an AI-powered contract analysis web application built with **Streamlit**. It ingests a contract PDF, classifies each paragraph into known legal clause types using a fine-tuned **DeBERTa** model, detects deviations from standard clause language using semantic similarity and rule-based invariants, and provides an interactive Q&A interface powered by a local **Ollama LLM** (llama3.2:3b).

The system is designed to assist non-legal users in understanding contract risk — it is explicitly **not legal advice**.

---

## Project Structure

```
contract_deviation_app/
├── app.py                          # Streamlit UI entry point
├── pipeline.py                     # Core ML pipeline (classification + deviation detection)
├── precompute_baselines.py         # One-time script to generate baseline .npy files
├── download_models.py              # Auto-downloads model artifacts from Google Drive
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version for deployment (3.10)
├── packages.txt                    # System packages (poppler-utils for PDF)
├── .streamlit/
│   └── config.toml                 # Streamlit server config (dark theme, upload size)
├── llm/
│   └── llm_client.py               # Ollama HTTP client wrapper
├── rag/
│   ├── rag_chain.py                # RAG engine builder (LLM + prompt template)
│   └── rag_context_builder.py      # Builds safe abstracted context for LLM
└── resources/
    ├── deberta-clause-final/       # Fine-tuned DeBERTa model (33 clause types)
    ├── clause_centroids.npy        # Per-clause embedding centroids (precomputed)
    ├── clause_thresholds.npy       # 95th percentile deviation thresholds
    ├── clause_applicability.npy    # 99th percentile applicability thresholds
    ├── clause_polarity.npy         # Polarity signal profiles per clause
    └── final_cleaned_version (2).csv  # Training dataset (3,688 labeled clause spans)
```

---

## Dataset

**File:** `resources/final_cleaned_version (2).csv`

- **Shape:** 3,688 rows × 3 columns
- **Columns:**
  - `Filename` — source contract document name
  - `Span` — clause text stored as a Python list string (e.g., `"['text here']"`)
  - `Clause` — ground-truth clause label (33 unique types)

**Clause Distribution (top 10):**

| Clause | Count |
|---|---|
| Anti-Assignment | 374 |
| Cap On Liability | 275 |
| License Grant | 255 |
| Audit Rights | 214 |
| Termination For Convenience | 183 |
| Post-Termination Services | 182 |
| Exclusivity | 180 |
| Insurance | 167 |
| Revenue/Profit Sharing | 166 |
| Minimum Commitment | 165 |

**All 33 Clause Types:**
Non-Disparagement, Anti-Assignment, Minimum Commitment, License Grant, Audit Rights, Cap On Liability, Warranty Duration, Most Favored Nation, Termination For Convenience, Revenue/Profit Sharing, Unlimited/All-You-Can-Eat-License, Uncapped Liability, Exclusivity, Affiliate License-Licensee, Change Of Control, Non-Transferable License, Rofr/Rofo/Rofn, Irrevocable Or Perpetual License, Competitive Restriction Exception, Non-Compete, Price Restrictions, Covenant Not To Sue, Volume Restriction, Joint Ip Ownership, Ip Ownership Assignment, Post-Termination Services, Insurance, Affiliate License-Licensor, No-Solicit Of Customers, No-Solicit Of Employees, Liquidated Damages, Third Party Beneficiary, Source Code Escrow

---

## File-by-File Breakdown

### `app.py` — Streamlit UI

The main entry point. Handles:
- PDF upload via sidebar
- Triggers `analyze_document()` from `pipeline.py`
- Stores all results in `st.session_state` for persistence across rerenders
- Renders 3 tabs:
  1. **Overview** — executive summary, metrics, clause-to-text mapping
  2. **Deviating Clauses** — expandable list of flagged clauses with reasons
  3. **Ask the Contract** — Q&A using ML retrieval + Ollama LLM

Key design decisions:
- Model loading is guarded by `@st.cache_resource` to avoid reloading on every interaction
- LLM narration is optional — if Ollama is unavailable, a deterministic fallback is used
- The LLM never sees raw contract text — only an abstracted summary object

---

### `pipeline.py` — Core ML Pipeline

The brain of the application. Contains:

#### Text Processing
- `generate_spans(text)` — splits PDF text on double newlines, filters spans < 80 chars
- `clean_text(text)` — lowercases, removes special chars
- `normalize_span(text)` — replaces party names with "party", numbers with "num", time units with "time"

#### Clause Classification
Uses a fine-tuned **DeBERTa-v3** model (`deberta-clause-final`) to classify each span into one of 33 clause types. Spans with confidence < 0.5 are labeled "Unknown".

#### Deviation Detection (4 signals)
1. **Semantic deviation** — cosine distance from clause centroid > 95th percentile threshold
2. **Polarity violation** — permission language ("freely", "without restriction") in clauses that normally use obligation language ("not", "shall not")
3. **License Grant invariant** — flags ownership/transfer language in license clauses
4. **Cap On Liability invariant** — flags "unlimited liability" / "without limitation" language

#### Question Answering (`ask_document`)
Three-tier retrieval:
1. Risk keywords → returns all deviating clauses
2. Clause name/alias match → direct lookup
3. Semantic similarity → top-k cosine similarity retrieval

#### Summary Builder (`build_contract_summary`)
Deterministic (no LLM). Produces a structured dict with overview stats, coverage, deviations, and confidence notes.

#### Narration (`narrate_contract_summary`)
Optional LLM call. If Ollama is unavailable, returns a safe fallback string.

---

### `precompute_baselines.py` — Baseline Generator

One-time script that:
1. Loads the training CSV
2. Embeds all spans using `sentence-transformers/all-mpnet-base-v2`
3. Computes per-clause centroids, 95th/99th percentile distance thresholds
4. Computes polarity signal profiles (frequency of "shall", "may", "not", etc.)
5. Saves 4 `.npy` files to `resources/`

This script must be re-run if the training data changes.

---

### `download_models.py` — Model Downloader

Checks if all required model artifacts exist locally. If not, downloads them from a Google Drive folder using `gdown`. Required paths:
- `resources/deberta-clause-final/` (model weights + tokenizer)
- `resources/clause_centroids.npy`
- `resources/clause_thresholds.npy`
- `resources/clause_applicability.npy`
- `resources/clause_polarity.npy`

---

### `llm/llm_client.py` — Ollama Client

A minimal HTTP wrapper around the Ollama REST API (`http://localhost:11434/api/generate`). Uses `llama3.2:3b` by default. Returns the `response` field from the JSON payload. Timeout is 120 seconds.

---

### `rag/rag_chain.py` — RAG Engine

Minimal RAG setup. Wraps the LLM client with a prompt template for contract explanation. The `run_rag_reasoning` function takes a pre-built context dict and a question, formats a prompt, and calls the LLM.

---

### `rag/rag_context_builder.py` — RAG Context Builder

Builds a safe, abstracted context object from `clause_df` and `contract_summary`. Crucially, **raw contract text is never passed to the LLM** — only structured metadata (clause names, deviation flags, severity hints, reasons). This prevents prompt injection and hallucination from raw legal text.

---

## End-to-End Workflow

```
PDF Upload
    │
    ▼
pdfplumber → extract text per page → join pages
    │
    ▼
generate_spans() → paragraph-level text chunks (spans)
    │
    ▼
DeBERTa classifier → clause label + confidence per span
    │
    ▼
SentenceTransformer → embed normalized spans
    │
    ▼
Deviation Detection:
  ├── cosine distance vs centroid (semantic)
  ├── polarity signal check
  └── hard invariant rules (License Grant, Cap On Liability)
    │
    ▼
build_contract_summary() → structured summary dict
    │
    ▼
narrate_contract_summary() → Ollama LLM (optional)
    │
    ▼
Streamlit UI renders:
  ├── Tab 1: Overview + clause-to-text mapping
  ├── Tab 2: Deviating clauses with reasons
  └── Tab 3: Q&A (ask_document + Ollama)
```

---

## Setup & Launch Guide

### Prerequisites

- Python 3.10
- [Ollama](https://ollama.com) installed and running locally
- `poppler-utils` (for PDF rendering on Linux/Mac; Windows uses bundled binary)

### Step 1 — Install Dependencies

```bash
cd contract_deviation_app
pip install -r requirements.txt
```

### Step 2 — Pull the LLM Model

```bash
ollama pull llama3.2:3b
ollama serve   # starts the API at http://localhost:11434
```

### Step 3 — Verify Model Artifacts

The app auto-downloads from Google Drive on first run. To manually trigger:

```bash
python download_models.py
```

Verify these exist:
```
resources/deberta-clause-final/
resources/clause_centroids.npy
resources/clause_thresholds.npy
resources/clause_applicability.npy
resources/clause_polarity.npy
```

### Step 4 — Launch the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### Step 5 — Test

1. Upload any contract PDF via the sidebar
2. Click "Analyze Contract"
3. Review the Overview tab for clause coverage and deviations
4. Check the Deviating Clauses tab for flagged sections
5. Ask questions in the "Ask the Contract" tab

---

## Known Issues & Limitations

1. **Hardcoded absolute paths** in `precompute_baselines.py` (Windows-specific)
2. **Only 2 hard invariants** — License Grant and Cap On Liability; other clauses rely solely on semantic distance
3. **No confidence calibration** — the 0.5 threshold is fixed, not learned
4. **Single-page span splitting** — only splits on `\n\n`, misses numbered subsections in some PDFs
5. **No export** — analysis results cannot be downloaded as PDF/CSV
6. **No multi-document comparison** — can only analyze one contract at a time
7. **LLM dependency** — Ollama must be running locally; no cloud LLM fallback
8. **No clause highlighting** — cannot show which page/position in the PDF a clause came from
9. **RAG chain not fully wired** — `rag_chain.py` and `rag_context_builder.py` exist but are not used in `app.py` (the app builds its own inline prompt)
10. **Dead commented code** — `pipeline.py` contains ~400 lines of commented-out legacy code

---

## Planned Enhancements

See the Enhancements section below for the full implementation roadmap.


---

## Enhancements Implemented

### 1. Cleaned up `pipeline.py`
Removed ~400 lines of dead commented-out legacy code. The file is now clean, readable, and production-ready.

### 2. Fixed hardcoded paths in `precompute_baselines.py`
Replaced Windows-absolute paths with `os.path.dirname(os.path.abspath(__file__))` so the script works on any OS from any directory.

### 3. Severity scoring
Added `get_severity()` in `pipeline.py` that maps deviation reasons to High/Medium/Low severity. Severity is stored in `clause_df` and propagated through the summary and Q&A.

### 4. Severity badges in UI
Color-coded HTML badges (red=High, orange=Medium, green=Low) appear next to every deviation in the Overview, Deviating Clauses tab, and Q&A answers.

### 5. Progress bar during analysis
`analyze_document()` now accepts a `progress_callback(step, total, message)` parameter. The UI shows a live progress bar while classifying spans.

### 6. Export to CSV and JSON
Two new functions in `pipeline.py`:
- `export_results_csv()` — full analysis as a downloadable CSV
- `export_results_json()` — structured JSON with summary + per-span results

Download buttons appear in the sidebar after analysis.

### 7. Analytics tab (new Tab 3)
A dedicated analytics tab with:
- Average confidence per clause (bar chart)
- Clause frequency distribution (bar chart)
- Full deviation summary table (sortable dataframe)

### 8. Ollama health check
On startup, the app pings `http://localhost:11434/api/tags`. If Ollama is not running, a warning is shown in the sidebar and the Q&A tab uses a deterministic fallback instead of crashing.

### 9. Expanded clause aliases
Added aliases for Anti-Assignment, Audit Rights, Non-Compete, Exclusivity, Insurance, and IP Ownership Assignment in the Q&A intent matching.

### 10. Improved span generation
`generate_spans()` now also splits long paragraphs (>1000 chars) on numbered subsections (`\d+\.\d+`), improving clause detection on contracts with dense numbered sections.

### 11. Span counts in clause coverage
The Overview tab now shows `Clause Name (N spans)` instead of just the clause name, giving a quick sense of how much of the contract each clause type covers.

### 12. Deviations sorted by severity
The Deviating Clauses tab now sorts High severity items to the top, with summary metrics (total, high, medium counts).

### 13. Deterministic Q&A fallback
When Ollama is unavailable, the Ask tab returns a structured text answer based on the retrieved evidence instead of an error.

---

## Quick Reference — Running the App

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama (optional but recommended)
ollama pull llama3.2:3b
ollama serve

# 3. Launch
streamlit run app.py
```

App opens at http://localhost:8501
