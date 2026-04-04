# Contract Intelligence Framework

> AI-powered contract clause detection, deviation analysis, and explainable Q&A — built for non-legal users.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://contractiq.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What It Does

Upload any contract PDF and the system will:

- **Detect and classify** every clause into one of 33 legal clause types using a fine-tuned DeBERTa model
- **Flag deviations** from standard contract language using semantic similarity, polarity analysis, and hard invariant rules
- **Explain risks** with severity scoring (High / Medium / Low) and plain-English explanations
- **Answer questions** about the contract in natural language via Groq LLM (cloud) or Ollama (local)
- **Export results** as CSV or JSON for further review

---

## Live Demo

**[contractiq.streamlit.app](https://contractiq.streamlit.app)**

> Upload a contract PDF → get instant clause analysis, deviation flags, and AI-powered Q&A.

---

## Architecture

```
PDF Upload
    │
    ▼
pdfplumber → extract text per page
    │
    ▼
Span Extraction
  ├── Split on double newlines (paragraph boundaries)
  └── Further split long paragraphs on numbered subsections
    │
    ▼
DeBERTa Clause Classifier (fine-tuned, 33 clause types)
  └── Confidence < 0.5 → labeled "Unknown"
    │
    ▼
SentenceTransformer Embeddings (all-mpnet-base-v2)
    │
    ▼
Deviation Detection (4 signals)
  ├── Semantic distance from clause centroid (95th percentile threshold)
  ├── Polarity violation (permission language in obligation clauses)
  ├── License Grant invariant (ownership/transfer language)
  └── Cap On Liability invariant (uncapped liability language)
    │
    ▼
Severity Scoring → High / Medium / Low
    │
    ▼
Streamlit UI
  ├── Overview tab (executive summary, risk snapshot, clause mapping)
  ├── Deviating Clauses tab (sorted by severity)
  ├── Analytics tab (confidence charts, frequency, deviation table)
  └── Ask the Contract tab (Groq / Ollama Q&A)
```

---

## Clause Types Detected (33 total)

| Category | Clauses |
|---|---|
| IP & Licensing | License Grant, Non-Transferable License, Irrevocable Or Perpetual License, Unlimited/All-You-Can-Eat-License, Ip Ownership Assignment, Joint Ip Ownership, Affiliate License-Licensee, Affiliate License-Licensor |
| Liability & Risk | Cap On Liability, Uncapped Liability, Liquidated Damages, Insurance |
| Termination | Termination For Convenience, Post-Termination Services |
| Competition | Non-Compete, Exclusivity, Competitive Restriction Exception, Non-Disparagement |
| Assignment & Transfer | Anti-Assignment, Change Of Control |
| Financial | Revenue/Profit Sharing, Minimum Commitment, Price Restrictions, Volume Restriction, Most Favored Nation |
| Legal Protections | Covenant Not To Sue, Rofr/Rofo/Rofn, Source Code Escrow, Third Party Beneficiary |
| Operational | Audit Rights, Warranty Duration, No-Solicit Of Customers, No-Solicit Of Employees |

---

## Dataset

The clause classifier was trained on **3,688 labeled contract spans** from real commercial agreements.

| Column | Description |
|---|---|
| `Filename` | Source contract document |
| `Span` | Clause text (stored as Python list string) |
| `Clause` | Ground-truth label (33 types) |

Top clause distribution: Anti-Assignment (374), Cap On Liability (275), License Grant (255), Audit Rights (214), Termination For Convenience (183).

---

## Project Structure

```
contract_deviation_app/
├── app.py                          # Streamlit UI — all tabs and interactions
├── pipeline.py                     # Core ML pipeline (classification + deviation)
├── precompute_baselines.py         # One-time script to regenerate .npy baselines
├── download_models.py              # Auto-downloads model artifacts from Google Drive
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version hint
├── packages.txt                    # System packages (poppler-utils)
├── .streamlit/
│   ├── config.toml                 # Dark theme, upload size config
│   └── secrets.toml                # API keys (gitignored, local only)
├── llm/
│   └── llm_client.py               # Groq + Ollama client with auto-detection
├── rag/
│   ├── rag_chain.py                # RAG engine (LLM + prompt template)
│   └── rag_context_builder.py      # Safe abstracted context builder
└── resources/
    ├── deberta-clause-final/       # Fine-tuned DeBERTa model weights + tokenizer
    ├── clause_centroids.npy        # Per-clause embedding centroids
    ├── clause_thresholds.npy       # 95th percentile deviation thresholds
    ├── clause_applicability.npy    # 99th percentile applicability thresholds
    ├── clause_polarity.npy         # Polarity signal profiles per clause
    └── final_cleaned_version (2).csv  # Training dataset
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- Git
- [Ollama](https://ollama.com) (optional — for local LLM)

### 1. Clone the repository

```bash
git clone https://github.com/Adithiyanpv/contract-intelligence-framework.git
cd contract-intelligence-framework
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys (optional — for LLM Q&A)

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 5. Download model artifacts

The app auto-downloads on first run. To trigger manually:

```bash
python download_models.py
```

This downloads to `resources/`:
- `deberta-clause-final/` — fine-tuned DeBERTa model
- `clause_centroids.npy`, `clause_thresholds.npy`, `clause_applicability.npy`, `clause_polarity.npy`

### 6. (Optional) Set up local Ollama LLM

```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2:3b
ollama serve
```

The app auto-detects Ollama if running. Groq takes priority if both are available.

### 7. Launch the app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## Deployed Version (Streamlit Cloud)

The app is deployed at **[contractiq.streamlit.app](https://contractiq.streamlit.app)** and runs independently of any local machine.

### How deployment works

1. Code is hosted on GitHub at `Adithiyanpv/contract-intelligence-framework`
2. Streamlit Cloud pulls from the `main` branch automatically on every push
3. Model artifacts are downloaded from Google Drive on first boot via `download_models.py`
4. The Groq API key is stored in Streamlit Cloud secrets (never in the repository)

### Updating the deployed app

Any push to `main` triggers an automatic redeploy:

```bash
git add .
git commit -m "your changes"
git push origin main
```

### Adding secrets on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Select your app → **Settings** → **Secrets**
3. Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

4. Click **Save** — the app reboots automatically

### LLM on Streamlit Cloud

Ollama cannot run on Streamlit Cloud (it requires a local server). The app uses **Groq** as the cloud LLM. Without a Groq key, the Q&A tab uses deterministic fallbacks (clause retrieval without natural language generation).

---

## Regenerating Baselines

If you update the training dataset, regenerate the `.npy` baseline files:

```bash
cd contract_deviation_app
python precompute_baselines.py
```

This recomputes:
- Clause embedding centroids
- 95th / 99th percentile distance thresholds
- Polarity signal profiles

Commit the new `.npy` files and push — the deployed app will pick them up on next boot.

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Clause Classification | DeBERTa-v3 (fine-tuned, HuggingFace Transformers) |
| Semantic Embeddings | sentence-transformers/all-mpnet-base-v2 |
| PDF Parsing | pdfplumber |
| Cloud LLM | Groq API (llama-3.1-8b-instant) |
| Local LLM | Ollama (llama3.2:3b) |
| ML Utilities | scikit-learn, numpy, pandas |
| Model Download | gdown (Google Drive) |

---

## Disclaimer

This system is designed to assist in contract review and does not constitute legal advice. All analysis is based on learned patterns from reference data. The absence of a detected deviation does not guarantee standard or low-risk language. Always consult a qualified legal professional for binding contract decisions.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
