"""
Precompute baseline embeddings, centroids, thresholds, polarity profiles,
and keyword profiles from the training dataset.

Run from contract_deviation_app/ directory:
    python precompute_baselines.py
"""

import os
import re
import ast
import numpy as np
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "resources", "final_cleaned_version (2).csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "resources")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\,\;\:\-]", "", text)
    return text.strip()


def normalize_span(text):
    text = re.sub(r"\b(company|licensor|licensee|producer|vendor|client|customer|partner)\b", "party", text)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", "num", text)
    text = re.sub(r"\$[\d,\.]+", "num", text)
    text = re.sub(r"\b(day|days|month|months|year|years|week|weeks)\b", "time", text)
    return text


def extract_span_text(span):
    try:
        lst = ast.literal_eval(span)
        if isinstance(lst, list) and len(lst) > 0:
            return " ".join(str(x) for x in lst if x)
    except Exception:
        pass
    return str(span) if pd.notna(span) else None


def polarity_profile(texts):
    signals = ["shall", "may", "must", "not", "without", "freely", "restrict", "prohibit", "permit"]
    total = max(len(texts), 1)
    return {s: sum(s in t for t in texts) / total for s in signals}


def keyword_profile(texts, top_n=30):
    """Extract top discriminative keywords per clause using TF-IDF-like scoring."""
    from collections import Counter
    word_counts = Counter()
    for t in texts:
        words = re.findall(r'\b[a-z]{3,}\b', t.lower())
        word_counts.update(set(words))  # set to avoid counting duplicates per doc

    # Filter stopwords
    stopwords = {
        "the", "and", "for", "that", "this", "with", "shall", "party",
        "any", "all", "such", "may", "not", "its", "are", "been",
        "have", "has", "will", "from", "each", "other", "upon", "under",
        "into", "than", "then", "when", "which", "where", "who", "num", "time"
    }
    filtered = {w: c for w, c in word_counts.items() if w not in stopwords and c >= 2}
    top_keywords = [w for w, _ in sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    return top_keywords


def main():
    print(f"Loading dataset: {TRAIN_CSV_PATH}")
    df = pd.read_csv(TRAIN_CSV_PATH)
    df["span_text"] = df["Span"].apply(extract_span_text)
    df = df.dropna(subset=["span_text"]).reset_index(drop=True)
    df["norm_span"] = df["span_text"].apply(lambda x: normalize_span(clean_text(x)))

    print(f"Loaded {len(df)} rows, {df['Clause'].nunique()} clause types")

    print("Computing embeddings…")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(df["norm_span"].tolist(), batch_size=32, show_progress_bar=True)

    clause_embeddings = defaultdict(list)
    for emb, clause in zip(embeddings, df["Clause"]):
        clause_embeddings[clause].append(emb)

    centroids, thresholds, applicability, polarity_profiles, keyword_profiles = {}, {}, {}, {}, {}

    for clause, embs in clause_embeddings.items():
        embs_arr = np.vstack(embs)
        centroid = embs_arr.mean(axis=0)
        dists = cosine_distances(embs_arr, centroid.reshape(1, -1)).flatten()

        centroids[clause] = centroid
        # Use 90th percentile for more sensitive deviation detection
        thresholds[clause] = float(np.percentile(dists, 90))
        applicability[clause] = float(np.percentile(dists, 99))

        clause_texts = df[df["Clause"] == clause]["norm_span"].tolist()
        polarity_profiles[clause] = polarity_profile(clause_texts)
        keyword_profiles[clause] = keyword_profile(clause_texts)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "clause_centroids.npy"), centroids)
    np.save(os.path.join(OUTPUT_DIR, "clause_thresholds.npy"), thresholds)
    np.save(os.path.join(OUTPUT_DIR, "clause_applicability.npy"), applicability)
    np.save(os.path.join(OUTPUT_DIR, "clause_polarity.npy"), polarity_profiles)
    np.save(os.path.join(OUTPUT_DIR, "clause_keywords.npy"), keyword_profiles)

    print(f"✅ Baselines saved to {OUTPUT_DIR}/")
    print(f"   Clauses: {sorted(centroids.keys())}")

    # Print sample thresholds for inspection
    print("\nSample thresholds (90th percentile):")
    for c in ["License Grant", "Cap On Liability", "Anti-Assignment", "Termination For Convenience"]:
        if c in thresholds:
            print(f"  {c}: {thresholds[c]:.4f}")


if __name__ == "__main__":
    main()
