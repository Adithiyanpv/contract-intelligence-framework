"""
Precompute baseline embeddings, centroids, thresholds, and polarity profiles
from the training dataset. Run this script once from the contract_deviation_app
directory whenever the training data changes.

Usage:
    python precompute_baselines.py
"""

import os
import numpy as np
import pandas as pd
import ast
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

# ---- Paths (relative to this script's directory) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "resources", "final_cleaned_version (2).csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "resources")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def normalize_span(text):
    text = re.sub(r"\b(company|licensor|licensee|producer|ma|ent)\b", "party", text)
    text = re.sub(r"\b\d+(\.\d+)?\b", "num", text)
    text = re.sub(r"\b(day|days|month|months|year|years)\b", "time", text)
    return text


def extract_span_text(span):
    try:
        lst = ast.literal_eval(span)
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0]
    except Exception:
        return None
    return None


def polarity_profile(texts):
    signals = ["shall", "may", "must", "not", "without", "freely"]
    total = max(len(texts), 1)
    return {s: sum(s in t for t in texts) / total for s in signals}


def main():
    print(f"Loading dataset from: {TRAIN_CSV_PATH}")
    df = pd.read_csv(TRAIN_CSV_PATH)
    df["span_text"] = df["Span"].apply(extract_span_text)
    df = df.dropna(subset=["span_text"]).reset_index(drop=True)
    df["norm_span"] = df["span_text"].apply(lambda x: normalize_span(clean_text(x)))

    print(f"Dataset loaded: {len(df)} rows, {df['Clause'].nunique()} clause types")

    print("Computing embeddings (this may take a few minutes)…")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(
        df["norm_span"].tolist(),
        batch_size=32,
        show_progress_bar=True
    )

    clause_embeddings = defaultdict(list)
    for emb, clause in zip(embeddings, df["Clause"]):
        clause_embeddings[clause].append(emb)

    centroids, thresholds, applicability, polarity_profiles = {}, {}, {}, {}

    for clause, embs in clause_embeddings.items():
        embs_arr = np.vstack(embs)
        centroid = embs_arr.mean(axis=0)
        dists = cosine_distances(embs_arr, centroid.reshape(1, -1)).flatten()

        centroids[clause] = centroid
        thresholds[clause] = float(np.percentile(dists, 95))
        applicability[clause] = float(np.percentile(dists, 99))

        clause_texts = df[df["Clause"] == clause]["norm_span"].tolist()
        polarity_profiles[clause] = polarity_profile(clause_texts)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "clause_centroids.npy"), centroids)
    np.save(os.path.join(OUTPUT_DIR, "clause_thresholds.npy"), thresholds)
    np.save(os.path.join(OUTPUT_DIR, "clause_applicability.npy"), applicability)
    np.save(os.path.join(OUTPUT_DIR, "clause_polarity.npy"), polarity_profiles)

    print(f"✅ Baselines saved to {OUTPUT_DIR}/")
    print(f"   Clauses processed: {list(centroids.keys())}")


if __name__ == "__main__":
    main()
