import re
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --- 1. Load and cache models ---
@st.cache_resource(show_spinner="Loading ranking models...")
def load_models():
    # Bi-encoder: fast embeddings used for novelty
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    # Cross-encoder: slower but much more accurate query/abstract relevance
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return bi_encoder, cross_encoder


bi_model, cross_model = load_models()

# Final score weights (sum to 1.0, then scaled to a 0-5 range)
WEIGHTS = {
    "relevance": 0.50,  # Cross-encoder relevance (most important)
    "llm_score": 0.20,  # Gemini's relevance rating
    "novelty": 0.10,
    "venue": 0.10,
    "citations": 0.10,
}
SCORE_SCALE = 5.0

# --- 2. Top-tier venues ---
TOP_TIER_VENUES = {
    "neurips", "icml", "iclr", "cvpr", "iccv", "eccv", "acl", "emnlp", "naacl",
    "nature", "science", "pnas", "jama", "new england journal of medicine",
    "ieee", "acm", "lancet", "cell",
}


# Whole-word match so e.g. "acl" does not match "oracle"
TOP_TIER_PATTERN = re.compile(r"\b(" + "|".join(re.escape(v) for v in TOP_TIER_VENUES) + r")\b")


def get_venue_score(venue_name: Optional[str]) -> float:
    """1.0 for top-tier venues, 0.5 for other published venues, 0.3 for preprints, 0 if unknown."""
    if not venue_name or not isinstance(venue_name, str) or venue_name == "Unknown":
        return 0.0
    venue_lower = venue_name.lower()
    if TOP_TIER_PATTERN.search(venue_lower):
        return 1.0
    if "arxiv" in venue_lower or "preprint" in venue_lower:
        return 0.3
    return 0.5


# --- 3. Novelty ---
def calculate_novelty(df: pd.DataFrame) -> pd.DataFrame:
    """Novelty = 1 - similarity to the most similar other paper in the set."""
    if df.empty or "summary" not in df.columns or bi_model is None:
        df["novelty"] = 0.0
        return df

    summaries = df["summary"].fillna("").astype(str).tolist()
    similarity_matrix = cosine_similarity(bi_model.encode(summaries))
    np.fill_diagonal(similarity_matrix, 0)
    df["novelty"] = 1 - similarity_matrix.max(axis=1)
    return df


# --- 4. Cross-encoder relevance ---
def compute_cross_encoder_scores(query: str, summaries: List[str]) -> List[float]:
    if not cross_model or not query:
        return [0.0] * len(summaries)
    return cross_model.predict([[query, doc] for doc in summaries])


def exact_title_matches(df: pd.DataFrame, query: str) -> pd.Series:
    """True for papers whose title contains the exact query; these are always ranked first."""
    if df.empty or not query:
        return pd.Series(False, index=df.index)
    query_norm = query.lower().strip()
    return df["title"].fillna("").astype(str).str.lower().str.contains(query_norm, regex=False)


def _min_max(series: pd.Series) -> np.ndarray:
    values = series.astype(float).to_numpy()
    lo, hi = values.min(), values.max()
    if hi > lo:
        return (values - lo) / (hi - lo)
    return np.full(len(values), 0.5)


# --- 5. Main ranking function ---
def rank_papers(input_data: Union[pd.DataFrame, List[Dict]], query: str, user_requested_count: int = 5) -> List[Dict]:
    df = pd.DataFrame(input_data) if isinstance(input_data, list) else input_data.copy()
    if df.empty:
        return []

    # Lowercase column names so "Title" and "title" are treated the same
    df.columns = [c.lower() for c in df.columns]
    for col in ("title", "summary"):
        if col not in df.columns:
            df[col] = ""

    for col in ("citationcount", "llm_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df.columns else 0.0

    # A. Semantic relevance
    summaries = df["summary"].fillna("").astype(str).tolist()
    df["relevance"] = compute_cross_encoder_scores(query, summaries)

    # B. Novelty
    if "novelty" not in df.columns:
        df = calculate_novelty(df)

    # C. Venue
    df["venue"] = df["venue"].fillna("").astype(str) if "venue" in df.columns else ""
    df["venuescore"] = df["venue"].apply(get_venue_score)

    # D. Weighted final score. Each component is kept (0-1) so the UI can show a breakdown.
    df["score_relevance"] = _min_max(df["relevance"])
    df["score_llm_score"] = (df["llm_score"] / 10.0).clip(0, 1)
    df["score_novelty"] = _min_max(df["novelty"])
    df["score_venue"] = df["venuescore"]
    df["score_citations"] = _min_max(np.log1p(df["citationcount"]))
    df["finalscore"] = sum(WEIGHTS[name] * df[f"score_{name}"] for name in WEIGHTS) * SCORE_SCALE

    df["exact_match"] = exact_title_matches(df, query)
    df = df.sort_values(by=["exact_match", "finalscore"], ascending=False).reset_index(drop=True)

    return df.head(user_requested_count).to_dict(orient="records")
