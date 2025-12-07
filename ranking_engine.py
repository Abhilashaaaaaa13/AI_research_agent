import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Optional, Union
import streamlit as st

# --- 1. Load & Cache Models ---
@st.cache_resource
def load_models():
    # Bi-encoder for fast retrieval/novelty
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Cross-encoder for HIGH ACCURACY ranking
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') 
    
    return bi_encoder, cross_encoder

# Load both models
bi_model, cross_model = load_models()
scaler = MinMaxScaler()

# --- 2. Top-tier venues (Restored) ---
TOP_TIER_VENUES = {
    'neurips', 'icml', 'iclr', 'cvpr', 'iccv', 'eccv', 'acl', 'emnlp', 'naacl',
    'nature', 'science', 'pnas', 'jama', 'new england journal of medicine', 
    'ieee', 'acm', 'lancet', 'cell'
}

def get_venue_score(venue_name: Optional[str]) -> float:
    if not venue_name or not isinstance(venue_name, str):
        return 0.0
    venue_lower = venue_name.lower()
    for top_venue in TOP_TIER_VENUES:
        if top_venue in venue_lower:
            return 1.0
    return 0.5

# --- 3. Calculate novelty ---
def calculate_novelty(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'summary' not in df.columns or bi_model is None:
        df['novelty'] = 0.0
        return df

    summaries = df['summary'].fillna("").astype(str).tolist()
    embeddings = bi_model.encode(summaries)
    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, 0)
    df['novelty'] = 1 - similarity_matrix.max(axis=1)
    return df

# --- 4. Cross-Encoder Re-Ranking ---
def compute_cross_encoder_scores(query: str, summaries: List[str]) -> List[float]:
    if not cross_model or not query:
        return [0.0] * len(summaries)
    
    pairs = [[query, doc] for doc in summaries]
    scores = cross_model.predict(pairs)
    return scores
# new exact match booster
def boost_exact_matches(df: pd.DataFrame, query: str)-> pd.DataFrame:
    """If the user's query appears exactly in the title, give it a HUGE score boost"""
    if df.empty or not query:
        return df
    #normalize for comparison
    query_norm = query.lower().strip()
    #massive booost to ensure it jumpt to #1
    mask = df['title'].str.lower().str.contains(query_norm,regex=False)
    if mask.any():
        df.loc[mask,'finalscore'] += 50.0
    return df
# --- 5. Main ranking function ---
def rank_papers(input_data: Union[pd.DataFrame, List[Dict]], query: str, user_requested_count: int = 5) -> List[Dict]:
    if isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    else:
        df = input_data.copy()
    
    if df.empty:
        return []

    # Clean columns (Forces lowercase, prevents KeyError 'Title')
    df.columns = [c.lower() for c in df.columns]
    
    # Handle numeric columns safely
    df['citationcount'] = pd.to_numeric(df.get('citationcount', 0), errors='coerce').fillna(0)
    df['llm_score'] = pd.to_numeric(df.get('llm_score', 0), errors='coerce').fillna(0)

    # --- A. Semantic Relevance (Cross-Encoder) ---
    summaries = df['summary'].fillna("").astype(str).tolist()
    ce_scores = compute_cross_encoder_scores(query, summaries)
    df['relevance'] = ce_scores

    # --- B. Novelty ---
    if 'novelty' not in df.columns:
        df = calculate_novelty(df)

    # --- C. Venue & Normalization ---
    df['venue'] = df.get('venue', 'Unknown')
    df['venuescore'] = df['venue'].apply(get_venue_score)

    def safe_normalize(series):
        if series.max() > series.min():
            return scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        return np.full(len(series), 0.5)

    # Normalize metrics
    df['norm_relevance'] = safe_normalize(df['relevance'])
    df['norm_novelty'] = safe_normalize(df['novelty'])
    df['norm_venue'] = df['venuescore']
    df['norm_citations'] = safe_normalize(np.log1p(df['citationcount']))
    
    # Normalize LLM Score (0-10 -> 0.0-1.0)
    df['norm_llm'] = df['llm_score'] / 10.0

    # --- D. Final Weighted Score ---
    # Updated weights to include 'llm_score'
    weights = {
        'relevance': 0.50,  # Cross-Encoder (Most important)
        'llm_score': 0.20,  # Gemini's Opinion
        'novelty': 0.10,
        'venue': 0.10,
        'citations': 0.10
    }

    df['finalscore'] = (
        weights['relevance'] * df['norm_relevance'] +
        weights['llm_score'] * df['norm_llm'] +
        weights['novelty'] * df['norm_novelty'] +
        weights['venue'] * df['norm_venue'] +
        weights['citations'] * df['norm_citations']
    ) * 5.0
#apply boost
    df = boost_exact_matches(df,query)
    # Sort
    df = df.sort_values(by='finalscore', ascending=False).reset_index(drop=True)
    
    # --- E. Strict Slicing ---
    result_df = df.head(user_requested_count)
    
    return result_df.to_dict(orient='records')