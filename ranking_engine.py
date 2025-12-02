import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Optional, Union
import streamlit as st # REQUIRED for caching

# --- 1. OPTIMIZATION: Cache the AI Model ---
# This prevents the app from reloading the model (2GB+) on every interaction.
# Without this, your app will freeze for 3-5 seconds on every button click.
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"⚠️ Error loading SentenceTransformer: {e}")
        return None

model = load_embedding_model()
scaler = MinMaxScaler()

# Top-tier academic venues list
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

def calculate_novelty(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'summary' not in df.columns or model is None:
        df['novelty'] = 0.0
        return df

    # Ensure all summaries are strings to prevent crashes
    summaries = df['summary'].fillna("").astype(str).tolist()
    embeddings = model.encode(summaries)
    
    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, 0)
    
    # Novelty = 1 - (How similar is this paper to its closest neighbor?)
    max_similarity = similarity_matrix.max(axis=1)
    df['novelty'] = 1 - max_similarity
    return df

def embed_query_variants(query: str) -> np.ndarray:
    if model is None: return np.zeros((1, 384))
    if not query or not isinstance(query, str): query = "general research"
    
    variants = [q.strip() for q in query.split(" OR ") if q.strip()]
    if not variants: variants = [query]

    embeddings = model.encode(variants)
    if embeddings.ndim == 1: embeddings = embeddings.reshape(1, -1)
    return embeddings

def rank_papers(input_data: Union[pd.DataFrame, List[Dict]], query: str, weights: Optional[Dict[str, float]] = None) -> List[Dict]:
    """
    Ranks papers based on Semantic Relevance, Novelty, LLM Score, and Citations.
    """
    # 1. Convert Input to DataFrame
    if isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    else:
        df = input_data.copy()

    if df.empty: return []

    # 2. Standardize Columns
    df.columns = [c.lower() for c in df.columns]
    
    # --- CRITICAL FIX: Force Numeric Types ---
    # APIs often send numbers as strings ("10" instead of 10). This fixes math errors.
    df['citationcount'] = pd.to_numeric(df.get('citationcount', 0), errors='coerce').fillna(0)
    df['llm_score'] = pd.to_numeric(df.get('llm_score', 0), errors='coerce').fillna(0)
    
    # 3. Novelty
    if 'novelty' not in df.columns:
        df = calculate_novelty(df)

    # 4. Semantic Relevance (The "Meaning" Match)
    if model is not None and 'summary' in df.columns:
        query_embeddings = embed_query_variants(query)
        summaries = df['summary'].fillna("").astype(str).tolist()
        summary_embeddings = model.encode(summaries)
        
        sim_matrix = cosine_similarity(summary_embeddings, query_embeddings)
        df['relevance'] = sim_matrix.max(axis=1)
    else:
        df['relevance'] = 0.0

    # 5. Metadata Scores
    df['venue'] = df.get('venue', 'Unknown')
    df['venuescore'] = df['venue'].apply(get_venue_score)

    # 6. Normalization Helper
    def safe_normalize(series):
        if series.max() > series.min():
            return scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        return np.full(len(series), 0.5)

    df['norm_relevance'] = safe_normalize(df['relevance'])
    df['norm_novelty'] = safe_normalize(df['novelty'])
    df['norm_llm_score'] = df['llm_score'] / 5.0 
    df['norm_venue'] = df['venuescore']
    
    # Log transform citations (handle 0 vs 1000 citations smoothly)
    df['log_citations'] = np.log1p(df['citationcount'])
    df['norm_citations'] = safe_normalize(df['log_citations'])

    # 7. Weighted Scoring Formula
    default_weights = {
        'relevance': 0.50,  # Content match is most important
        'novelty': 0.10,    # Reward unique papers
        'llm_score': 0.30,  # Trust Gemini's filter
        'venue': 0.05,
        'citations': 0.05   # Trust impactful papers
    }
    final_weights = {**default_weights, **(weights or {})}

    df['finalscore'] = (
        final_weights['relevance'] * df['norm_relevance'] +
        final_weights['novelty'] * df['norm_novelty'] +
        final_weights['llm_score'] * df['norm_llm_score'] +
        final_weights['venue'] * df['norm_venue'] +
        final_weights['citations'] * df['norm_citations']
    )

    # 8. Sort and Return
    df = df.sort_values(by='finalscore', ascending=False).reset_index(drop=True)
    
    return df.to_dict(orient='records')