import arxiv
import pandas as pd
from typing import List, Dict
import requests
import re
import time
import random

# --- CONSTANTS ---
OPENALEX_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_URL = "https://api.crossref.org/works"

def reconstruct_abstract(inverted_index: Dict) -> str:
    """Rebuild abstract from OpenAlex inverted index"""
    if not inverted_index:
        return ""
    word_list = [(pos, word) for word, positions in inverted_index.items() for pos in positions]
    return " ".join(word for _, word in sorted(word_list, key=lambda x: x[0]))

def normalize_title(title: str) -> str:
    """Removes symbols and casing to find duplicates"""
    if not title:
        return ""
    return re.sub(r'[^a-z0-9]', '', title.lower())

# --- 1. Crossref (NEW BACKUP SOURCE) ---
def fetch_crossref_papers(query: str, max_results: int = 5) -> List[Dict]:
    papers = []
    params = {
        "query": query,
        "rows": max_results,
        "select": "title,abstract,author,published-print,URL,is-referenced-by-count"
    }
    try:
        # Crossref likes a user-agent
        headers = {'User-Agent': 'ResearchAgent/1.0 (mailto:student@example.com)'}
        res = requests.get(CROSSREF_URL, params=params, headers=headers, timeout=10)
        
        if res.status_code == 200:
            data = res.json()
            items = data.get('message', {}).get('items', [])
            
            for item in items:
                # Titles in Crossref are lists
                title_list = item.get('title', [])
                title = title_list[0] if title_list else "Untitled"
                
                # Authors
                authors = [f"{a.get('given', '')} {a.get('family', '')}" for a in item.get('author', [])]
                
                # Abstract (sometimes XML formatted, keep it simple)
                abstract = item.get('abstract', 'No abstract available in Crossref.')
                abstract = re.sub('<[^<]+?>', '', abstract) # Remove XML tags
                
                papers.append({
                    "id": item.get("DOI"),
                    "source": "Crossref",
                    "title": title,
                    "summary": abstract,
                    "authors": authors,
                    "published": str(item.get("published-print", {}).get("date-parts", [[2020]])[0][0]),
                    "pdf_url": item.get("URL"),
                    "citationcount": item.get("is-referenced-by-count", 0)
                })
    except Exception as e:
        print(f"⚠️ Crossref Error: {e}")
        
    return papers

# --- 2. Semantic Scholar (With Retry Logic) ---
def fetch_semanticscholar_papers(query: str, max_results: int = 5) -> List[Dict]:
    papers = []
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,authors,year,openAccessPdf,citationCount,url"
    }
    
    # Retry up to 3 times if we get a 429
    for attempt in range(3):
        try:
            # Sleep increases with attempts (1s, 2s, 4s)
            time.sleep(1 + attempt) 
            
            res = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=10)
            
            if res.status_code == 429:
                print(f"⚠️ Semantic Scholar Busy (429). Retrying {attempt+1}/3...")
                continue # Try loop again
            
            if res.status_code == 200:
                data = res.json()
                for item in data.get('data', []):
                    papers.append({
                        "id": item.get("paperId"),
                        "source": "Semantic Scholar",
                        "title": item.get("title") or "Untitled",
                        "summary": item.get("abstract") or "No abstract available.",
                        "authors": [a.get("name") for a in item.get("authors", [])],
                        "published": str(item.get("year")),
                        "pdf_url": item.get("openAccessPdf", {}).get("url") or item.get("url"),
                        "citationcount": item.get("citationCount", 0)
                    })
                break # Success, exit loop
                
        except Exception as e:
            print(f"⚠️ Semantic Scholar Error: {e}")
            break
            
    return papers

# --- 3. OpenAlex (Fixed Params) ---
def fetch_openalex_papers(query: str, max_results: int = 5) -> List[Dict]:
    papers = []
    # Removed specific search_fields to avoid 400 Bad Request
    params = {
        "search": query,
        "per_page": max_results,
        "filter": "from_publication_date:2010-01-01",
        "sort": "relevance_score:desc"
    }
    try:
        response = requests.get(OPENALEX_URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for item in data.get('results', []):
                abstract = reconstruct_abstract(item.get('abstract_inverted_index'))
                papers.append({
                    "id": item.get('id'),
                    "source": "OpenAlex",
                    "title": item.get('display_name') or "Untitled",
                    "summary": abstract or "No abstract available.",
                    "authors": [a.get('author', {}).get('display_name') for a in item.get('authorships', [])],
                    "published": str(item.get('publication_year')),
                    "pdf_url": item.get('open_access', {}).get('oa_url') or item.get('doi'),
                    "citationcount": item.get('cited_by_count', 0)
                })
    except Exception as e:
        print(f"⚠️ OpenAlex Error: {e}")
    return papers

# --- 4. ArXiv (Standard) ---
def fetch_arxiv_papers(query: str, max_results: int = 5) -> List[Dict]:
    papers = []
    client = arxiv.Client()
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        for result in client.results(search):
            papers.append({
                "id": result.entry_id.split("/abs/")[-1],
                "source": "ArXiv",
                "title": result.title.replace('\n', ' '),
                "summary": result.summary.replace('\n', ' '),
                "authors": [a.name for a in result.authors],
                "published": result.published.strftime('%Y-%m-%d'),
                "pdf_url": result.pdf_url,
                "citationcount": 0
            })
    except Exception as e:
        print(f"⚠️ ArXiv Error: {e}")
    return papers

# --- MAIN FETCH FUNCTION ---
def fetch_recent_papers(queries: List[str], max_results: int = 10) -> List[Dict]:
    """
    Fetch from ArXiv, OpenAlex, Semantic Scholar, AND Crossref.
    """
    # Ask for enough papers from each source
    limit_per_source = max(5, int(max_results))
    print(f"--- FETCHING: Searching {len(queries)} queries on 4 APIs ---")
    
    all_papers = []
    seen_titles = set()
    
    for query in queries:
        # 1. ArXiv
        all_papers.extend(fetch_arxiv_papers(query, limit_per_source))
        
        # 2. OpenAlex
        all_papers.extend(fetch_openalex_papers(query, limit_per_source))
        
        # 3. Semantic Scholar (with sleep to avoid 429)
        time.sleep(1) 
        all_papers.extend(fetch_semanticscholar_papers(query, limit_per_source))
        
        # 4. Crossref (NEW)
        time.sleep(1)
        all_papers.extend(fetch_crossref_papers(query, limit_per_source))

    # Deduplicate
    unique_papers = []
    for paper in all_papers:
        norm_title = normalize_title(paper['title'])[:60]
        if norm_title not in seen_titles and norm_title != "":
            unique_papers.append(paper)
            seen_titles.add(norm_title)
            
    print(f"✅ Fetch Complete: {len(unique_papers)} unique papers found.")
    return unique_papers