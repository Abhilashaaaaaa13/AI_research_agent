import arxiv
import pandas as pd
from typing import List,Dict
import requests
OPENALEX_URL = "https://api.openalex.org/works"

def reconstruct_abstract(inverted_index: Dict)->str:
    """OpenAlex stores abstracts as an inverted index to save spaces.This helper rebuildes the text from {word: [pos1,pos2]}."""
    if not inverted_index:
        return ""
    
    word_list = []
    for word,positions in inverted_index.items():
        for pos in positions:
            word_list.append((pos,word))
    
    #sort by position to reconstryct the sentence
    sorted_words = sorted(word_list,key=lambda x:x[0])
    return " ".join([word for _, word in sorted_words])

def fetch_openalex_papers(query:str,max_results:int=5)->List[Dict]:
    """drectly hits the openalex api endpoint"""
    papers=[]
    params = {
        "search":query,
        "per_page":max_results,
        "filter": "from_publication_date:2020-01-01",
        "sort": "relevance_score:desc",
    }
    try:
        response = requests.get(OPENALEX_URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for item in data.get('results', []):
                
                # 1. Title
                title = item.get('display_name') or item.get('title') or "Untitled"
                
                # 2. Abstract (Reconstruct it)
                abstract = reconstruct_abstract(item.get('abstract_inverted_index'))
                if not abstract: 
                    abstract = "No abstract available."

                # 3. Authors
                authors = [a.get('author', {}).get('display_name') for a in item.get('authorships', [])]

                # 4. URL (Try PDF first, then DOI, then Landing Page)
                pdf_url = item.get('open_access', {}).get('oa_url')
                if not pdf_url:
                    pdf_url = item.get('doi')
                if not pdf_url:
                    pdf_url = item.get('id')

                papers.append({
                    "id": item.get('id'),
                    "source": "OpenAlex",
                    "title": title,
                    "published": str(item.get('publication_year')),
                    "authors": authors,
                    "summary": abstract,
                    "pdf_url": pdf_url,
                    # OpenAlex gives real citation counts
                    "citationcount": item.get('cited_by_count', 0) 
                })
    except Exception as e:
        print(f"⚠️ OpenAlex Error for '{query}': {e}")
        
    return papers

def fetch_arxiv_papers(query: str, max_results: int = 5) -> List[Dict]:
    """
    Uses the ArXiv API (via wrapper or direct endpoint).
    """
    papers = []
    client = arxiv.Client()
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        for result in client.results(search):
            papers.append({
                "id": result.entry_id.split("/abs/")[-1],
                "source": "ArXiv",
                "title": result.title.replace('\n', ' '),
                "published": result.published.strftime('%Y-%m-%d'),
                "authors": [a.name for a in result.authors],
                "summary": result.summary.replace('\n', ' '),
                "pdf_url": result.pdf_url,
                # ArXiv API doesn't provide citation counts, defaulting to 0
                "citationcount": 0 
            })
    except Exception as e:
        print(f"⚠️ ArXiv Error for '{query}': {e}")
        
    return papers

def fetch_recent_papers(queries :List[str],max_results: int =10)->List[Dict]:
    """The Main Function:
    1. Takes multiple search queries (e.g. "AI Agents", "LLM Security")
    2. Hits ArXiv AND OpenAlex for each.
    3. Merges and removes duplicates. """
    print(f"--- FETCHING: Searching {len(queries)} queries on ArXiv + OpenAlex ---")
    
    all_papers = []
    seen_titles = set()
    # Split the budget: if max=10, get ~5 from each to ensure diversity
    limit_per_source = max(3, int(max_results * 0.6)) 

    for query in queries:
        # Fetch from both
        arxiv_res = fetch_arxiv_papers(query, limit_per_source)
        openalex_res = fetch_openalex_papers(query, limit_per_source)
        
        # Combine
        combined = arxiv_res + openalex_res
        
        for paper in combined:
            # Simple normalization to catch duplicates like 
            # "Attention Is All You Need" vs "Attention is all you need"
            norm_title = paper['title'].lower().strip()[:60]
            
            if norm_title not in seen_titles:
                all_papers.append(paper)
                seen_titles.add(norm_title)

    print(f"✅ Fetch Complete: {len(all_papers)} unique papers found.")
    return all_papers