import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional

import arxiv
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OPENALEX_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_URL = "https://api.crossref.org/works"
REQUEST_TIMEOUT = 10

# Optional free API keys. Both services rate-limit anonymous traffic heavily, so a key
# makes results from these two sources much more reliable. Get them at:
#   https://openalex.org/rest-api  and  https://www.semanticscholar.org/product/api
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
DEFAULT_MIN_YEAR = 2010


def reconstruct_abstract(inverted_index: Dict) -> str:
    """Rebuild an abstract from an OpenAlex inverted index."""
    if not inverted_index:
        return ""
    word_list = [(pos, word) for word, positions in inverted_index.items() for pos in positions]
    return " ".join(word for _, word in sorted(word_list, key=lambda x: x[0]))


def normalize_title(title: str) -> str:
    """Strip symbols and casing so near-identical titles can be deduplicated."""
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(title).lower())


def _clean_authors(names: Iterable) -> List[str]:
    return [str(n).strip() for n in names if n and str(n).strip()]


def _clean_doi(doi: Optional[str]) -> str:
    if not doi:
        return ""
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", str(doi)).strip()


def paper_year(paper: Dict) -> Optional[int]:
    """Extract a 4-digit year from a paper's 'published' field."""
    match = re.match(r"(\d{4})", str(paper.get("published") or ""))
    return int(match.group(1)) if match else None


# --- 1. Crossref ---
def fetch_crossref_papers(query: str, max_results: int = 5, min_year: int = DEFAULT_MIN_YEAR) -> List[Dict]:
    papers = []
    params = {
        "query": query,
        "rows": max_results,
        "filter": f"from-pub-date:{min_year}",
        "select": "DOI,title,abstract,author,issued,published-print,URL,is-referenced-by-count,container-title",
    }
    try:
        # Crossref asks clients to identify themselves with a User-Agent
        headers = {"User-Agent": "DeepResearchAgent/1.0"}
        res = requests.get(CROSSREF_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        if res.status_code != 200:
            logger.warning("Crossref returned HTTP %s", res.status_code)
            return papers

        for item in res.json().get("message", {}).get("items", []):
            title_list = item.get("title") or []
            venue_list = item.get("container-title") or []

            # Crossref abstracts are JATS XML; strip the tags
            abstract = item.get("abstract") or "No abstract available."
            abstract = re.sub(r"<[^<]+?>", "", abstract).strip()

            date_info = item.get("published-print") or item.get("issued") or {}
            date_parts = date_info.get("date-parts") or [[None]]
            year = date_parts[0][0] if date_parts and date_parts[0] else None

            papers.append({
                "id": item.get("DOI"),
                "doi": _clean_doi(item.get("DOI")),
                "source": "Crossref",
                "title": title_list[0] if title_list else "Untitled",
                "summary": abstract,
                "authors": _clean_authors(
                    f"{a.get('given', '')} {a.get('family', '')}" for a in item.get("author") or []
                ),
                "published": str(year) if year else "N/A",
                "venue": venue_list[0] if venue_list else "",
                "pdf_url": item.get("URL"),
                "citationcount": item.get("is-referenced-by-count", 0),
            })
    except Exception as e:
        logger.warning("Crossref error: %s", e)

    return papers


# --- 2. Semantic Scholar (with retry on rate limits) ---
def fetch_semanticscholar_papers(query: str, max_results: int = 5, min_year: int = DEFAULT_MIN_YEAR) -> List[Dict]:
    papers = []
    params = {
        "query": query,
        "limit": max_results,
        "year": f"{min_year}-",
        "fields": "title,abstract,authors,year,openAccessPdf,citationCount,url,venue,externalIds",
    }

    for attempt in range(3):
        try:
            # Back off a little more on each attempt (1s, 2s, 3s)
            time.sleep(1 + attempt)
            headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
            res = requests.get(SEMANTIC_SCHOLAR_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT)

            if res.status_code == 429:
                logger.info("Semantic Scholar rate limited, retry %d/3", attempt + 1)
                continue

            if res.status_code == 200:
                for item in res.json().get("data") or []:
                    # openAccessPdf and externalIds are frequently null, so guard before calling .get
                    pdf_info = item.get("openAccessPdf") or {}
                    external_ids = item.get("externalIds") or {}
                    papers.append({
                        "id": item.get("paperId"),
                        "doi": _clean_doi(external_ids.get("DOI")),
                        "source": "Semantic Scholar",
                        "title": item.get("title") or "Untitled",
                        "summary": item.get("abstract") or "No abstract available.",
                        "authors": _clean_authors(a.get("name") for a in item.get("authors") or []),
                        "published": str(item.get("year") or "N/A"),
                        "venue": item.get("venue") or "",
                        "pdf_url": pdf_info.get("url") or item.get("url"),
                        "citationcount": item.get("citationCount") or 0,
                    })
            else:
                logger.warning("Semantic Scholar returned HTTP %s", res.status_code)
            break

        except Exception as e:
            logger.warning("Semantic Scholar error: %s", e)
            break

    return papers


# --- 3. OpenAlex ---
def fetch_openalex_papers(query: str, max_results: int = 5, min_year: int = DEFAULT_MIN_YEAR) -> List[Dict]:
    papers = []
    params = {
        "search": query,
        "per_page": max_results,
        "filter": f"from_publication_date:{min_year}-01-01",
        "sort": "relevance_score:desc",
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    try:
        response = requests.get(OPENALEX_URL, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            logger.warning("OpenAlex returned HTTP %s", response.status_code)
            return papers

        for item in response.json().get("results", []):
            abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
            open_access = item.get("open_access") or {}
            venue_source = (item.get("primary_location") or {}).get("source") or {}
            papers.append({
                "id": item.get("id"),
                "doi": _clean_doi(item.get("doi")),
                "source": "OpenAlex",
                "title": item.get("display_name") or "Untitled",
                "summary": abstract or "No abstract available.",
                "authors": _clean_authors(
                    (a.get("author") or {}).get("display_name") for a in item.get("authorships") or []
                ),
                "published": str(item.get("publication_year") or "N/A"),
                "venue": venue_source.get("display_name") or "",
                "pdf_url": open_access.get("oa_url") or item.get("doi"),
                "citationcount": item.get("cited_by_count") or 0,
            })
    except Exception as e:
        logger.warning("OpenAlex error: %s", e)
    return papers


# --- 4. arXiv ---
def fetch_arxiv_papers(query: str, max_results: int = 5, min_year: int = DEFAULT_MIN_YEAR) -> List[Dict]:
    papers = []
    client = arxiv.Client()
    try:
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        for result in client.results(search):
            # The arXiv API has no date filter, so filter by year here
            if result.published.year < min_year:
                continue
            papers.append({
                "id": result.entry_id.split("/abs/")[-1],
                "doi": _clean_doi(result.doi),
                "source": "arXiv",
                "title": result.title.replace("\n", " "),
                "summary": result.summary.replace("\n", " "),
                "authors": [a.name for a in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "venue": result.journal_ref or "arXiv preprint",
                "pdf_url": result.pdf_url,
                "citationcount": 0,
            })
    except Exception as e:
        logger.warning("arXiv error: %s", e)
    return papers


SOURCES: Dict[str, Callable[..., List[Dict]]] = {
    "arXiv": fetch_arxiv_papers,
    "OpenAlex": fetch_openalex_papers,
    "Semantic Scholar": fetch_semanticscholar_papers,
    "Crossref": fetch_crossref_papers,
}


def _merge_duplicate(kept: Dict, duplicate: Dict):
    """Fill gaps in an already-kept paper with data from a duplicate found in another source."""
    kept["citationcount"] = max(kept.get("citationcount") or 0, duplicate.get("citationcount") or 0)
    for field in ("doi", "venue", "pdf_url"):
        if not kept.get(field) and duplicate.get(field):
            kept[field] = duplicate[field]
    if str(kept.get("summary", "")).startswith("No abstract") and duplicate.get("summary"):
        kept["summary"] = duplicate["summary"]
    also_in = kept.setdefault("also_in", [])
    if duplicate["source"] != kept["source"] and duplicate["source"] not in also_in:
        also_in.append(duplicate["source"])


# --- Main fetch function ---
def fetch_recent_papers(
    queries: List[str],
    max_results: int = 10,
    sources: Optional[List[str]] = None,
    min_year: Optional[int] = None,
) -> List[Dict]:
    """Search the selected sources (all four by default), then merge duplicates by title."""
    queries = [q for q in queries if q]
    selected = [SOURCES[name] for name in (sources or SOURCES) if name in SOURCES]
    if not queries or not selected:
        return []
    min_year = int(min_year or DEFAULT_MIN_YEAR)

    # Spread the requested budget across queries, with a sensible floor per source
    limit_per_source = max(5, math.ceil(int(max_results) / len(queries)) * 2)
    logger.info("Fetching: %d queries x %d sources", len(queries), len(selected))

    all_papers: List[Dict] = []
    # Sources are queried in parallel; queries run one after another so
    # Semantic Scholar's shared rate limit is not hit by concurrent requests.
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        for query in queries:
            futures = [pool.submit(source, query, limit_per_source, min_year) for source in selected]
            for future in futures:
                all_papers.extend(future.result())

    unique_papers: Dict[str, Dict] = {}
    for paper in all_papers:
        year = paper_year(paper)
        if year and year < min_year:
            continue
        norm_title = normalize_title(paper.get("title"))[:60]
        if not norm_title:
            continue
        if norm_title in unique_papers:
            _merge_duplicate(unique_papers[norm_title], paper)
        else:
            unique_papers[norm_title] = paper

    logger.info("Fetch complete: %d unique papers", len(unique_papers))
    return list(unique_papers.values())
