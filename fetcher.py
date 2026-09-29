import html
import logging
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import arxiv
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15
DEFAULT_MIN_YEAR = 2010
USER_AGENT = "DeepResearchAgent/1.0"

# Optional free API keys. Without them these services still work, but they rate-limit
# anonymous traffic, so results from them can be missing. Where to get them is in the README.
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
CORE_API_KEY = os.getenv("CORE_API_KEY", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")


class SourceError(Exception):
    """A database could not be searched (HTTP error, rate limit, bad response)."""


# ------------------ Helpers ------------------
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


def _strip_tags(text: Optional[str]) -> str:
    """Remove HTML/XML tags and entities from titles and abstracts."""
    return html.unescape(re.sub(r"<[^<]+?>", "", text or "")).strip()


def paper_year(paper: Dict) -> Optional[int]:
    """Extract a 4-digit year from a paper's 'published' field."""
    match = re.match(r"(\d{4})", str(paper.get("published") or ""))
    return int(match.group(1)) if match else None


def _get(url: str, params: Dict = None, headers: Dict = None) -> requests.Response:
    """GET a URL and raise SourceError with a readable reason when it fails."""
    all_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    try:
        res = requests.get(url, params=params, headers=all_headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        raise SourceError(f"connection failed ({type(e).__name__})") from e
    if res.status_code == 429:
        raise SourceError("rate limited")
    if res.status_code != 200:
        raise SourceError(f"HTTP {res.status_code}")
    return res


def _paper(source: str, **fields) -> Dict:
    """Build a paper record with every field present, so all sources share one shape."""
    paper = {
        "id": "", "doi": "", "source": source, "title": "Untitled", "summary": "No abstract available.",
        "authors": [], "published": "N/A", "venue": "", "pdf_url": "", "citationcount": 0,
    }
    paper.update({k: v for k, v in fields.items() if v not in (None, "", [])})
    return paper


# ------------------ 1. arXiv ------------------
def fetch_arxiv_papers(query: str, max_results: int, min_year: int) -> List[Dict]:
    papers = []
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    try:
        results = list(arxiv.Client().results(search))
    except Exception as e:
        raise SourceError(str(e)[:80]) from e
    for result in results:
        # The arXiv API has no date filter, so filter by year here
        if result.published.year < min_year:
            continue
        papers.append(_paper(
            "arXiv",
            id=result.entry_id.split("/abs/")[-1],
            doi=_clean_doi(result.doi),
            title=result.title.replace("\n", " "),
            summary=result.summary.replace("\n", " "),
            authors=[a.name for a in result.authors],
            published=result.published.strftime("%Y-%m-%d"),
            venue=result.journal_ref or "arXiv preprint",
            pdf_url=result.pdf_url,
        ))
    return papers


# ------------------ 2. OpenAlex ------------------
def fetch_openalex_papers(query: str, max_results: int, min_year: int) -> List[Dict]:
    params = {
        "search": query,
        "per_page": max_results,
        "filter": f"from_publication_date:{min_year}-01-01",
        "sort": "relevance_score:desc",
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    data = _get("https://api.openalex.org/works", params).json()

    papers = []
    for item in data.get("results", []):
        open_access = item.get("open_access") or {}
        venue_source = (item.get("primary_location") or {}).get("source") or {}
        papers.append(_paper(
            "OpenAlex",
            id=item.get("id"),
            doi=_clean_doi(item.get("doi")),
            title=item.get("display_name"),
            summary=reconstruct_abstract(item.get("abstract_inverted_index")),
            authors=_clean_authors((a.get("author") or {}).get("display_name") for a in item.get("authorships") or []),
            published=str(item.get("publication_year") or ""),
            venue=venue_source.get("display_name"),
            pdf_url=open_access.get("oa_url") or item.get("doi"),
            citationcount=item.get("cited_by_count") or 0,
        ))
    return papers


# ------------------ 3. Semantic Scholar ------------------
def fetch_semanticscholar_papers(query: str, max_results: int, min_year: int) -> List[Dict]:
    params = {
        "query": query,
        "limit": max_results,
        "year": f"{min_year}-",
        "fields": "title,abstract,authors,year,openAccessPdf,citationCount,url,venue,externalIds",
    }
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}

    # The anonymous pool is shared by everyone, so retry a few times with a growing delay
    for attempt in range(3):
        time.sleep(1 + attempt)
        try:
            data = _get("https://api.semanticscholar.org/graph/v1/paper/search", params, headers).json()
            break
        except SourceError as e:
            if str(e) != "rate limited" or attempt == 2:
                raise

    papers = []
    for item in data.get("data") or []:
        # openAccessPdf and externalIds are frequently null
        pdf_info = item.get("openAccessPdf") or {}
        external_ids = item.get("externalIds") or {}
        papers.append(_paper(
            "Semantic Scholar",
            id=item.get("paperId"),
            doi=_clean_doi(external_ids.get("DOI")),
            title=item.get("title"),
            summary=item.get("abstract"),
            authors=_clean_authors(a.get("name") for a in item.get("authors") or []),
            published=str(item.get("year") or ""),
            venue=item.get("venue"),
            pdf_url=pdf_info.get("url") or item.get("url"),
            citationcount=item.get("citationCount") or 0,
        ))
    return papers


# ------------------ 4. Crossref ------------------
def fetch_crossref_papers(query: str, max_results: int, min_year: int) -> List[Dict]:
    params = {
        "query": query,
        "rows": max_results,
        "filter": f"from-pub-date:{min_year}",
        "select": "DOI,title,abstract,author,issued,published-print,URL,is-referenced-by-count,container-title",
    }
    data = _get("https://api.crossref.org/works", params).json()

    papers = []
    for item in data.get("message", {}).get("items", []):
        date_info = item.get("published-print") or item.get("issued") or {}
        date_parts = date_info.get("date-parts") or [[None]]
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        papers.append(_paper(
            "Crossref",
            id=item.get("DOI"),
            doi=_clean_doi(item.get("DOI")),
            title=_strip_tags((item.get("title") or [""])[0]),
            summary=_strip_tags(item.get("abstract")),  # Crossref abstracts are JATS XML
            authors=_clean_authors(f"{a.get('given', '')} {a.get('family', '')}" for a in item.get("author") or []),
            published=str(year or ""),
            venue=(item.get("container-title") or [""])[0],
            pdf_url=item.get("URL"),
            citationcount=item.get("is-referenced-by-count") or 0,
        ))
    return papers


# ------------------ 5. PubMed ------------------
def _xml_text(element: Optional[ET.Element]) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def fetch_pubmed_papers(query: str, max_results: int, min_year: int) -> List[Dict]:
    """Biomedical and life-science literature from the US National Library of Medicine."""
    key = {"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}
    search = _get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        {"db": "pubmed", "term": f"({query}) AND {min_year}:3000[dp]", "retmax": max_results,
         "retmode": "json", "sort": "relevance", **key},
    ).json()
    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    # esearch only returns IDs; efetch returns the full records (with abstracts) as XML
    records = _get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        {"db": "pubmed", "id": ",".join(ids), "retmode": "xml", **key},
    )
    try:
        root = ET.fromstring(records.content)
    except ET.ParseError as e:
        raise SourceError("invalid XML response") from e

    papers = []
    for article in root.iter("PubmedArticle"):
        pmid = _xml_text(article.find(".//MedlineCitation/PMID"))
        abstract = " ".join(
            (f"{part.get('Label')}: " if part.get("Label") else "") + _xml_text(part)
            for part in article.findall(".//Abstract/AbstractText")
        )
        authors = []
        for author in article.findall(".//AuthorList/Author"):
            name = f"{_xml_text(author.find('ForeName'))} {_xml_text(author.find('LastName'))}".strip()
            authors.append(name or _xml_text(author.find("CollectiveName")))
        pub_date = article.find(".//Journal/JournalIssue/PubDate")
        year = _xml_text(pub_date.find("Year")) if pub_date is not None else ""
        if not year and pub_date is not None:
            year = _xml_text(pub_date.find("MedlineDate"))[:4]
        ids_by_type = {i.get("IdType"): _xml_text(i) for i in article.findall(".//PubmedData/ArticleIdList/ArticleId")}
        pmc = ids_by_type.get("pmc")

        papers.append(_paper(
            "PubMed",
            id=f"PMID:{pmid}",
            doi=ids_by_type.get("doi"),
            title=_xml_text(article.find(".//ArticleTitle")),
            summary=abstract,
            authors=_clean_authors(authors),
            published=year,
            venue=_xml_text(article.find(".//Journal/Title")),
            # Link to the free full text in PubMed Central when there is one
            pdf_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/" if pmc else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        ))
    return papers


# ------------------ 6. Europe PMC ------------------
def fetch_europepmc_papers(query: str, max_results: int, min_year: int) -> List[Dict]:
    """Life sciences: PubMed, PubMed Central and preprint servers such as bioRxiv and medRxiv."""
    data = _get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        {"query": f"({query}) AND PUB_YEAR:[{min_year} TO 3000]", "format": "json",
         "resultType": "core", "pageSize": max_results},
    ).json()

    papers = []
    for item in data.get("resultList", {}).get("result", []):
        urls = (item.get("fullTextUrlList") or {}).get("fullTextUrl", [])
        # Prefer an open-access PDF, then any open-access link, then the Europe PMC page
        open_urls = [u for u in urls if u.get("availabilityCode") in ("OA", "F")]
        pdf = next((u["url"] for u in open_urls if u.get("documentStyle") == "pdf"), None)
        link = pdf or (open_urls[0]["url"] if open_urls else None)
        is_preprint = item.get("source") == "PPR"
        journal = ((item.get("journalInfo") or {}).get("journal") or {}).get("title")
        papers.append(_paper(
            "Europe PMC",
            id=f"{item.get('source')}:{item.get('id')}",
            doi=_clean_doi(item.get("doi")),
            title=_strip_tags(item.get("title")),
            summary=_strip_tags(item.get("abstractText")),
            authors=_clean_authors((item.get("authorString") or "").rstrip(".").split(", ")),
            published=str(item.get("pubYear") or ""),
            venue="Preprint" if is_preprint else journal,
            pdf_url=link or f"https://europepmc.org/article/{item.get('source')}/{item.get('id')}",
            citationcount=item.get("citedByCount") or 0,
        ))
    return papers


# ------------------ 7. OpenAIRE ------------------
def fetch_openaire_papers(query: str, max_results: int, min_year: int) -> List[Dict]:
    """European open-science graph: publications from repositories, journals and funders worldwide."""
    data = _get(
        "https://api.openaire.eu/graph/v1/researchProducts",
        {"search": query, "type": "publication", "pageSize": max_results,
         "fromPublicationDate": f"{min_year}-01-01", "sortBy": "relevance DESC"},
    ).json()

    papers = []
    for item in data.get("results", []):
        dois = [p.get("value") for p in item.get("pids") or [] if p.get("scheme") == "doi"]
        urls = [url for inst in item.get("instances") or [] for url in inst.get("urls") or []]
        citations = ((item.get("indicators") or {}).get("citationImpact") or {}).get("citationCount")
        papers.append(_paper(
            "OpenAIRE",
            id=item.get("id"),
            doi=_clean_doi(dois[0]) if dois else "",
            title=_strip_tags(item.get("mainTitle")),
            summary=_strip_tags((item.get("descriptions") or [""])[0]),
            authors=_clean_authors(a.get("fullName") for a in item.get("authors") or []),
            published=item.get("publicationDate"),
            venue=(item.get("container") or {}).get("name") or item.get("publisher"),
            pdf_url=urls[0] if urls else "",
            citationcount=int(citations or 0),
        ))
    return papers


# ------------------ 8. CORE ------------------
def fetch_core_papers(query: str, max_results: int, min_year: int) -> List[Dict]:
    """The world's largest collection of open-access papers, aggregated from repositories."""
    headers = {"Authorization": f"Bearer {CORE_API_KEY}"} if CORE_API_KEY else {}
    data = _get(
        "https://api.core.ac.uk/v3/search/works",
        {"q": f"({query}) AND yearPublished>={min_year}", "limit": max_results},
        headers,
    ).json()

    papers = []
    for item in data.get("results") or []:
        journal = next((j.get("title") for j in item.get("journals") or [] if j.get("title")), None)
        display = next((l.get("url") for l in item.get("links") or [] if l.get("type") == "display"), None)
        papers.append(_paper(
            "CORE",
            id=f"CORE:{item.get('id')}",
            doi=_clean_doi(item.get("doi")),
            title=_strip_tags(item.get("title")),
            summary=_strip_tags(item.get("abstract")),
            authors=_clean_authors(a.get("name") for a in item.get("authors") or []),
            published=str(item.get("yearPublished") or ""),
            venue=journal or item.get("publisher"),
            pdf_url=item.get("downloadUrl") or display,
            citationcount=item.get("citationCount") or 0,
        ))
    return papers


# ------------------ 9. DOAJ ------------------
def fetch_doaj_papers(query: str, max_results: int, min_year: int) -> List[Dict]:
    """Directory of Open Access Journals: peer-reviewed, fully open-access journals.
    The DOAJ API has no date filter; min_year is applied afterwards in fetch_recent_papers."""
    # Require every word; DOAJ otherwise matches any single word of the query
    terms = " AND ".join(re.sub(r"[^\w\s-]", " ", query).split())
    data = _get(f"https://doaj.org/api/search/articles/{requests.utils.quote(terms)}", {"pageSize": max_results}).json()

    papers = []
    for result in data.get("results", []):
        bib = result.get("bibjson") or {}
        doi = next((i.get("id") for i in bib.get("identifier") or [] if i.get("type") == "doi"), "")
        links = [l.get("url") for l in bib.get("link") or [] if l.get("url")]
        papers.append(_paper(
            "DOAJ",
            id=result.get("id"),
            doi=_clean_doi(doi),
            title=_strip_tags(bib.get("title")),
            summary=_strip_tags(bib.get("abstract")),
            authors=_clean_authors(a.get("name") for a in bib.get("author") or []),
            published=str(bib.get("year") or ""),
            venue=(bib.get("journal") or {}).get("title"),
            pdf_url=links[0] if links else (f"https://doi.org/{_clean_doi(doi)}" if doi else ""),
        ))
    return papers


# ------------------ Source registry ------------------
SOURCES: Dict[str, Callable[[str, int, int], List[Dict]]] = {
    "arXiv": fetch_arxiv_papers,
    "OpenAlex": fetch_openalex_papers,
    "Semantic Scholar": fetch_semanticscholar_papers,
    "Crossref": fetch_crossref_papers,
    "PubMed": fetch_pubmed_papers,
    "Europe PMC": fetch_europepmc_papers,
    "OpenAIRE": fetch_openaire_papers,
    "CORE": fetch_core_papers,
    "DOAJ": fetch_doaj_papers,
}

# Shown in the UI so users know what each database covers
SOURCE_INFO: Dict[str, Dict[str, str]] = {
    "arXiv": {"covers": "Preprints in CS, physics, maths, statistics and more", "url": "https://arxiv.org"},
    "OpenAlex": {"covers": "250M+ works across every field, with citation counts", "url": "https://openalex.org"},
    "Semantic Scholar": {"covers": "200M+ papers, strongest in CS and biomedicine", "url": "https://www.semanticscholar.org"},
    "Crossref": {"covers": "DOI records from most journal publishers", "url": "https://www.crossref.org"},
    "PubMed": {"covers": "36M+ biomedical and life-science citations", "url": "https://pubmed.ncbi.nlm.nih.gov"},
    "Europe PMC": {"covers": "Life sciences, including bioRxiv and medRxiv preprints", "url": "https://europepmc.org"},
    "OpenAIRE": {"covers": "European open-science graph of repositories and journals", "url": "https://explore.openaire.eu"},
    "CORE": {"covers": "The largest collection of open-access papers", "url": "https://core.ac.uk"},
    "DOAJ": {"covers": "Peer-reviewed, fully open-access journals", "url": "https://doaj.org"},
}


def _merge_duplicate(kept: Dict, duplicate: Dict):
    """Fill gaps in an already-kept paper with data from a duplicate found in another source."""
    kept["citationcount"] = max(kept.get("citationcount") or 0, duplicate.get("citationcount") or 0)
    for field in ("doi", "venue", "pdf_url"):
        if not kept.get(field) and duplicate.get(field):
            kept[field] = duplicate[field]
    if str(kept.get("summary", "")).startswith("No abstract") and not str(duplicate.get("summary", "")).startswith("No abstract"):
        kept["summary"] = duplicate["summary"]
    also_in = kept.setdefault("also_in", [])
    if duplicate["source"] != kept["source"] and duplicate["source"] not in also_in:
        also_in.append(duplicate["source"])


def _run_source(name: str, query: str, limit: int, min_year: int) -> Tuple[str, List[Dict], Optional[str]]:
    try:
        return name, SOURCES[name](query, limit, min_year), None
    except Exception as e:
        logger.warning("%s failed: %s", name, e)
        return name, [], str(e) if isinstance(e, SourceError) else f"error ({type(e).__name__})"


# ------------------ Main fetch function ------------------
def fetch_recent_papers(
    queries: List[str],
    max_results: int = 10,
    sources: Optional[List[str]] = None,
    min_year: Optional[int] = None,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Search the selected databases (all by default) and merge duplicates by title.

    Returns (papers, stats) where stats[source] = {"found": int, "error": str | None}.
    """
    queries = [q for q in queries if q]
    selected = [name for name in (sources or SOURCES) if name in SOURCES]
    if not queries or not selected:
        return [], {}
    min_year = int(min_year or DEFAULT_MIN_YEAR)

    # Keep the total number of fetched papers roughly constant however many databases are selected
    limit_per_source = max(3, math.ceil(int(max_results) * 6 / (len(queries) * len(selected))))
    logger.info("Fetching: %d queries x %d sources, %d each", len(queries), len(selected), limit_per_source)

    stats = {name: {"found": 0, "error": None} for name in selected}
    all_papers: List[Dict] = []
    # Databases are queried in parallel; queries run one after another so
    # rate-limited services are not hit by several concurrent requests.
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        for query in queries:
            for name, papers, error in pool.map(lambda n: _run_source(n, query, limit_per_source, min_year), selected):
                stats[name]["found"] += len(papers)
                if error:
                    stats[name]["error"] = error
                all_papers.extend(papers)

    # Only report an error when a database returned nothing at all
    for entry in stats.values():
        if entry["found"]:
            entry["error"] = None

    unique_papers: Dict[str, Dict] = {}
    for paper in all_papers:
        year = paper_year(paper)
        if year and year < min_year:
            continue
        norm_title = normalize_title(paper.get("title"))[:60]
        if not norm_title or norm_title == "untitled":
            continue
        if norm_title in unique_papers:
            _merge_duplicate(unique_papers[norm_title], paper)
        else:
            unique_papers[norm_title] = paper

    logger.info("Fetch complete: %d unique papers", len(unique_papers))
    return list(unique_papers.values()), stats
