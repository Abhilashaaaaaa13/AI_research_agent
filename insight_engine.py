import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import pymupdf
import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
logger = logging.getLogger(__name__)

# Both defaults are available on the Gemini API free tier. Free-tier rate limits are per model,
# so splitting work between two models roughly doubles the requests available per minute.
MAIN_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")  # trends, gaps, roadmap, summary, chat
FAST_MODEL = os.getenv("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite")  # query refinement, keywords, scoring
MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_RETRIES = 2
# Every failed LLM call returns text starting with this marker, so callers can tell
# an error apart from real output and avoid saving it as a result.
LLM_ERROR_PREFIX = "[Gemini unavailable] "


# ------------------ LLM Clients ------------------
def _chat_model(model: str, api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        # Google recommends the default temperature of 1.0 for Gemini 3 models
        temperature=1.0,
        # Fail fast on rate limits so the fallback model can take over
        max_retries=1,
        timeout=120,
    )


def load_models():
    """Return (main_llm, fast_llm). Each falls back to the other model if a call fails,
    which mostly happens when one model hits its free-tier rate limit.
    Returns (None, None) when no API key is configured."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not found")
        return None, None

    main = _chat_model(MAIN_MODEL, api_key)
    fast = _chat_model(FAST_MODEL, api_key)
    return main.with_fallbacks([fast]), fast.with_fallbacks([main])


def content_to_text(content: Any) -> str:
    """Gemini may return either a string or a list of content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def generate_response(prompt: str, llm, json_mode: bool = False) -> str:
    """Call the LLM and return cleaned text."""
    if not llm:
        return LLM_ERROR_PREFIX + "The language model is not configured. Set GOOGLE_API_KEY and restart the app."
    try:
        messages = [
            SystemMessage(content="You must follow formatting instructions EXACTLY."),
            HumanMessage(content=prompt),
        ]
        content = content_to_text(_invoke_with_retry(llm, messages).content).strip()
        if json_mode:
            content = content.replace("```json", "").replace("```", "").strip()
        return content
    except Exception as e:
        logger.error("LLM error: %s", e)
        if _is_rate_limit(e):
            return LLM_ERROR_PREFIX + "The free-tier rate limit was reached. Please wait a minute and try again."
        if _is_connection_error(e):
            return LLM_ERROR_PREFIX + "Could not reach the Gemini API. Check your internet connection and try again."
        return LLM_ERROR_PREFIX + f"{e}"


def is_llm_error(text) -> bool:
    return isinstance(text, str) and text.startswith(LLM_ERROR_PREFIX)


def _is_connection_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(s in text for s in ("getaddrinfo", "connection", "timed out", "name resolution", "network"))


def _is_rate_limit(error: Exception) -> bool:
    text = str(error)
    return "429" in text or "ResourceExhausted" in type(error).__name__ or "quota" in text.lower()


def _invoke_with_retry(llm, messages):
    """Invoke the LLM, waiting and retrying when the per-minute rate limit is hit."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if not _is_rate_limit(e) or attempt == MAX_RETRIES:
                raise
            # Gemini includes the suggested wait time in the error message
            match = re.search(r"retry in ([\d.]+)s", str(e))
            delay = min(float(match.group(1)) + 1 if match else 20.0 * (attempt + 1), 65.0)
            logger.info("Rate limited, retrying in %.0fs", delay)
            time.sleep(delay)


def parse_json_list(text: str) -> Optional[list]:
    """Extract the first JSON array found in an LLM response."""
    match = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, list) else None
    except json.JSONDecodeError:
        return None


def _title(row) -> str:
    return str(row.get("title") or row.get("Title") or "Untitled")


def _summary(row) -> str:
    return str(row.get("summary") or row.get("Summary") or "")


# ------------------ PDF Utilities ------------------
def get_full_text(pdf_url: Optional[str]) -> Optional[str]:
    """Download a PDF and return the text of its last 20% (limitations / future work usually live there)."""
    if not pdf_url or not isinstance(pdf_url, str):
        return None
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(pdf_url, stream=True, timeout=15, headers=headers)
        if "application/pdf" not in res.headers.get("Content-Type", ""):
            return None
        # Skip very large files before downloading the body
        if int(res.headers.get("Content-Length") or 0) > MAX_PDF_BYTES:
            return None

        with pymupdf.open(stream=res.content, filetype="pdf") as doc:
            start_page = int(doc.page_count * 0.80)
            return "".join(doc[i].get_text() for i in range(start_page, doc.page_count))
    except Exception:
        return None


# ------------------ Keyword Utilities ------------------
def extract_keywords(topic: str, llm, max_terms: int = 6) -> List[str]:
    prompt = f"""
Extract {max_terms} specific keywords for: "{topic}"

RULES:
- Adapt to the specific domain (Physics, CS, Medicine, History, etc.).
- Include ONLY the most critical domain nouns.
- Do NOT include generic words like "system", "approach", "using".
- Respond ONLY with a JSON array of lowercase strings.

Example (Physics): ["quantum", "entanglement", "superposition"]
Example (Medicine): ["cardiology", "arrhythmia", "atrial"]
"""
    keywords = parse_json_list(generate_response(prompt, llm, json_mode=True))
    if keywords:
        return [str(k).lower().strip() for k in keywords if str(k).strip()]
    return [w.lower() for w in topic.split() if len(w) > 3][:max_terms] or [topic.lower()]


def keyword_filter(papers_df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
    """Keep papers whose title or abstract mentions at least one keyword."""
    if papers_df.empty:
        return papers_df

    def match(row):
        text = (_title(row) + " " + _summary(row)).lower()
        return any(kw in text for kw in keywords)

    filtered = papers_df[papers_df.apply(match, axis=1)].reset_index(drop=True)

    # If the filter is too strict and removes everything, fall back to the full list
    if filtered.empty:
        logger.info("Keyword filter removed every paper; keeping all fetched papers")
        return papers_df

    return filtered


# ------------------ LLM-Based Relevance Scoring ------------------
def filter_papers_with_llm(papers_df: pd.DataFrame, topic: str, client, top_n: int = 20) -> pd.DataFrame:
    """Score papers 0-10 for relevance and return them with an 'llm_score' column."""
    if papers_df.empty:
        papers_df["llm_score"] = 0.0
        return papers_df

    top_papers = papers_df.head(top_n).reset_index(drop=True).copy()

    paper_context = "\n\n".join(
        f"ID:{i}\nTitle:{_title(row)}\nAbstract:{_summary(row)[:600]}"
        for i, (_, row) in enumerate(top_papers.iterrows())
    )

    prompt = f"""
You are a research relevance evaluator.
Topic: "{topic}"

TASK:
Score each paper (0-10) based on relevance to the topic.

SCORING:
- 0-3: Paper is about a completely different field or too generic.
- 4-7: Paper is related but not a direct match.
- 8-10: Paper is a precise match for the user's specific query.

RESPONSE FORMAT:
Output ONLY a JSON array: [{{ "ID": 0, "Score": 9 }}, {{ "ID": 1, "Score": 0 }}]

PAPERS:
{paper_context}
"""

    result = parse_json_list(generate_response(prompt, client, json_mode=True))
    if not result:
        logger.warning("LLM scoring failed; using a neutral score")
        top_papers["llm_score"] = 5.0
        return top_papers

    score_map = {}
    for item in result:
        try:
            score_map[int(item["ID"])] = float(item["Score"])
        except (KeyError, TypeError, ValueError):
            continue

    top_papers["llm_score"] = [score_map.get(i, 0.0) for i in range(len(top_papers))]
    return top_papers.sort_values(by="llm_score", ascending=False).reset_index(drop=True)


# ------------------ Topic Query Refinement ------------------
def refine_topic_query(raw_topic: str, llm) -> List[str]:
    prompt = f"""
You are a research query optimizer.
Task: Create 4 specific, technical search queries for: "{raw_topic}"

RULES:
- Adapt to the specific domain (e.g., if Physics, use physics terminology; if AI, use CS terminology).
- Ensure queries cover "State of the Art", "Review/Survey", and "Specific Implementations".
- Output ONLY a valid JSON list of strings.
"""
    queries = parse_json_list(generate_response(prompt, llm, json_mode=True))
    if queries:
        return [str(q).strip() for q in queries if str(q).strip()]
    return [raw_topic]


# ------------------ Trend Analysis ------------------
def find_trends(top_papers_df: pd.DataFrame, llm, num_trends: int = 3) -> str:
    """Extract trends supported by the ranked papers."""
    if top_papers_df.empty:
        return "No papers available to analyze trends."

    paper_context = "\n\n".join(
        f"- {_title(row)}: {_summary(row)[:700]}" for _, row in top_papers_df.iterrows()
    )

    prompt = f"""
You are an expert research analyst.
TASK: Identify exactly {num_trends} key trends based ONLY on the provided papers.

REQUIREMENTS:
1. **Source**: Derive trends ONLY from the papers below. Do not use outside knowledge.
2. **Detail**: Write a distinct, high-quality description (150-300 words) for EACH trend.
3. **Citations**: Cite the specific papers that support each trend.

FORMAT (Markdown):
### 1. Trend Name
**Description:** detailed explanation...

**Derived from:** list of paper titles

PAPERS TO ANALYZE:
{paper_context}
"""
    return generate_response(prompt, llm)


# ------------------ Gap Identification ------------------
def _analyze_gaps(paper: Dict, llm, num_gaps: int) -> Dict:
    """Find the gaps in one paper, using its full PDF when it can be downloaded."""
    text = get_full_text(paper.get("pdf_url"))
    used_pdf = bool(text)
    if not text:
        text = _summary(paper)[:3000]

    title = _title(paper)
    prompt = f"""
You are a critical peer reviewer evaluating a research paper.
TASK: Identify {num_gaps} **specific technical limitations, methodological flaws, or unaddressed scopes** in this paper.

RULES:
1. **Precision**: Do not say "it is slow". Say "suffers from **high latency** in real-time scenarios".
2. **Highlighting**: **Bold** the key technical terms that represent the gap.
3. **Context**: Explain WHY this is a gap based on the provided text.

FORMAT:
- **Gap 1**: description with **bold keywords**
- **Gap 2**: description with **bold keywords**

PAPER TO REVIEW:
Title: {title}
Content: {text[:4000]}
"""
    return {
        "source": title,
        "gaps": generate_response(prompt, llm),
        "pdf_url": paper.get("pdf_url"),
        "used_pdf": used_pdf,
    }


def find_gaps_with_citations(top_papers_df: pd.DataFrame, llm, num_gaps: int = 2) -> List[Dict]:
    """Analyze the top 3 papers in parallel (PDF downloads dominate the running time)."""
    papers = top_papers_df.head(3).to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=len(papers) or 1) as pool:
        return list(pool.map(lambda paper: _analyze_gaps(paper, llm, num_gaps), papers))


# ------------------ Roadmap Generation ------------------
def suggest_roadmap(topic: str, gaps: List[Dict], llm) -> str:
    gaps_text = "\n\n".join(f"{item['source']}:\n{item['gaps']}" for item in gaps)

    prompt = f"""
You are a senior research advisor. Create a 4-phase research roadmap for the topic "{topic}"
that addresses the gaps below.

FORMAT (Markdown):
### Phase 1: Title
- **Goal:** one sentence
- **Key tasks:** 2-4 bullet points
- **Addresses gap:** which gap(s) this tackles
- **Expected outcome:** one sentence

(Repeat for phases 2-4.)

GAPS:
{gaps_text}
"""
    return generate_response(prompt, llm)


# ------------------ Final Summary ------------------
def generate_final_summary(result: Dict, llm) -> str:
    gaps = result.get("gaps") or []
    gaps_text = "\n\n".join(f"{g['source']}:\n{g['gaps']}" for g in gaps) if isinstance(gaps, list) else str(gaps)

    prompt = f"""
Write an executive research summary in Markdown for the topic "{result.get('topic')}".

Use these sections:
### Overview
### Key Trends
### Open Problems
### Recommended Next Steps

Keep it concise (400-600 words) and grounded in the material below.

TRENDS:
{result.get('trends')}

GAPS:
{gaps_text}

ROADMAP:
{result.get('final_plan')}
"""
    return generate_response(prompt, llm)


# ------------------ Paper Comparison ------------------
def compare_papers(papers: List[Dict], topic: str, llm) -> str:
    """Build a side-by-side comparison of 2-4 papers."""
    if len(papers) < 2:
        return "Select at least two papers to compare."

    paper_context = "\n\n".join(
        f"[{i}] {_title(p)} ({p.get('published', 'N/A')}, {p.get('venue') or p.get('source', '')})\n"
        f"Abstract: {_summary(p)[:1500]}"
        for i, p in enumerate(papers, start=1)
    )

    prompt = f"""
You are a research analyst comparing papers on "{topic}".

TASK: Compare the papers below using ONLY their abstracts.

FORMAT (Markdown):
1. A table with one column per paper (header: [number] and a short title) and these rows:
   Problem, Approach / Method, Data or Setting, Key Results, Limitations, Best suited for
   Keep each cell under 25 words. Write "Not stated" when the abstract does not say.
2. A section "### Key differences" with 3-5 bullet points.
3. A section "### Which to read first" with one short paragraph.

PAPERS:
{paper_context}
"""
    return generate_response(prompt, llm)


# ------------------ Research Chatbot ------------------
def answer_user_query(user_query: str, papers: List[Dict], llm, session: Optional[Dict] = None) -> str:
    if not papers:
        return "No papers are available for reference yet."

    source_context = "\n".join(
        f"[{i}] {_title(p)}\n"
        f"Published: {p.get('published', 'N/A')} | Source: {p.get('source', 'N/A')} | "
        f"Citations: {int(float(p.get('citationcount') or 0))} | "
        f"Authors: {', '.join(map(str, (p.get('authors') or [])[:5]))}\n"
        f"Abstract: {_summary(p)}\n---"
        for i, p in enumerate(papers, start=1)
    )

    # Include whatever analysis has already been produced in this session
    findings = []
    session = session or {}
    if session.get("trends"):
        findings.append(f"TRENDS:\n{session['trends']}")
    if session.get("gaps"):
        findings.append("GAPS:\n" + "\n".join(f"{g['source']}: {g['gaps']}" for g in session["gaps"]))
    if session.get("roadmap"):
        findings.append(f"ROADMAP:\n{session['roadmap']}")
    findings_text = "\n\n".join(findings) or "None yet."

    prompt = f"""
You are a factual academic assistant.
USER QUESTION: "{user_query}"

PAPERS (cite as [number]):
{source_context}

SESSION FINDINGS:
{findings_text}

Answer concisely using ONLY the material above. Cite papers by their [number] and title.
If the answer is not in the material, say so.
"""
    return generate_response(prompt, llm)
