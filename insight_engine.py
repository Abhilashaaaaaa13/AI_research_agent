import os
import requests
import fitz  # PyMuPDF
from dotenv import load_dotenv
import pandas as pd
import json
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# --- IMPORT HELPERS ---
try:
    import fetcher
    import ranking_engine
except ImportError:
    print("⚠️ Warning: fetcher.py or ranking_engine.py not found.")

load_dotenv()

# ------------------ LLM Client ------------------
def load_model():
    """Connects to Google Gemini API."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found!")
        return None

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
    )

def generate_response(prompt: str, llm, json_mode: bool = False) -> str:
    """Generate and clean LLM responses."""
    if not llm:
        return "LLM Client not initialized correctly."
    try:
        messages = [SystemMessage(content="You must follow formatting instructions EXACTLY."),
                    HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        content = response.content.strip()

        if json_mode:
            content = content.replace("```json", "").replace("```", "").strip()
        return content

    except Exception as e:
        return f"Error generating response: {e}"


# ------------------ PDF Utilities ------------------
def get_full_text(pdf_url: str) -> str:
    """Extracts last 20% text from PDF."""
    if not pdf_url:
        return None
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(pdf_url, stream=True, timeout=15, headers=headers)

        if "application/pdf" not in res.headers.get("Content-Type", ""):
            return None

        doc = fitz.open(stream=res.content, filetype="pdf")
        start_page = int(doc.page_count * 0.80)
        text = "".join(doc[i].get_text() for i in range(start_page, doc.page_count))
        return text

    except Exception:
        return None

# ------------------ Keyword Utilities (UNIVERSAL) ------------------
def extract_keywords(topic: str, llm, max_terms: int = 6) -> List[str]:
    # Updated prompt to handle ANY domain
    prompt = f"""
Extract {max_terms} specific keywords for: "{topic}"

RULES:
- Adapt to the specific domain (Physics, CS, Medicine, History, etc.).
- Include ONLY the most critical domain nouns.
- Do NOT include generic words like "system", "approach", "using".
- Respond ONLY as a Python list of strings.

Example (Physics): ["quantum", "entanglement", "superposition"]
Example (Medicine): ["cardiology", "arrhythmia", "atrial"]
"""
    try:
        keywords = json.loads(generate_response(prompt, llm, json_mode=True))
        return [k.lower().strip() for k in keywords]
    except:
        return [topic.split()[0].lower()]

def keyword_filter(papers_df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
    if papers_df.empty:
        return papers_df
    
    def match(row):
        t = str(row.get('title', row.get('Title', '')))
        s = str(row.get('summary', row.get('Summary', '')))
        text = (t + " " + s).lower()
        return any(kw in text for kw in keywords)
    
    filtered = papers_df[papers_df.apply(match, axis=1)].reset_index(drop=True)
    
    # --- CRITICAL FIX: SAFETY FALLBACK ---
    # If the filter removes ALL papers (too strict), return the original list.
    if filtered.empty:
        print("⚠️ Keyword filter was too strict. Returning all fetched papers.")
        return papers_df
        
    return filtered

# ------------------ LLM-Based Paper Filtering (UNIVERSAL) ------------------
def filter_papers_with_llm(papers_df: pd.DataFrame, topic: str, client, top_n: int = 20):
    """Rate papers and return dataframe with 'llm_score'."""
    if papers_df.empty:
        papers_df['llm_score'] = 0.0
        return papers_df.head(top_n)
    
    top_papers = papers_df.head(top_n).reset_index(drop=True).copy()

    paper_context = "\n\n".join(
        [f"ID:{i}\nTitle:{row.get('title', row.get('Title', 'Untitled'))}\nAbstract:{str(row.get('summary', row.get('Summary', '')))[:600]}"
         for i, (_, row) in enumerate(top_papers.iterrows())]
    )

    # Updated Prompt to be Domain-Agnostic
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

    try:
        result_text = generate_response(prompt, client, json_mode=True)
        result = json.loads(result_text)
        score_map = {item['ID']: item['Score'] for item in result}
        
        top_papers['llm_score'] = top_papers.index.map(score_map).fillna(0)
        top_papers = top_papers.sort_values(by='llm_score', ascending=False).reset_index(drop=True)
        return top_papers.head(top_n)

    except Exception as e:
        print(f"⚠️ Error in LLM Scoring: {e}")
        top_papers['llm_score'] = 5.0
        return top_papers.head(top_n)


# ------------------ Topic Query Refinement (UNIVERSAL) ------------------
def refine_topic_query(raw_topic: str, llm) -> List[str]:
    prompt = f"""
You are a research query optimizer.
Task: Create 4 specific, technical search queries for: "{raw_topic}"

RULES:
- Adapt to the specific domain (e.g., if Physics, use physics terminology; if AI, use CS terminology).
- Ensure queries cover "State of the Art", "Review/Survey", and "Specific Implementations".
- Output ONLY a valid JSON list of strings.
"""
    try:
        queries = json.loads(generate_response(prompt, llm, json_mode=True))
        return [q.strip() for q in queries]
    except Exception:
        return [raw_topic]

# ------------------ Trend Analysis ------------------
# --- Trend Analysis (UPDATED) ---
def find_trends(top_papers_df: pd.DataFrame, llm, num_trends: int = 3) -> str:
    """
    Extracts specific trends from the provided papers.
    """
    if top_papers_df.empty:
        return "No papers available to analyze trends."

    # We only use the papers we already have (Ranked Papers)
    paper_context = "\n\n".join(
        [f"- {row.get('title', row.get('Title', 'Untitled'))}: {str(row.get('summary', row.get('Summary', '')))[:700]}" 
         for _, row in top_papers_df.iterrows()]
    )

    prompt = f"""
You are an expert research analyst.
TASK: Identify exactly {num_trends} key trends based ONLY on the provided papers.

REQUIREMENTS:
1. **Source**: You must derive trends ONLY from the papers below. Do not use outside knowledge.
2. **Detail**: Write a distinct, high-quality description (200-400 words) for EACH trend.
3. **Citations**: You MUST cite the specific papers that support each trend.

FORMAT:
1. **Trend Name**
   - **Description**: [200-400 word detailed explanation...]
   - **Derived From**: [List specific paper titles here]

PAPERS TO ANALYZE:
{paper_context}
"""

    return generate_response(prompt, llm)


# ------------------ Gap Identification ------------------
def find_gaps_with_citations(top_papers_df: pd.DataFrame, llm, num_gaps: int = 2):
    gaps = []
    # Analyze top 3 ranked papers
    for _, p in top_papers_df.head(3).iterrows():
        # Handle PDF and Keys safely
        pdf_url = p.get('pdf_url', p.get('openAccessPdf', {}).get('url'))
        text = get_full_text(pdf_url)
        
        if not text:
            text = str(p.get('summary', p.get('Summary', '')))[:3000]

        title = p.get('title', p.get('Title', 'Unknown Title'))

        # --- ENHANCED PROMPT FOR PRECISION & HIGHLIGHTING ---
        prompt = f"""
You are a critical Peer Reviewer evaluating a research paper.
TASK: Identify {num_gaps} **specific technical limitations, methodological flaws, or unaddressed scopes** in this paper.

RULES for OUTPUT:
1. **Precision**: Do not say "it is slow". Say "suffers from **high latency** in real-time scenarios".
2. **Highlighting**: You MUST **bold** key technical terms representing the gap.
3. **Context**: Explain WHY this is a gap based on the provided text.

FORMAT:
- **Gap 1**: [Description with **bold keywords**]
- **Gap 2**: [Description with **bold keywords**]

PAPER TO REVIEW:
Title: {title}
Content: {text[:4000]}
"""

        response = generate_response(prompt, llm)
        gaps.append({"source": title, "gaps": response})

    return gaps


# ------------------ Roadmap Generation ------------------
def suggest_roadmap(topic: str, gaps: List[Dict], llm):
    gaps_text = "\n".join(
        [f"{item['source']}:\n{item['gaps']}" for item in gaps]
    )

    prompt = f"""
Create a 4-step research roadmap based on these gaps:
{gaps_text}
"""
    return generate_response(prompt, llm)

# ------------------ Final Summary ------------------
def generate_final_summary(result: Dict, llm):
    prompt = f"""
Create a final research summary for:
Topic: {result.get('topic')}
Trends: {result.get('trends')}
Gaps: {result.get('gaps')}
Plan: {result.get('final_plan')}
"""
    return generate_response(prompt, llm)


# ------------------ Research Chatbot ------------------
def answer_user_query(user_query: str, papers: List[Dict], llm) -> str:
    if not papers:
        return "No papers available for reference."

    source_context = "\n".join([
        f"Paper: {p.get('title', p.get('Title', 'Untitled'))}\nAbstract: {p.get('summary', p.get('Summary', ''))}\n---"
        for p in papers
    ])

    prompt = f"""
You are a factual academic assistant.
USER QUESTION: "{user_query}"
KNOWLEDGE BASE:
{source_context}
Answer concisely using ONLY the papers. Cite them.
"""

    return generate_response(prompt, llm)