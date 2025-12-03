import os
import requests
import fitz  # PyMuPDF
from dotenv import load_dotenv
import pandas as pd
import json
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

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
        temperature=0.2,  # Lower temp for factual output
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

def extract_keywords(topic: str, llm, max_terms: int = 8) -> List[str]:
    prompt = f"""
Extract {max_terms} technical and domain-specific keywords for searching research
papers directly related to: "{topic}"

RULES:
- Focus on core terminology used in REAL research papers
- Include specific methods, applications, datasets, and evaluation metrics
- No generic words like "survey", "technology", "deep learning"
- Respond ONLY as a Python list of strings

Example:
["lung nodule detection", "CT scan anomaly", "autoencoder", "unsupervised"]
"""
    try:
        keywords = json.loads(generate_response(prompt, llm, json_mode=True))
        return [k.lower().strip() for k in keywords]
    except:
        return [topic.lower()]

def keyword_filter(papers_df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
    def match(row):
        text = (row['Title'] + " " + row['Summary']).lower()
        return any(kw in text for kw in keywords)
    return papers_df[papers_df.apply(match, axis=1)].reset_index(drop=True)

# ------------------ LLM-Based Paper Filtering ------------------
def filter_papers_with_llm(papers_df: pd.DataFrame, topic: str, client, top_n: int = 20):
    """Rate papers and always return up to top_n by relevance."""
    if papers_df.empty:
        papers_df['LLM_Score'] = 0.0
        return papers_df.head(top_n)
    
    top_papers = papers_df.head(top_n).reset_index(drop=True).copy()

    paper_context = "\n\n".join(
        [f"ID:{i}\nTitle:{row['Title']}\nAbstract:{row['Summary'][:600]}"
         for i, (_, row) in enumerate(top_papers.iterrows())]
    )

    prompt = f"""
You are a research paper relevance evaluator.

Topic of Interest: "{topic}"

TASK:
For EACH paper below:
- Read ONLY the title and abstract.
- Assign a relevance score from 1 to 10 based on how well the paper fits the topic.

SCORING GUIDELINES:
-10: Perfect match
-8-9: Highly relevant
-5-7: Related but broad
-0-4: Irrelevant

RESPONSE FORMAT:
Output ONLY a JSON array with objects like:
[{{ "ID": 0, "Score": 9 }}, {{ "ID": 1, "Score": 3 }}]

PAPERS:
{paper_context}
"""

    try:
        result_text = generate_response(prompt, client, json_mode=True)
        result = json.loads(result_text)
        score_map = {item['ID']: item['Score'] for item in result}
        top_papers['LLM_Score'] = top_papers.index.map(score_map).fillna(0)

        # Sort by score descending
        top_papers = top_papers.sort_values(by='LLM_Score', ascending=False).reset_index(drop=True)

        # Take exactly top_n
        return top_papers.head(top_n)

    except Exception:
        # Fallback: assign default score, return top_n
        top_papers['LLM_Score'] = 5.0
        return top_papers.head(top_n)




# ------------------ Topic Query Refinement ------------------
def refine_topic_query(raw_topic: str, llm) -> List[str]:
    """Genertes SPECIFIC, TECHNICAL search queries. Avoids broad terms like 'Intro to AI"""
    prompt = f"""
You are a research query optimization system.
Convert the user's topic into 3 specific, technical search queries for ArXiv/Google Scholar.
Topic: "{raw_topic}"
RULES:
    1. Use domain-specific terminology (e.g., instead of "AI code", use "LLM code generation capabilities").
    2. Focus on "State of the Art", "Survey", "Implementation", or "Novel Architecture".
    3. Do NOT make broad queries. Be precise.
Output ONLY a valid JSON list of query strings. No extra text.
"""

    try:
        queries = json.loads(generate_response(prompt, llm, json_mode=True))
        return [q.strip() for q in queries]
    except Exception:
        return [raw_topic]

def answer_user_query(user_query: str, papers: List[Dict], llm) -> str:
    if not papers:
        return "No highly relevant papers were found for this query."

    source_context = "\n\n".join([
        f"Paper: {p['Title']}\nAbstract: {p['Summary'][:700]}"
        for p in papers
    ])

    prompt = f"""
Answer ONLY if information exists in the provided papers.

User Question: "{user_query}"

RULES:
- Cite ONLY the relevant papers directly
- If vague or unsupported: respond 
  "The current papers do not provide this information."
- Max: 5 concise bullet points

Context:
{source_context}
"""

    return generate_response(prompt, llm)

# ------------------ Trend Analysis ------------------
def find_trends(top_papers_df: pd.DataFrame, llm, num_trends: int = 5) -> str:
    """
    Extracts top N trends from provided papers with source citation.
    """
    # Use all papers, not just top 10
    paper_context = "\n\n".join(
        [f"- {row['Title']}: {row['Summary'][:700]}" for _, row in top_papers_df.iterrows()]
    )

    prompt = f"""
You are an expert research analyst.

TASK:
Identify the TOP {num_trends} **specific, emerging, or recurring research trends** 
in the following AI papers. For each trend, provide:

- Trend_Name: short, 2–5 words
- Description: 1–2 sentences strictly based on the papers
- Supporting_Papers: List of titles of papers where this trend is visible

RULES:
- Use ONLY the provided papers. Do NOT hallucinate.
- Trends must be concise, actionable, and technical.
- Response format: JSON list of objects like:
[
  {{
    "Trend_Name": "XYZ",
    "Description": "...",
    "Supporting_Papers": ["Paper A", "Paper B"]
  }},
  ...
]

PAPERS:
{paper_context}
"""

    # Call LLM and parse response
    response_text = generate_response(prompt, llm, json_mode=True)
    try:
        trends_list = json.loads(response_text)
        # Format as string for display
        formatted_trends = ""
        for idx, t in enumerate(trends_list, 1):
            formatted_trends += f"{idx}. {t['Trend_Name']}\n"
            formatted_trends += f"   Description: {t['Description']}\n"
            formatted_trends += f"   Source Papers: {', '.join(t['Supporting_Papers'])}\n\n"
        return formatted_trends.strip()
    except Exception:
        return "Could not extract trends properly. Ensure papers have abstracts and summaries."



# ------------------ Gap Identification ------------------
def find_gaps_with_citations(top_papers_df: pd.DataFrame, llm, num_gaps: int = 2):
    """
    RETURNS:
      A list of dicts: [{"source": <paper title>, "gaps": <text>}, ...]
    NOTE (IMPORTANT): The keys 'source' and 'gaps' are chosen to match the Streamlit UI display.
    """
    gaps = []
    for _, p in top_papers_df.head(3).iterrows():
        text = get_full_text(p.get('pdf_url'))
        if not text:
            text = p.get('Summary', '')[:3000]

        prompt = f"""
Identify up to {num_gaps} research gaps or limitations in the paper below.

RULES:
- Base findings ONLY on provided text
- Each gap must be 1–2 concise sentences
- MUST cite specific hints from the text (but summarize, do not quote directly)

OUTPUT FORMAT:
- Gap 1: ...
- Gap 2: ...

Paper Title: {p['Title']}
Text: {text[:4000]}
"""

        response = generate_response(prompt, llm)
        # Normalize output into the keys that Streamlit expects:
        gaps.append({"source": p["Title"], "gaps": response})

    return gaps


# ------------------ Roadmap Generation ------------------
def suggest_roadmap(topic: str, gaps: List[Dict], llm):
    gaps_text = "\n".join(
        [f"{item['source']}:\n{item['gaps']}" for item in gaps]
    )

    prompt = f"""
You are a PhD research strategist.

Using ONLY the following research gaps:
{gaps_text}

Create a 4-step research roadmap.  
Each step must include:
- Step_Name (2–5 words)
- Objective (one sentence)
- What must be done (2 concise bullet points)

Example format:
1. Step_Name
   Objective: ...
   - bullet
   - bullet

Ensure logical progression from theory → methodology → experiments → evaluation.
"""

    return generate_response(prompt, llm)


# ------------------ Final Summary ------------------
def generate_final_summary(result: Dict, llm):
    prompt = f"""
Create a final research summary including:

Topic: {result.get('topic')}
Key Trends: {result.get('trends')}
Most Critical Gaps: {result.get('gaps')}
Proposed Research Plan: {result.get('final_plan')}

OUTPUT:
Concise professional summary. Bullet points allowed.
"""

    return generate_response(prompt, llm)


# ------------------ Research Chatbot ------------------
def answer_user_query(user_query: str, papers: List[Dict], llm) -> str:
    if not papers:
        return "No papers available for reference."

    source_context = "\n".join([
        f"Paper: {p.get('title')}\nAbstract: {p.get('summary')}\n---"
        for p in papers
    ])

    prompt = f"""
You are a factual academic assistant.

USER QUESTION:
"{user_query}"

KNOWLEDGE BASE (Use ONLY this information):
{source_context}

RESPONSE RULES:
- If answer is found in the papers → cite paper titles exactly:
  Example: According to "Paper A", …
- If unknown → say: "Based on the current papers, this information is not available."
- Keep response concise (4–6 sentences max).
- Do NOT add outside knowledge.

Provide a single, clear answer now.
"""

    return generate_response(prompt, llm)
