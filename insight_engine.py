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
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2,  # Lower temp for factual output
        convert_system_message_to_human=True
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


# ------------------ LLM-Based Paper Filtering ------------------
def filter_papers_with_llm(papers_df: pd.DataFrame, topic: str, client, top_n: int = 20):
    if papers_df.empty:
        return papers_df.assign(LLM_Score=0.0)

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
- Assign a relevance score from 1 to 5 based on how well the paper fits the topic.

SCORING GUIDELINES:
5 = Directly focused on the topic and addresses it in depth.
4 = Strongly relevant; topic is a major part of the study.
3 = Indirectly relevant; related methods or applications.
2 = Very weak relevance; only small or hypothetical connection.
1 = Not relevant at all.

RESPONSE FORMAT:
Output ONLY a JSON array with objects like:
[
  {{ "ID": 0, "Score": 5 }},
  {{ "ID": 1, "Score": 3 }}
]

NO text before or after JSON.

PAPERS:
{paper_context}
"""


    try:
        result = json.loads(generate_response(prompt, client, json_mode=True))
        score_map = {item['ID']: item['Score'] for item in result}
        top_papers['LLM_Score'] = top_papers.index.map(score_map).fillna(0)
        scored_df = papers_df.merge(top_papers[['ArXiv ID', 'LLM_Score']], on='ArXiv ID', how='left')
        scored_df['LLM_Score'] = scored_df['LLM_Score'].fillna(0)
        return scored_df
    except:
        return papers_df.assign(LLM_Score=2.5)


# ------------------ Topic Query Refinement ------------------
def refine_topic_query(raw_topic: str, llm) -> List[str]:
    prompt = f"""
You are a research query optimization system.

Convert the topic below into 3–5 highly effective academic search queries
that:
- Include domain-specific keywords and synonyms
- Avoid being too broad
- Are suitable for Google Scholar, ArXiv, and Scopus

Topic: "{raw_topic}"

Output ONLY a valid JSON list of query strings. No extra text.
"""


    try:
        queries = json.loads(generate_response(prompt, llm, json_mode=True))
        return [q.strip() for q in queries]
    except:
        return [raw_topic]


# ------------------ Trend Analysis ------------------
def find_trends(top_papers_df: pd.DataFrame, llm, num_trends: int = 3) -> str:
    paper_context = "\n\n".join(
        [f"- {row['Title']}: {row['Summary'][:700]}"
         for _, row in top_papers_df.head(10).iterrows()]
    )

    prompt = f"""
You are analyzing emerging research trends from AI papers.

Identify the TOP {num_trends} most recurring or growing trends.

For each trend, provide this structure:
- Trend_Name: short, 2–5 words
- Description: 2 short sentences
- Supporting_Papers: List of 2–3 titles

Output format example:
1. Trend_Name: XYZ
   Description: ...
   Supporting_Papers: ["Paper A", "Paper B"]

Use only the provided papers:
{paper_context}
"""

    return generate_response(prompt, llm)


# ------------------ Gap Identification ------------------
def find_gaps_with_citations(top_papers_df: pd.DataFrame, llm, num_gaps: int = 2):
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
        gaps.append({"source_paper": p["Title"], "gaps_text": response})

    return gaps


# ------------------ Roadmap Generation ------------------
def suggest_roadmap(topic: str, gaps: List[Dict], llm):
    gaps_text = "\n".join(
        [f"{item['source_paper']}:\n{item['gaps_text']}" for item in gaps]
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
