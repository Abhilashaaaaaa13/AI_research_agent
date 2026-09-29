import operator
import os
import sqlite3
from typing import Annotated, Dict, List, TypedDict

import pandas as pd
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from fetcher import fetch_recent_papers
from insight_engine import (
    LLM_ERROR_PREFIX,
    answer_user_query,
    compare_papers,
    extract_keywords,
    filter_papers_with_llm,
    find_gaps_with_citations,
    find_trends,
    generate_final_summary,
    is_llm_error,
    keyword_filter,
    load_models,
    refine_topic_query,
    suggest_roadmap,
)
from ranking_engine import rank_papers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DB = os.path.join(BASE_DIR, "checkpoints.sqlite")

# Number of top candidates (after embedding-based pre-ranking) sent to the LLM for scoring
LLM_CANDIDATE_POOL = 20

# llm: analysis model; fast_llm: lightweight model for queries, keywords and scoring
llm, fast_llm = load_models()


# --- State ---
class AgentState(TypedDict, total=False):
    # Which stage the UI wants to run: research | trends | gaps | roadmap | summary | compare | chat
    action: str
    topic: str
    max_results: int
    sources: List[str]  # Which databases to search (all when empty)
    min_year: int  # Earliest publication year to include
    num_trends: int
    num_gaps: int
    refined_queries: List[str]
    raw_papers: List[dict]
    ranked_papers: List[dict]
    trends: str
    gaps: List[dict]
    roadmap: str
    analysis_report: str
    compare_indices: List[int]  # Positions in ranked_papers selected for comparison
    comparison: str
    status: str
    keywords: List[str]
    # Chat is stored as plain {"role", "content"} dicts: the SQLite checkpointer writes each
    # step's input and output into JSON metadata, so LangChain message objects would fail to save.
    question: str
    messages: Annotated[List[Dict[str, str]], operator.add]


def _ensure_ok(text: str) -> str:
    """Raise when the LLM call failed, so the error is shown in the UI instead of being saved as a result."""
    if is_llm_error(text):
        raise RuntimeError(text[len(LLM_ERROR_PREFIX):])
    return text


# --- Nodes ---
def refine_node(state: AgentState):
    queries = refine_topic_query(state["topic"], fast_llm)
    return {"refined_queries": queries, "status": f"Generated {len(queries)} search queries"}


def fetch_node(state: AgentState):
    user_request = state.get("max_results") or 5
    fetch_limit = min(user_request * 3, 50)
    papers = fetch_recent_papers(
        state["refined_queries"],
        max_results=fetch_limit,
        sources=state.get("sources"),
        min_year=state.get("min_year"),
    )
    if not papers:
        return {"raw_papers": [], "status": "No papers found"}
    return {"raw_papers": papers, "status": f"Fetched {len(papers)} unique papers"}


def keyword_extract_node(state: AgentState):
    keywords = extract_keywords(state["topic"], fast_llm)
    return {"keywords": keywords, "status": f"Keywords: {', '.join(keywords)}"}


def keyword_filter_node(state: AgentState):
    raw_papers = state.get("raw_papers") or []
    keywords = state.get("keywords") or []

    if not raw_papers:
        return {"raw_papers": [], "status": "No papers to filter"}
    if not keywords:
        return {"raw_papers": raw_papers, "status": "No keywords, skipped filtering"}

    filtered_df = keyword_filter(pd.DataFrame(raw_papers), keywords)
    return {
        "raw_papers": filtered_df.to_dict(orient="records"),
        "status": f"{len(filtered_df)} papers passed the keyword filter",
    }


def rank_node(state: AgentState):
    raw_papers = state.get("raw_papers") or []
    topic = state["topic"]
    user_request = state.get("max_results") or 5

    if not raw_papers:
        return {"ranked_papers": [], "status": "No papers to rank"}

    # Pre-rank every paper with embeddings so the LLM scores the best candidates,
    # not just whichever papers happened to be fetched first.
    pool_size = max(LLM_CANDIDATE_POOL, user_request * 2)
    candidates = rank_papers(raw_papers, query=topic, user_requested_count=pool_size)

    df_scored = filter_papers_with_llm(pd.DataFrame(candidates), topic, fast_llm, top_n=pool_size)
    final_ranked = rank_papers(df_scored, query=topic, user_requested_count=user_request)
    return {"ranked_papers": final_ranked, "status": f"Selected the top {len(final_ranked)} papers"}


def trends_node(state: AgentState):
    papers = state.get("ranked_papers") or []
    if not papers:
        return {"trends": "No papers available.", "status": "Skipped trends"}

    num_trends = state.get("num_trends") or 3
    trends = _ensure_ok(find_trends(pd.DataFrame(papers), llm, num_trends=num_trends))
    return {"trends": trends, "status": "Identified trends"}


def gaps_node(state: AgentState):
    papers = state.get("ranked_papers") or []
    if not papers:
        return {"gaps": [], "status": "Skipped gaps"}

    num_gaps = state.get("num_gaps") or 2
    gaps = find_gaps_with_citations(pd.DataFrame(papers), llm, num_gaps=num_gaps)
    if gaps and all(is_llm_error(g["gaps"]) for g in gaps):
        _ensure_ok(gaps[0]["gaps"])
    return {"gaps": gaps, "status": "Analyzed research gaps"}


def roadmap_node(state: AgentState):
    gaps = state.get("gaps") or []
    roadmap = _ensure_ok(suggest_roadmap(state["topic"], gaps, llm)) if gaps else "No gaps found to generate a roadmap."
    return {"roadmap": roadmap, "status": "Generated research roadmap"}


def summary_node(state: AgentState):
    result_dict = {
        "topic": state["topic"],
        "trends": state.get("trends"),
        "gaps": state.get("gaps"),
        "final_plan": state.get("roadmap"),
    }
    report = _ensure_ok(generate_final_summary(result_dict, llm))
    return {"analysis_report": report, "status": "Final report ready"}


def compare_node(state: AgentState):
    papers = state.get("ranked_papers") or []
    selected = [papers[i] for i in state.get("compare_indices") or [] if 0 <= i < len(papers)]
    comparison = _ensure_ok(compare_papers(selected, state["topic"], llm))
    return {"comparison": comparison, "status": f"Compared {len(selected)} papers"}


def chatbot_node(state: AgentState):
    query = state.get("question") or "Summarize the papers."
    response_text = _ensure_ok(answer_user_query(query, state.get("ranked_papers") or [], llm, state))
    return {
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response_text},
        ],
        "status": "Answered query",
    }


# --- Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("refine", refine_node)
workflow.add_node("fetch", fetch_node)
workflow.add_node("keyword_extract", keyword_extract_node)
workflow.add_node("keyword_filter", keyword_filter_node)
workflow.add_node("rank", rank_node)
workflow.add_node("extract_trends", trends_node)
workflow.add_node("extract_gaps", gaps_node)
workflow.add_node("create_roadmap", roadmap_node)
workflow.add_node("summary", summary_node)
workflow.add_node("compare", compare_node)
workflow.add_node("chatbot", chatbot_node)

# Each UI action runs exactly one stage, so later stages never run before the user asks for them.
ACTION_TO_NODE = {
    "research": "refine",
    "trends": "extract_trends",
    "gaps": "extract_gaps",
    "roadmap": "create_roadmap",
    "summary": "summary",
    "compare": "compare",
    "chat": "chatbot",
}


def route_input(state: AgentState):
    return ACTION_TO_NODE.get(state.get("action") or "research", "refine")


workflow.add_conditional_edges(START, route_input, {node: node for node in ACTION_TO_NODE.values()})

# The research stage is a small pipeline; every other stage is a single node.
workflow.add_edge("refine", "fetch")
workflow.add_edge("fetch", "keyword_extract")
workflow.add_edge("keyword_extract", "keyword_filter")
workflow.add_edge("keyword_filter", "rank")
for terminal in ["rank", "extract_trends", "extract_gaps", "create_roadmap", "summary", "compare", "chatbot"]:
    workflow.add_edge(terminal, END)

# --- Persistence ---
# Sessions are stored in a local SQLite file so past research can be reopened from the sidebar.
conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
memory = SqliteSaver(conn)

graph = workflow.compile(checkpointer=memory)


def delete_session(thread_id: str):
    """Remove every saved checkpoint for a research session."""
    memory.delete_thread(thread_id)
