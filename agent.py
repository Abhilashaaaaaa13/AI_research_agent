import pandas as pd
from typing import TypedDict, List, Dict, Any, Annotated
import operator
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# --- 1. IMPORT FUNCTIONS ---
from fetcher import fetch_recent_papers
from ranking_engine import rank_papers
from insight_engine import (
    load_model,
    refine_topic_query,
    filter_papers_with_llm,
    find_trends,              
    find_gaps_with_citations, 
    suggest_roadmap,         
    generate_final_summary,   
    answer_user_query,
    extract_keywords,       # added
    keyword_filter          # added
)

# Initialize LLM
llm = load_model()

# --- 2. Define State ---
class AgentState(TypedDict):
    topic: str
    max_results: int
    refined_queries: List[str]
    raw_papers: List[dict]
    ranked_papers: List[dict]
    trends: str
    gaps: List[dict]
    roadmap: str
    analysis_report: str
    status: str
    messages: Annotated[List[BaseMessage], operator.add]

# --- 3. Nodes ---

def refine_node(state: AgentState):
    topic = state["topic"]
    queries = refine_topic_query(topic, llm)
    return {"refined_queries": queries, "status": "Refined Queries"}

def rank_node(state: AgentState):
    raw_papers = state["raw_papers"]
    topic = state["topic"]
    user_request = state.get("max_results") or 5

    # convert to DataFrame
    df_raw = pd.DataFrame(raw_papers)
    if df_raw.empty:
        return {"ranked_papers": [], "status": "No Papers Found"}

    df_raw = df_raw.rename(columns={'title': 'Title', 'summary': 'Summary', 'id': 'ArXiv ID'})

    # filter using LLM
    df_scored = filter_papers_with_llm(df_raw, topic, llm)

    # rank papers
    final_ranked = rank_papers(df_scored, query=topic)

    # slice top N
    top_papers = final_ranked[:user_request]

    return {"ranked_papers": top_papers, "status": "Ranked Papers"}

        

def fetch_node(state: AgentState):
    queries = state["refined_queries"]
    user_request = state.get("max_results") or 5

    fetch_limit = min(user_request * 3, 50)  # fetch more to allow filtering

    papers = fetch_recent_papers(queries, max_results=fetch_limit)

    # fallback if fetch fails
    if not papers:
        print("DEBUG: No papers fetched, adding dummy fallback")
        papers = [{"title": state["topic"], "summary": "No papers found", "id": "N/A", "pdf_url": None}]

    return {"raw_papers": papers, "status": f"Fetched {len(papers)} Raw Papers"}


def trends_node(state: AgentState):
    papers = state["ranked_papers"]
    df_ranked = pd.DataFrame(papers)
    
    if df_ranked.empty or len(df_ranked) < 3:  # safety check
        trends = "Not enough papers to find trends."
    else:
        df_ranked = df_ranked.rename(columns={'title': 'Title', 'summary': 'Summary'})
        trends = find_trends(df_ranked, llm)
        
    return {"trends": trends, "status": "Identified Trends"}

def gaps_node(state: AgentState):
    papers = state["ranked_papers"]
    df_ranked = pd.DataFrame(papers)
    
    if not df_ranked.empty:
        df_ranked = df_ranked.rename(columns={'title': 'Title', 'summary': 'Summary', 'pdf_url': 'pdf_url'})
        gaps = find_gaps_with_citations(df_ranked, llm)
    else:
        gaps = []
        
    return {"gaps": gaps, "status": "Analyzed PDFs for Gaps"}

def roadmap_node(state: AgentState):
    topic = state["topic"]
    gaps = state["gaps"]

    if gaps:
        roadmap = suggest_roadmap(topic, gaps, llm)
    else:
        roadmap = "No gaps found to generate roadmap."
        
    return {"roadmap": roadmap, "status": "Generated Research Roadmap"}

def summary_node(state: AgentState):
    result_dict = {
        "topic": state["topic"],
        "trends": state["trends"],
        "gaps": state["gaps"],
        "final_plan": state["roadmap"]
    }

    report = generate_final_summary(result_dict, llm)
    return {"analysis_report": report, "status": "Final Report Ready"}

def chatbot_node(state: AgentState):
    messages = state["messages"]
    ranked_papers = state["ranked_papers"]
    last_user_msg = messages[-1]
    query = last_user_msg.content if isinstance(last_user_msg, HumanMessage) else str(last_user_msg.content)

    response_text = answer_user_query(query, ranked_papers, llm)

    return {
        "messages": [AIMessage(content=response_text)],
        "status": "Answered Query"
    }

# --- 4. Build Graph (Routing) ---
workflow = StateGraph(AgentState)

workflow.add_node("refine", refine_node)
workflow.add_node("fetch", fetch_node)
workflow.add_node("rank", rank_node)
workflow.add_node("extract_trends", trends_node)
workflow.add_node("extract_gaps", gaps_node)
workflow.add_node("create_roadmap", roadmap_node)
workflow.add_node("summary", summary_node)
workflow.add_node("chatbot", chatbot_node)

def route_input(state: AgentState):
    if state.get("messages") and len(state["messages"]) > 0:
        if state.get("ranked_papers"):
            return "chatbot"
    return "refine"

workflow.set_conditional_entry_point(
    route_input,
    {
        "refine": "refine",
        "chatbot": "chatbot"
    }
)

workflow.add_edge("refine", "fetch")
workflow.add_edge("fetch", "rank")
workflow.add_edge("rank", "extract_trends")
workflow.add_edge("extract_trends", "extract_gaps")
workflow.add_edge("extract_gaps", "create_roadmap")
workflow.add_edge("create_roadmap", "summary")
workflow.add_edge("summary", END)
workflow.add_edge("chatbot", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
