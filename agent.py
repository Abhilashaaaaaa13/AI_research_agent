import pandas as pd
import sqlite3  # <--- Ye Built-in hai, install mat karna
from typing import TypedDict, List, Annotated
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver 
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
    extract_keywords,
    keyword_filter
)

# Initialize LLM
llm = load_model()

# --- 2. Define State ---
class AgentState(TypedDict):
    topic: str
    max_results: int
    num_trends: int
    num_gaps: int
    refined_queries: List[str]
    raw_papers: List[dict]
    ranked_papers: List[dict]
    trends: str
    gaps: List[dict]
    roadmap: str
    analysis_report: str
    status: str
    keywords: List[str]
    field_filter: List[str]
    messages: Annotated[List[BaseMessage], operator.add]

# --- 3. Nodes ---
def refine_node(state: AgentState):
    topic = state["topic"]
    queries = refine_topic_query(topic, llm)
    return {"refined_queries": queries, "status": "Refined Queries"}

def fetch_node(state: AgentState):
    if state.get("raw_papers") and len(state["raw_papers"]) > 0:
        return {"raw_papers": state["raw_papers"], "status": "Using Cached Papers"}

    queries = state["refined_queries"]
    user_request = state.get("max_results") or 5
    fetch_limit = min(user_request * 3, 50)  
    papers = fetch_recent_papers(queries, max_results=fetch_limit)
    
    if not papers:
        return {"raw_papers": [], "status": "No papers found"}
        
    return {"raw_papers": papers, "status": f"Fetched {len(papers)} Raw Papers"}

def keyword_extract_node(state: AgentState):
    keywords = extract_keywords(state["topic"], llm)
    return {"keywords": keywords, "status": "Extracted Keywords"}

def keyword_filter_node(state: AgentState):
    raw_papers = state["raw_papers"]
    keywords = state["keywords"]
    
    if not raw_papers:
        return {"raw_papers": [], "status": "No papers to filter"}
    if not keywords:
        return {"raw_papers": raw_papers, "status": "No keywords, skipped filter"}

    df = pd.DataFrame(raw_papers)
    filtered_df = keyword_filter(df, keywords)
    
    if filtered_df.empty:
        return {"raw_papers": raw_papers, "status": "Filter too strict, reverted to raw"}
        
    return {
        "raw_papers": filtered_df.to_dict(orient='records'),
        "status": f"Filtered to {len(filtered_df)} papers"
    }

def rank_node(state: AgentState):
    if state.get("ranked_papers") and len(state["ranked_papers"]) > 0:
         return {"ranked_papers": state["ranked_papers"], "status": "Using Cached Rankings"}

    raw_papers = state["raw_papers"]
    topic = state["topic"]
    user_request = state.get("max_results") or 5
    
    if not raw_papers:
        return {"ranked_papers": [], "status": "No Papers to Rank"}
        
    df_raw = pd.DataFrame(raw_papers)
    df_scored = filter_papers_with_llm(df_raw, topic, llm)
    final_ranked = rank_papers(df_scored, query=topic, user_requested_count=user_request)
    
    return {"ranked_papers": final_ranked, "status": "Ranked & Sliced Papers"}

def trends_node(state: AgentState):
    if state.get("trends"):
        return {"trends": state["trends"], "status": "Using Cached Trends"}

    papers = state["ranked_papers"]
    if not papers:
        return {"trends": "No papers available.", "status": "Skipped Trends"}
        
    df_ranked = pd.DataFrame(papers)
    num_trends = state.get("num_trends") or 3
    trends = find_trends(df_ranked, llm, num_trends=num_trends)
    return {"trends": trends, "status": "Identified Trends"}

def gaps_node(state: AgentState):
    if state.get("gaps"):
        return {"gaps": state["gaps"], "status": "Using Cached Gaps"}

    papers = state["ranked_papers"]
    if not papers:
        return {"gaps": [], "status": "Skipped Gaps"}
        
    df_ranked = pd.DataFrame(papers)
    num_gaps = state.get("num_gaps") or 2
    gaps = find_gaps_with_citations(df_ranked, llm, num_gaps=num_gaps)
    return {"gaps": gaps, "status": "Analyzed for Gaps"}

def roadmap_node(state: AgentState):
    topic = state["topic"]
    gaps = state["gaps"]
    roadmap = suggest_roadmap(topic, gaps, llm) if gaps else "No gaps found to generate roadmap."
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
    if messages and isinstance(messages[-1], HumanMessage):
        query = messages[-1].content
    else:
        query = str(messages[-1].content) if messages else "Summary"
    response_text = answer_user_query(query, ranked_papers, llm)
    return {"messages": [AIMessage(content=response_text)], "status": "Answered Query"}

# --- 4. Build Graph ---
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
workflow.add_node("chatbot", chatbot_node)

def route_input(state: AgentState):
    if state.get("messages") and len(state["messages"]) > 0:
        return "chatbot"
    if state.get("ranked_papers") and len(state["ranked_papers"]) > 0:
        # Smart Routing: Check what is missing and go there
        if not state.get("trends"):
            return "extract_trends"
        if not state.get("gaps") and state.get("num_gaps"): # If user asked for gaps
             return "extract_gaps"
        # If we have everything, default to trends (UI handles specific calls)
        return "extract_trends"
    return "refine"

workflow.set_conditional_entry_point(
    route_input,
    {
        "refine": "refine", 
        "chatbot": "chatbot", 
        "extract_trends": "extract_trends",
        "extract_gaps": "extract_gaps"
    }
)

workflow.add_edge("refine", "fetch")
workflow.add_edge("fetch", "keyword_extract")
workflow.add_edge("keyword_extract", "keyword_filter")
workflow.add_edge("keyword_filter", "rank")
workflow.add_edge("rank", "extract_trends")
workflow.add_edge("extract_trends", "extract_gaps")
workflow.add_edge("extract_gaps", "create_roadmap")
workflow.add_edge("create_roadmap", "summary")
workflow.add_edge("summary", END)
workflow.add_edge("chatbot", END)

# --- PERSISTENCE SETUP ---
# Ye line 'checkpoints.sqlite' naam ki file banayegi
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

graph = workflow.compile(checkpointer=memory)