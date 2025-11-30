import pandas as pd
from typing import TypedDict, List, Dict, Any, Annotated
import operator
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# --- 1. YAHAN SE FETCH HOTE HAIN FUNCTIONS ---
from fetcher import fetch_recent_papers
from ranking_engine import rank_papers
from insight_engine import (
    load_model,
    refine_topic_query,
    filter_papers_with_llm,
    find_trends,              # Trends function
    find_gaps_with_citations, # Gaps function
    suggest_roadmap,         # Roadmap function
    generate_final_summary,   # Summary function
    answer_user_query         # Chatbot function
)

# --- 2. Define State ---
class AgentState(TypedDict):
    topic: str
    refined_queries: List[str]
    raw_papers: List[dict]
    ranked_papers: List[dict]
    trends: str
    gaps: List[dict]
    roadmap: str
    analysis_report: str
    status: str
    messages: Annotated[List[BaseMessage], operator.add]

# Initialize LLM
llm = load_model()

# --- 3. Nodes (Jo Functions ko call karte hain) ---
def refine_node(state: AgentState):
    topic = state["topic"]
    queries = refine_topic_query(topic, llm)
    return {"refined_queries": queries, "status": "Refined Queries"}

def fetch_node(state: AgentState):
    queries = state["refined_queries"]
    papers = fetch_recent_papers(queries, max_results=8)
    return {"raw_papers": papers, "status": f"Fetched {len(papers)} Papers"}

def rank_node(state: AgentState):
    raw_papers = state["raw_papers"]
    topic = state["topic"]

    # DataFrame conversion zaroori hai kyunki insight_engine DF expect karta hai
    df_raw = pd.DataFrame(raw_papers)

    # Column rename for safety
    df_raw = df_raw.rename(columns={'title': 'Title', 'summary': 'Summary', 'id': 'ArXiv ID'})

    # Insight Engine ka function call
    df_scored = filter_papers_with_llm(df_raw, topic, llm)

    # Ranking Engine ka function call
    final_ranked = rank_papers(df_scored, query=topic)

    return {"ranked_papers": final_ranked, "status": "Ranked Papers"}

def trends_node(state: AgentState):
    papers = state["ranked_papers"]
    df_ranked = pd.DataFrame(papers)
    df_ranked = df_ranked.rename(columns={'title': 'Title', 'summary': 'Summary'})

    # Yahan 'insight_engine' se 'find_trends' use ho raha hai
    trends = find_trends(df_ranked, llm)
    return {"trends": trends, "status": "Identified Trends"}

def gaps_node(state: AgentState):
    papers = state["ranked_papers"]
    df_ranked = pd.DataFrame(papers)
    df_ranked = df_ranked.rename(columns={'title': 'Title', 'summary': 'Summary', 'pdf_url': 'pdf_url'})

    # Yahan 'insight_engine' se 'find_gaps_with_citations' use ho raha hai
    gaps = find_gaps_with_citations(df_ranked, llm)
    # gaps is expected to be a list of {"source": <title>, "gaps": <text>}
    return {"gaps": gaps, "status": "Analyzed PDFs for Gaps"}

def roadmap_node(state: AgentState):
    topic = state["topic"]
    gaps = state["gaps"]

    # Yahan 'insight_engine' se 'suggest_roadmap' use ho raha hai
    roadmap = suggest_roadmap(topic, gaps, llm)
    return {"roadmap": roadmap, "status": "Generated Research Roadmap"}

def summary_node(state: AgentState):
    result_dict = {
        "topic": state["topic"],
        "trends": state["trends"],
        "gaps": state["gaps"],
        "final_plan": state["roadmap"]
    }

    # Yahan 'insight_engine' se 'generate_final_summary' use ho raha hai
    report = generate_final_summary(result_dict, llm)
    return {"analysis_report": report, "status": "Final Report Ready"}

def chatbot_node(state: AgentState):
    messages = state["messages"]
    ranked_papers = state["ranked_papers"]
    last_user_msg = messages[-1]
    query = last_user_msg.content if isinstance(last_user_msg, HumanMessage) else str(last_user_msg.content)

    # Yahan 'insight_engine' se 'answer_user_query' use ho raha hai
    response_text = answer_user_query(query, ranked_papers, llm)

    return {
        "messages": [AIMessage(content=response_text)],
        "status": "Answered Query"
    }

# --- 4. Build Graph (Routing) ---
workflow = StateGraph(AgentState)

# Nodes add kar rahe hain (Unique names ke saath)
workflow.add_node("refine", refine_node)
workflow.add_node("fetch", fetch_node)
workflow.add_node("rank", rank_node)
workflow.add_node("extract_trends", trends_node)
workflow.add_node("extract_gaps", gaps_node)
workflow.add_node("create_roadmap", roadmap_node)
workflow.add_node("summary", summary_node)
workflow.add_node("chatbot", chatbot_node)

# Conditional Logic for Chat vs Research
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

# Flow Connections
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
