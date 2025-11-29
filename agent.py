import pandas as pd
from typing import TypedDict, List, Dict, Any, Annotated
import operator
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import your tools
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
    answer_user_query  # Added import
)

# --- 1. Define State ---
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
    # Add messages for chat history (using operator.add to append history)
    messages: Annotated[List[BaseMessage], operator.add]

# Initialize LLM once
llm = load_model()

# --- 2. Define Nodes ---

def refine_node(state: AgentState):
    """Refines the user topic into search queries."""
    topic = state["topic"]
    queries = refine_topic_query(topic, llm)
    return {"refined_queries": queries, "status": "Refined Queries"}

def fetch_node(state: AgentState):
    """Fetches papers from ArXiv/OpenAlex."""
    queries = state["refined_queries"]
    papers = fetch_recent_papers(queries, max_results=8)
    return {"raw_papers": papers, "status": f"Fetched {len(papers)} Papers"}

def rank_node(state: AgentState):
    """Scores and Ranks papers."""
    raw_papers = state["raw_papers"]
    topic = state["topic"]
    
    df_raw = pd.DataFrame(raw_papers)
    df_scored = filter_papers_with_llm(df_raw, topic, llm)
    final_ranked = rank_papers(df_scored, query=topic)
    
    return {"ranked_papers": final_ranked, "status": "Ranked Papers"}

def trends_node(state: AgentState):
    """Identifies trends."""
    papers = state["ranked_papers"]
    df_ranked = pd.DataFrame(papers)
    trends = find_trends(df_ranked, llm)
    return {"trends": trends, "status": "Identified Trends"}

def gaps_node(state: AgentState):
    """Finds gaps by reading PDFs."""
    papers = state["ranked_papers"]
    df_ranked = pd.DataFrame(papers)
    gaps = find_gaps_with_citations(df_ranked, llm)
    return {"gaps": gaps, "status": "Analyzed PDFs for Gaps"}

def roadmap_node(state: AgentState):
    """Generates a roadmap."""
    topic = state["topic"]
    gaps = state["gaps"]
    roadmap = suggest_roadmap(topic, gaps, llm)
    return {"roadmap": roadmap, "status": "Generated Research Roadmap"}

def summary_node(state: AgentState):
    """Synthesizes the final report."""
    topic = state["topic"]
    trends = state["trends"]
    gaps = state["gaps"]
    roadmap = state["roadmap"]
    
    result_dict = {
        "topic": topic,
        "trends": trends,
        "gaps": gaps,
        "final_plan": roadmap
    }
    
    report = generate_final_summary(result_dict, llm)
    return {"analysis_report": report, "status": "Final Report Ready"}

def chatbot_node(state: AgentState):
    """
    Handles Q&A after research is complete.
    Uses the 'ranked_papers' context to answer questions.
    """
    messages = state["messages"]
    ranked_papers = state["ranked_papers"]
    
    # Get the last user message
    last_user_msg = messages[-1]
    if isinstance(last_user_msg, HumanMessage):
        query = last_user_msg.content
    else:
        query = str(last_user_msg.content)
        
    # Generate answer using insight engine tool
    response_text = answer_user_query(query, ranked_papers, llm)
    
    return {
        "messages": [AIMessage(content=response_text)],
        "status": "Answered Query"
    }

# --- 3. Build Graph ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("refine", refine_node)
workflow.add_node("fetch", fetch_node)
workflow.add_node("rank", rank_node)
workflow.add_node("trends", trends_node)
workflow.add_node("gaps", gaps_node)
workflow.add_node("roadmap", roadmap_node)
workflow.add_node("summary", summary_node)
workflow.add_node("chatbot", chatbot_node)

# --- Define Routing Logic ---
def route_input(state: AgentState):
    """
    Decides whether to start new research or chat based on input.
    - If 'messages' exists and we have a report, go to Chatbot.
    - Else, start Research flow.
    """
    if state.get("messages") and len(state["messages"]) > 0:
        # Assuming if we are chatting, research is likely done or we have context
        if state.get("ranked_papers"):
            return "chatbot"
            
    return "refine"

# Set Conditional Entry Point
workflow.set_conditional_entry_point(
    route_input,
    {
        "refine": "refine",
        "chatbot": "chatbot"
    }
)

# Linear Research Sequence
workflow.add_edge("refine", "fetch")
workflow.add_edge("fetch", "rank")
workflow.add_edge("rank", "trends")   
workflow.add_edge("trends", "gaps")   
workflow.add_edge("gaps", "roadmap")  
workflow.add_edge("roadmap", "summary") 
workflow.add_edge("summary", END)

# Chatbot Loop (Single turn then wait for next input)
workflow.add_edge("chatbot", END)

# Compile with Persistence
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)