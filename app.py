import streamlit as st
import uuid
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
import time
# --- Page Configuration ---
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# Import graph
try:
    from agent import graph
except ImportError:
    st.error("⚠️ agent.py not found.")

# --- 🌙 Dark Mode CSS ---
st.markdown("""
<style>
/* Background */
.stApp { background-color: #0d1117 !important; color: #ffffff !important; }

/* Section Header */
.section-header { font-size: 1.4rem; font-weight: 700; color: #ffffff !important; margin-top: 25px; margin-bottom: 15px; border-bottom: 3px solid #6366f1; padding-bottom: 5px; }

/* Paper Card */
.paper-card { background-color: #1e1e1e !important; padding: 15px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.4); margin-bottom: 12px; border-left: 5px solid #6366f1; }
.paper-card h5 { color: white !important; font-weight: 700; margin-bottom: 6px; }
.paper-card p { color: #d1d5db !important; font-size: 0.9em; margin: 0; }
.paper-meta { font-size: 0.85em; color: #9ca3af !important; margin-bottom: 10px !important; }

/* Buttons */
div.stButton > button { background-color: #30363d !important; color: #ffffff !important; border-radius: 8px; border: 1px solid #6366f1; }
div.stButton > button:hover { background-color: #6366f1 !important; }

/* Inputs */
input[type="text"], input[type="number"] { background-color: #1e1e1e !important; color: white !important; border: 1px solid #4b5563 !important; }
</style>
""", unsafe_allow_html=True)
#helper:stream text--
def stream_text(text):
    """Simulates streaming for typewriter effect"""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.002)

# sidebar:persistence
st.sidebar.title("🗂️ Research History")
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

st.sidebar.markdown(f"**Current Session ID:**")
st.sidebar.code(st.session_state.thread_id)

st.sidebar.markdown("---")
st.sidebar.subheader("Resume Chat")
existing_thread = st.sidebar.text_input("Enter past Session ID to resume:")

if st.sidebar.button("Load Session"):
    if existing_thread:
        st.session_state.thread_id = existing_thread
        #fetch state from db
        config = {"configurable" : {"thread_id":existing_thread}}
        state_snapshot = graph.get_state(config)
        if state_snapshot.values:
            val = state_snapshot.values
            #hydrate ui state 
            st.session_state.research_started = True
            st.session_state.topic = val.get("topic","")
            st.session_state.ranked_papers = val.get("ranked_papers")
            st.session_state.gaps_content = val.get("gaps")
            st.session_state.roadmap_content = val.get("roadmap")
            st.session_state.final_report = val.get("analysis_report")
            st.session_state.messages = val.get("messages",[])

            #update visibility flags
            if st.session_state.trends_content: st.session_state.show_trends = True
            if st.session_state.gaps_content: st.session_state.show_gaps = True
            if st.session_state.roadmap_content: st.session_state.show_roadmap = True
            if st.session_state.final_report: st.session_state.show_summary = True

            st.success("Session Loaded!")
            st.rerun()
        else:
            st.sidebar.error("Session ID not found.")
# --- Session State ---

if "messages" not in st.session_state: st.session_state.messages = []
if "research_started" not in st.session_state: st.session_state.research_started = False

for key in ["ranked_papers", "trends_content", "gaps_content", "roadmap_content", "final_report"]:
    if key not in st.session_state: st.session_state[key] = None

for key in ["show_trends", "show_gaps", "show_roadmap", "show_summary"]:
    if key not in st.session_state: st.session_state[key] = False

# --- 1. INITIAL INPUT ---
if not st.session_state.research_started:
    st.title("🔬 Deep AI Research Agent")
    topic = st.text_input("Enter Research Topic", placeholder="e.g. Generative AI Agents")
    
    num_papers_str = st.text_input("How many papers to fetch?", value="5")
    
    if st.button("🚀 Fetch & Rank Papers", type="primary"):
        if not topic:
             st.error("Please enter a topic.")
        elif not num_papers_str.isdigit():
            st.error("Please enter a valid number (e.g. 5).")
        else:
            st.session_state.topic = topic
            st.session_state.max_results = int(num_papers_str)
            st.session_state.research_started = True
            st.rerun()

# --- EXECUTE STEP 1: Fetch & Rank ---
if st.session_state.research_started and st.session_state.ranked_papers is None:
    st.title(f"🧠 Researching: {st.session_state.topic}")
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.status("📚 Fetching & ranking best papers...", expanded=True) as stbox:
        inputs = {
            "topic": st.session_state.topic,
            "max_results": st.session_state.max_results
        }
        
        for event in graph.stream(inputs, config=config):
            for node, values in event.items():
                if node == "fetch":
                    stbox.write(f"Fetched raw papers...")
                if node == "rank":
                    st.session_state.ranked_papers = values["ranked_papers"]
                    stbox.write("Ranked based on relevance & citations 📈")
                    stbox.update(label="⭐ Ranking Complete!", state="complete")
                    st.rerun()

# --- DISPLAY STEP 1: Ranked Papers ---
if st.session_state.ranked_papers:
    st.markdown('<div class="section-header">1️⃣ Top Ranked Papers</div>', unsafe_allow_html=True)

    for p in st.session_state.ranked_papers:
        title = p.get('title') or p.get('Title') or "Untitled"
        summary = p.get('summary') or p.get('Summary') or "No summary"
        pub = p.get('published', 'N/A')
        pdf = p.get('pdf_url', '#')
        score = p.get('finalscore', 0)
        
        raw_authors = p.get('authors')
        authors_text = ", ".join([str(a) for a in raw_authors if a]) if isinstance(raw_authors, list) else str(raw_authors)

        card_html = f"""
        <div class="paper-card">
            <h5>{title}</h5>
            <p class="paper-meta"><b>🗓 {pub}</b> | ✍ {authors_text}</p>
            <details><summary style="cursor:pointer; color:#6366f1;">Read Summary</summary><p style="margin-top:5px;">{summary}</p></details>
            <br>
            <div style="display:flex; justify-content:space-between;">
                <span><b>⭐ Score:</b> {score:.2f}</span>
                <a href="{pdf}" target="_blank" style="color:#6366f1; text-decoration:none;">🔗 Open PDF</a>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    # --- INPUT FOR TRENDS ---
    if not st.session_state.show_trends:
        st.markdown("---")
        st.subheader("📉 Extract Trends")
        
        c1, c2 = st.columns([1, 3])
        with c1:
            num_trends_input = st.number_input("How many trends to find?", min_value=1, max_value=10, value=3)
        
        with c2:
            st.write("") 
            st.write("") 
            if st.button("Extract Trends"):
                st.session_state.num_trends = int(num_trends_input)
                st.session_state.show_trends = True
                st.session_state.trends_content = None
                st.rerun()

# --- EXECUTE STEP 2: Extract Trends ---
if st.session_state.show_trends and st.session_state.trends_content is None:
    graph_input = {
        "topic": st.session_state.topic,
        "ranked_papers": st.session_state.ranked_papers, 
        "num_trends": st.session_state.num_trends
    }
    
    with st.spinner("📉 Analyzing papers for detailed trends..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        for event in graph.stream(graph_input, config=config):
            if "extract_trends" in event:
                st.session_state.trends_content = event["extract_trends"]["trends"]
                st.rerun()

# --- DISPLAY STEP 2: Trends ---
if st.session_state.trends_content:
    st.markdown('<div class="section-header">2️⃣ Trends in Research</div>', unsafe_allow_html=True)
    st.markdown(st.session_state.trends_content)

    # --- INPUT FOR GAPS (ADDED HERE) ---
    if not st.session_state.show_gaps:
        st.markdown("---")
        st.subheader("🧩 Extract Research Gaps")
        
        c1, c2 = st.columns([1, 3])
        with c1:
            # Here we ask for gaps per paper
            num_gaps_input = st.number_input("Gaps per paper?", min_value=1, max_value=5, value=2)
        
        with c2:
            st.write("") 
            st.write("")
            if st.button("Extract Gaps"):
                st.session_state.num_gaps = int(num_gaps_input)
                st.session_state.show_gaps = True
                st.session_state.gaps_content = None
                st.rerun()

# --- EXECUTE STEP 3: Gaps ---
if st.session_state.show_gaps and st.session_state.gaps_content is None:
    
    # --- CRITICAL: Pass papers AND num_gaps ---
    graph_input = {
        "topic": st.session_state.topic,
        "ranked_papers": st.session_state.ranked_papers,
        "num_gaps": st.session_state.num_gaps
    }
    
    with st.spinner("🔍 Finding Gaps..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        for event in graph.stream(graph_input, config=config):
            if "extract_gaps" in event:
                st.session_state.gaps_content = event["extract_gaps"]["gaps"]
                st.rerun()

# --- DISPLAY STEP 3: Gaps ---
if st.session_state.gaps_content:
    st.markdown('<div class="section-header">3️⃣ Research Gaps</div>', unsafe_allow_html=True)
    for g in st.session_state.gaps_content:
        st.markdown(f"**{g['source']}**")
        st.info(g['gaps'])

    if not st.session_state.show_roadmap:
        if st.button("🗺️ Generate Roadmap"):
            st.session_state.show_roadmap = True
            st.rerun()

# --- EXECUTE STEP 4: Roadmap ---
if st.session_state.show_roadmap and st.session_state.roadmap_content is None:
    with st.spinner("🛠 Creating Roadmap..."):
        for event in graph.stream({"topic": st.session_state.topic, "gaps": st.session_state.gaps_content},
                                  config={"configurable": {"thread_id": st.session_state.thread_id}}):
            if "create_roadmap" in event:
                st.session_state.roadmap_content = event["create_roadmap"]["roadmap"]
                st.rerun()

# --- DISPLAY STEP 4: Roadmap ---
if st.session_state.roadmap_content:
    st.markdown('<div class="section-header">4️⃣ Roadmap</div>', unsafe_allow_html=True)
    st.markdown(st.session_state.roadmap_content)

    if not st.session_state.show_summary:
        if st.button("📑 Final Summary"):
            st.session_state.show_summary = True
            st.rerun()

# --- EXECUTE STEP 5: Summary ---
if st.session_state.show_summary and st.session_state.final_report is None:
    with st.spinner("✍ Creating Summary..."):
        input_state = {
            "topic": st.session_state.topic,
            "trends": st.session_state.trends_content,
            "gaps": st.session_state.gaps_content,
            "roadmap": st.session_state.roadmap_content
        }
        for event in graph.stream(input_state, config={"configurable": {"thread_id": st.session_state.thread_id}}):
            if "summary" in event:
                st.session_state.final_report = event["summary"]["analysis_report"]
                st.rerun()

# --- DISPLAY STEP 5: Summary & Chat ---
if st.session_state.final_report:
    st.markdown('<div class="section-header">5️⃣ Final Summary</div>', unsafe_allow_html=True)
    st.markdown(st.session_state.final_report)

    st.markdown("---")
    st.subheader("💬 Q&A Chatbot")
    
    for msg in st.session_state.messages:
        st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

    if prompt := st.chat_input("Ask questions..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)
        
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        output = graph.invoke({"messages": [HumanMessage(content=prompt)], "ranked_papers": st.session_state.ranked_papers}, config=config)
        
        ai_msg = output["messages"][-1].content
        st.session_state.messages.append(AIMessage(content=ai_msg))
        st.chat_message("assistant").write(ai_msg)
        #streaming effect
        st.chat_message("assistant").write_stream(stream_text(ai_msg))