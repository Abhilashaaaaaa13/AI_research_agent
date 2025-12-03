import streamlit as st
import uuid
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

# --- Page Configuration ---
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed" # Sidebar ko by default band rakha hai
)

from agent import graph

# --- 🌙 Dark Mode CSS ---
st.markdown("""
<style>
/* Background */
.stApp {
    background-color: #0d1117 !important;
    color: #ffffff !important;
}

/* Section Header */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff !important;
    margin-top: 25px;
    margin-bottom: 15px;
    border-bottom: 3px solid #6366f1;
    padding-bottom: 5px;
}

/* Paper Card */
.paper-card {
    background-color: #1e1e1e !important;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.4);
    margin-bottom: 12px;
    border-left: 5px solid #6366f1;
}
.paper-card h5 { color: white !important; font-weight: 700; margin-bottom: 6px; }
.paper-card p { color: #d1d5db !important; font-size: 0.9em; margin: 0; }
.paper-meta { font-size: 0.85em; color: #9ca3af !important; margin-bottom: 10px !important; }

/* Buttons */
div.stButton > button {
    background-color: #30363d !important;
    color: #ffffff !important;
    border-radius: 8px;
    border: 1px solid #6366f1;
}
div.stButton > button:hover {
    background-color: #6366f1 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state: st.session_state.messages = []
if "topic" not in st.session_state: st.session_state.topic = ""
if "research_started" not in st.session_state: st.session_state.research_started = False

for key in ["ranked_papers", "trends_content", "gaps_content",
            "roadmap_content", "final_report"]:
    if key not in st.session_state:
        st.session_state[key] = None

for key in ["show_trends", "show_gaps", "show_roadmap", "show_summary"]:
    if key not in st.session_state:
        st.session_state[key] = False

# --- (Sidebar Code Removed as per your preference) ---
# Agar "New Research" button chahiye to main page pe laga sakte hain,
# par abhi ke liye sidebar hata diya hai taki clean dikhe.

# --- Input UI ---
if not st.session_state.research_started:
    st.title("🔬 Deep AI Research Agent")
    topic = st.text_input("Enter Research Topic", placeholder="e.g. Generative AI Agents")
    
    if topic:
        st.write("Great! Now, how many papers should I fetch?")
        
        # FIX: Value is empty string ("") so user MUST type
        num_papers_str = st.text_input("Type a number (e.g. 5)", value="")
        
        if st.button("🚀 Fetch & Rank Papers", type="primary"):
            
            # Validation: Check if empty OR not a number
            if not num_papers_str.strip():
                 st.error("Please enter the number of papers.")
            elif not num_papers_str.isdigit():
                st.error("Please enter a valid number (e.g. 3, 5, 10).")
            else:
                st.session_state.topic = topic
                st.session_state.max_results = int(num_papers_str)
                st.session_state.research_started = True
                st.rerun()
    

# --- Step 1: Fetch + Rank Only ---
if st.session_state.research_started and st.session_state.ranked_papers is None:

    st.title(f"🧠 Researching: {st.session_state.topic}")
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.status("📚 Fetching & ranking best papers...", expanded=True) as stbox:
        # Pass inputs correctly
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

# --- Step 1 UI Display ---
# --- Step 1 UI Display ---
if st.session_state.ranked_papers:
    st.markdown('<div class="section-header">1️⃣ Top Ranked Papers</div>', unsafe_allow_html=True)

    for p in st.session_state.ranked_papers:

        # Safety Handling
        raw_authors = p.get('authors')
        if isinstance(raw_authors, list):
            authors_text = ", ".join([str(a) for a in raw_authors if a])
        elif isinstance(raw_authors, str):
            authors_text = raw_authors
        else:
            authors_text = "Unknown Authors"

        pub_date = p.get('published', 'Unknown Date')
        citations = p.get('citationcount', 0) or 0
        score = p.get('finalscore', 0) or 0
        pdf_link = p.get('pdf_url', '#')

        card_html = f"""
        <div class="paper-card">
            <h5>{p['title']}</h5>
            <p class="paper-meta">
                <b>🗓 Published:</b> {pub_date} &nbsp;|&nbsp;
                <b>✍ Authors:</b> {authors_text}
            </p>
            <details>
                <summary style="cursor:pointer; color:#6366f1;"><b>📜 Read Summary</b></summary>
                <p style="margin-top:5px; font-style:italic;">{p['summary']}</p>
            </details>
            <br>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>
                    <b>⭐ Score:</b> {score:.2f} &nbsp;|&nbsp;
                    <b>📊 Citations:</b> {citations}
                </span>
                <a href="{pdf_link}" target="_blank"
                    style="background-color:#6366f1; color:white; padding:6px 12px; 
                    border-radius:5px; text-decoration:none;">
                    🔗 Open PDF
                </a>
            </div>
        </div>
        """

        st.markdown(card_html, unsafe_allow_html=True)
    if not st.session_state.show_trends:
        if st.button("📉 Extract Trends"):
            st.session_state.show_trends = True
            st.session_state.trends_content = None
            st.rerun()    

    
# ---- USER CLICKS THIS BUTTON TO EXTRACT TRENDS ----


# --- Step 2: Trends ---
if st.session_state.show_trends and st.session_state.trends_content is None:
    with st.spinner("📉 Extracting Trends..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        for event in graph.stream({"topic": st.session_state.topic}, config=config):
            if "extract_trends" in event:
                st.session_state.trends_content = event["extract_trends"]["trends"]
                st.rerun()

if st.session_state.trends_content:
    st.markdown('<div class="section-header">2️⃣ Trends in Research</div>', unsafe_allow_html=True)
    st.markdown(st.session_state.trends_content)

    if not st.session_state.show_gaps:
        if st.button("🧩 Show Research Gaps"):
            st.session_state.show_gaps = True
            st.rerun()

# --- Step 3: Gaps ---
if st.session_state.show_gaps and st.session_state.gaps_content is None:
    with st.spinner("🔍 Finding Gaps..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        for event in graph.stream({"topic": st.session_state.topic}, config=config):
            if "extract_gaps" in event:
                st.session_state.gaps_content = event["extract_gaps"]["gaps"]
                st.rerun()

if st.session_state.gaps_content:
    st.markdown('<div class="section-header">3️⃣ Research Gaps</div>', unsafe_allow_html=True)
    for g in st.session_state.gaps_content:
        st.markdown(f"**{g['source']}** → {g['gaps']}")

    if not st.session_state.show_roadmap:
        if st.button("🗺️ Generate Roadmap"):
            st.session_state.show_roadmap = True
            st.rerun()

# --- Step 4: Roadmap ---
if st.session_state.show_roadmap and st.session_state.roadmap_content is None:
    with st.spinner("🛠 Creating Roadmap..."):
        for event in graph.stream({"topic": st.session_state.topic},
                                  config={"configurable": {"thread_id": st.session_state.thread_id}}):
            if "create_roadmap" in event:
                st.session_state.roadmap_content = event["create_roadmap"]["roadmap"]
                st.rerun()

if st.session_state.roadmap_content:
    st.markdown('<div class="section-header">4️⃣ Roadmap</div>', unsafe_allow_html=True)
    st.markdown(st.session_state.roadmap_content)

    if not st.session_state.show_summary:
        if st.button("📑 Final Summary"):
            st.session_state.show_summary = True
            st.rerun()

# --- Step 5: Summary & Chatbot ---
if st.session_state.show_summary and st.session_state.final_report is None:
    with st.spinner("✍ Creating Summary..."):
        for event in graph.stream({"topic": st.session_state.topic},
                                  config={"configurable": {"thread_id": st.session_state.thread_id}}):
            if "summary" in event:
                st.session_state.final_report = event["summary"]["analysis_report"]
                st.rerun()

if st.session_state.final_report:
    st.markdown('<div class="section-header">5️⃣ Final Summary</div>', unsafe_allow_html=True)
    st.markdown(st.session_state.final_report)

    st.markdown("---")
    st.subheader("💬 Q&A Chatbot")

    for msg in st.session_state.messages:
        st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

    if prompt := st.chat_input("Ask questions about this topic"):
        st.session_state.messages.append(HumanMessage(content=prompt))
        for event in graph.stream({"messages": [HumanMessage(content=prompt)]},
                                  config={"configurable": {"thread_id": st.session_state.thread_id}}):
            if "chatbot" in event:
                ai_msg = event["chatbot"]["messages"][-1].content
                st.chat_message("assistant").write(ai_msg)
                st.session_state.messages.append(AIMessage(content=ai_msg))