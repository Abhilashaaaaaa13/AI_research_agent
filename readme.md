🧬 Deep Research Agent (AI Scientist)

An autonomous research assistant powered by LangGraph, Gemini 1.5, and Streamlit.

This agent automates the heavy lifting of academic research. It fetches real-time papers from ArXiv & OpenAlex, ranks them using vector embeddings, reads full PDFs to identify research gaps, and synthesizes a strategic roadmap—all in a seamless, interactive dashboard.

🚀 Key Features

🔍 Multi-Source Retrieval: Aggregates papers from ArXiv and OpenAlex for high coverage.

🧠 Intelligent Ranking: Uses a hybrid scoring system:

Semantic Match: SentenceTransformer embeddings (all-MiniLM-L6-v2).

AI Evaluation: Gemini 1.5 scores papers for relevance (1-5).

Impact Metrics: Factors in citation counts and venue prestige.

📉 Trend Analysis: Identifies emerging technical shifts and patterns across multiple abstracts.

🧩 Gap Analysis (PDF Reading): Automatically downloads and reads full PDFs to extract specific limitations and future work suggestions.

🗺️ Strategic Roadmap: Generates a phased execution plan based on identified gaps.

💬 Context-Aware Chat: Chat with the entire research session (papers + findings) to ask follow-up questions.

🌊 LangGraph Streaming: Visualizes the agent's "thinking process" step-by-step in the UI.

🛠️ Architecture

The system is built on a modular, stateful architecture:

Brain (insight_engine.py): Handles LLM interactions (Gemini), PDF parsing (PyMuPDF), and prompt logic.

Body (agent.py): Uses LangGraph to orchestrate the workflow nodes (Fetch -> Rank -> Trends -> Gaps -> Summary). Handles state persistence.

Limbs (fetcher.py & ranking_engine.py): Pure utility modules for API calls and vector math/ranking logic.

Face (app.py): A modern Streamlit interface with progressive data revealing and session management.

📦 Installation

Prerequisites

Python 3.9+

A Google Gemini API Key (Free tier works).

1. Clone the Repository

git clone [https://github.com/yourusername/deep-research-agent.git](https://github.com/yourusername/deep-research-agent.git)
cd deep-research-agent


2. Create a Virtual Environment

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


4. Configure Environment

Create a .env file in the root directory:

GOOGLE_API_KEY=your_actual_api_key_here


🏃‍♂️ Usage

Run the Streamlit application:

streamlit run app.py


Enter a Topic: Type a research interest (e.g., "Multi-Agent Reinforcement Learning").

Watch it Work: The agent will stream its progress (Refining Query -> Fetching -> Ranking).

Explore Results:

Ranked Papers: View the top papers sorted by relevance score.

Trends & Gaps: Click the buttons to reveal deep insights.

Roadmap: See the proposed research strategy.

Chat: Use the chat interface at the bottom to ask questions like "What specific method did Paper 3 use?".

📂 Project Structure

├── agent.py            # LangGraph state machine & node logic
├── app.py              # Streamlit UI (Frontend)
├── fetcher.py          # ArXiv & OpenAlex API connectors
├── insight_engine.py   # LLM prompts, PDF reading, & Analysis logic
├── ranking_engine.py   # Vector embeddings & Scoring math
├── requirements.txt    # Python dependencies
└── .env                # API Keys (Not shared)


🛡️ Troubleshooting

404 Model Not Found: Ensure your .env key is valid. If gemini-1.5-flash is unavailable in your region, edit insight_engine.py to use gemini-pro.

PDF Read Errors: Some papers restrict automated downloads. The agent will fallback to analyzing the abstract if the PDF download fails.

🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

📄 License

MIT License