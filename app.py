import csv
import html
import io
import json
import os
import re
import sys
import uuid
from datetime import datetime

import streamlit as st

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")
RESULT_KEYS = ["ranked_papers", "trends_content", "gaps_content", "roadmap_content", "final_report", "comparison"]
DEFAULT_MIN_YEAR = 2010
STAGES = [
    ("ranked_papers", "Papers"),
    ("trends_content", "Trends"),
    ("gaps_content", "Gaps"),
    ("roadmap_content", "Roadmap"),
    ("final_report", "Summary"),
]
EXAMPLE_TOPICS = [
    "Retrieval-augmented generation",
    "CRISPR off-target effects",
    "Graph neural networks for drug discovery",
    "Quantum error correction",
]
SUGGESTED_QUESTIONS = [
    "What methods do these papers use?",
    "Which datasets or benchmarks are used?",
    "What are the main open problems?",
    "Explain paper [1] in simple terms.",
]
# Sort keys receive (rank, paper) pairs
SORT_OPTIONS = {
    "Rank": lambda item: item[0],
    "Newest": lambda item: -(paper_year(item[1]) or 0),
    "Most cited": lambda item: -float(item[1].get("citationcount") or 0),
}
SCORE_LABELS = [
    ("score_relevance", "Relevance", "50%"),
    ("score_llm_score", "Gemini rating", "20%"),
    ("score_novelty", "Novelty", "10%"),
    ("score_venue", "Venue", "10%"),
    ("score_citations", "Citations", "10%"),
]
# Progress label shown while the *next* research node is running
NEXT_STEP_LABEL = {
    "refine": "Searching the selected databases",
    "fetch": "Extracting domain keywords",
    "keyword_extract": "Filtering papers by keyword",
    "keyword_filter": "Scoring and ranking papers",
}


# ------------------ Styles ------------------
st.markdown(
    """
<style>
.block-container { padding-top: 2.2rem; max-width: 1100px; }
footer { visibility: hidden; }

.hero-title {
    font-size: 2.6rem; font-weight: 800; line-height: 1.15; margin: 0 0 .3rem;
    background: linear-gradient(90deg, #8B93FF, #5CE1E6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { opacity: .75; font-size: 1.05rem; margin-bottom: 1.6rem; max-width: 720px; }
.eyebrow { text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; font-weight: 700; opacity: .55; }
.topic-title { font-size: 1.9rem; font-weight: 750; margin: .1rem 0 .8rem; line-height: 1.25; }

.stepper { display: flex; gap: .5rem; flex-wrap: wrap; margin: .2rem 0 1.4rem; }
.step {
    display: flex; align-items: center; gap: .45rem; padding: .35rem .8rem .35rem .4rem;
    border-radius: 999px; font-size: .85rem; border: 1px solid rgba(128,128,160,.3);
}
.step .dot {
    width: 1.4rem; height: 1.4rem; border-radius: 50%; display: inline-flex; align-items: center;
    justify-content: center; font-size: .72rem; font-weight: 700; background: rgba(128,128,160,.25);
}
.step.done { border-color: rgba(43,179,163,.55); }
.step.done .dot { background: #2BB3A3; color: #fff; }
.step.current { border-color: #8B93FF; background: rgba(139,147,255,.12); }
.step.current .dot { background: #8B93FF; color: #fff; }
.step.todo { opacity: .5; }

.section-title { font-size: 1.35rem; font-weight: 700; margin: 2.2rem 0 .15rem; }
.section-sub { opacity: .65; font-size: .9rem; margin-bottom: .9rem; }

.paper-card {
    border: 1px solid rgba(128,128,160,.25); background: rgba(128,128,160,.06);
    border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: .75rem; transition: border-color .15s;
}
.paper-card:hover { border-color: rgba(139,147,255,.6); }
.paper-head { display: flex; gap: .9rem; align-items: flex-start; }
.rank {
    flex: none; width: 2rem; height: 2rem; border-radius: 10px; background: rgba(139,147,255,.18);
    color: #A5ACFF; font-weight: 800; display: flex; align-items: center; justify-content: center;
}
.paper-body { flex: 1; min-width: 0; }
.paper-title { font-weight: 650; font-size: 1.02rem; line-height: 1.35; margin: 0; }
.paper-title a { color: inherit; text-decoration: none; }
.paper-title a:hover { color: #A5ACFF; }
.paper-meta { font-size: .82rem; opacity: .72; margin: .35rem 0 .1rem; }
.badge {
    display: inline-block; padding: .08rem .5rem; border-radius: 6px; font-size: .72rem;
    font-weight: 600; margin-right: .35rem; background: rgba(128,128,160,.18);
}
.badge.src { background: rgba(92,225,230,.14); color: #5CE1E6; }
.badge.ok { background: rgba(43,179,163,.16); color: #3FD1BF; }
.score-row { display: flex; align-items: center; gap: .6rem; font-size: .8rem; margin-top: .55rem; flex-wrap: wrap; }
.score-bar { flex: 1; min-width: 80px; max-width: 220px; height: 6px; border-radius: 99px; background: rgba(128,128,160,.2); overflow: hidden; }
.score-bar span { display: block; height: 100%; background: linear-gradient(90deg, #8B93FF, #5CE1E6); }
.score-row a { color: #A5ACFF; text-decoration: none; margin-left: auto; }
.paper-card details summary { cursor: pointer; font-size: .85rem; color: #A5ACFF; margin-top: .55rem; }
.paper-card details p { font-size: .9rem; opacity: .85; margin: .4rem 0 0; line-height: 1.55; }

.feature {
    border: 1px solid rgba(128,128,160,.25); border-radius: 14px; padding: 1rem;
    background: rgba(128,128,160,.05); height: 100%;
}
.feature .num { font-size: .75rem; font-weight: 700; color: #8B93FF; }
.feature .name { font-weight: 650; margin: .25rem 0; }
.feature .desc { font-size: .85rem; opacity: .7; }

.gap-source { font-weight: 650; margin-bottom: .3rem; }
.source-card {
    border: 1px solid rgba(128,128,160,.25); border-radius: 12px; padding: .7rem .9rem;
    background: rgba(128,128,160,.05); margin-bottom: .6rem; height: calc(100% - .6rem);
}
.source-card a { font-weight: 650; color: inherit; text-decoration: none; }
.source-card a:hover { color: #A5ACFF; }
.source-card div { font-size: .8rem; opacity: .7; margin-top: .15rem; }
.coverage { width: 100%; border-collapse: collapse; font-size: .85rem; }
.coverage th { text-align: left; font-weight: 600; opacity: .6; padding: .35rem .5rem; border-bottom: 1px solid rgba(128,128,160,.3); }
.coverage td { padding: .35rem .5rem; border-bottom: 1px solid rgba(128,128,160,.15); }
.coverage td.num { text-align: right; font-variant-numeric: tabular-nums; }
.coverage th.num { text-align: right; }
.coverage a { color: inherit; }
.status-ok { color: #3FD1BF; }
.status-err { color: #F2A65A; }
.topic-meta { font-size: .85rem; opacity: .65; margin: -.5rem 0 .8rem; }
.breakdown { display: grid; grid-template-columns: max-content 1fr max-content; gap: .3rem .7rem;
    align-items: center; font-size: .8rem; margin-top: .5rem; max-width: 420px; }
.breakdown .bar { height: 5px; border-radius: 99px; background: rgba(128,128,160,.2); overflow: hidden; }
.breakdown .bar span { display: block; height: 100%; background: #8B93FF; }
.breakdown .w { opacity: .55; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------ History ------------------
def load_history() -> dict:
    """Load past sessions as {thread_id: {"topic": str, "updated": iso-date}}."""
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    # Older versions stored only the topic string
    return {
        t_id: entry if isinstance(entry, dict) else {"topic": str(entry), "updated": ""}
        for t_id, entry in raw.items()
    }


def save_to_history(thread_id: str, topic: str):
    history = load_history()
    history.pop(thread_id, None)  # Re-insert so the most recent session is last
    history[thread_id] = {"topic": topic or "Untitled research", "updated": datetime.now().isoformat(timespec="minutes")}
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except OSError:
        pass


def remove_from_history(thread_id: str):
    history = load_history()
    history.pop(thread_id, None)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except OSError:
        pass


# ------------------ Agent ------------------
try:
    with st.spinner("Loading research models (first run can take a minute)..."):
        import agent
        from agent import delete_session, graph
        from fetcher import SOURCE_INFO, SOURCES, paper_year
        from insight_engine import FAST_MODEL, MAIN_MODEL, content_to_text
except Exception as e:
    st.error(f"Could not start the research agent: {e}")
    st.stop()

ss = st.session_state
ALL_SOURCES = list(SOURCES)


def thread_config() -> dict:
    return {"configurable": {"thread_id": ss.thread_id}}


def reset_session():
    ss.thread_id = str(uuid.uuid4())
    ss.topic = ""
    ss.messages = []
    ss.pending_research = False
    ss.pending_question = None
    ss.error = None
    ss.sources = list(ALL_SOURCES)
    ss.source_stats = {}
    ss.min_year = DEFAULT_MIN_YEAR
    ss.compare_titles = []
    for key in RESULT_KEYS:
        ss[key] = None


def to_chat_dict(message) -> dict:
    """Normalize stored chat messages (older sessions saved LangChain message objects)."""
    if isinstance(message, dict):
        return message
    role = "user" if getattr(message, "type", "") == "human" else "assistant"
    return {"role": role, "content": str(getattr(message, "content", message))}


def load_session(thread_id: str):
    reset_session()
    ss.thread_id = thread_id
    try:
        values = graph.get_state(thread_config()).values or {}
    except Exception:
        values = {}
    ss.topic = values.get("topic") or load_history().get(thread_id, {}).get("topic", "")
    ss.ranked_papers = values.get("ranked_papers")
    ss.trends_content = values.get("trends")
    ss.gaps_content = values.get("gaps")
    ss.roadmap_content = values.get("roadmap")
    ss.final_report = values.get("analysis_report")
    ss.comparison = values.get("comparison")
    ss.sources = values.get("sources") or list(ALL_SOURCES)
    ss.source_stats = values.get("source_stats") or {}
    ss.min_year = values.get("min_year") or DEFAULT_MIN_YEAR
    papers = values.get("ranked_papers") or []
    ss.compare_titles = [papers[i].get("title", "") for i in values.get("compare_indices") or [] if i < len(papers)]
    ss.messages = [to_chat_dict(m) for m in values.get("messages") or []]


if "thread_id" not in ss:
    reset_session()


def run_research():
    inputs = {
        "action": "research",
        "topic": ss.topic,
        "max_results": ss.max_results,
        "sources": ss.sources,
        "min_year": ss.min_year,
    }
    ranked = []
    with st.status("Generating search queries...", expanded=True) as status:
        for event in graph.stream(inputs, config=thread_config()):
            for node, values in event.items():
                values = values or {}
                status.write(f"✓ {values.get('status', node)}")
                if node in NEXT_STEP_LABEL:
                    status.update(label=NEXT_STEP_LABEL[node] + "...")
                if node == "fetch":
                    ss.source_stats = values.get("source_stats") or {}
                if node == "rank":
                    ranked = values.get("ranked_papers") or []
        status.update(label=f"Selected {len(ranked)} papers", state="complete", expanded=False)
    ss.ranked_papers = ranked


def run_stage(action: str, result_key: str, state_key: str, spinner: str, **inputs):
    """Run a single graph stage and store its output in session state."""
    with st.spinner(spinner):
        try:
            values = graph.invoke({"action": action, **inputs}, config=thread_config())
            ss[result_key] = values.get(state_key)
            ss.error = None
            save_to_history(ss.thread_id, ss.topic)
        except Exception as e:
            ss.error = f"The {action} step failed: {e}"


def stream_chat_reply(question: str):
    """Yield the chatbot's answer token by token while Gemini generates it."""
    streamed = False
    try:
        for chunk, meta in graph.stream(
            {"action": "chat", "question": question}, config=thread_config(), stream_mode="messages"
        ):
            if meta.get("langgraph_node") == "chatbot":
                text = content_to_text(chunk.content)
                if text:
                    streamed = True
                    yield text
    except Exception as e:
        yield f"Sorry, something went wrong: {e}"
        return
    if not streamed:
        # Nothing was streamed (e.g. an error message was returned instead), so show the saved reply
        messages = graph.get_state(thread_config()).values.get("messages") or []
        if messages:
            yield str(messages[-1]["content"])


# ------------------ Rendering helpers ------------------
def esc(value) -> str:
    return html.escape(str(value)) if value is not None else ""


def section(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def render_stepper():
    parts, current_found = [], False
    for i, (key, label) in enumerate(STAGES, start=1):
        done = ss[key] is not None and (key != "ranked_papers" or bool(ss[key]))
        if done:
            cls, mark = "done", "✓"
        elif not current_found:
            cls, mark, current_found = "current", str(i), True
        else:
            cls, mark = "todo", str(i)
        parts.append(f'<div class="step {cls}"><span class="dot">{mark}</span>{label}</div>')
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def author_list(p: dict) -> list:
    raw = p.get("authors")
    if isinstance(raw, list):
        return [str(a) for a in raw if a]
    return [str(raw)] if isinstance(raw, str) and raw else []


def format_authors(p: dict) -> str:
    authors = author_list(p)
    if not authors:
        return "Unknown authors"
    return ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")


def text_field(p: dict, key: str) -> str:
    """Return a string field, treating missing values and pandas NaN as empty."""
    value = p.get(key)
    return value if isinstance(value, str) and value != "Unknown" else ""


def render_score_breakdown(p: dict) -> str:
    # Sessions created before the breakdown existed do not have these fields
    if any(key not in p for key, _, _ in SCORE_LABELS):
        return ""
    rows = []
    for key, label, weight in SCORE_LABELS:
        value = max(0.0, min(1.0, float(p.get(key) or 0)))
        rows.append(
            f'<span>{label} <span class="w">{weight}</span></span>'
            f'<div class="bar"><span style="width:{value * 100:.0f}%"></span></div>'
            f"<span>{value:.2f}</span>"
        )
    return f'<details><summary>Score breakdown</summary><div class="breakdown">{"".join(rows)}</div></details>'


def render_paper(rank: int, p: dict):
    title = p.get("title") or p.get("Title") or "Untitled"
    summary = p.get("summary") or p.get("Summary") or "No abstract available."
    url = text_field(p, "pdf_url")
    venue = text_field(p, "venue")
    score = float(p.get("finalscore") or 0)
    citations = int(float(p.get("citationcount") or 0))
    width = max(0, min(100, score / 5 * 100))

    title_html = f'<a href="{esc(url)}" target="_blank">{esc(title)}</a>' if url else esc(title)
    link_html = f'<a href="{esc(url)}" target="_blank">Open paper ↗</a>' if url else ""
    citation_html = f"<span>{citations:,} citations</span>" if citations else ""
    venue_html = f" · <i>{esc(venue)}</i>" if venue else ""
    also_in = p.get("also_in") if isinstance(p.get("also_in"), list) else []
    also_html = "".join(f'<span class="badge">{esc(s)}</span>' for s in also_in)

    st.markdown(
        f"""
<div class="paper-card"><div class="paper-head">
  <div class="rank">{rank}</div>
  <div class="paper-body">
    <p class="paper-title">{title_html}</p>
    <div class="paper-meta">
      <span class="badge src">{esc(p.get("source", "Unknown"))}</span>{also_html}
      <span class="badge">{esc(p.get("published", "N/A"))}</span>
      {esc(format_authors(p))}{venue_html}
    </div>
    <div class="score-row">
      <span>Score <b>{score:.2f}</b>/5</span>
      <div class="score-bar"><span style="width:{width:.0f}%"></span></div>
      {citation_html}
      {link_html}
    </div>
    <details><summary>Abstract</summary><p>{esc(summary)}</p></details>
    {render_score_breakdown(p)}
  </div>
</div></div>
""",
        unsafe_allow_html=True,
    )


# ------------------ Exports ------------------
def file_slug() -> str:
    return re.sub(r"[^a-z0-9]+", "-", ss.topic.lower()).strip("-")[:40] or "research"


def build_report() -> str:
    lines = [f"# Research report: {ss.topic}", "", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", ""]
    if ss.ranked_papers:
        lines += ["## Top papers", ""]
        for i, p in enumerate(ss.ranked_papers, start=1):
            url = text_field(p, "pdf_url")
            venue = text_field(p, "venue") or p.get("source", "")
            lines.append(
                f"{i}. **{p.get('title', 'Untitled')}**. {format_authors(p)} ({p.get('published', 'N/A')}), {venue}"
                + (f" - {url}" if url else "")
            )
        lines.append("")
    if ss.trends_content:
        lines += ["## Trends", "", ss.trends_content, ""]
    if ss.gaps_content:
        lines += ["## Research gaps", ""]
        for g in ss.gaps_content:
            lines += [f"### {g['source']}", "", g["gaps"], ""]
    if ss.roadmap_content:
        lines += ["## Roadmap", "", ss.roadmap_content, ""]
    if ss.final_report:
        lines += ["## Summary", "", ss.final_report, ""]
    if ss.comparison:
        lines += ["## Paper comparison", "", ss.comparison, ""]
    return "\n".join(lines)


def bibtex_escape(value) -> str:
    return str(value).replace("{", "(").replace("}", ")")


def bibtex_key(p: dict, used: set) -> str:
    authors = author_list(p)
    last_name = re.sub(r"[^A-Za-z]", "", authors[0].split()[-1]) if authors and authors[0].split() else ""
    title_words = [re.sub(r"[^A-Za-z]", "", w) for w in str(p.get("title", "")).split()]
    first_word = next((w for w in title_words if len(w) > 3), "work")
    base = f"{(last_name or 'anon').lower()}{paper_year(p) or 'nd'}{first_word.lower()}"
    key, n = base, 2
    while key in used:
        key, n = f"{base}{n}", n + 1
    used.add(key)
    return key


def build_bibtex(papers: list) -> str:
    entries, used = [], set()
    for p in papers:
        venue = text_field(p, "venue")
        is_preprint = not venue or "arxiv" in venue.lower() or "preprint" in venue.lower()
        entry_type = "misc" if is_preprint else "article"
        fields = {
            "title": p.get("title"),
            "author": " and ".join(author_list(p)),
            "year": paper_year(p),
            ("howpublished" if is_preprint else "journal"): venue,
            "doi": text_field(p, "doi"),
            "url": text_field(p, "pdf_url"),
        }
        if p.get("source") == "arXiv":
            fields["eprint"] = p.get("id")
            fields["archiveprefix"] = "arXiv"
        body = ",\n".join(f"  {k} = {{{bibtex_escape(v)}}}" for k, v in fields.items() if v)
        entries.append(f"@{entry_type}{{{bibtex_key(p, used)},\n{body}\n}}")
    return "\n\n".join(entries) + "\n"


def build_csv(papers: list) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["rank", "title", "authors", "year", "venue", "source", "citations", "score", "doi", "url"])
    for i, p in enumerate(papers, start=1):
        writer.writerow([
            i,
            p.get("title", ""),
            "; ".join(author_list(p)),
            paper_year(p) or "",
            text_field(p, "venue"),
            p.get("source", ""),
            int(float(p.get("citationcount") or 0)),
            round(float(p.get("finalscore") or 0), 3),
            text_field(p, "doi"),
            text_field(p, "pdf_url"),
        ])
    return buffer.getvalue()


def regenerate_button(stage: str, result_key: str, state_key: str, spinner: str, **inputs):
    if st.button("Regenerate", key=f"regen_{stage}", icon=":material/refresh:", type="tertiary"):
        run_stage(stage, result_key, state_key, spinner, **inputs)
        st.rerun()


def ask_suggestion(key: str):
    ss.pending_question = ss.get(key)


# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("### 🧬 Deep Research")
    if st.button("New research", icon=":material/add:", type="primary", width="stretch"):
        reset_session()
        st.rerun()

    history = load_history()
    st.markdown("#### Recent")
    if not history:
        st.caption("Your past research sessions will appear here.")
    for t_id, entry in list(history.items())[::-1][:30]:
        topic = entry.get("topic", "Untitled research")
        label = topic if len(topic) <= 42 else topic[:40] + "…"
        is_current = t_id == ss.thread_id
        if st.button(label, key=f"hist_{t_id}", width="stretch", disabled=is_current, help=topic):
            load_session(t_id)
            st.rerun()

    if ss.topic:
        with st.popover("Delete this session", icon=":material/delete:", width="stretch"):
            st.caption("This permanently removes the papers, analysis and chat for this session.")
            if st.button("Delete permanently", type="primary", key="confirm_delete"):
                try:
                    delete_session(ss.thread_id)
                except Exception:
                    pass
                remove_from_history(ss.thread_id)
                reset_session()
                st.rerun()

    st.divider()
    st.caption(f"Searching {len(ALL_SOURCES)} databases: {', '.join(ALL_SOURCES)}")
    st.caption(f"Models (free tier): {MAIN_MODEL} · {FAST_MODEL}")

if agent.llm is None:
    st.warning("GOOGLE_API_KEY is not set, so analysis steps will not work. Add it to your .env file and restart.")


# ------------------ Landing page ------------------
def use_example_topic():
    if ss.example_topic:
        ss.topic_input = ss.example_topic


if not ss.topic:
    st.markdown('<div class="hero-title">Deep Research Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Search four academic databases, rank the most relevant papers, '
        "and turn them into trends, open research gaps and a roadmap, then ask follow-up questions.</div>",
        unsafe_allow_html=True,
    )

    with st.form("research_form", border=True):
        topic = st.text_input("Research topic", key="topic_input", placeholder="e.g. Multi-agent reinforcement learning")
        num_papers = st.slider("Number of papers", min_value=3, max_value=20, value=6)
        with st.expander("Advanced options"):
            sources = st.multiselect(
                "Databases", ALL_SOURCES, default=ALL_SOURCES,
                help="Where to search. Biomedical topics benefit from PubMed and Europe PMC; "
                "CS and physics from arXiv.",
            )
            min_year = st.number_input(
                "Published from (year)", min_value=1950, max_value=datetime.now().year, value=DEFAULT_MIN_YEAR, step=1
            )
        submitted = st.form_submit_button(
            "Start research", icon=":material/travel_explore:", type="primary", width="stretch"
        )

    st.pills("Try an example", EXAMPLE_TOPICS, key="example_topic", on_change=use_example_topic)

    if submitted:
        if not topic.strip():
            st.error("Please enter a research topic.")
        elif not sources:
            st.error("Select at least one database.")
        else:
            ss.topic = topic.strip()
            ss.max_results = num_papers
            ss.sources = sources
            ss.min_year = int(min_year)
            ss.pending_research = True
            save_to_history(ss.thread_id, ss.topic)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    features = [
        ("Find", f"Searches {len(ALL_SOURCES)} free academic databases with AI-refined queries."),
        ("Rank", "Scores papers by relevance, Gemini's rating, novelty, venue and citations."),
        ("Analyze", "Finds cross-paper trends and reads full PDFs to spot research gaps."),
        ("Compare", "Puts papers side by side and builds a roadmap and summary."),
        ("Ask & export", "Chat with the papers, then export Markdown, BibTeX or CSV."),
    ]
    for col, (i, (name, desc)) in zip(st.columns(len(features)), enumerate(features, start=1)):
        col.markdown(
            f'<div class="feature"><div class="num">STEP {i}</div><div class="name">{name}</div>'
            f'<div class="desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )

    section("Where the papers come from", f"{len(ALL_SOURCES)} free academic databases, searched in parallel.")
    source_cols = st.columns(3)
    for i, name in enumerate(ALL_SOURCES):
        info = SOURCE_INFO.get(name, {})
        source_cols[i % 3].markdown(
            f'<div class="source-card"><a href="{esc(info.get("url", "#"))}" target="_blank">{esc(name)}</a>'
            f'<div>{esc(info.get("covers", ""))}</div></div>',
            unsafe_allow_html=True,
        )
    st.stop()


# ------------------ Research view ------------------
head_left, head_right = st.columns([4, 1], vertical_alignment="bottom")
with head_left:
    st.markdown('<div class="eyebrow">Research topic</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="topic-title">{esc(ss.topic)}</div>', unsafe_allow_html=True)
    source_text = "all databases" if set(ss.sources) == set(ALL_SOURCES) else ", ".join(ss.sources)
    st.markdown(f'<div class="topic-meta">{esc(source_text)} · published from {ss.min_year}</div>', unsafe_allow_html=True)
with head_right:
    if ss.ranked_papers:
        with st.popover("Export", icon=":material/download:", width="stretch"):
            st.download_button(
                "Report (Markdown)", data=build_report(), file_name=f"{file_slug()}.md",
                mime="text/markdown", width="stretch",
            )
            st.download_button(
                "References (BibTeX)", data=build_bibtex(ss.ranked_papers), file_name=f"{file_slug()}.bib",
                mime="application/x-bibtex", width="stretch",
            )
            st.download_button(
                "Paper list (CSV)", data=build_csv(ss.ranked_papers), file_name=f"{file_slug()}.csv",
                mime="text/csv", width="stretch",
            )
render_stepper()

if ss.error:
    st.error(ss.error)

# --- Step 1: fetch and rank ---
if ss.pending_research:
    try:
        run_research()
        ss.error = None
    except Exception as e:
        ss.error = f"Research failed: {e}"
    ss.pending_research = False
    save_to_history(ss.thread_id, ss.topic)
    st.rerun()

if ss.ranked_papers is None:
    st.info("This session has no results yet.")
    if st.button("Run research", type="primary"):
        ss.max_results = ss.get("max_results", 6)
        ss.pending_research = True
        st.rerun()
    st.stop()

if not ss.ranked_papers:
    st.warning("No papers matched this topic. Try a broader topic, more databases or an earlier year.")
    if st.button("Start a new search", type="primary"):
        reset_session()
        st.rerun()
    st.stop()

papers = ss.ranked_papers
section("Top ranked papers", "Ranked by cross-encoder relevance, Gemini's relevance rating, novelty, venue and citations.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Papers", len(papers))
m2.metric("Sources", len({p.get("source") for p in papers}))
m3.metric("Avg. score", f"{sum(float(p.get('finalscore') or 0) for p in papers) / len(papers):.2f}")
m4.metric("Citations", f"{sum(int(float(p.get('citationcount') or 0)) for p in papers):,}")


def render_source_coverage():
    """Table of every searched database: papers found, papers in the final list, and status."""
    stats = ss.source_stats or {}
    names = list(stats) or ss.sources
    rows = []
    for name in names:
        entry = stats.get(name, {})
        in_list = sum(
            1 for p in papers
            if p.get("source") == name or name in (p.get("also_in") if isinstance(p.get("also_in"), list) else [])
        )
        if entry.get("error"):
            status = f'<span class="status-err">{esc(entry["error"])}</span>'
        elif stats:
            status = '<span class="status-ok">OK</span>'
        else:
            status = "—"  # Sessions created before per-database stats were recorded
        found = entry.get("found", "—") if stats else "—"
        url = SOURCE_INFO.get(name, {}).get("url", "#")
        rows.append(
            f'<tr><td><a href="{esc(url)}" target="_blank">{esc(name)}</a></td>'
            f'<td class="num">{found}</td><td class="num">{in_list}</td><td>{status}</td></tr>'
        )
    st.markdown(
        '<table class="coverage"><tr><th>Database</th><th class="num">Found</th>'
        f'<th class="num">In your list</th><th>Status</th></tr>{"".join(rows)}</table>',
        unsafe_allow_html=True,
    )
    if any(e.get("error") == "rate limited" for e in stats.values()):
        st.caption("Rate-limited databases work reliably with a free API key; see the README.")


with st.expander("Where these papers came from", icon=":material/database:"):
    render_source_coverage()

t1, t2 = st.columns([1, 3], vertical_alignment="bottom")
sort_by = t1.selectbox("Sort by", list(SORT_OPTIONS), key="sort_by")
available_sources = sorted({p.get("source", "Unknown") for p in papers})
shown_sources = t2.pills(
    "Show sources", available_sources, selection_mode="multi", default=available_sources,
    key=f"source_filter_{ss.thread_id}",
)
items = sorted(
    [(rank, p) for rank, p in enumerate(papers, start=1) if p.get("source", "Unknown") in (shown_sources or [])],
    key=SORT_OPTIONS[sort_by],
)
if not items:
    st.caption("No papers match the selected sources.")
for rank, paper in items:
    render_paper(rank, paper)

# --- Compare papers ---
section("Compare papers", "Pick 2–4 papers for a side-by-side look at their methods, data, results and limitations.")
with st.container(border=True):
    labels = [f"[{i}] {p.get('title', 'Untitled')}" for i, p in enumerate(papers, start=1)]
    chosen = st.multiselect(
        "Papers to compare", labels, max_selections=4, key=f"compare_{ss.thread_id}",
        default=[label for label, p in zip(labels, papers) if p.get("title") in ss.compare_titles],
    )
    if st.button("Compare", icon=":material/compare_arrows:", type="primary", disabled=len(chosen) < 2):
        indices = [labels.index(label) for label in chosen]
        ss.compare_titles = [papers[i].get("title") for i in indices]
        run_stage("compare", "comparison", "comparison", "Comparing papers...", compare_indices=indices)
        st.rerun()
    if ss.comparison:
        st.markdown(ss.comparison)

# --- Step 2: trends ---
section("Trends", "Recurring directions across the ranked papers, with the papers that support each one.")
with st.container(border=True):
    if ss.trends_content:
        st.markdown(ss.trends_content)
        regenerate_button("trends", "trends_content", "trends", "Re-analyzing trends...", num_trends=ss.get("num_trends", 3))
    else:
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        num_trends = c1.slider("How many trends?", 1, 8, 3)
        if c2.button("Extract trends", icon=":material/trending_up:", type="primary", width="stretch"):
            ss.num_trends = num_trends
            run_stage("trends", "trends_content", "trends", "Analyzing papers for trends...", num_trends=num_trends)
            st.rerun()

# --- Step 3: gaps ---
if ss.trends_content:
    section("Research gaps", "Limitations found in the top 3 papers. Full PDFs are read when they are openly available.")
    if ss.gaps_content:
        for g in ss.gaps_content:
            with st.container(border=True):
                badge = '<span class="badge ok">Full PDF</span>' if g.get("used_pdf") else '<span class="badge">Abstract only</span>'
                st.markdown(f'<div class="gap-source">{esc(g["source"])} {badge}</div>', unsafe_allow_html=True)
                st.markdown(g["gaps"])
        regenerate_button("gaps", "gaps_content", "gaps", "Re-reading papers...", num_gaps=ss.get("num_gaps", 2))
    else:
        with st.container(border=True):
            c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
            num_gaps = c1.slider("Gaps per paper", 1, 5, 2)
            if c2.button("Find gaps", icon=":material/search_insights:", type="primary", width="stretch"):
                ss.num_gaps = num_gaps
                run_stage("gaps", "gaps_content", "gaps", "Reading papers and finding gaps...", num_gaps=num_gaps)
                st.rerun()

# --- Step 4: roadmap ---
if ss.gaps_content:
    section("Roadmap", "A phased research plan that addresses the gaps above.")
    if ss.roadmap_content:
        with st.container(border=True):
            st.markdown(ss.roadmap_content)
            regenerate_button("roadmap", "roadmap_content", "roadmap", "Redrafting the roadmap...")
    elif st.button("Generate roadmap", icon=":material/route:", type="primary"):
        run_stage("roadmap", "roadmap_content", "roadmap", "Drafting the roadmap...")
        st.rerun()

# --- Step 5: summary ---
if ss.roadmap_content:
    section("Executive summary", "Everything above condensed into one report.")
    if ss.final_report:
        with st.container(border=True):
            st.markdown(ss.final_report)
            regenerate_button("summary", "final_report", "analysis_report", "Rewriting the summary...")
    elif st.button("Write summary", icon=":material/summarize:", type="primary"):
        run_stage("summary", "final_report", "analysis_report", "Writing the summary...")
        st.rerun()

# --- Chat ---
section("Ask the papers", "Answers come from the ranked papers and the findings above, with citations.")
for msg in ss.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

suggestion_key = f"suggestion_{len(ss.messages)}"
st.pills(
    "Suggested questions", SUGGESTED_QUESTIONS, key=suggestion_key,
    on_change=ask_suggestion, args=(suggestion_key,), label_visibility="collapsed",
)

prompt = st.chat_input("Ask a question about these papers...")
if not prompt and ss.pending_question:
    prompt, ss.pending_question = ss.pending_question, None

if prompt:
    ss.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        reply = st.write_stream(stream_chat_reply(prompt))
    ss.messages.append({"role": "assistant", "content": reply if isinstance(reply, str) else "".join(map(str, reply))})
    save_to_history(ss.thread_id, ss.topic)
    st.rerun()
