# 🧬 Deep Research Agent

An AI research assistant built with **LangGraph**, **Google Gemini 3** (free tier only) and **Streamlit**.

Give it a topic and it searches nine free academic databases, ranks the most relevant papers, and turns them into
trends, research gaps, a roadmap and an executive summary. You can compare papers side by side, ask follow-up
questions, and export everything as Markdown, BibTeX or CSV.

Everything runs on free services: the Gemini API free tier, free public paper databases, and ranking models
that run locally.

## Features

**Search**
- **Nine free databases**, queried in parallel with the topic plus three short AI-refined queries:

  | Database | Covers | Key |
  | --- | --- | --- |
  | [arXiv](https://arxiv.org) | Preprints in CS, physics, maths, statistics and more | none |
  | [OpenAlex](https://openalex.org) | 250M+ works across every field, with citation counts | optional, free |
  | [Semantic Scholar](https://www.semanticscholar.org) | 200M+ papers, strongest in CS and biomedicine | optional, free |
  | [Crossref](https://www.crossref.org) | DOI records from most journal publishers | none |
  | [PubMed](https://pubmed.ncbi.nlm.nih.gov) | 36M+ biomedical and life-science citations | optional, free |
  | [Europe PMC](https://europepmc.org) | Life sciences, including bioRxiv and medRxiv preprints | none |
  | [OpenAIRE](https://explore.openaire.eu) | European open-science graph of repositories and journals | none |
  | [CORE](https://core.ac.uk) | The largest collection of open-access papers | optional, free |
  | [DOAJ](https://doaj.org) | Peer-reviewed, fully open-access journals | none |

- **Search options**: choose which databases to use and the earliest publication year.
- **Source report**: the results page shows, for every database, how many papers it returned, how many made
  your final list, and whether it failed or was rate limited.
- **Duplicate merging**: when a paper appears in several databases, the copies are merged (highest citation
  count, venue, DOI and PDF link are kept) and the card shows where else it was found.

**Ranking**
- Each paper gets a 0–5 score, and every card shows a **score breakdown**:

  | Component | Weight |
  | --- | --- |
  | Cross-encoder relevance (`ms-marco-MiniLM-L-6-v2`, runs locally) | 50% |
  | Gemini relevance rating | 20% |
  | Novelty (embedding distance to the other papers) | 10% |
  | Venue (top-tier > published > preprint) | 10% |
  | Citations (log-scaled) | 10% |

  Papers whose title contains the exact topic are always ranked first.
- **Sort** by rank, newest or most cited, and **filter** by source.

**Analysis**
- **Compare papers**: pick 2–4 papers for a table of problem, method, data, results and limitations.
- **Trends** across the ranked papers, with citations.
- **Research gaps** in the top 3 papers, analyzed in parallel. Open-access PDFs are downloaded and their final
  sections read; otherwise the abstract is used.
- **Roadmap and summary**: a 4-phase research plan and an executive summary.
- **Regenerate** any analysis step to get a new version.

**Chat and export**
- **Chat** with the papers and findings, streamed token by token, with suggested questions to get started.
- **Export** a Markdown report, BibTeX references (for LaTeX / Zotero) or a CSV paper list.

**Sessions**
- Every session is saved to SQLite and can be reopened from the sidebar, or deleted.

## Architecture

| File | Role |
| --- | --- |
| `app.py` | Streamlit UI: landing page, progress stepper, paper cards, analysis, chat and exports |
| `agent.py` | LangGraph state machine. Each UI action (`research`, `trends`, `gaps`, `roadmap`, `summary`, `compare`, `chat`) runs only its own stage |
| `fetcher.py` | API clients for the nine databases, per-database stats, and duplicate merging |
| `ranking_engine.py` | Local embedding models and the scoring formula |
| `insight_engine.py` | Gemini clients, prompts, PDF reading, keyword filtering and analysis |
| `.streamlit/config.toml` | UI theme |

The `research` action runs a small pipeline:

```
refine queries -> fetch -> extract keywords -> keyword filter -> rank
```

## Models

The app only uses models that are available on the **free** Gemini API tier. Free-tier rate limits apply per
model, so the work is split between two models, and each automatically falls back to the other when it is
rate limited.

| Model | Used for | Override with |
| --- | --- | --- |
| `gemini-3.8-flash` | trends, gaps, roadmap, summary, comparison, chat | `GEMINI_MODEL` |
| `gemini-3.5-flash-lite` | query refinement, keywords, relevance scoring | `GEMINI_FAST_MODEL` |

To stay free, create your key in [Google AI Studio](https://aistudio.google.com/apikey) **without** enabling
billing on its Google Cloud project. With billing disabled, requests past the free limits are rejected rather
than charged.

## Setup

Requires Python 3.12+ (tested on 3.13).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key

# Optional, free: more reliable results from these databases (they rate-limit anonymous use)
# OPENALEX_API_KEY=...            https://openalex.org/rest-api
# SEMANTIC_SCHOLAR_API_KEY=...    https://www.semanticscholar.org/product/api
# CORE_API_KEY=...                https://core.ac.uk/services/api
# NCBI_API_KEY=...                https://www.ncbi.nlm.nih.gov/account/ (for PubMed)

# Optional model overrides (keep them on free-tier models)
# GEMINI_MODEL=gemini-3.8-flash
# GEMINI_FAST_MODEL=gemini-3.5-flash-lite
```

## Usage

```bash
streamlit run app.py
```

1. Enter a topic (or pick an example), choose how many papers you want, and optionally open
   **Advanced options** to pick databases and a start year.
2. Review the ranked papers: sort, filter, open the score breakdown, or compare a few side by side.
3. Work through **Trends → Gaps → Roadmap → Summary**.
4. Ask questions in the chat at the bottom of the page.
5. Use **Export** to download a Markdown report, BibTeX or CSV.

The first run downloads the two sentence-transformer models (about 200 MB).

## Deployment

The `Procfile` runs the app on platforms such as Railway or Heroku:

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Set `GOOGLE_API_KEY` (and optionally the other keys above) as environment variables on the platform.
`runtime.txt` and `.python-version` select Python 3.13.

## Troubleshooting

- **"The free-tier rate limit was reached"**: the Gemini free tier allows only a few requests per minute. The
  app switches to the other model and retries automatically; if it still fails, wait a minute and try again.
- **A database shows "rate limited"** in *Where these papers came from*: OpenAlex, Semantic Scholar and CORE
  rate-limit anonymous traffic. Add their free API keys to `.env`; the other databases still return results.
- **Gaps say "Abstract only"**: the paper's PDF is not openly downloadable, so only its abstract was analyzed.

## License

MIT
