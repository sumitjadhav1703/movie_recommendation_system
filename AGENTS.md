# AGENTS.md — Movie Recommendation System

Guidance for humans and coding agents working in this repository.

## What this project is

A **job-demo** content-based movie recommender:

- **Frontend:** Streamlit (`app.py`) — browse, search, open a movie, see recommendations
- **Backend:** FastAPI (`main.py`) — TF-IDF similarity over local pickles + TMDB for posters/metadata
- **Deployed:** Streamlit Cloud (UI) + Render (API)

Live URLs are listed in `README.md`.

## Hard rules (do not break)

1. **Do not retrain or rewrite the ML model.** Leave the TF-IDF approach, notebook training logic, and pickle *semantics* alone unless the owner explicitly asks.
2. **Do not edit pickle file contents** (`df.pkl`, `indices.pkl`, `tfidf.pkl`, `tfidf_matrix.pkl`) as part of normal UI/API work.
3. Keep changes **demo-simple**: no Redis, no auth, no collaborative filtering, no microservices split unless requested.
4. Prefer **clear comments** on functions/routes so the owner can reread the code later.

## Architecture (runtime)

```text
Streamlit (app.py)
    |  REST (API_BASE)
    v
FastAPI (main.py)
    |-- local pickles --> title lookup + TF-IDF cosine scores
    |-- TMDB API -------> home feed, search, details, posters, genre discover
```

### Key files

| File | Role |
|------|------|
| `app.py` | Streamlit UI only — calls the API, renders grids/details |
| `main.py` | FastAPI — loads pickles once, exposes routes, talks to TMDB |
| `*.pkl` | Frozen ML artifacts from offline training |
| `movie_recommendations_system.ipynb` | Offline exploration / training (not used at request time) |
| `requirements.txt` | Shared deps for API + Streamlit |
| `.streamlit/config.toml` | Streamlit theme |

### Recommendation flow (important)

1. User picks a movie by **TMDB id** in the UI.
2. Details load from `/movie/id/{tmdb_id}`.
3. Similar movies use local TF-IDF by **title match** into `df.pkl` / `tfidf_matrix.pkl` (dataset has titles, not TMDB ids).
4. Posters for TF-IDF titles are attached via TMDB search (can be parallelized; still title-based).
5. “More like this” uses TMDB discover by genre id.

**Implication:** Duplicate/ambiguous titles can mismatch. Prefer routes that take `tmdb_id` for details/genre; accept title-based TF-IDF as a demo limitation.

## Real API routes (source of truth: `main.py`)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/home` | TMDB home feed (`category`, `limit`) |
| GET | `/tmdb/search` | Raw TMDB keyword search |
| GET | `/movie/id/{tmdb_id}` | Movie details |
| GET | `/recommend/genre` | Genre discover by `tmdb_id` |
| GET | `/recommend/tfidf` | Local TF-IDF titles + scores only |
| GET | `/movie/search` | Bundle: details + TF-IDF + genre (title query) |
| GET | `/movie/id/{tmdb_id}/recommendations` | Bundle by TMDB id (preferred for details page) |

Swagger: `{API_BASE}/docs`

## Environment

- `TMDB_API_KEY` required for the API (`.env` locally; host secrets in deploy).
- Frontend `API_BASE`: env `MOVIE_API_BASE` if set, else production Render URL; use `http://127.0.0.1:8000` for local API.

## Safe improvement areas

- Streamlit layout, theme, CSS, loading/error UX
- TMDB HTTP client reuse, timeouts, **parallel** poster fetches
- Comments, README accuracy, dependency pins (`requests` for Streamlit)
- Small route helpers that do not change similarity math

## Out of scope unless asked

- Changing `TfidfVectorizer` params, tags recipe, or cosine formula
- Replacing pickles / adding user accounts / Redis / new ML models

## Local run

```bash
# API
uvicorn main:app --reload

# UI (separate terminal)
MOVIE_API_BASE=http://127.0.0.1:8000 streamlit run app.py
```

## Testing expectations for agents

1. Change one vertical slice at a time (docs → API → UI).
2. Smoke-test: health, home feed, search, details, recommendations.
3. Before shipping a PR, run a **cold** review/test pass (fresh agent with no prior chat memory) that only reports what works vs what breaks.
4. Do not claim “done” if the UI cannot talk to a reachable API or if recommendation grids are empty for common titles (e.g. Inception) without explaining why (cold start / missing key / title miss).
