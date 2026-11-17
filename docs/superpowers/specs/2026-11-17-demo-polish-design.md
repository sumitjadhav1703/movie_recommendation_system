# Demo Polish Design — Movie Recommender

**Date:** 2026-11-17  
**Goal:** Make this job-demo app look and feel stronger, keep the backend clean, leave the ML model alone.

## Constraints

- Demo project for portfolio/interviews — simple, readable, commented.
- No Redis / complex caching / auth / new ML.
- Do not change TF-IDF training, similarity math, or pickle contents.
- Parallel TMDB poster fetches are allowed.
- Test after each chunk; cold test agent before PR.

## Approach (chosen)

**Polish in place:** keep single-file Streamlit + single-file FastAPI. Improve UI theme/layout, fix obvious bugs, tidy TMDB fetching, document accurately.

## Documentation

- `AGENTS.md` — agent-facing map of architecture, routes, hard rules.
- `README.md` — fix endpoint examples and project structure; stay concise.
- This design note — scope for the polish pass.

## Backend (light)

1. Shared `httpx.AsyncClient` for TMDB (reuse connections).
2. Parallel `attach_tmdb_card_by_title` via `asyncio.gather`.
3. Prefer recommendations by `tmdb_id` for the details page (avoid rematching the wrong title).
4. Comments on every public helper/route.
5. Leave pickle load + `tfidf_recommend_titles` math unchanged.

## Frontend (Streamlit)

1. Cinema-inspired dark theme via `.streamlit/config.toml` + restrained custom CSS.
2. Clear home → search → details → recommendations flow.
3. Poster grid with title + open action; better empty/error/loading states.
4. Configurable `API_BASE` via env for local vs production.
5. Fix broken `API_BASE or localhost` bug; replace deprecated image width args.

## Visual direction

- Dark charcoal background, soft amber accent (not purple gradients).
- Brand signal: “CineMatch” / Movie Recommender as the main header.
- Poster grids as the visual anchor; minimal chrome.
- One job per section: browse/search OR details+recs.

## Out of scope

- Retraining, collaborative filtering, user accounts, Redis, multi-page Streamlit package split, React rewrite.
