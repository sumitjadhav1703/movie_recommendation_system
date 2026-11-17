"""
FastAPI backend for the Movie Recommendation demo.

Responsibilities:
- Load frozen TF-IDF pickles once at startup (do not retrain here).
- Proxy/enrich TMDB data (home feed, search, details, posters).
- Combine local similarity scores with TMDB posters for the Streamlit UI.

Hard rule: do not change the TF-IDF math or pickle contents unless explicitly asked.
"""

from __future__ import annotations

import asyncio
import os
import pickle
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# ENV
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    # Fail early so deploys without a key are obvious.
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")

# =========================
# PICKLE PATHS / GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

# Populated during lifespan startup.
df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None

# Shared TMDB HTTP client (connection reuse). Set in lifespan.
http_client: Optional[httpx.AsyncClient] = None


# =========================
# RESPONSE MODELS
# =========================
class TMDBMovieCard(BaseModel):
    """Compact movie card used in grids (home, genre, TF-IDF posters)."""

    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    """Fuller TMDB details for the movie page."""

    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    """One local TF-IDF recommendation, optionally enriched with a TMDB card."""

    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    """Details + similar (TF-IDF) + genre discover in one payload for the UI."""

    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    """Normalize titles for case-insensitive local lookup."""
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    """Build a full TMDB image URL from a poster/backdrop path."""
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe TMDB GET using the shared async client.

    Network / non-200 responses become HTTP 502 so the UI can show a clear error.
    """
    if http_client is None:
        raise HTTPException(status_code=503, detail="HTTP client not ready")

    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        r = await http_client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"TMDB error {r.status_code}: {r.text[:500]}"
        )

    return r.json()


def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    """Map raw TMDB result dicts into TMDBMovieCard models (sync, no I/O)."""
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    """Fetch TMDB movie details by numeric id."""
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """Return the raw TMDB search payload (includes a `results` list)."""
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    """Return the first TMDB search hit for a title string, or None."""
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


async def genre_recommendations_for_details(
    details: TMDBMovieDetails, limit: int = 18
) -> List[TMDBMovieCard]:
    """
    Discover popular movies in the first genre of `details`.

    Demo-simple: uses only the first genre id (same behavior as before).
    """
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != details.tmdb_id]


# =========================
# TF-IDF HELPERS (math unchanged)
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    Normalize indices.pkl into a dict of lowercased title -> row index.

    Accepts a dict or a pandas Series-like mapping with .items().
    """
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception as exc:
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        ) from exc


def get_local_idx_by_title(title: str) -> int:
    """Resolve a movie title to a local dataframe row index, or 404."""
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Return top-N similar titles using cosine-style scores on the sparse TF-IDF matrix.

    Implementation note: scores = tfidf_matrix @ query_vector (same as original demo).
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """
    Look up a local recommendation title on TMDB to attach a poster card.

    Never raises: failures become None so one bad title does not break the list.
    """
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


async def build_tfidf_items(
    title_for_local: str, fallback_title: str, top_n: int
) -> List[TFIDFRecItem]:
    """
    Run local TF-IDF for a title and attach TMDB posters in parallel.

    Tries `title_for_local` first, then `fallback_title` if lookup fails.
    """
    recs: List[Tuple[str, float]] = []
    try:
        recs = tfidf_recommend_titles(title_for_local, top_n=top_n)
    except Exception:
        try:
            recs = tfidf_recommend_titles(fallback_title, top_n=top_n)
        except Exception:
            recs = []

    if not recs:
        return []

    # Parallel TMDB searches for posters (biggest easy latency win).
    cards = await asyncio.gather(
        *[attach_tmdb_card_by_title(t) for t, _ in recs],
        return_exceptions=False,
    )

    items: List[TFIDFRecItem] = []
    for (title, score), card in zip(recs, cards):
        items.append(TFIDFRecItem(title=title, score=score, tmdb=card))
    return items


async def build_recommendation_bundle(
    details: TMDBMovieDetails,
    query_label: str,
    tfidf_top_n: int,
    genre_limit: int,
) -> SearchBundleResponse:
    """Assemble details + TF-IDF + genre recommendations for the UI."""
    tfidf_items = await build_tfidf_items(
        title_for_local=details.title,
        fallback_title=query_label,
        top_n=tfidf_top_n,
    )
    genre_recs = await genre_recommendations_for_details(details, limit=genre_limit)
    return SearchBundleResponse(
        query=query_label,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )


def load_pickles() -> None:
    """Load frozen ML artifacts into process globals (called once at startup)."""
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Vectorizer is not used for online scoring; kept for parity with training artifacts.
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """App lifespan: open shared HTTP client, load pickles, then clean up."""
    global http_client
    http_client = httpx.AsyncClient(timeout=20.0)
    load_pickles()
    try:
        yield
    finally:
        await http_client.aclose()
        http_client = None


# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Movie Recommender API",
    version="3.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    """Liveness probe for deploys and local smoke tests."""
    return {"status": "ok"}


@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """
    Home feed posters for Streamlit.

    category:
      - trending -> /trending/movie/day
      - popular | top_rated | upcoming | now_playing -> /movie/{category}
    """
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return tmdb_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """
    Raw TMDB keyword search.

    Streamlit uses this for suggestion dropdowns and result grids.
    """
    return await tmdb_search_movies(query=query, page=page)


@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    """Movie details by TMDB id (preferred over title search)."""
    return await tmdb_movie_details(tmdb_id)


@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    """Genre-based TMDB discover for a known movie id."""
    details = await tmdb_movie_details(tmdb_id)
    return await genre_recommendations_for_details(details, limit=limit)


@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    """Debug / simple TF-IDF endpoint: titles + scores only (no posters)."""
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    Bundle by title string: best TMDB match + TF-IDF + genre.

    Prefer `/movie/id/{tmdb_id}/recommendations` when the UI already knows the id.
    """
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for query: {query}"
        )

    details = await tmdb_movie_details(int(best["id"]))
    return await build_recommendation_bundle(
        details=details,
        query_label=query,
        tfidf_top_n=tfidf_top_n,
        genre_limit=genre_limit,
    )


@app.get(
    "/movie/id/{tmdb_id}/recommendations",
    response_model=SearchBundleResponse,
)
async def recommendations_by_id(
    tmdb_id: int,
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    Bundle by TMDB id (used by the details page).

    Avoids rematching the wrong film when titles are ambiguous.
    Local TF-IDF still keys off the TMDB title string (dataset has no TMDB ids).
    """
    details = await tmdb_movie_details(tmdb_id)
    return await build_recommendation_bundle(
        details=details,
        query_label=details.title,
        tfidf_top_n=tfidf_top_n,
        genre_limit=genre_limit,
    )
