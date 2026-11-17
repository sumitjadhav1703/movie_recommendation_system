"""
Streamlit frontend for the Movie Recommendation demo (CineMatch).

Talks only to the FastAPI backend — never loads pickles or calls TMDB directly.
Views: home feed / keyword search → movie details → TF-IDF + genre recommendations.
"""

from __future__ import annotations

import html
import os
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env so local MOVIE_API_BASE / secrets work the same as the API process.
load_dotenv()

# =============================
# CONFIG
# =============================
# Prefer MOVIE_API_BASE for local dev; otherwise use the deployed Render API.
DEFAULT_API = "https://movie-recommendation-system-t86z.onrender.com"
API_BASE = (os.getenv("MOVIE_API_BASE") or DEFAULT_API).rstrip("/")
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================
# STYLES
# Cinema-dark chrome; posters carry the visual weight.
# =============================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Source+Sans+3:wght@400;600;700&display=swap');

html, body, [class*="css"] {
  font-family: "Source Sans 3", sans-serif;
}

.block-container {
  padding-top: 1.25rem;
  padding-bottom: 2.5rem;
  max-width: 1280px;
}

.cm-hero {
  margin: 0 0 0.35rem 0;
  font-family: "Bebas Neue", sans-serif;
  font-size: clamp(2.6rem, 5vw, 3.6rem);
  letter-spacing: 0.04em;
  line-height: 1;
  color: #F2F0EA;
}
.cm-hero span { color: #E8B86D; }

.cm-tagline {
  color: #9AA3B2;
  font-size: 1.02rem;
  margin-bottom: 1rem;
}

.cm-section {
  font-family: "Bebas Neue", sans-serif;
  letter-spacing: 0.06em;
  font-size: 1.55rem;
  color: #F2F0EA;
  margin: 0.4rem 0 0.75rem 0;
}

.cm-meta {
  color: #9AA3B2;
  font-size: 0.95rem;
}

.movie-title {
  font-size: 0.88rem;
  line-height: 1.2rem;
  min-height: 2.4rem;
  overflow: hidden;
  margin-top: 0.35rem;
  color: #E8ECF2;
}

.poster-shell {
  border-radius: 10px;
  overflow: hidden;
  background: linear-gradient(160deg, #1A1F29 0%, #12151C 100%);
  border: 1px solid rgba(232, 184, 109, 0.12);
  padding: 0.45rem;
}

.details-panel {
  border-radius: 14px;
  border: 1px solid rgba(232, 184, 109, 0.16);
  background: linear-gradient(145deg, rgba(26,31,41,0.95), rgba(14,17,22,0.9));
  padding: 1.1rem 1.25rem 1.25rem 1.25rem;
}

div[data-testid="stSidebar"] {
  background: #12151C;
}

/* Soften default Streamlit chrome a bit */
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# STATE + ROUTING
# =============================
HOME_CATEGORIES = ["trending", "popular", "top_rated", "now_playing", "upcoming"]

if "view" not in st.session_state:
    st.session_state.view = "home"  # home | details
if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None
# Keep feed category outside the selectbox widget so details → home does not reset it.
if "home_category" not in st.session_state:
    st.session_state.home_category = "trending"

qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_view in ("home", "details"):
    st.session_state.view = qp_view
if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except (TypeError, ValueError):
        pass


def goto_home() -> None:
    """Navigate to the home feed and clear the selected movie from the URL."""
    st.session_state.view = "home"
    st.query_params["view"] = "home"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()


def goto_details(tmdb_id: int) -> None:
    """Open the details view for a TMDB movie id."""
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))
    st.rerun()


# =============================
# API HELPERS
# =============================
@st.cache_data(ttl=60)
def _cached_api_get(path: str, params_key: Tuple[Tuple[str, Any], ...]) -> Any:
    """
    Cached happy-path GET. Raises on HTTP/network errors so failures are NOT cached
    (Render cold-start 502s must not stick for the full TTL).
    """
    params = dict(params_key) if params_key else None
    r = requests.get(f"{API_BASE}{path}", params=params, timeout=45)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def api_get_json(
    path: str, params: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Optional[str]]:
    """
    GET JSON from the FastAPI backend.

    Returns (data, None) on success, or (None, error_message) on failure.
    Successful bodies are cached briefly; errors always retry on the next call.
    """
    try:
        params_key = tuple(sorted((params or {}).items()))
        return _cached_api_get(path, params_key), None
    except Exception as e:
        return None, f"Request failed: {e}"


def poster_grid(
    cards: Optional[List[dict]], cols: int = 6, key_prefix: str = "grid"
) -> None:
    """Render a responsive poster grid with Open buttons."""
    if not cards:
        st.info("No movies to show.")
        return

    rows = (len(cards) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        colset = st.columns(cols)
        for c in range(cols):
            if idx >= len(cards):
                break
            m = cards[idx]
            idx += 1

            tmdb_id = m.get("tmdb_id")
            title = m.get("title", "Untitled")
            poster = m.get("poster_url")

            with colset[c]:
                st.markdown("<div class='poster-shell'>", unsafe_allow_html=True)
                if poster:
                    st.image(poster, use_column_width=True)
                else:
                    st.caption("No poster")
                if st.button("Open", key=f"{key_prefix}_{r}_{c}_{idx}_{tmdb_id}"):
                    if tmdb_id:
                        goto_details(int(tmdb_id))
                st.markdown(
                    f"<div class='movie-title'>{html.escape(str(title))}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)


def to_cards_from_tfidf_items(tfidf_items: Optional[List[dict]]) -> List[dict]:
    """Convert TF-IDF recommendation items into poster-grid cards."""
    cards: List[dict] = []
    for x in tfidf_items or []:
        tmdb = x.get("tmdb") or {}
        if tmdb.get("tmdb_id"):
            cards.append(
                {
                    "tmdb_id": tmdb["tmdb_id"],
                    "title": tmdb.get("title") or x.get("title") or "Untitled",
                    "poster_url": tmdb.get("poster_url"),
                }
            )
    return cards


def parse_tmdb_search_to_cards(
    data: Any, keyword: str, limit: int = 24
) -> Tuple[List[Tuple[str, int]], List[dict]]:
    """
    Normalize either raw TMDB search JSON or card lists into suggestions + cards.

    Supports:
      1) {"results":[{id,title,poster_path,...}]}
      2) [{tmdb_id,title,poster_url,...}]
    """
    keyword_l = keyword.strip().lower()
    raw_items: List[dict] = []

    if isinstance(data, dict) and "results" in data:
        for m in data.get("results") or []:
            title = (m.get("title") or "").strip()
            tmdb_id = m.get("id")
            poster_path = m.get("poster_path")
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": f"{TMDB_IMG}{poster_path}" if poster_path else None,
                    "release_date": m.get("release_date", ""),
                }
            )
    elif isinstance(data, list):
        for m in data:
            tmdb_id = m.get("tmdb_id") or m.get("id")
            title = (m.get("title") or "").strip()
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": m.get("poster_url"),
                    "release_date": m.get("release_date", ""),
                }
            )
    else:
        return [], []

    matched = [x for x in raw_items if keyword_l in x["title"].lower()]
    final_list = matched if matched else raw_items

    # Labels must be unique — duplicate "Title (year)" entries would collapse in a dict.
    suggestions: List[Tuple[str, int]] = []
    used_labels: set[str] = set()
    for x in final_list[:10]:
        year = (x.get("release_date") or "")[:4]
        base = f"{x['title']} ({year})" if year else x["title"]
        label = base if base not in used_labels else f"{base} · id {x['tmdb_id']}"
        used_labels.add(label)
        suggestions.append((label, x["tmdb_id"]))

    cards = [
        {"tmdb_id": x["tmdb_id"], "title": x["title"], "poster_url": x["poster_url"]}
        for x in final_list[:limit]
    ]
    return suggestions, cards


def render_header() -> None:
    """Brand-forward page header (hero signal for the demo)."""
    st.markdown(
        '<div class="cm-hero">CINE<span>MATCH</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cm-tagline">Find a film. Open it. Get similar picks from content-based TF-IDF plus TMDB genres.</div>',
        unsafe_allow_html=True,
    )


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("### Navigate")
    if st.button("Home", use_container_width=True):
        goto_home()

    st.markdown("---")
    if st.session_state.view == "home":
        st.markdown("### Home feed")
        current = st.session_state.home_category
        try:
            cat_index = HOME_CATEGORIES.index(current)
        except ValueError:
            cat_index = 0
        selected_category = st.selectbox(
            "Category",
            HOME_CATEGORIES,
            index=cat_index,
            key="home_category_select",
        )
        st.session_state.home_category = selected_category
    else:
        st.caption(
            f"Feed category saved: `{st.session_state.home_category}` "
            "(change it on Home)."
        )

    home_category = st.session_state.home_category
    grid_cols = st.slider("Grid columns", 4, 8, 6)
    st.markdown("---")
    st.caption(f"API: `{API_BASE}`")

# =============================
# HEADER
# =============================
render_header()
st.divider()

# ==========================================================
# VIEW: HOME
# ==========================================================
if st.session_state.view == "home":
    typed = st.text_input(
        "Search movies",
        placeholder="Try: inception, batman, spider…",
        label_visibility="visible",
    )

    if typed.strip():
        if len(typed.strip()) < 2:
            st.caption("Type at least 2 characters.")
        else:
            with st.spinner("Searching TMDB…"):
                data, err = api_get_json(
                    "/tmdb/search", params={"query": typed.strip()}
                )

            if err or data is None:
                st.error(f"Search failed: {err}")
            else:
                suggestions, cards = parse_tmdb_search_to_cards(
                    data, typed.strip(), limit=24
                )

                if suggestions:
                    labels = ["— Select a suggestion —"] + [s[0] for s in suggestions]
                    selected = st.selectbox(
                        "Suggestions",
                        labels,
                        index=0,
                        key="search_suggestion_box",
                    )
                    # Navigate by index (not label→id dict) so duplicate titles stay correct.
                    if selected != "— Select a suggestion —":
                        pick = labels.index(selected) - 1
                        goto_details(suggestions[pick][1])
                else:
                    st.info("No suggestions. Try another keyword.")

                st.markdown(
                    '<div class="cm-section">Results</div>', unsafe_allow_html=True
                )
                poster_grid(cards, cols=grid_cols, key_prefix="search_results")
    else:
        st.markdown(
            f'<div class="cm-section">Home — {home_category.replace("_", " ").title()}</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Loading feed…"):
            home_cards, err = api_get_json(
                "/home", params={"category": home_category, "limit": 24}
            )
        if err:
            st.error(f"Home feed failed: {err}")
            st.stop()
        if not home_cards:
            st.info("No movies in this feed right now.")
        else:
            poster_grid(home_cards, cols=grid_cols, key_prefix="home_feed")

# ==========================================================
# VIEW: DETAILS
# ==========================================================
elif st.session_state.view == "details":
    tmdb_id = st.session_state.selected_tmdb_id
    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("Back to Home"):
            goto_home()
        st.stop()

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.markdown(
            '<div class="cm-section">Movie details</div>', unsafe_allow_html=True
        )
    with top_r:
        if st.button("Back to Home", use_container_width=True):
            goto_home()

    with st.spinner("Loading details…"):
        data, err = api_get_json(f"/movie/id/{tmdb_id}")
    if err or not data:
        st.error(f"Could not load details: {err or 'Unknown error'}")
        st.stop()

    left, right = st.columns([1, 2.2], gap="large")

    with left:
        st.markdown("<div class='poster-shell'>", unsafe_allow_html=True)
        if data.get("poster_url"):
            st.image(data["poster_url"], use_column_width=True)
        else:
            st.caption("No poster")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='details-panel'>", unsafe_allow_html=True)
        st.markdown(f"## {html.escape(str(data.get('title', '') or ''))}")
        release = html.escape(str(data.get("release_date") or "—"))
        genres = html.escape(
            ", ".join([g["name"] for g in data.get("genres", [])]) or "—"
        )
        st.markdown(
            f"<div class='cm-meta'>Release: {release} · Genres: {genres}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### Overview")
        st.write(data.get("overview") or "No overview available.")
        st.markdown("</div>", unsafe_allow_html=True)

    if data.get("backdrop_url"):
        st.markdown('<div class="cm-section">Backdrop</div>', unsafe_allow_html=True)
        st.image(data["backdrop_url"], use_column_width=True)

    st.divider()
    st.markdown(
        '<div class="cm-section">Recommendations</div>', unsafe_allow_html=True
    )

    # Prefer id-based bundle. On older deploys, keep genre tied to this tmdb_id
    # (do not rematch a different film via /movie/search for genre).
    with st.spinner("Finding similar movies…"):
        bundle, err2 = api_get_json(
            f"/movie/id/{tmdb_id}/recommendations",
            params={"tfidf_top_n": 12, "genre_limit": 12},
        )
        if err2 or not bundle:
            title = (data.get("title") or "").strip()
            title_bundle, err_title = (None, "no title")
            if title:
                title_bundle, err_title = api_get_json(
                    "/movie/search",
                    params={
                        "query": title,
                        "tfidf_top_n": 12,
                        "genre_limit": 12,
                    },
                )
            genre_only, err_genre = api_get_json(
                "/recommend/genre", params={"tmdb_id": tmdb_id, "limit": 12}
            )
            if title_bundle or genre_only:
                bundle = {
                    "tfidf_recommendations": (title_bundle or {}).get(
                        "tfidf_recommendations", []
                    ),
                    "genre_recommendations": genre_only
                    or (title_bundle or {}).get("genre_recommendations", []),
                }
                err2 = None
            else:
                err2 = err_title or err_genre or err2

    if not err2 and bundle:
        st.markdown("##### Similar (TF-IDF)")
        poster_grid(
            to_cards_from_tfidf_items(bundle.get("tfidf_recommendations")),
            cols=grid_cols,
            key_prefix="details_tfidf",
        )
        st.markdown("##### More like this (Genre)")
        poster_grid(
            bundle.get("genre_recommendations", []),
            cols=grid_cols,
            key_prefix="details_genre",
        )
    else:
        st.warning(
            f"No recommendations available right now. ({err2 or 'unknown'})"
        )
