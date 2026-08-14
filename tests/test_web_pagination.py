"""Regression tests for the Streamlit catalog-browser pagination flow.

Bug fixed here: st.session_state["cat_page"] was written AFTER the
number_input widget with key="cat_page" was instantiated (inside the
prev/next button bodies), which raises:

    StreamlitAPIException: st.session_state.cat_page cannot be modified
    after the widget with key cat_page is instantiated

The fix moves all writes before instantiation (seed + clamp) and uses
on_click callbacks. These tests simulate the session-state flow without
launching Streamlit.
"""
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")

import importlib.util

APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "task5" / "earthquake-analysis" / "frontend" / "app.py"
)


@pytest.fixture(scope="module")
def app_source() -> str:
    return APP_PATH.read_text(encoding="utf-8")


def _simulate_session(total: int, page_size: int, clicks: list[str]):
    """Replicate the app's session-state pagination logic."""
    state = {}
    total_pages = max(1, (total + page_size - 1) // page_size)

    def run_once(next_click: str | None):
        # seed + clamp (must happen BEFORE widget instantiation)
        if "cat_page" not in state:
            state["cat_page"] = 1
        state["cat_page"] = min(state["cat_page"], total_pages)
        cur_page = state["cat_page"]
        # widget renders here with key="cat_page" ...
        # on_click callbacks fire before the NEXT run
        if next_click == "prev":
            state["cat_page"] = max(1, state["cat_page"] - 1)
        elif next_click == "next":
            state["cat_page"] = min(total_pages, state["cat_page"] + 1)
        return cur_page

    pages = [run_once(click) for click in clicks]
    return state["cat_page"], pages


class TestPaginationFlow:
    def test_next_advances_page(self):
        final, _ = _simulate_session(total=500, page_size=100, clicks=[None, "next", "next"])
        assert final == 3

    def test_prev_never_below_one(self):
        final, _ = _simulate_session(total=500, page_size=100, clicks=[None, "prev", "prev"])
        assert final == 1

    def test_next_clamped_at_last_page(self):
        final, _ = _simulate_session(total=250, page_size=100, clicks=[None, "next", "next", "next", "next"])
        assert final == 3  # ceil(250/100)

    def test_page_resets_smaller_total(self):
        # user on page 5, then new filter shrinks total_pages to 2
        state = {"cat_page": 5}
        total_pages = 2
        state["cat_page"] = min(state["cat_page"], total_pages)
        assert state["cat_page"] == 2


class TestSourceSafety:
    """Static guards: the forbidden pattern must not come back."""

    def test_no_write_after_widget_instantiation(self, app_source):
        # The old bug: inside button body, after number_input was created:
        #   st.session_state["cat_page"] = ...
        # Buttons must use on_click callbacks now.
        assert 'st.session_state["cat_page"] = min(total_pages, cur_page + 1)' not in app_source
        assert 'st.session_state["cat_page"] = max(1, cur_page - 1)' not in app_source
        assert 'on_click=_prev_page' in app_source
        assert 'on_click=_next_page' in app_source

    def test_seed_happens_before_widget(self, app_source):
        seed_pos = app_source.index('st.session_state["cat_page"] = 1')
        widget_pos = app_source.index('key="cat_page"')
        assert seed_pos < widget_pos, "cat_page must be seeded before the number_input widget"
