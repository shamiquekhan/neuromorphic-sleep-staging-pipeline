"""Session state helpers for the Streamlit application."""

import json
from pathlib import Path

import streamlit as st

from sleep_staging.config import CHECKPOINT_PATH, FALLBACK_CHECKPOINT_PATH, RESULTS_PATH, StudentConfig


def _resolve_checkpoint() -> Path:
    """Return the final checkpoint, falling back to base if needed."""
    if CHECKPOINT_PATH.exists():
        return CHECKPOINT_PATH
    if FALLBACK_CHECKPOINT_PATH.exists():
        return FALLBACK_CHECKPOINT_PATH
    raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}")


@st.cache_data
def load_final_metrics() -> dict:
    """Load the authoritative final metrics from results/final/final_metrics.json."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def init_session() -> None:
    """Initialize all session-state keys on first load."""
    defaults = {
        "selected_subject": None,
        "selected_epoch": 50,
        "selected_target_epoch": 9,
        "current_prediction": None,
        "current_sequence": None,
        "demo_mode": True,
        "run_inference": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_predictor():
    """Return the cached SleepStagePredictor.

    Uses st.cache_resource so the model is loaded once per server process.
    """
    from sleep_staging.inference import SleepStagePredictor

    @st.cache_resource
    def _load(ckpt_path: str):
        return SleepStagePredictor(
            checkpoint_path=ckpt_path,
            device="cpu",
        )

    return _load(str(_resolve_checkpoint()))
