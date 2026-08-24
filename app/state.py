"""Session state helpers for the Streamlit application."""

import streamlit as st

from sleep_staging.config import CHECKPOINT_PATH, StudentConfig


def init_session() -> None:
    """Initialize all session-state keys on first load."""
    defaults = {
        "selected_subject": None,
        "selected_epoch": 50,
        "selected_target_epoch": 9,
        "current_prediction": None,
        "current_sequence": None,
        "demo_mode": False,
        "run_inference": False,
        "model_mode": "base",
        "adapter_path": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_predictor(adapter_path: str | None = None):
    """Return the cached SleepStagePredictor.

    Uses st.cache_resource so the model is loaded once per server process.
    If adapter_path changes, a new predictor is created.
    """
    from sleep_staging.inference import SleepStagePredictor

    cache_key = f"predictor_{adapter_path}"

    @st.cache_resource
    def _load(ckpt_path: str, adapt_path: str | None):
        return SleepStagePredictor(
            checkpoint_path=ckpt_path,
            device="cpu",
            adapter_path=adapt_path,
        )

    return _load(str(CHECKPOINT_PATH), adapter_path)
