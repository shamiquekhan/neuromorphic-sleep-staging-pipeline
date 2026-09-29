"""Session state helpers for the Streamlit application."""

import json
from pathlib import Path

import streamlit as st

from sleep_staging.config import CHECKPOINT_PATH, RESULTS_PATH, StudentConfig


def _resolve_checkpoint() -> Path:
    """Return the deployed 99,477-parameter checkpoint."""
    if CHECKPOINT_PATH.exists():
        return CHECKPOINT_PATH
    raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}")


@st.cache_data
def load_final_metrics() -> dict:
    """Load the final submission metrics (results/final/final_metrics.json, EXP-FULL-AUG30)."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_primary_benchmark() -> dict:
    """Load the historical person-level benchmark (mean/std schema)."""
    path = RESULTS_PATH.parent.parent / "research" / "EXP-BENCH-PERSON" / "final_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_submission_per_class() -> dict:
    """Load per-class precision/recall/F1 for the final submission test set."""
    path = RESULTS_PATH.parent / "per_class_metrics.csv"
    if not path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(path)
    return {
        row["class"]: {
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1-score"],
        }
        for _, row in df.iterrows()
    }


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
