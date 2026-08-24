"""NeuroSleep — Streamlit application entry point.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure the project src/ is importable
_repo = Path(__file__).resolve().parents[1]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

import streamlit as st

from app.components import header, model_status_cards
from app.state import get_predictor, init_session

st.set_page_config(
    page_title="NeuroSleep",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

header()

predictor = get_predictor()
init_session()

model_status_cards(predictor)

st.info(
    "Use the **sidebar** to navigate between the Dashboard, Signal Viewer, "
    "Night Explorer, and Model Information pages."
)

st.markdown("---")
st.markdown(
    """
    **Quick start:**
    1. Go to **Signal Viewer** and select a recording + epoch.
    2. Click **RUN PREDICTION**.
    3. View the predicted stage and probability distribution.
    """
)
