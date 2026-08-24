"""NeuroSleep — Streamlit application entry point.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

import streamlit as st

st.set_page_config(
    page_title="NeuroSleep",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.components import inject_swiss_css, header, model_status_cards
from app.state import get_predictor, init_session

inject_swiss_css()
header()

predictor = get_predictor()
init_session()

model_status_cards(predictor)

st.markdown('<div class="swiss-divider-thick"></div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        <div class="swiss-section-title">How to use</div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.9rem; line-height: 1.7; color: #333;">
            <strong>1.</strong> Open <strong>Signal Viewer</strong> to select a Sleep-EDF recording and epoch.<br>
            <strong>2.</strong> Click <strong>RUN PREDICTION</strong> to classify the selected epoch.<br>
            <strong>3.</strong> View the predicted stage, probability distribution, and signal quality.<br>
            <strong>4.</strong> Open <strong>Night Explorer</strong> for a full-night hypnogram view.<br>
            <strong>5.</strong> Open <strong>Model Information</strong> for architecture and official results.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="swiss-section-title">System</div>
        <table class="swiss-table">
            <tr><td>Dataset</td><td>Sleep-EDF</td></tr>
            <tr><td>Window</td><td>10 &times; 30 s</td></tr>
            <tr><td>Sampling</td><td>100 Hz</td></tr>
            <tr><td>Model</td><td>Improved Student</td></tr>
            <tr><td>Parameters</td><td>99,477</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="swiss-divider"></div>', unsafe_allow_html=True)

st.markdown(
    """
    <div style="font-family: 'Inter', sans-serif; font-size: 0.7rem; letter-spacing: 0.1em;
         text-transform: uppercase; color: #999; text-align: center; padding: 1rem 0;">
        VIT Bhopal University &nbsp;&middot;&nbsp; Neuromorphic Sleep Stage Scoring Pipeline
    </div>
    """,
    unsafe_allow_html=True,
)
