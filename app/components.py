"""Reusable Streamlit UI components."""

import streamlit as st

from sleep_staging.config import STAGE_COLORS


def header() -> None:
    """Render the application header."""
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <h1 style="margin:0; font-size:2.2rem; letter-spacing:0.05em; color:#1a1a2e;">
                NEUROSLEEP
            </h1>
            <p style="margin:0; font-size:0.95rem; color:#666;">
                Neuromorphic Sleep Stage Scoring &mdash; Sleep-EDF Research Demonstrator
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_status_cards(predictor) -> None:
    """Display model / parameters / device / status cards."""
    info = predictor.model_info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MODEL", info["name"])
    c2.metric("PARAMETERS", f"{info['parameters']:,}")
    c3.metric("DEVICE", info["device"].upper())
    c4.metric("STATUS", "READY")


def prediction_block(result) -> None:
    """Display the current prediction prominently."""
    color = STAGE_COLORS.get(result.stage, "#333")
    st.markdown(
        f"""
        <div style="text-align:center; padding:1.5rem; border-radius:12px;
                     background:linear-gradient(135deg, {color}15, {color}08);
                     border:2px solid {color}40; margin:1rem 0;">
            <p style="margin:0; font-size:0.85rem; color:#888; text-transform:uppercase;
                       letter-spacing:0.1em;">Current Sleep Stage</p>
            <p style="margin:0.3rem 0; font-size:3rem; font-weight:700; color:{color};">
                {result.stage}
            </p>
            <p style="margin:0; font-size:1.1rem; color:#555;">
                Confidence: {result.confidence:.1%} &nbsp;&bull;&nbsp;
                Latency: {result.latency_ms:.1f} ms
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def qc_panel(qc_results: list) -> None:
    """Render signal quality status for each channel."""
    all_pass = all(r.passed for r in qc_results)
    status = "READY FOR INFERENCE" if all_pass else "QC WARNING"

    with st.expander("Signal Quality", expanded=not all_pass):
        for r in qc_results:
            icon = "PASS" if r.passed else "FAIL"
            st.text(f"  {r.channel_name:<15s} {icon}")
        if all_pass:
            st.success(f"QC STATUS: {status}")
        else:
            st.warning(
                "QC WARNING — This input contains a quality-control flag. "
                "Prediction is still displayed for research demonstration."
            )
