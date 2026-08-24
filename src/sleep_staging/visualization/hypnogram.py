"""Hypnogram visualization for sleep-night explorer."""

from typing import Any

import numpy as np

from ..config import STAGE_COLORS, STAGE_NAMES


def create_hypnogram(
    predictions: list[int],
    stage_names: dict[int, str] | None = None,
    epoch_seconds: int = 30,
    title: str = "Predicted Hypnogram",
) -> dict[str, Any]:
    """Create a Plotly hypnogram figure from a sequence of predictions.

    Args:
        predictions: List of integer stage indices (0–4).
        stage_names: Mapping from index to name.
        epoch_seconds: Duration of each epoch.
        title: Plot title.

    Returns:
        Plotly figure dict.
    """
    import plotly.graph_objects as go

    if stage_names is None:
        stage_names = STAGE_NAMES

    n_epochs = len(predictions)
    times = np.arange(n_epochs) * epoch_seconds / 60.0  # minutes
    stages_numeric = np.array(predictions)

    colors = [STAGE_COLORS.get(STAGE_NAMES[p], "#888") for p in predictions]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times,
        y=stages_numeric,
        mode="lines+markers",
        marker=dict(size=4, color=colors),
        line=dict(color="#555", width=1),
        hovertemplate="Time: %{x:.1f} min<br>Stage: %{text}<extra></extra>",
        text=[stage_names.get(p, "?") for p in predictions],
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Time (min)",
        yaxis=dict(
            tickvals=list(STAGE_NAMES.keys()),
            ticktext=list(STAGE_NAMES.values()),
            autorange="reversed",
        ),
        height=300,
        margin=dict(l=60, r=20, t=40, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig
