"""Plotly signal visualization for PSG channels — Swiss Style."""

from typing import Any

import numpy as np

STAGE_COLORS = {
    "Wake": "#E63946",
    "N1": "#FFA07A",
    "N2": "#4ECDC4",
    "N3": "#2C73D2",
    "REM": "#9B59B6",
}

CHANNEL_NAMES = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG", "EMG"]

SWISS_FONT = "Inter, Helvetica Neue, Arial, sans-serif"


def create_signal_figure(
    epoch: np.ndarray,
    channel_names: list[str] | None = None,
    fs: int = 100,
    epoch_seconds: int = 30,
    title: str = "PSG Epoch",
) -> dict[str, Any]:
    """Create a Plotly figure dict for a single epoch — Swiss Style.

    Args:
        epoch: ``[n_channels, n_samples]`` array.
        channel_names: Labels for each channel.
        fs: Sampling rate.
        epoch_seconds: Duration of the epoch.
        title: Plot title.

    Returns:
        Plotly figure dict suitable for ``st.plotly_chart``.
    """
    import plotly.graph_objects as go

    if channel_names is None:
        channel_names = CHANNEL_NAMES[:epoch.shape[0]]

    n_channels = epoch.shape[0]
    time_axis = np.linspace(0, epoch_seconds, epoch.shape[-1])

    fig = go.Figure()

    colors = ["#111111", "#555555", "#E63946", "#9B59B6"]

    for i in range(n_channels):
        offset = (n_channels - 1 - i) * 3.5
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=epoch[i] + offset,
            name=channel_names[i],
            line=dict(color=colors[i % len(colors)], width=0.7),
            hovertemplate=f"{channel_names[i]}<br>Time: %{{x:.1f}}s<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family=SWISS_FONT, size=13, color="#111"),
        ),
        xaxis_title="Time (s)",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(family=SWISS_FONT, size=10, color="#999"),
            titlefont=dict(family=SWISS_FONT, size=10, color="#999"),
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=380,
        margin=dict(l=20, r=20, t=40, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family=SWISS_FONT, size=10, color="#555"),
        ),
        hovermode="x unified",
    )

    return fig


def create_probability_figure(probabilities: dict[str, float]) -> dict[str, Any]:
    """Create a horizontal bar chart of class probabilities — Swiss Style."""
    import plotly.graph_objects as go

    stages = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [STAGE_COLORS.get(s, "#888") for s in stages]

    fig = go.Figure(go.Bar(
        x=values,
        y=stages,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        textfont=dict(family=SWISS_FONT, size=10, color="#555"),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Probability",
        xaxis=dict(
            range=[0, 1.05],
            tickfont=dict(family=SWISS_FONT, size=10, color="#999"),
            titlefont=dict(family=SWISS_FONT, size=10, color="#999"),
        ),
        yaxis=dict(
            tickfont=dict(family=SWISS_FONT, size=11, color="#111"),
        ),
        height=220,
        margin=dict(l=60, r=20, t=10, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.3,
    )

    return fig
