"""Plotly hypnogram visualization — Swiss Style."""

from typing import Any

SWISS_FONT = "Inter, Helvetica Neue, Arial, sans-serif"


def create_hypnogram(
    stage_indices: list[int],
    stage_names: dict[int, str],
    title: str = "Predicted Hypnogram",
) -> dict[str, Any]:
    """Create a Plotly hypnogram figure.

    Args:
        stage_indices: List of integer stage indices for each epoch.
        stage_names: Mapping from index to stage name.
        title: Plot title.

    Returns:
        Plotly figure dict.
    """
    import plotly.graph_objects as go

    epochs = list(range(len(stage_indices)))
    stage_labels = [stage_names.get(i, "?") for i in stage_indices]

    stage_to_y = {name: idx for idx, name in sorted(stage_names.items())}
    y_values = [stage_to_y.get(s, 0) for s in stage_labels]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=y_values,
        mode="lines",
        line=dict(color="#111111", width=1),
        text=stage_labels,
        hovertemplate="Epoch %{x}<br>%{text}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family=SWISS_FONT, size=13, color="#111"),
        ),
        xaxis=dict(
            title="Epoch",
            tickfont=dict(family=SWISS_FONT, size=10, color="#999"),
            titlefont=dict(family=SWISS_FONT, size=10, color="#999"),
            showgrid=False,
        ),
        yaxis=dict(
            tickvals=list(stage_to_y.values()),
            ticktext=list(stage_to_y.keys()),
            tickfont=dict(family=SWISS_FONT, size=10, color="#555"),
            showgrid=False,
            autorange="reversed",
        ),
        height=300,
        margin=dict(l=60, r=20, t=40, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig
