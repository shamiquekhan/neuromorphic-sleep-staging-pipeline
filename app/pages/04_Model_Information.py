"""Model Information — Architecture, results, and reproducibility."""

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

import streamlit as st

from app.components import inject_swiss_css, header, section_title
from app.state import get_predictor, load_final_metrics
from sleep_staging.config import CHECKPOINT_PATH

st.set_page_config(page_title="NeuroSleep — Model Information", page_icon=None, layout="wide")
inject_swiss_css()
header()

predictor = get_predictor()
info = predictor.model_info
metrics = load_final_metrics()

st.markdown('<div class="divider-thick"></div>', unsafe_allow_html=True)

# ── Architecture + Properties ───────────────────────────────────────────
st.markdown(
    f'<div class="swiss-grid-2">'
    # Left: Properties
    f'<div>'
    f'<div class="sz-label" style="margin-bottom:0.75rem;">Properties</div>'
    f'<table class="swiss-table">'
    f"<tr><td>Parameters</td><td>{info['parameters']:,}</td></tr>"
    f"<tr><td>Classes</td><td>5 (Wake, N1, N2, N3, REM)</td></tr>"
    f"<tr><td>Context</td><td>10 &times; 30 s = 300 s</td></tr>"
    f"<tr><td>Sampling Rate</td><td>100 Hz</td></tr>"
    f"<tr><td>Device</td><td>{info['device'].upper()}</td></tr>"
    f"<tr><td>Checkpoint</td><td>{CHECKPOINT_PATH.name}</td></tr>"
    f"</table>"
    f"</div>"
    # Right: Architecture
    f'<div>'
    f'<div class="sz-label" style="margin-bottom:0.75rem;">Architecture</div>'
    f'<div class="arch-diagram">'
    f"PSG Input (4 channels)\n"
    f"      |\n"
    f"      v\n"
    f"Multi-Resolution Stem\n"
    f"      |\n"
    f"      v\n"
    f"Depthwise-Separable CNN\n"
    f"      |\n"
    f"      +---+\n"
    f"      |   |\n"
    f"      v   v\n"
    f"CNN   Gabor FEB\n"
    f"      |   |\n"
    f"      +---+\n"
    f"          v\n"
    f"    Feature Fusion\n"
    f"          |\n"
    f"          v\n"
    f"     2-layer GRU\n"
    f"          |\n"
    f"          v\n"
    f"    5-class head\n"
    f"          |\n"
    f"          v\n"
    f"Wake / N1 / N2 / N3 / REM"
    f"</div>"
    f"</div>"
    f"</div>",
    unsafe_allow_html=True,
)

# ── Official Results ────────────────────────────────────────────────────
if metrics:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    acc = metrics.get("accuracy", {})
    kappa = metrics.get("cohen_kappa", {})
    macro = metrics.get("macro_f1", {})
    weighted = metrics.get("weighted_f1", {})

    section_title("Official Results — 15-Subject 4-Fold CV")

    st.markdown(
        f'<div class="swiss-grid-4">'
        f'<div><div class="sz-label">Accuracy</div><div class="sz-display">{acc.get("mean",0):.1%}</div>'
        f'<div class="sz-caption">&plusmn; {acc.get("std",0):.1%}</div></div>'
        f'<div><div class="sz-label">Cohen&rsquo;s &kappa;</div><div class="sz-display">{kappa.get("mean",0):.3f}</div>'
        f'<div class="sz-caption">&plusmn; {kappa.get("std",0):.3f}</div></div>'
        f'<div><div class="sz-label">Macro F1</div><div class="sz-display">{macro.get("mean",0):.3f}</div>'
        f'<div class="sz-caption">&plusmn; {macro.get("std",0):.3f}</div></div>'
        f'<div><div class="sz-label">Weighted F1</div><div class="sz-display">{weighted.get("mean",0):.3f}</div>'
        f'<div class="sz-caption">&plusmn; {weighted.get("std",0):.3f}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Per-class
    per_class = metrics.get("per_class", {})
    if per_class:
        section_title("Per-Class Performance")
        rows = (
            '<table class="swiss-table">'
            '<tr><th>Stage</th><th>F1</th><th>Precision</th><th>Recall</th></tr>'
        )
        for stage in ["Wake", "N1", "N2", "N3", "REM"]:
            if stage in per_class:
                s = per_class[stage]
                rows += (
                    f'<tr><td>{stage}</td>'
                    f'<td>{s["f1_mean"]:.3f} &plusmn; {s["f1_std"]:.3f}</td>'
                    f'<td>{s["precision_mean"]:.3f}</td>'
                    f'<td>{s["recall_mean"]:.3f}</td></tr>'
                )
        rows += "</table>"
        st.markdown(rows, unsafe_allow_html=True)

    # Fold breakdown
    fold_metrics = metrics.get("fold_metrics", [])
    if fold_metrics:
        section_title("Fold Breakdown")
        test_subjects = ["SC4001", "SC4002", "SC4011", "SC4012"]
        rows = (
            '<table class="swiss-table">'
            '<tr><th>Fold</th><th>Test Subject</th><th>Accuracy</th><th>Kappa</th><th>Macro F1</th></tr>'
        )
        for fm in fold_metrics:
            subj = test_subjects[fm["fold"] - 1]
            rows += (
                f'<tr><td>{fm["fold"]}</td><td>{subj}</td>'
                f'<td>{fm["accuracy"]:.1%}</td><td>{fm["kappa"]:.3f}</td>'
                f'<td>{fm["macro_f1"]:.3f}</td></tr>'
            )
        rows += "</table>"
        st.markdown(rows, unsafe_allow_html=True)
else:
    st.warning("Final metrics not found. Run scripts/evaluate_final_model.py")

# ── Reproducibility ─────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
section_title("Reproducibility")

try:
    import torch, mne, streamlit as stlib
    st.markdown(
        '<table class="swiss-table" style="max-width:400px;">'
        f"<tr><td>PyTorch</td><td>{torch.__version__}</td></tr>"
        f"<tr><td>MNE</td><td>{mne.__version__}</td></tr>"
        f"<tr><td>Streamlit</td><td>{stlib.__version__}</td></tr>"
        "</table>",
        unsafe_allow_html=True,
    )
except ImportError:
    st.info("Install all dependencies for full version info.")

st.markdown(
    '<div class="sz-body" style="margin-top:1rem;">'
    "<strong>Dataset:</strong> Sleep-EDF Expanded, 15 subjects (PhysioNet)<br>"
    "<strong>Training window:</strong> 10 &times; 30 s epochs (300 s context)<br>"
    "<strong>Final model:</strong> Improved Student (Full Fine-Tuning)<br>"
    "<strong>Training:</strong> All-position supervision + N1/REM class weighting<br>"
    "<strong>Cross-validation:</strong> 4-fold subject-level CV"
    "</div>",
    unsafe_allow_html=True,
)
