"""Model Information — Architecture, results, and reproducibility."""

import json
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

import streamlit as st

from app.components import inject_swiss_css, header, section_title
from app.state import get_predictor
from sleep_staging.config import CHECKPOINT_PATH

st.set_page_config(page_title="NeuroSleep — Model Information", page_icon=None, layout="wide")
inject_swiss_css()
header()

predictor = get_predictor()
info = predictor.model_info

# Load 100-subject aggregate metrics
AGGREGATE_PATH = _repo / "results" / "100_subject_adaptation" / "final" / "aggregate_metrics.json"
metrics_100 = {}
if AGGREGATE_PATH.exists():
    with open(AGGREGATE_PATH) as f:
        metrics_100 = json.load(f)

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

# ── 100-Subject Benchmark Results ───────────────────────────────────────
if metrics_100 and "full_finetune" in metrics_100:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    ft = metrics_100["full_finetune"]
    frozen = metrics_100.get("frozen", {})
    lora = metrics_100.get("lora_cnn_head", {})

    section_title("100-Subject Benchmark — Full Fine-Tuning (Authoritative)")

    o = ft["overall"]
    st.markdown(
        f'<div class="swiss-grid-4">'
        f'<div><div class="sz-label">Accuracy</div><div class="sz-display">{o["accuracy"]["mean"]:.1%}</div>'
        f'<div class="sz-caption">&plusmn; {o["accuracy"]["std"]:.1%}</div></div>'
        f'<div><div class="sz-label">Cohen&rsquo;s &kappa;</div><div class="sz-display">{o["kappa"]["mean"]:.3f}</div>'
        f'<div class="sz-caption">&plusmn; {o["kappa"]["std"]:.3f}</div></div>'
        f'<div><div class="sz-label">Macro F1</div><div class="sz-display">{o["macro_f1"]["mean"]:.3f}</div>'
        f'<div class="sz-caption">&plusmn; {o["macro_f1"]["std"]:.3f}</div></div>'
        f'<div><div class="sz-label">Weighted F1</div><div class="sz-display">{o["weighted_f1"]["mean"]:.3f}</div>'
        f'<div class="sz-caption">&plusmn; {o["weighted_f1"]["std"]:.3f}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Per-class
    if "per_class" in ft:
        section_title("Per-Class Performance (Full Fine-Tuning)")
        rows = (
            '<table class="swiss-table">'
            '<tr><th>Stage</th><th>F1</th><th>Precision</th><th>Recall</th></tr>'
        )
        for stage in ["Wake", "N1", "N2", "N3", "REM"]:
            if stage in ft["per_class"]:
                s = ft["per_class"][stage]
                rows += (
                    f'<tr><td>{stage}</td>'
                    f'<td>{s["f1"]["mean"]:.3f} &plusmn; {s["f1"]["std"]:.3f}</td>'
                    f'<td>{s["precision"]["mean"]:.3f}</td>'
                    f'<td>{s["recall"]["mean"]:.3f}</td></tr>'
                )
        rows += "</table>"
        st.markdown(rows, unsafe_allow_html=True)

    # Adaptation comparison
    if frozen and lora:
        section_title("Adaptation Method Comparison")
        rows = (
            '<table class="swiss-table">'
            '<tr><th>Model</th><th>Params</th><th>Accuracy</th><th>&kappa;</th><th>Macro F1</th></tr>'
        )
        for label, data, params in [
            ("Frozen", frozen, "0"),
            ("LoRA CNN+Head", lora, "1,448"),
            ("Full Fine-Tuning", ft, "99,477"),
        ]:
            o = data["overall"]
            rows += (
                f'<tr><td>{label}</td><td>{params}</td>'
                f'<td>{o["accuracy"]["mean"]:.1%} &plusmn; {o["accuracy"]["std"]:.1%}</td>'
                f'<td>{o["kappa"]["mean"]:.3f} &plusmn; {o["kappa"]["std"]:.3f}</td>'
                f'<td>{o["macro_f1"]["mean"]:.3f} &plusmn; {o["macro_f1"]["std"]:.3f}</td></tr>'
            )
        rows += "</table>"
        st.markdown(rows, unsafe_allow_html=True)

else:
    # Fallback: try loading from results/final/
    from app.state import load_final_metrics
    metrics = load_final_metrics()
    if metrics:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        acc = metrics.get("accuracy", {})
        kappa = metrics.get("cohen_kappa", {})
        macro = metrics.get("macro_f1", {})
        weighted = metrics.get("weighted_f1", {})

        section_title("Results (Development — 15-Subject 4-Fold CV)")
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
    else:
        st.warning("No results found. Run the benchmark scripts first.")

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
    "<strong>Dataset:</strong> Sleep-EDF Expanded, 92 subjects (PhysioNet)<br>"
    "<strong>Training window:</strong> 10 &times; 30 s epochs (300 s context)<br>"
    "<strong>Final model:</strong> Improved Student (Full Fine-Tuning, 99,477 params)<br>"
    "<strong>Training:</strong> All-position supervision + N1/REM class weighting (2x)<br>"
    "<strong>Evaluation:</strong> 10-fold subject-level CV, 3 seeds (42, 43, 44)"
    "</div>",
    unsafe_allow_html=True,
)
