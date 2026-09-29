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

# ── Primary Benchmark (92-Subject, from scratch, seed 42) ─────────────────
from app.state import load_primary_benchmark

primary_metrics = load_primary_benchmark()
if primary_metrics:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    o = primary_metrics
    acc, kappa = o.get("accuracy", {}), o.get("cohen_kappa", {})
    macro, weighted = o.get("macro_f1", {}), o.get("weighted_f1", {})

    section_title("Person-Level Primary Benchmark — From Scratch (Seed 42)")
    st.caption(
        "EXP-BENCH-PERSON · 92-record eligible cohort (52 persons; 100 "
        "downloaded, 8 wake-only excluded) · 10-fold person-level CV · "
        "seeds 42/43/44 (30 folds). Authoritative numbers: docs/results.md"
    )
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

    if "per_class" in o:
        section_title("Per-Class Performance (Primary Benchmark)")
        rows = (
            '<table class="swiss-table">'
            '<tr><th>Stage</th><th>F1</th><th>Precision</th><th>Recall</th></tr>'
        )
        for stage in ["Wake", "N1", "N2", "N3", "REM"]:
            if stage in o["per_class"]:
                s = o["per_class"][stage]
                rows += (
                    f'<tr><td>{stage}</td>'
                    f'<td>{s["f1_mean"]:.3f} &plusmn; {s["f1_std"]:.3f}</td>'
                    f'<td>{s["precision_mean"]:.3f}</td>'
                    f'<td>{s["recall_mean"]:.3f}</td></tr>'
                )
        rows += "</table>"
        st.markdown(rows, unsafe_allow_html=True)

# ── Final Submission (EXP-FULL-AUG30, person-level holdout) ──────────────────
from app.state import load_final_metrics, load_submission_per_class

submission = load_final_metrics()
if submission:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    section_title("Final Submission — EXP-FULL-AUG30 (Person-Level Holdout)")
    st.caption(
        "Complete Sleep-EDF Expanded corpus (197 recordings / 100 subjects) → "
        "person-level 70/15/16 split (69/15/16 subjects, seed 42), "
        "≤30 epochs with patience-5 early stopping on validation macro F1 "
        "(best epoch 12, stopped 17/30) → single evaluation on 16 held-out "
        "test subjects (7,220 windows / 72,200 labels, stride-10). "
        "Authoritative numbers: results/final/final_metrics.json"
    )
    st.markdown(
        f'<div class="swiss-grid-4">'
        f'<div><div class="sz-label">Accuracy</div><div class="sz-display">{submission.get("test_accuracy", 0):.1%}</div>'
        f'<div class="sz-caption">16 test subjects</div></div>'
        f'<div><div class="sz-label">Cohen&rsquo;s &kappa;</div><div class="sz-display">{submission.get("cohen_kappa", 0):.3f}</div>'
        f'<div class="sz-caption">person-level holdout</div></div>'
        f'<div><div class="sz-label">Macro F1</div><div class="sz-display">{submission.get("macro_f1", 0):.3f}</div>'
        f'<div class="sz-caption">5 classes</div></div>'
        f'<div><div class="sz-label">Weighted F1</div><div class="sz-display">{submission.get("weighted_f1", 0):.3f}</div>'
        f'<div class="sz-caption">epoch-weighted</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    per_class = load_submission_per_class()
    if per_class:
        rows = (
            '<table class="swiss-table" style="max-width:520px;">'
            '<tr><th>Stage</th><th>Precision</th><th>Recall</th><th>F1</th></tr>'
        )
        for stage in ["Wake", "N1", "N2", "N3", "REM"]:
            if stage in per_class:
                c = per_class[stage]
                rows += (
                    f'<tr><td>{stage}</td>'
                    f'<td>{c["precision"]:.3f}</td>'
                    f'<td>{c["recall"]:.3f}</td>'
                    f'<td>{c["f1"]:.3f}</td></tr>'
                )
        rows += "</table>"
        st.markdown(rows, unsafe_allow_html=True)

if not primary_metrics and not submission:
    st.warning(
        "No results found. Run the notebooks (01→05) first."
    )

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
    "<strong>Dataset:</strong> Sleep-EDF Expanded complete corpus — 197 recordings / 100 subjects (PhysioNet)<br>"
    "<strong>Training window:</strong> 10 &times; 30 s epochs (300 s context)<br>"
    "<strong>Primary model:</strong> Improved Student (from scratch, 99,477 params)<br>"
    "<strong>Training:</strong> Supervised class-weighted cross-entropy (from scratch)<br>"
    "<strong>Evaluation:</strong> Final: person-level 70/15/16 holdout, seed 42 &middot; "
    "Historical: 10-fold person-level CV, seeds 42/43/44"
    "</div>",
    unsafe_allow_html=True,
)
