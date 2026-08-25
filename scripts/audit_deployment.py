#!/usr/bin/env python3
"""Deployment Audit Script.

Checks that all deployment artifacts are present and correct.

Usage:
    python scripts/audit_deployment.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

results = []


def check(description: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"  {status} {description}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results.append(condition)
    return condition


def main():
    print("=" * 60)
    print("  NEUROSLEEP — DEPLOYMENT AUDIT")
    print("=" * 60)

    # ── Final Checkpoint ──────────────────────────────────────────────
    print("\n[1] Final Checkpoint")
    ckpt_path = REPO / "artifacts" / "final" / "student_full_finetuned.pt"
    check("Final checkpoint exists", ckpt_path.exists(), str(ckpt_path))

    if ckpt_path.exists():
        import torch
        from sleep_staging.models.improved_student import ImprovedStudent
        from sleep_staging.models import count_parameters
        from sleep_staging.config import StudentConfig

        model = ImprovedStudent()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

        n_params = count_parameters(model)
        check("99,477 parameters", n_params == 99477, f"got {n_params}")

        config = StudentConfig()
        check("10 x 4 x 3000 input",
              config.seq_len == 10 and config.n_channels == 4 and config.samples_per_epoch == 3000)
        check("5 classes", config.n_classes == 5)

    # ── Final Metrics ─────────────────────────────────────────────────
    print("\n[2] Final Metrics")
    metrics_path = REPO / "results" / "final" / "final_metrics.json"
    check("final_metrics.json exists", metrics_path.exists())

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        acc = metrics.get("accuracy", {}).get("mean", 0)
        kappa = metrics.get("cohen_kappa", {}).get("mean", 0)
        macro = metrics.get("macro_f1", {}).get("mean", 0)

        check("Accuracy present", acc > 0, f"{acc:.3f}")
        check("Kappa present", kappa > 0, f"{kappa:.3f}")
        check("Macro F1 present", macro > 0, f"{macro:.3f}")

        per_class = metrics.get("per_class", {})
        check("Per-class metrics present", len(per_class) == 5, f"got {len(per_class)} classes")

        fold_metrics = metrics.get("fold_metrics", [])
        check("Fold metrics present", len(fold_metrics) == 4, f"got {len(fold_metrics)} folds")

    # ── Configuration ─────────────────────────────────────────────────
    print("\n[3] Configuration")
    config_path = REPO / "configs" / "final.yaml"
    check("final.yaml exists", config_path.exists())

    # ── Deployment Directory ──────────────────────────────────────────
    print("\n[4] Deployment Directory")
    deploy_dir = REPO / "deployment"
    check("deployment/ exists", deploy_dir.exists())
    check("deployment/app.py exists", (deploy_dir / "app.py").exists())
    check("deployment/Dockerfile exists", (deploy_dir / "Dockerfile").exists())
    check("deployment/requirements.txt exists", (deploy_dir / "requirements.txt").exists())
    check("deployment/config/inference.yaml exists", (deploy_dir / "config" / "inference.yaml").exists())

    # ── Streamlit ─────────────────────────────────────────────────────
    print("\n[5] Streamlit App")
    app_dir = REPO / "app"
    check("app/ directory exists", app_dir.exists())
    check("app/streamlit_app.py exists", (app_dir / "streamlit_app.py").exists())
    check("app/state.py exists", (app_dir / "state.py").exists())
    check("app/components.py exists", (app_dir / "components.py").exists())

    # Check for LoRA references in Streamlit
    lora_found = False
    for py_file in app_dir.rglob("*.py"):
        content = py_file.read_text()
        if "LoRA" in content or "lora" in content:
            lora_found = True
            print(f"  {WARN} LoRA reference in {py_file.name}")
    check("No LoRA references in Streamlit", not lora_found)

    # ── Hugging Face ──────────────────────────────────────────────────
    print("\n[6] Hugging Face")
    hf_model = REPO / "huggingface" / "neuromorphic-sleep-staging"
    hf_space = REPO / "huggingface" / "neurosleep-demo"
    check("HF model directory exists", hf_model.exists())
    check("HF model README.md exists", (hf_model / "README.md").exists())
    check("HF Space directory exists", hf_space.exists())
    check("HF Space README.md exists", (hf_space / "README.md").exists())

    # ── Kaggle ────────────────────────────────────────────────────────
    print("\n[7] Kaggle")
    kaggle_dir = REPO / "kaggle"
    check("kaggle/ directory exists", kaggle_dir.exists())
    check("Kaggle notebook exists", (kaggle_dir / "neurosleep_final.ipynb.py").exists())

    # ── Documentation ─────────────────────────────────────────────────
    print("\n[8] Documentation")
    check("README.md exists", (REPO / "README.md").exists())
    check("MODEL_REPORT.md exists", (REPO / "MODEL_REPORT.md").exists())
    check("docs/results.md exists", (REPO / "docs" / "results.md").exists())

    # Check README for LoRA
    readme = (REPO / "README.md").read_text()
    check("README has no LoRA references", "LoRA" not in readme)

    # ── Tests ─────────────────────────────────────────────────────────
    print("\n[9] Tests")
    tests_dir = REPO / "tests"
    check("tests/ directory exists", tests_dir.exists())
    n_tests = len(list(tests_dir.glob("test_*.py")))
    check("Test files present", n_tests > 0, f"found {n_tests} test files")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"  {PASS} ALL {total} CHECKS PASSED")
    else:
        failed = total - passed
        print(f"  {FAIL} {failed} of {total} checks failed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
