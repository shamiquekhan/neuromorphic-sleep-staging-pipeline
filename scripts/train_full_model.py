"""Full model training with all-position supervision + class weights.

Trains the Improved Student from scratch using:
- All 10 sequence positions supervised (not just last)
- Inverse-frequency class weights
- 4-fold held-out-subject CV
- Full per-class evaluation

Usage:
    python scripts/train_full_model.py --seed 42
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, confusion_matrix, classification_report

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from sleep_staging.config import CACHE_DIR, STAGE_NAMES, StudentConfig
from sleep_staging.models import ImprovedStudent, count_parameters

FOLDS = [
    {"train": ["SC4002", "SC4011", "SC4012"], "test": "SC4001"},
    {"train": ["SC4001", "SC4011", "SC4012"], "test": "SC4002"},
    {"train": ["SC4001", "SC4002", "SC4012"], "test": "SC4011"},
    {"train": ["SC4001", "SC4002", "SC4011"], "test": "SC4012"},
]
SEQ_LEN = 10
STRIDE = 5
N_CLASSES = 5


class SeqDataset(Dataset):
    def __init__(self, sequences, labels_per_pos):
        self.sequences = sequences
        self.labels_per_pos = labels_per_pos  # [N, 10]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels_per_pos[idx]


def load_fold_sequences(subj_list, seq_len=SEQ_LEN, stride=STRIDE):
    """Load sequences with ALL position labels (not just last)."""
    X_all, Y_all = [], []
    for subj in subj_list:
        d = np.load(CACHE_DIR / f"{subj}_night0.npz")
        epochs, labels = d["epochs"], d["labels"]
        for i in range(0, len(labels) - seq_len + 1, stride):
            X_all.append(epochs[i : i + seq_len])
            Y_all.append(labels[i : i + seq_len])
    return np.array(X_all, dtype=np.float32), np.array(Y_all, dtype=np.int64)


def build_weight_tensor(labels_flat, n1_boost=1.0, rem_boost=1.0):
    """Build CE class weights with optional minority-class boosts."""
    counts = np.bincount(labels_flat, minlength=N_CLASSES).astype(np.float32)
    w = counts.sum() / (counts + 1.0)
    w = w / w.mean()
    w[1] *= n1_boost   # N1
    w[4] *= rem_boost   # REM
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)  # yb: [B, 10]
        logits = model(xb)  # [B, 10, 5]
        # Supervise ALL 10 positions
        loss = criterion(logits.reshape(-1, N_CLASSES), yb.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        preds = logits.argmax(-1)  # [B, 10]
        correct += (preds == yb).sum().item()
        total += yb.numel()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)  # [B, 10, 5]
        # Evaluate at last position (index 9) — matches deployment
        all_preds.append(logits[:, -1, :].argmax(-1).cpu().numpy())
        all_labels.append(yb[:, -1].numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    kap = cohen_kappa_score(labels, preds)
    mf1 = f1_score(labels, preds, average="macro", zero_division=0)
    wf1 = f1_score(labels, preds, average="weighted", zero_division=0)
    per_class = {}
    for i in range(N_CLASSES):
        name = STAGE_NAMES[i]
        mask = labels == i
        tp = int(((preds == i) & mask).sum())
        total = int(mask.sum())
        pred_total = int((preds == i).sum())
        recall = tp / total if total > 0 else 0.0
        precision = tp / pred_total if pred_total > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": total,
            "predicted": pred_total,
            "correct": tp,
        }
    cm = confusion_matrix(labels, preds, labels=list(range(N_CLASSES)))
    stage_list = [STAGE_NAMES[i] for i in range(N_CLASSES)]
    report = classification_report(labels, preds, target_names=stage_list, zero_division=0)
    return {
        "accuracy": round(float(acc), 6),
        "kappa": round(float(kap), 6),
        "macro_f1": round(float(mf1), 6),
        "weighted_f1": round(float(wf1), 6),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


def run_fold(fold_idx, fold, n1_boost, rem_boost, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load all-position data
    X_train, Y_train = load_fold_sequences(fold["train"])
    X_test, Y_test = load_fold_sequences([fold["test"]], stride=1)

    # Temporal validation split (last 20%)
    n_val = int(len(X_train) * 0.2)
    X_val, Y_val = X_train[-n_val:], Y_train[-n_val:]
    X_tr, Y_tr = X_train[:-n_val], Y_train[:-n_val]

    # Weight tensor from flattened training labels
    train_labels_flat = Y_tr.reshape(-1)
    class_weights = build_weight_tensor(train_labels_flat, n1_boost=n1_boost, rem_boost=rem_boost).to(device)

    train_loader = DataLoader(SeqDataset(X_tr, Y_tr), batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(SeqDataset(X_val, Y_val), batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(SeqDataset(X_test, Y_test), batch_size=16, shuffle=False, num_workers=0)

    model = ImprovedStudent(StudentConfig()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_val_mf1 = -1
    best_state = None
    no_improve = 0

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_preds, val_labels = evaluate(model, val_loader, device)
        val_mf1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        scheduler.step()

        if val_mf1 > best_val_mf1:
            best_val_mf1 = val_mf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 3:
                break

    # Evaluate on test
    model.load_state_dict(best_state)
    test_preds, test_labels = evaluate(model, test_loader, device)
    metrics = compute_metrics(test_preds, test_labels)

    return metrics, best_val_mf1, best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n1-boost", type=float, default=1.0, help="Extra N1 weight multiplier")
    parser.add_argument("--rem-boost", type=float, default=1.0, help="Extra REM weight multiplier")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | N1 boost: {args.n1_boost}x | REM boost: {args.rem_boost}x | Seed: {args.seed}", flush=True)
    print("=" * 80, flush=True)

    all_metrics = []
    best_overall_state = None
    best_overall_mf1 = -1

    for fi, fold in enumerate(FOLDS):
        t0 = time.time()
        metrics, val_mf1, fold_state = run_fold(fi, fold, args.n1_boost, args.rem_boost, device, args.seed)
        elapsed = time.time() - t0

        print(f"\nFold {fi+1} ({fold['test']}) — {elapsed:.0f}s", flush=True)
        print(f"  val_mf1={val_mf1:.4f} | acc={metrics['accuracy']:.4f} kap={metrics['kappa']:.4f} mf1={metrics['macro_f1']:.4f}", flush=True)
        for s_name in [STAGE_NAMES[i] for i in range(N_CLASSES)]:
            pc = metrics["per_class"][s_name]
            print(f"  {s_name:>4s}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} (true={pc['support']}, pred={pc['predicted']}, correct={pc['correct']})", flush=True)

        all_metrics.append(metrics)

        if val_mf1 > best_overall_mf1:
            best_overall_mf1 = val_mf1
            best_overall_state = fold_state
            best_fold = fi

        # Save intermediate results after each fold
        out_dir = _root / "results" / "full_model"
        out_dir.mkdir(parents=True, exist_ok=True)
        intermediate = out_dir / f"partial_seed{args.seed}.json"
        with open(intermediate, "w") as f:
            json.dump({"completed_folds": fi + 1, "per_fold": all_metrics}, f, indent=2)

    # Save best checkpoint
    if best_overall_state is not None:
        ckpt_dir = _root / "artifacts" / "full_model_trained"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "student_full_trained.pt"
        torch.save(best_overall_state, ckpt_path)
        print(f"\nBest checkpoint saved: {ckpt_path} (from fold {best_fold+1})", flush=True)

    # Aggregate
    print("\n" + "=" * 80, flush=True)
    print("AGGREGATE RESULTS (4-fold held-out-subject CV)", flush=True)
    print("=" * 80, flush=True)

    accs = [m["accuracy"] for m in all_metrics]
    kaps = [m["kappa"] for m in all_metrics]
    mf1s = [m["macro_f1"] for m in all_metrics]
    wf1s = [m["weighted_f1"] for m in all_metrics]

    print(f"Accuracy:     {np.mean(accs):.4f} +/- {np.std(accs):.4f}", flush=True)
    print(f"Kappa:        {np.mean(kaps):.4f} +/- {np.std(kaps):.4f}", flush=True)
    print(f"Macro F1:     {np.mean(mf1s):.4f} +/- {np.std(mf1s):.4f}", flush=True)
    print(f"Weighted F1:  {np.mean(wf1s):.4f} +/- {np.std(wf1s):.4f}", flush=True)

    print(f"\n{'Stage':>6s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}", flush=True)
    print("-" * 40, flush=True)
    for si in range(N_CLASSES):
        s = STAGE_NAMES[si]
        precs = [m["per_class"][s]["precision"] for m in all_metrics]
        recs = [m["per_class"][s]["recall"] for m in all_metrics]
        f1s = [m["per_class"][s]["f1"] for m in all_metrics]
        sups = [m["per_class"][s]["support"] for m in all_metrics]
        print(f"{s:>6s} {np.mean(precs):>8.4f} {np.mean(recs):>8.4f} {np.mean(f1s):>8.4f} {np.mean(sups):>8.1f}", flush=True)

    # Aggregate confusion matrix
    all_cm = np.zeros((5, 5), dtype=int)
    for m in all_metrics:
        all_cm += np.array(m["confusion_matrix"])
    print(f"\nAggregate Confusion Matrix:", flush=True)
    stage_list = [STAGE_NAMES[i] for i in range(N_CLASSES)]
    print(f"{'':>8s}", end="")
    for s in stage_list:
        print(f"{s:>8s}", end="")
    print(flush=True)
    for i in range(5):
        print(f"{STAGE_NAMES[i]:>8s}", end="")
        for j in range(5):
            print(f"{all_cm[i][j]:>8d}", end="")
        print(flush=True)

    # Save results
    out_dir = _root / "results" / "full_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"full_n1b{args.n1_boost}_remb{args.rem_boost}_s{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump({
            "config": {"n1_boost": args.n1_boost, "rem_boost": args.rem_boost, "seed": args.seed},
            "aggregate": {
                "accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
                "kappa": {"mean": float(np.mean(kaps)), "std": float(np.std(kaps))},
                "macro_f1": {"mean": float(np.mean(mf1s)), "std": float(np.std(mf1s))},
                "weighted_f1": {"mean": float(np.mean(wf1s)), "std": float(np.std(wf1s))},
            },
            "per_class": {
                STAGE_NAMES[si]: {
                    "precision": {"mean": float(np.mean([m["per_class"][STAGE_NAMES[si]]["precision"] for m in all_metrics])),
                                  "std": float(np.std([m["per_class"][STAGE_NAMES[si]]["precision"] for m in all_metrics]))},
                    "recall": {"mean": float(np.mean([m["per_class"][STAGE_NAMES[si]]["recall"] for m in all_metrics])),
                               "std": float(np.std([m["per_class"][STAGE_NAMES[si]]["recall"] for m in all_metrics]))},
                    "f1": {"mean": float(np.mean([m["per_class"][STAGE_NAMES[si]]["f1"] for m in all_metrics])),
                           "std": float(np.std([m["per_class"][STAGE_NAMES[si]]["f1"] for m in all_metrics]))},
                }
                for si in range(N_CLASSES)
            },
            "confusion_matrix": all_cm.tolist(),
            "per_fold": all_metrics,
        }, f, indent=2)
    print(f"\nResults saved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
