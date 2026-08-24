"""N1-focused training with class-aware sampling and weighted loss."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import cohen_kappa_score, f1_score

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from sleep_staging.config import CACHE_DIR, STAGE_NAMES, StudentConfig
from sleep_staging.models import ImprovedStudent, count_parameters

ALL_SUBJ = ["SC4001", "SC4002", "SC4011", "SC4012"]
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
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def load_fold_sequences(subj_list, seq_len=SEQ_LEN, stride=STRIDE):
    X_all, Y_all = [], []
    for subj in subj_list:
        d = np.load(CACHE_DIR / f"{subj}_night0.npz")
        epochs, labels = d["epochs"], d["labels"]
        for i in range(0, len(labels) - seq_len + 1, stride):
            X_all.append(epochs[i:i+seq_len])
            Y_all.append(labels[i + seq_len - 1])
    return np.array(X_all, dtype=np.float32), np.array(Y_all, dtype=np.int64)


def make_sampler(labels, oversample_factor=1):
    counts = np.bincount(labels, minlength=N_CLASSES)
    weights_per_class = 1.0 / (counts + 1.0)
    weights_per_class[1] *= oversample_factor
    weights_per_class = weights_per_class / weights_per_class.mean()
    sample_weights = weights_per_class[labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)


def build_weight_tensor(labels, n1_weight=1.0):
    counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float32)
    w = counts.sum() / (counts + 1.0)
    w = w / w.mean()
    w[1] *= n1_weight
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
    def forward(self, input, target):
        ce = nn.functional.cross_entropy(input, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0; correct = 0; total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)[:, -1, :]
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(-1) == yb).sum().item()
        total += len(yb)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)[:, -1, :]
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(preds, labels):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    acc = accuracy_score(labels, preds)
    kap = cohen_kappa_score(labels, preds)
    mf1 = f1_score(labels, preds, average="macro", zero_division=0)
    wf1 = f1_score(labels, preds, average="weighted", zero_division=0)
    per_class = {}
    for i in range(N_CLASSES):
        name = STAGE_NAMES[i]
        mask = labels == i
        tp = ((preds == i) & mask).sum()
        total = mask.sum()
        pred_total = (preds == i).sum()
        recall = tp / total if total > 0 else 0.0
        precision = tp / pred_total if pred_total > 0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
        per_class[name] = {"precision": round(float(precision), 4), "recall": round(float(recall), 4),
                           "f1": round(float(f1), 4), "support": int(total)}
    cm = confusion_matrix(labels, preds, labels=list(range(N_CLASSES)))
    return {"accuracy": round(float(acc), 6), "kappa": round(float(kap), 6),
            "macro_f1": round(float(mf1), 6), "weighted_f1": round(float(wf1), 6),
            "per_class": per_class, "confusion_matrix": cm.tolist()}


def run_fold(fold_idx, fold, n1_weight, oversample, device, seed, use_focal=False):
    torch.manual_seed(seed); np.random.seed(seed)
    X_train, Y_train = load_fold_sequences(fold["train"])
    X_test, Y_test = load_fold_sequences([fold["test"]], stride=1)
    n_val = int(len(X_train) * 0.2)
    X_val, Y_val = X_train[-n_val:], Y_train[-n_val:]
    X_tr, Y_tr = X_train[:-n_val], Y_train[:-n_val]

    sampler = make_sampler(Y_tr, oversample_factor=oversample)
    train_loader = DataLoader(SeqDataset(X_tr, Y_tr), batch_size=16, sampler=sampler, num_workers=0)
    val_loader = DataLoader(SeqDataset(X_val, Y_val), batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(SeqDataset(X_test, Y_test), batch_size=16, shuffle=False, num_workers=0)

    model = ImprovedStudent(StudentConfig()).to(device)
    class_weights = build_weight_tensor(Y_tr, n1_weight=n1_weight).to(device)
    if use_focal:
        criterion = FocalLoss(weight=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

    best_val_mf1 = -1; best_state = None; no_improve = 0
    for epoch in range(12):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_preds, val_labels = evaluate(model, val_loader, device)
        val_mf1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        scheduler.step()
        if val_mf1 > best_val_mf1:
            best_val_mf1 = val_mf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 4: break

    model.load_state_dict(best_state)
    test_preds, test_labels = evaluate(model, test_loader, device)
    return compute_metrics(test_preds, test_labels), best_val_mf1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n1-weight", type=float, default=4.0)
    parser.add_argument("--oversample", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focal", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device} | N1 weight: {args.n1_weight}x | Oversample: {args.oversample}x | Seed: {args.seed} | Focal: {args.focal}", flush=True)
    print("=" * 70, flush=True)

    all_metrics = []
    for fi, fold in enumerate(FOLDS):
        t0 = time.time()
        metrics, val_mf1 = run_fold(fi, fold, args.n1_weight, args.oversample, device, args.seed, use_focal=args.focal)
        elapsed = time.time() - t0
        n1 = metrics["per_class"]["N1"]
        print(f"Fold {fi+1} ({fold['test']}): val_mf1={val_mf1:.4f} | test_acc={metrics['accuracy']:.4f} kap={metrics['kappa']:.4f} | N1 P={n1['precision']:.3f} R={n1['recall']:.3f} F1={n1['f1']:.3f} | {elapsed:.0f}s", flush=True)
        all_metrics.append(metrics)

    accs = [m["accuracy"] for m in all_metrics]
    kaps = [m["kappa"] for m in all_metrics]
    n1_f1s = [m["per_class"]["N1"]["f1"] for m in all_metrics]
    n1_recs = [m["per_class"]["N1"]["recall"] for m in all_metrics]
    n1_pcs = [m["per_class"]["N1"]["precision"] for m in all_metrics]

    print("\n" + "=" * 70, flush=True)
    print(f"Accuracy:     {np.mean(accs):.4f} +/- {np.std(accs):.4f}", flush=True)
    print(f"Kappa:        {np.mean(kaps):.4f} +/- {np.std(kaps):.4f}", flush=True)
    print(f"N1 Precision: {np.mean(n1_pcs):.4f} +/- {np.std(n1_pcs):.4f}", flush=True)
    print(f"N1 Recall:    {np.mean(n1_recs):.4f} +/- {np.std(n1_recs):.4f}", flush=True)
    print(f"N1 F1:        {np.mean(n1_f1s):.4f} +/- {np.std(n1_f1s):.4f}", flush=True)

    out_dir = _root / "results" / "n1_fix"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"n1w{args.n1_weight}_os{args.oversample}_s{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump({"config": vars(args), "aggregate": {
            "accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
            "kappa": {"mean": float(np.mean(kaps)), "std": float(np.std(kaps))},
            "n1_precision": {"mean": float(np.mean(n1_pcs)), "std": float(np.std(n1_pcs))},
            "n1_recall": {"mean": float(np.mean(n1_recs)), "std": float(np.std(n1_recs))},
            "n1_f1": {"mean": float(np.mean(n1_f1s)), "std": float(np.std(n1_f1s))},
        }, "per_fold": all_metrics}, f, indent=2)
    print(f"\nSaved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
