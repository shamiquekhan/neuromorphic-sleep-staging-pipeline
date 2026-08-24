"""4-Fold Held-Out-Subject CV for LoRA adaptation."""
import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sleep_staging.config import StudentConfig, CHECKPOINT_PATH
from sleep_staging.models import ImprovedStudent
from sleep_staging.data.loader import load_cached_subject
from sleep_staging.evaluation import compute_all_metrics
from sleep_staging.adaptation import LoRAConfig, apply_lora, count_lora_parameters

SEED = 42
SEQ_LEN, STRIDE, BATCH, EPOCHS, LR = 10, 5, 16, 10, 3e-4
ALL_SUBJ = ["SC4001", "SC4002", "SC4011", "SC4012"]
FOLDS = [
    {"train": ["SC4002","SC4011","SC4012"], "test": "SC4001"},
    {"train": ["SC4001","SC4011","SC4012"], "test": "SC4002"},
    {"train": ["SC4001","SC4002","SC4012"], "test": "SC4011"},
    {"train": ["SC4001","SC4002","SC4011"], "test": "SC4012"},
]

np.random.seed(SEED); torch.manual_seed(SEED)
config = StudentConfig()
sd = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

# Build all sequences
print("Building sequences...", flush=True)
X_all, Y_all, idx_map, cur = [], [], {}, 0
for s in ALL_SUBJ:
    d = load_cached_subject(s)
    n = 0
    for i in range(0, len(d["epochs"]) - SEQ_LEN + 1, STRIDE):
        X_all.append(d["epochs"][i:i+SEQ_LEN])
        Y_all.append(d["labels"][i+SEQ_LEN-1])
        n += 1
    idx_map[s] = list(range(cur, cur+n)); cur += n
    print(f"  {s}: {n}", flush=True)

X = torch.from_numpy(np.array(X_all, np.float32))
Y = np.array(Y_all, np.int64)

def train_model(x_tr, y_tr, method, rank=None):
    """Train and return best model state."""
    model = ImprovedStudent(config)
    model.load_state_dict(sd, strict=True)
    
    if method == "frozen":
        return model.state_dict(), 0
    
    if method == "lora":
        lora_cfg = LoRAConfig(rank=rank, alpha=rank*2, target_modules=["head"], dropout=0.05)
        model = apply_lora(model, lora_cfg)
        n_p = count_lora_parameters(model)["trainable"]
    else:  # full_ft
        for p in model.parameters(): p.requires_grad = True
        n_p = sum(p.numel() for p in model.parameters())
    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    n_val = int(len(x_tr) * 0.2)
    xv, yv = x_tr[-n_val:], y_tr[-n_val:]
    xt, yt = x_tr[:-n_val], y_tr[:-n_val]
    
    best_k, best_s = -1, None
    for ep in range(1, EPOCHS+1):
        model.train()
        perm = torch.randperm(len(xt)).numpy()
        for i in range(0, len(xt), BATCH):
            idx = perm[i:i+BATCH]
            xb = xt[idx]
            yb = torch.from_numpy(yt[idx].copy())
            optimizer.zero_grad()
            criterion(model(xb)[:, -1, :], yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            yp = model(xv)[:, -1, :].argmax(-1).numpy()
        k = cohen_kappa_score(yv, yp)
        if k > best_k:
            best_k = k
            best_s = {k_: v_.clone() for k_, v_ in model.state_dict().items()}
    
    if best_s: model.load_state_dict(best_s)
    return model.state_dict(), n_p

def eval_model(state_dict, x_te, y_te, method="frozen", rank=None):
    model = ImprovedStudent(config)
    if method == "lora":
        lora_cfg = LoRAConfig(rank=rank, alpha=rank*2, target_modules=["head"], dropout=0.05)
        model = apply_lora(model, lora_cfg)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.inference_mode():
        yp = model(x_te)[:, -1, :].argmax(-1).numpy()
    return compute_all_metrics(y_te, yp)

# ── Run CV ──
all_results = []
for fi, fold in enumerate(FOLDS, 1):
    print(f"\n=== Fold {fi}: test={fold['test']} ===", flush=True)
    tr_idx = [i for s in fold["train"] for i in idx_map[s]]
    te_idx = idx_map[fold["test"]]
    x_tr, y_tr = X[tr_idx], Y[tr_idx]
    x_te, y_te = X[te_idx], Y[te_idx]
    print(f"  Train: {len(x_tr)}, Test: {len(x_te)}", flush=True)
    
    r = {}
    # Frozen
    state, _ = train_model(x_tr, y_tr, "frozen")
    r["frozen"] = eval_model(state, x_te, y_te)
    print(f"  frozen: κ={r['frozen']['kappa']:.4f}", flush=True)
    
    # Full FT
    state, n = train_model(x_tr, y_tr, "full_ft")
    r["full_ft"] = eval_model(state, x_te, y_te)
    r["full_ft"]["trainable"] = n
    print(f"  full_ft: κ={r['full_ft']['kappa']:.4f} ({n} params)", flush=True)
    
    # LoRA r=2,4,8
    for rank in [2, 4, 8]:
        state, n = train_model(x_tr, y_tr, "lora", rank)
        r[f"lora_r{rank}"] = eval_model(state, x_te, y_te, "lora", rank)
        r[f"lora_r{rank}"]["trainable"] = n
        print(f"  lora_r{rank}: κ={r[f'lora_r{rank}']['kappa']:.4f} ({n} params)", flush=True)
    
    all_results.append(r)

# ── Summary ──
methods = ["frozen","full_ft","lora_r2","lora_r4","lora_r8"]
labels = {"frozen":"Frozen","full_ft":"Full FT","lora_r2":"LoRA r=2","lora_r4":"LoRA r=4","lora_r8":"LoRA r=8"}

print("\n" + "="*80, flush=True)
print("  4-FOLD CV RESULTS", flush=True)
print("="*80, flush=True)
print(f"  {'Method':<12} {'Params':>8} {'Accuracy':>16} {'κ':>16} {'Macro F1':>16} {'W.F1':>16}", flush=True)
print("  "+"-"*76, flush=True)

summary = {}
for m in methods:
    acc = [r[m]["accuracy"] for r in all_results]
    kap = [r[m]["kappa"] for r in all_results]
    mf1 = [r[m]["macro_f1"] for r in all_results]
    wf1 = [r[m]["weighted_f1"] for r in all_results]
    summary[m] = {"acc":(np.mean(acc),np.std(acc)), "kap":(np.mean(kap),np.std(kap)),
                  "mf1":(np.mean(mf1),np.std(mf1)), "wf1":(np.mean(wf1),np.std(wf1))}
    t = all_results[0][m].get("trainable", 0)
    print(f"  {labels[m]:<12} {t:>8,} {np.mean(acc):.4f}±{np.std(acc):.4f}  "
          f"{np.mean(kap):.4f}±{np.std(kap):.4f}  {np.mean(mf1):.4f}±{np.std(mf1):.4f}  "
          f"{np.mean(wf1):.4f}±{np.std(wf1):.4f}", flush=True)

print(f"\n  Per-fold κ:", flush=True)
print(f"  {'Method':<12} {'F1':>8} {'F2':>8} {'F3':>8} {'F4':>8} {'Mean':>8}", flush=True)
print("  "+"-"*52, flush=True)
for m in methods:
    k = [r[m]["kappa"] for r in all_results]
    print(f"  {labels[m]:<12} {k[0]:>8.4f} {k[1]:>8.4f} {k[2]:>8.4f} {k[3]:>8.4f} {np.mean(k):>8.4f}", flush=True)

# Save
output = {
    "summary": {m: {k: {"mean": float(v[0]), "std": float(v[1])} for k,v in s.items()} for m,s in summary.items()},
    "per_fold": [{m: {k: float(v) if isinstance(v,(np.floating,float)) else v for k,v in r[m].items()} for m in methods} for r in all_results],
}
Path("results/lora_cv_results.json").write_text(json.dumps(output, indent=2, default=str))
print("\n  Saved: results/lora_cv_results.json", flush=True)
