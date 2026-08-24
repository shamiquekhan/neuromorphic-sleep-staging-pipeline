"""Loss functions and training utilities for knowledge distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced sleep-stage classification."""

    def __init__(self, weight: torch.Tensor, gamma: float = 1.5):
        super().__init__()
        self.w = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        C = logits.shape[-1]
        ce = F.cross_entropy(
            logits.reshape(-1, C), labels.reshape(-1),
            weight=self.w.to(logits.device), reduction="none",
        )
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()


class DistillationObjective(nn.Module):
    """Combined CE + KL + feature-alignment loss for knowledge distillation."""

    def __init__(self, class_weights=None, temperature: float = 4.0,
                 alpha_ce: float = 1.0, alpha_kl: float = 1.0, alpha_feat: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_kl = alpha_kl
        self.alpha_feat = alpha_feat

    def forward(self, student_logits, student_features,
                teacher_logits, teacher_features, targets):
        b, t, c = student_logits.shape
        s = student_logits.reshape(b * t, c)
        teacher = teacher_logits.reshape(b * t, c).detach()
        target = targets.reshape(b * t)

        ce = self.ce(s, target)

        temp = self.temperature
        kl = F.kl_div(
            F.log_softmax(s / temp, dim=-1),
            F.softmax(teacher / temp, dim=-1),
            reduction="batchmean",
        ) * (temp ** 2)

        s_feat = student_features.reshape(b * t, -1)
        t_feat = teacher_features.reshape(b * t, -1).detach()
        common_dim = min(s_feat.shape[-1], t_feat.shape[-1])
        feat = F.mse_loss(s_feat[:, :common_dim], t_feat[:, :common_dim])

        total = self.alpha_ce * ce + self.alpha_kl * kl + self.alpha_feat * feat
        return total, {"ce": float(ce.detach()), "kl": float(kl.detach()), "feature": float(feat.detach())}


def compute_class_weights(labels, n_classes: int = 5) -> torch.Tensor:
    """Compute log-inverse frequency class weights from training labels."""
    counts = torch.bincount(labels, minlength=n_classes).float() + 1.0
    weights = torch.log(counts.sum() / counts)
    return weights / weights.mean()
