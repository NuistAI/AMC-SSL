import math
import torch
from torch.nn import functional as F

def lampc_loss(features, centers, labels, mask=None, tau=.05, beta=10.):






    if not math.isfinite(float(tau)) or float(tau) <= 0:
        raise ValueError("tau must be finite and > 0")
    if not math.isfinite(float(beta)) or float(beta) < 0:
        raise ValueError("beta must be finite and >= 0")
    if features.ndim != 4 or labels.shape != features.shape[:1] + features.shape[2:]:
        raise ValueError("Expected features B,D,H,W and labels B,H,W")
    if centers.ndim != 2 or centers.shape[1] != features.shape[1]:
        raise ValueError("Prototype dimension does not match features")
    pixels = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    targets = labels.reshape(-1).long()
    selected = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask.reshape(-1).eq(1)
    if selected.numel() != targets.numel():
        raise ValueError("mask shape does not match labels")
    if not selected.any():
        return features.sum() * 0.0
    pixels, targets = pixels[selected], targets[selected]
    if targets.min() < 0 or targets.max() >= centers.shape[0]:
        raise ValueError("Selected label outside prototype class range")
    similarities = F.normalize(pixels.float(), dim=1) @ F.normalize(centers.float(), dim=1).T
    correct = similarities.gather(1, targets[:, None]).squeeze(1)

    eps = torch.finfo(correct.dtype).eps
    correct = correct.clamp(-1.0 + eps, 1.0 - eps)
    margin = torch.exp(-float(beta) * (1.0 - correct))
    sine = torch.sqrt((1.0 - correct.square()).clamp_min(eps))
    rotated = correct * torch.cos(margin) - sine * torch.sin(margin)
    logits = similarities.scatter(1, targets[:, None], rotated[:, None]) / float(tau)
    return F.nll_loss(F.log_softmax(logits, dim=1), targets)
