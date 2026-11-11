import torch
from torch.nn import functional as F

def uniform_centers(num_classes, features=16, steps=10000, t=.07, seed=0):






    if int(steps) < 1 or int(num_classes) < 2 or int(features) < 1 or float(t) <= 0:
        raise ValueError("Invalid uniform-center configuration")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn(int(num_classes), int(features), generator=generator,
                    dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=1e-3)
    best_loss, best = float("inf"), None
    with torch.enable_grad():
        for _ in range(int(steps)):
            normalized = F.normalize(x, dim=1)
            loss = torch.logsumexp(normalized @ normalized.T / float(t), dim=-1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Uniform-center loss is nonfinite")
            if loss.item() < best_loss:
                best_loss = loss.item()
                best = normalized.detach().clone()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return best
