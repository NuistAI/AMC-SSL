import math

import torch
from torch import nn
from torch.nn import functional as F

from .LAMPC import lampc_loss as angular_prototype_loss
from .SAMPC import semi_loss


def freeze_bn_stats(model):

    count = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


def freeze_sampc_stats_and_dropout(model):





    bn=freeze_bn_stats(model);dropout=0
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.eval();dropout+=1
    return {'bn':bn,'dropout':dropout}


def _integer(value, name, minimum):
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError("%s must be an integer >= %s" % (name, minimum))
    return int(number)


def unsupervised_weight(epoch_one_based, warm_up=10, ramp_epochs=10):

    epoch = _integer(epoch_one_based, "epoch_one_based", 1)
    warm = _integer(warm_up, "warm_up", 0)
    ramp = _integer(ramp_epochs, "ramp_epochs", 1)
    return min(1.0, max(0.0, float(epoch - warm) / ramp))


def _finite_tensor(value, name):
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise FloatingPointError("Nonfinite %s" % name)


def stable_semi_loss(logits, features, teacher_logits, labels, lbs, centers,
                     config, epoch_one_based):











    scale = unsupervised_weight(epoch_one_based, config.get("warm_up", 10),
                                config.get("ramp_epochs", 10))
    weights = {key: float(config.get(key, default))
               for key, default in (("r1", .5), ("r2", 1.), ("r3", .2))}
    if not all(math.isfinite(value) and value >= 0.0 for value in weights.values()):
        raise ValueError("r1/r2/r3 must be finite and nonnegative")

    if scale == 0.0:
        with torch.no_grad():
            raw_loss, raw_terms = semi_loss(logits, features, teacher_logits,
                                            labels, lbs, centers, config)
    else:
        raw_loss, raw_terms = semi_loss(logits, features, teacher_logits,
                                        labels, lbs, centers, config)
    _finite_tensor(raw_loss, "raw semi-supervised loss")
    for key, value in raw_terms.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise FloatingPointError("Nonfinite raw loss diagnostic: %s" % key)

    if scale == 1.0:
        loss = raw_loss
        supervised_value = raw_terms["ce"] + weights["r2"] * raw_terms["lampc"]
    else:
        n_labeled = int(lbs)
        supervised = F.cross_entropy(logits[:n_labeled], labels[:n_labeled].long())
        supervised = supervised + weights["r2"] * angular_prototype_loss(
            features[:n_labeled], centers, labels[:n_labeled],
            tau=float(config.get("tau", .05)), beta=float(config.get("beta", 10.)))
        _finite_tensor(supervised, "supervised loss")
        supervised_value = float(supervised.detach().item())



        loss = supervised if scale == 0.0 else (1.0 - scale) * supervised + scale * raw_loss
    _finite_tensor(loss, "effective semi-supervised loss")

    terms = dict(raw_terms)
    effective = float(loss.detach().item())
    terms.update({
        "raw_loss": float(raw_loss.detach().item()),
        "loss": effective,
        "effective_loss": effective,
        "supervised_loss": supervised_value,
        "unsupervised_scale": scale,
        "effective_ce": raw_terms["ce"],
        "effective_lampc": weights["r2"] * raw_terms["lampc"],
        "effective_consistency": scale * weights["r1"] * raw_terms["consistency"],
        "effective_sampc": scale * weights["r3"] * raw_terms["sampc"],
        "effective_r1": scale * weights["r1"],
        "effective_r2": weights["r2"],
        "effective_r3": scale * weights["r3"],
    })
    return loss, terms


def clip_gradients(model, max_norm=1.0):





    limit = float(max_norm)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_norm must be finite and > 0")
    parameters = [parameter for parameter in model.parameters()
                  if parameter.grad is not None]
    if not parameters:
        return 0.0
    for parameter in parameters:
        _finite_tensor(parameter.grad, "gradient")
    norm = nn.utils.clip_grad_norm_(parameters, max_norm=limit, norm_type=2.0,
                                   error_if_nonfinite=True)
    _finite_tensor(norm, "global gradient norm")
    return float(norm.detach().item())
