import copy
import torch
from torch.nn import functional as F
from .LAMPC import lampc_loss
from .EntropyFiltering import filter_pseudo_labels
from .FeatureAlignmentConsistency import align_teacher, masked_consistency_loss
from .OCTInspiredTransformation import spatial_teacher_view
from .augmentations import weak_view

def sampc_loss(features, centers, pseudo_labels, reliable, tau=.05, beta=10.):

    selected = int(reliable.sum().item())
    total = reliable.numel()
    if total == 0:
        return features.sum() * 0.0
    loss = lampc_loss(features, centers, pseudo_labels, reliable, tau=tau, beta=beta)
    return loss * (float(selected) / float(total))

def semi_loss(logits, feat, teacher_logits, labels, labeled_bs, centers, config):







    n_labeled = int(labeled_bs)
    if not (0 < n_labeled <= logits.shape[0]):
        raise ValueError("labeled_bs must be in [1, batch_size]")
    if logits.shape != teacher_logits.shape or labels.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("Incompatible student/teacher/label shapes")
    tau, beta = float(config.get("tau", .05)), float(config.get("beta", 10.))
    r1, r2, r3 = (float(config.get(k, v)) for k, v in (("r1", .5), ("r2", 1.), ("r3", .2)))
    threshold = float(config.get("entropy_threshold", .1))
    ce = F.cross_entropy(logits[:n_labeled], labels[:n_labeled].long())
    lampc = lampc_loss(feat[:n_labeled], centers,
                                   labels[:n_labeled], tau=tau, beta=beta)
    zero = logits.sum() * 0.0
    consistency, sampc = zero, feat.sum() * 0.0
    selected_count, total_count = 0, 0
    if n_labeled < logits.shape[0]:
        target_logits = teacher_logits[n_labeled:].detach()
        pseudo_labels, reliable = filter_pseudo_labels(target_logits, threshold)
        total_count, selected_count = reliable.numel(), int(reliable.sum().item())
        consistency = masked_consistency_loss(logits[n_labeled:], target_logits, reliable)
        sampc = sampc_loss(feat[n_labeled:], centers, pseudo_labels, reliable, tau=tau, beta=beta)
    loss = ce + r1 * consistency + r2 * lampc + r3 * sampc
    terms = {"ce": float(ce.detach()), "consistency": float(consistency.detach()),
             "lampc": float(lampc.detach()), "sampc": float(sampc.detach()),
             "loss": float(loss.detach()), "selected_unlabeled_pixels": selected_count,
             "total_unlabeled_pixels": total_count,
             "selected_fraction": selected_count / total_count if total_count else 0.0}
    return loss, terms

def optimizer_parameters(model):
    return model.parameters()

def create_teacher(model):
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher

def training_step(model, images, labels, centers, config, teacher=None, epoch=1):
    from .stability import stable_semi_loss
    features, logits = model.features(weak_view(images))
    with torch.no_grad():
        transformed, parameters = spatial_teacher_view(images)
        teacher_features, _ = teacher.features(transformed)
        teacher_logits = teacher.head(align_teacher(teacher_features, parameters))
    loss, terms = stable_semi_loss(logits, features, teacher_logits, labels,
                                  int(config["labeled_bs"]), centers, config, epoch)
    terms.update(student_logit_abs_max=float(logits.detach().abs().max()),
                 teacher_logit_abs_max=float(teacher_logits.abs().max()))
    return loss, terms, centers

def update_teacher(teacher, student, global_step, ema_decay):
    decay = min(1. - 1. / (global_step + 1), ema_decay)
    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
            teacher_parameter.mul_(decay).add_(student_parameter, alpha=1-decay)
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            teacher_buffer.copy_(student_buffer)
