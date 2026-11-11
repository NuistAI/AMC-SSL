from torch.nn import functional as F

def filter_pseudo_labels(teacher_logits, threshold=0.1):

    target_prob = F.softmax(teacher_logits.float(), dim=1)
    entropy = -(target_prob * target_prob.clamp_min(1e-8).log()).sum(dim=1)
    reliable = entropy <= threshold
    return target_prob.argmax(dim=1), reliable
