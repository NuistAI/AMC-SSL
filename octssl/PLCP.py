from torch.nn import functional as F
from .LAMPC import lampc_loss
from .HungarianMatching import hungarian_assign, empirical_centers

def optimizer_parameters(model):
    return model.parameters()

def _loss_and_centers(model, images, labels, centers, config):
    features, logits = model.features(images)
    matched = hungarian_assign(centers, empirical_centers(features, labels, model.num_classes))
    loss = F.cross_entropy(logits, labels) + config["plcp_weight"] * lampc_loss(
        features, matched, labels, tau=config["tau"], beta=config["beta"])
    return loss, matched

def training_step(model, images, labels, centers, config, teacher=None, epoch=1):
    loss, matched = _loss_and_centers(model, images, labels, centers, config)
    return loss, {}, matched

def validation_loss(model, images, labels, centers, config):
    return _loss_and_centers(model, images, labels, centers, config)[0]
