from .AMC import amc_loss
from .augmentations import uicp_view

def optimizer_parameters(model):
    return model.encoder.parameters()

def training_step(model, images, labels, centers, config, teacher=None, epoch=1):
    view = uicp_view(images)
    loss = amc_loss(model.encode(images), model.encode(view))
    return loss, {}, centers

def validation_loss(model, images, labels, centers, config):
    return training_step(model, images, labels, centers, config)[0]
