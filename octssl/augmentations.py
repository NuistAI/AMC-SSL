import random
import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image
import torch
from torchvision import transforms
class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image):
        image = np.asarray(image)
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        image = to_pil_image(image)
        return image



def weak_view(x):

    transform = transforms.Compose([transforms.ToPILImage(),
        RandomGaussianBlur(),
        transforms.RandomApply([transforms.ColorJitter(.4,.4,.4,.1)], p=.8),
        transforms.ToTensor()])
    return torch.stack([transform(v.cpu()) for v in x]).to(x.device)


def uicp_view(x):
    x=weak_view(x)
    spatial=transforms.Compose([transforms.RandomAffine(10,translate=(.02,.02)),
        transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomRotation(10)])
    return torch.stack([spatial(v) for v in x])
