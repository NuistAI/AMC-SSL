import torch
from torch import nn
from . import backbone
from .network_config import get_trans_unet_config
class OursModel(nn.Module):

    def __init__(self, num_classes, image_size):
        super().__init__()
        source = backbone
        config = get_trans_unet_config()
        config.n_classes = int(num_classes)
        self.image_size = tuple(int(v) for v in image_size)
        self.num_classes = int(num_classes)
        self.feature_dim = int(config.feature_num)
        self.encoder = source.UNet_encode(config, img_size=self.image_size,
                                          num_classes=self.num_classes)
        self.decoder = source.TransUNet_decode(config, img_size=self.image_size,
                                               num_classes=self.num_classes)
        self.head = source.TransUNet_head(config)
        self.architecture_label = "migrated UNet_encode + TransUNet_decode + TransUNet_head"

    def encode(self, x):
        encoded, _ = self.encoder(x)
        return encoded

    def features(self, x):
        encoded, skips = self.encoder(x)
        pixels, _ = self.decoder(encoded, skips)
        return pixels, self.head(pixels)

    def forward(self, x):
        return self.features(x)[1]

def build_model(method, num_classes, image_size):
    if method.lower() != "ours":
        raise ValueError("This release contains only Ours")
    if len(image_size)!=2 or any(int(v)<=0 or int(v)%16 for v in image_size):
        raise ValueError("image_size must contain positive multiples of 16")
    if int(num_classes)<2: raise ValueError("num_classes must be >= 2")
    return OursModel(num_classes,image_size)
