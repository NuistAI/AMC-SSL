import torch
from torch import nn
from torch.nn import functional as F

class ArcMarginProduct(nn.Module):
    def __init__(self, weak_features, base_features, s=1.0, beta=30.0, m_init=0.50):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m_init = m_init
        self.beta = beta
        self.weight = weak_features.view(weak_features.size(0), -1)
        self.base = base_features.view(base_features.size(0), -1)

        self.m = self.m_init

    def forward(self):
        k = self.weight
        b = self.base
        cosine = F.linear(F.normalize(self.base), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        self.m = torch.exp(-self.beta * (1 - cosine))

        self.cos_m = torch.cos(self.m)
        self.sin_m = torch.sin(self.m)

        phi = cosine * self.cos_m - sine * self.sin_m
        one_hot = torch.eye(cosine.size(0), device=cosine.device)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss = nn.CrossEntropyLoss()(output, one_hot)

        return cosine, loss


def amc_loss(features1, features2):





    _, loss = ArcMarginProduct(features1, features2,
                                               s=1.0, beta=30.0)()
    return loss
