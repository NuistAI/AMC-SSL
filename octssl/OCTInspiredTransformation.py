import random
import torch
from torchvision.transforms import functional as TF

def spatial_teacher_view(x):

    pars=[]; out=[]
    for v in x:
        angle=random.uniform(-.5,.5); tx=random.uniform(-.001,.001)*v.shape[2]
        ty=random.uniform(-.001,.001)*v.shape[1]; flip=random.random()<.5
        if flip: v=TF.hflip(v)
        v=TF.affine(v,0,[tx,ty],1,[0.0])
        v=TF.affine(v,angle,[0.,0.],1,[0.0])
        out.append(v); pars.append((angle,tx,ty,flip))
    return torch.stack(out), pars
