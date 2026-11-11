import torch
from torchvision.transforms import functional as TF

def align_teacher(logits, pars):
    out=[]
    for v,(a,tx,ty,flip) in zip(logits,pars):
        v=TF.affine(v,-a,[0.,0.],1,[0.0])
        v=TF.affine(v,0,[-tx,-ty],1,[0.0])
        if flip: v=TF.hflip(v)
        out.append(v)
    return torch.stack(out)

def masked_consistency_loss(student_logits, target_logits, reliable):

    squared = (student_logits.float() - target_logits.float()).square().sum(dim=1)
    return (squared * reliable).mean()
