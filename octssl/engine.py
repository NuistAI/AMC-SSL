import os,random,hashlib
from pathlib import Path
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch
from torch.utils.data import DataLoader,Subset
from .augmentations import weak_view, uicp_view
from .OCTInspiredTransformation import spatial_teacher_view
from .FeatureAlignmentConsistency import align_teacher
from . import UICP, PLCP
ROOT=Path(os.environ.get("OCTSSL_OUTPUT_DIR",str(Path(__file__).resolve().parents[1]/"outputs"))).resolve()
STOP_EXIT_CODE=None
def implementation_hash():
    files=sorted(Path(__file__).resolve().parent.glob("*.py"))
    return hashlib.sha256("".join(p.name+sha(p) for p in files).encode()).hexdigest()
def request_stop(code=143):
    global STOP_EXIT_CODE
    STOP_EXIT_CODE = int(code)


def stop_if_requested():
    if STOP_EXIT_CODE is not None:
        raise SystemExit(STOP_EXIT_CODE)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'].cpu())
    if s['cuda']: torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])


def build_data(config, split, labeled=None):
    from .data import build_dataset
    return build_dataset(config, split=split, labeled=labeled)


def loader(dataset, batch, shuffle=False, limit=None):
    if limit is not None: dataset = Subset(dataset, list(range(min(limit,len(dataset)))))
    return DataLoader(dataset,batch_size=batch,shuffle=shuffle,num_workers=0,drop_last=False)










def mixed_batches(labeled, unlabeled, batch, lbs, dry=False):

    lorder=np.random.permutation(len(labeled)); uorder=[]
    steps=max(1,len(lorder)//lbs)
    if dry: steps=1
    for step in range(steps):
        ix=lorder[step*lbs:(step+1)*lbs].tolist()
        if len(ix)<lbs: ix=np.random.choice(len(labeled),lbs,replace=True).tolist()
        count=batch-lbs
        while len(uorder)<count: uorder.extend(np.random.permutation(len(unlabeled)).tolist())
        us=uorder[:count]; uorder=uorder[count:]
        items=[labeled[i] for i in ix]+[unlabeled[i] for i in us]
        yield {'image':torch.stack([d['image'] for d in items]),
               'label':torch.stack([d['label'] if k<lbs else torch.zeros_like(d['label'])
                                    for k,d in enumerate(items)])}


def overlap_rows(model,dataset,device,batch=2,limit=None):
    model.eval(); ds=dataset if limit is None else Subset(dataset,range(min(limit,len(dataset))))
    rows=[]
    with torch.no_grad():
        for b in loader(ds,batch):
            pred=model(b['image'].to(device)).argmax(1).cpu()
            for j,y in enumerate(b['label']):
                d=[]; u=[]
                for c in range(1,model.num_classes):
                    a=pred[j]==c; t=y==c; inter=(a&t).sum().item()
                    d.append((2*inter+1)/(a.sum().item()+t.sum().item()+1))
                    u.append((inter+1)/((a|t).sum().item()+1))
                row={'Image':b['id'][j],'MDice':100*float(np.mean(d)),'MIoU':100*float(np.mean(u))}
                row.update({'Class%d_Dice'%(c+1):100*v for c,v in enumerate(d)})
                row.update({'Class%d_IoU'%(c+1):100*v for c,v in enumerate(u)})
                rows.append(row)
    return rows


def aggregate(rows):
    keys=[k for k in rows[0] if k!='Image']
    return {k:float(np.mean([r[k] for r in rows])) for k in keys}
def evaluate_stage(model,heads,dataset,stage,config,centers,device,dry=False):
    saved=rng_state();seed_all(config["seed"]+901)
    model.eval()
    try:
        if stage=="sampc":
            result=aggregate(overlap_rows(model,dataset,device,config.get("eval_batch_size",2),2 if dry else None))
            return -result["MIoU"],result
        losses=[]
        with torch.no_grad():
            for b in loader(dataset,2,limit=2 if dry else None):
                x=b["image"].to(device);y=b["label"].to(device)
                module = {"uicp": UICP, "plcp": PLCP}[stage]
                loss = module.validation_loss(model, x, y, centers, config)
                losses.append(float(loss))
        if not losses: raise ValueError("Empty validation dataset")
        value=float(np.mean(losses))
        return value,{"val_loss":value}
    finally:
        restore_rng(saved)
