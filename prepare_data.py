import argparse,hashlib,json,random
from pathlib import Path

def validate_protocol(protocol,dataset,ratio):
    data=protocol["datasets"][dataset];splits=data["splits"]
    subjects={k:{r["subject"] for r in splits[k]} for k in ("train","val","test")}
    for a,b in (("train","val"),("train","test"),("val","test")):
        if subjects[a]&subjects[b]: raise ValueError("Subject leakage: "+a+"/"+b)
    records=[r for k in ("train","val","test") for r in splits[k]]
    if any(not splits[k] for k in ("train","val","test")): raise ValueError("Empty data partition")
    if len({r["id"] for r in records})!=len(records): raise ValueError("Duplicate image IDs")
    labeled=set(data["ratios"][str(int(ratio))]["labeled_subjects"])
    if not labeled or not labeled<subjects["train"]: raise ValueError("Need labeled and unlabeled training subjects")
    if data["num_classes"]!=(9 if dataset=="HCMS" else 6): raise ValueError("Class count mismatch")
    if data["image_size"]!=([128,1024] if dataset=="HCMS" else [256,256]): raise ValueError("Image size mismatch")
    return data

def read_ids(path):
    return [Path(v.split()[0]).stem for v in path.read_text(encoding="utf8").splitlines() if v.strip()]

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset",choices=["HCMS","OCTA500"],required=True)
    p.add_argument("--data-root",type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--subject-splits",type=Path);g.add_argument("--split-dir",type=Path)
    p.add_argument("--label-ratio",type=int,default=20);p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();root=a.data_root.resolve()
    if not 0<a.label_ratio<100: p.error("label ratio must be between 0 and 100")
    lists=None
    if a.subject_splits:
        source=json.loads(a.subject_splits.read_text(encoding="utf8"))
        groups={k:set(map(str,source[k])) for k in ("train","val","test")}
        labeled=set(map(str,source.get("labeled",[])))
    else:
        lists={k:read_ids(a.split_dir/name) for k,name in
               (("train","Train_images.txt"),("val","Val_images.txt"),("test","Test_images.txt"))}
        groups={k:{name.split("_")[0] for name in v} for k,v in lists.items()}
        lp=a.split_dir/"Label_Train_images.txt"
        labeled={name.split("_")[0] for name in read_ids(lp)} if lp.exists() else set()
    for k in groups:
        if not groups[k]: raise ValueError("Empty subject partition "+k)
    for a1,b1 in (("train","val"),("train","test"),("val","test")):
        if groups[a1]&groups[b1]: raise ValueError("Subject overlap: "+a1+"/"+b1)
    if not labeled:
        count=max(1,int(len(groups["train"])*a.label_ratio/100+.5))
        if count>=len(groups["train"]): raise ValueError("Insufficient unlabeled subjects")
        labeled=set(random.Random(a.seed).sample(sorted(groups["train"]),count))
    if not labeled<groups["train"]: raise ValueError("Labeled subjects must be a proper subset of train")
    splits={}
    for split,members in groups.items():
        records=[]
        if a.dataset=="HCMS":
            names=sorted(set(lists[split])) if lists else sorted(p.stem for p in (root/"image").glob("*.png") if p.stem.split("_")[0] in members)
            for name in names:
                records.append(dict(id=name,subject=name.split("_")[0],kind="hcms_png",
                    image_path=str(root/"image"/(name+".png")),label_path=str(root/"label"/(name+".png"))))
        else:
            import numpy as np
            for subject in sorted(members):
                folder=root/"OCTA_6mm"/"OCT"/subject
                available=sorted(int(p.stem) for p in folder.glob("*.bmp") if p.stem.isdigit())
                if available!=list(range(1,401)): raise ValueError("Expected 400 raw BMP slices for "+subject)
                for number in (np.rint(np.linspace(0,399,25)).astype(int)+1):
                    n=int(number)
                    records.append(dict(id=subject+"_raw"+str(n-1).zfill(4),subject=subject,
                        kind="octa_raw_mat",slice_index=n-1,image_path=str(folder/(str(n)+".bmp")),
                        label_path=str(root/"GroundTruth"/"GT_Layers"/(subject+".mat"))))
        if {r["subject"] for r in records}!=members: raise ValueError("Missing images for subjects in "+split)
        splits[split]=records
    for record in [r for v in splits.values() for r in v]:
        for key in ("image_path","label_path"):
            if not Path(record[key]).is_file(): raise FileNotFoundError(record[key])
    ratio=dict(nominal_percent=a.label_ratio,actual_percent=100*len(labeled)/len(groups["train"]),
               labeled_subjects=sorted(labeled),unlabeled_subjects=sorted(groups["train"]-labeled))
    result={"protocol_version":"public_external_split_v1","seed":a.seed,"datasets":{a.dataset:
            dict(dataset=a.dataset,num_classes=9 if a.dataset=="HCMS" else 6,
                 image_size=[128,1024] if a.dataset=="HCMS" else [256,256],
                 splits=splits,ratios={str(a.label_ratio):ratio})}}
    validate_protocol(result,a.dataset,a.label_ratio)
    raw=(json.dumps(result,indent=2,sort_keys=True)+"\n").encode()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists() and a.output.read_bytes()!=raw: raise FileExistsError("Refusing to overwrite a different protocol")
    if not a.output.exists(): a.output.write_bytes(raw)
    print(json.dumps({"protocol":str(a.output.resolve()),"sha256":hashlib.sha256(raw).hexdigest(),
                      "images":{k:len(v) for k,v in splits.items()},"actual_labeled_percent":ratio["actual_percent"]}))

if __name__=="__main__": main()
