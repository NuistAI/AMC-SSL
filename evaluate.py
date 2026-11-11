import argparse,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda");p.add_argument("--batch-size",type=int,default=2)
    a=p.parse_args()
    if a.batch_size<1: p.error("batch size must be positive")
    import torch
    if a.device=="cuda" and not torch.cuda.is_available(): p.error("CUDA is unavailable; use --device cpu")
    from octssl.models import build_model
    from octssl.engine import build_data,overlap_rows,aggregate
    from octssl.config import CLASS_NAMES
    from prepare_data import validate_protocol
    state=torch.load(str(a.checkpoint),map_location="cpu",weights_only=False)
    config=dict(state["config"]);config.update(protocol_path=str(a.protocol.resolve()),split_file=None)
    validate_protocol(json.loads(a.protocol.read_text(encoding="utf8")),config["dataset"],config["label_ratio"])
    model=build_model("Ours",config["num_classes"],config["image_size"]).to(a.device)
    model.load_state_dict(state["model"],strict=True)
    result=aggregate(overlap_rows(model,build_data(config,"test"),torch.device(a.device),a.batch_size))
    report={name+"_Dice":result["Class%d_Dice"%(i+1)] for i,name in enumerate(CLASS_NAMES[config["dataset"]])}
    report.update(MDice=result["MDice"],MIOU=result["MIoU"],SelectedEpoch=state["epoch"]+1,
                  Units="percent",Dataset=config["dataset"],LabelRatio=config["label_ratio"])
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
