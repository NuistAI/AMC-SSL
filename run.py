import argparse,json,os,signal
from pathlib import Path
import sys
sys.dont_write_bytecode=True

def acquire_run_lock(path):
    lock=Path(path).open("a+b")
    try:
        if os.name=="nt":
            import msvcrt
            if lock.seek(0,2)==0:
                lock.write(b"\0");lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    return lock

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset",choices=["HCMS","OCTA500"],required=True)
    p.add_argument("--protocol",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--label-ratio",type=int,default=20);p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--smoke",action="store_true");p.add_argument("--check-only",action="store_true")
    a=p.parse_args();protocol=a.protocol.resolve();output=a.output.resolve()
    from prepare_data import validate_protocol
    data=validate_protocol(json.loads(protocol.read_text(encoding="utf8")),a.dataset,a.label_ratio)
    missing={r[k] for rows in data["splits"].values() for r in rows for k in ("image_path","label_path") if not Path(r[k]).is_file()}
    if missing: raise FileNotFoundError("Missing data files (first five): "+repr(sorted(missing)[:5]))
    if a.check_only:
        print(json.dumps({"dataset":a.dataset,"images":{k:len(v) for k,v in data["splits"].items()},
                          "labeled_subjects":len(data["ratios"][str(a.label_ratio)]["labeled_subjects"]),
                          "ready":True}));return
    os.environ["OCTSSL_OUTPUT_DIR"]=str(output)
    from octssl import stable_engine as engine
    from octssl.config import stage_config
    from octssl.checkpoint_io import write_json,sha,retire_latest
    output.mkdir(parents=True,exist_ok=True)
    lock=acquire_run_lock(output/"runner.lock")
    signals=[signal.SIGINT,signal.SIGTERM]
    if hasattr(signal,"SIGUSR1"): signals.append(signal.SIGUSR1)
    for signum in signals:
        signal.signal(signum,lambda s,f:engine.request_stop(128+s))
    split=output/"split.json"
    descriptor=dict(dataset=a.dataset,nominal_percent=a.label_ratio,protocol_path=str(protocol),protocol_sha256=sha(protocol))
    if split.exists() and json.loads(split.read_text())!=descriptor: raise ValueError("Existing output uses a different data protocol")
    write_json(split,descriptor)
    implementation=engine.implementation_hash()
    for stage in engine.STAGE_MODULES:
        config=stage_config(a.dataset,protocol,split,a.label_ratio,a.seed,stage,a.smoke)
        stage_dir=output/"checkpoints"/stage;result_file=stage_dir/"result.json"
        if result_file.exists():
            result=json.loads(result_file.read_text())
            if result["config"]!=config or result["implementation_sha256"]!=implementation:
                raise RuntimeError("Completed stage has different configuration/code; use a new output")
            if result["status"]!="completed": raise RuntimeError("Invalid stage result")
            receipt=stage_dir/"retired_checkpoint.json"
            retired=json.loads(receipt.read_text()).get("files",{}) if receipt.exists() else {}
            for name,digest in result["artifact_hashes"].items():
                path=stage_dir/name
                if path.exists():
                    if sha(path)!=digest: raise RuntimeError("Completed stage artifact changed: "+name)
                elif retired.get(name,{}).get("sha256")!=digest: raise RuntimeError("Missing stage artifact: "+name)
            retire_latest(stage_dir)
            print(json.dumps({"stage":stage,"status":"already_completed"}),flush=True);continue
        result=engine.run_experiment(config,stage_dir)
        artifact_hashes={Path(v).name:sha(v) for v in result["artifacts"]}
        write_json(result_file,dict(status="completed",config=config,
                   implementation_sha256=implementation,metrics=result["metrics"],
                   checkpoint_sha256=sha(stage_dir/"best.pt"),artifact_hashes=artifact_hashes,raw_measured=True))
        retire_latest(stage_dir)
        print(json.dumps({"stage":stage,"metrics":result["metrics"]}),flush=True)
    write_json(output/"completed.json",{"status":"completed","dataset":a.dataset,"label_ratio":a.label_ratio})
    print("Completed. Final checkpoint: "+str(output/"checkpoints"/"sampc"/"best.pt"))

if __name__=="__main__": main()
