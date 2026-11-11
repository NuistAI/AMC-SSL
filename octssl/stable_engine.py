import copy, hashlib, json, math, os, time
from collections import defaultdict
from pathlib import Path
import torch
from . import engine as E
from .models import build_model
from . import UICP, PLCP, SAMPC
from .UniformHypersphericalCenters import uniform_centers
from .stability import freeze_sampc_stats_and_dropout, clip_gradients
from .checkpoint_io import atomic_save, light_bundle, resolve_upstream, retire_stage, write_json, sha


STAGE_MODULES = {"uicp": UICP, "plcp": PLCP, "sampc": SAMPC}

ROOT=E.ROOT
implementation_hash=E.implementation_hash
request_stop=E.request_stop
stop_if_requested=E.stop_if_requested

def evaluate(model,val,config,centers,device):
    stop_if_requested()
    return E.evaluate_stage(model,[],val,config['stage'],config,centers,device,bool(config.get('smoke_mode')))

def run_experiment(config,run_dir,dry_run=False):
    if dry_run: raise ValueError('Use explicit isolated smoke configs')
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a CUDA-capable GPU')
    stop_if_requested()
    config=copy.deepcopy(config);run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    if ROOT.resolve() not in run_dir.resolve().parents:raise ValueError('Output outside revision root')
    stage=config['stage'];stage_module=STAGE_MODULES[stage];smoke=bool(config.get('smoke_mode'));seed=int(config['seed'])
    E.seed_all(seed);torch.set_num_threads(int(os.environ.get('OCTSSL_NUM_THREADS','2')))
    device=torch.device('cuda');torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    latest=run_dir/'latest.pt';best_path=run_dir/'best.pt'
    resume=torch.load(str(latest),map_location='cpu',weights_only=False) if latest.exists() else None
    upstream=config.get('upstream_experiment_id');upstream_path=None;upstream_sha=None
    if upstream:upstream_path,upstream_sha=resolve_upstream(ROOT,upstream)
    fingerprint={'config':config,'implementation':implementation_hash(),
        'split_sha256':sha(config['split_file']),'upstream_sha256':upstream_sha}
    fp_sha=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    model=build_model('Ours',config['num_classes'],config['image_size']).to(device)
    centers=None
    if resume is None and upstream:
        if upstream_path is None:raise RuntimeError('Fresh stage needs available upstream weights')
        parent=torch.load(str(upstream_path),map_location='cpu',weights_only=False)
        model.load_state_dict(parent['model'],strict=True);centers=parent.get('centers');del parent
    elif resume is not None:
        if resume['fingerprint']!=fp_sha:raise RuntimeError('Resume fingerprint changed')
        model.load_state_dict(resume['model'],strict=True);centers=resume['centers']
    if stage!='uicp':
        if centers is None:centers=uniform_centers(config['num_classes'],model.feature_dim,steps=10 if smoke else config['uniform_steps'],t=.07,seed=seed)
        centers=centers.to(device).detach()
    canonical_centers=centers.detach().clone() if stage=='sampc' else None
    teacher=SAMPC.create_teacher(model) if stage=='sampc' else None
    optimizer=torch.optim.Adam(stage_module.optimizer_parameters(model),lr=config['learning_rate'],weight_decay=0)
    train=E.build_data(config,'train');val=E.build_data(config,'val')
    labeled=E.build_data(config,'train',True);unlabeled=E.build_data(config,'train',False)
    batch=int(config['batch_size']);lbs=int(config['labeled_bs'])
    if batch<=lbs or not len(labeled) or not len(unlabeled):raise ValueError('Invalid sampler')
    epoch_start=0;best=float('inf');best_epoch=-1;stale=0;global_step=0
    baseline_metrics={};last_epoch=-1
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer'])
        if teacher:teacher.load_state_dict(resume['teacher'],strict=True)
        epoch_start=resume['epoch']+1;best=resume['best'];best_epoch=resume['best_epoch']
        stale=resume['stale'];global_step=resume['global_step'];baseline_metrics=resume.get('baseline_metrics',{})
        E.restore_rng(resume['rng'])


        record=resume.get('epoch_record')
        if record is not None:
            log=run_dir/'epochs.jsonl'
            existing=[json.loads(v) for v in log.read_text().splitlines() if v.strip()] if log.exists() else []
            same=[v for v in existing if v['epoch']==record['epoch']]
            if same and same[-1]!=record:raise RuntimeError('Epoch log/checkpoint mismatch')
            if not same:
                if existing and existing[-1]['epoch']>=record['epoch']:raise RuntimeError('Epoch log order mismatch')
                with log.open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
        oldbest=torch.load(str(best_path),map_location='cpu',weights_only=False) if best_path.exists() else None
        if oldbest is None or oldbest['best']!=best or oldbest['best_epoch']!=best_epoch:
            if resume['epoch']!=best_epoch:raise RuntimeError('Cannot reconstruct missing prior best')
            atomic_save(best_path,light_bundle(resume))
        del oldbest,resume
    if (run_dir/'provenance.json').exists():
        if json.loads((run_dir/'provenance.json').read_text())!=fingerprint:raise RuntimeError('Provenance mismatch')
    else:write_json(run_dir/'provenance.json',fingerprint)
    def bundle(epoch):
        return {'schema':2,'checkpoint_kind':'resume','config':config,'fingerprint':fp_sha,
            'model':model.state_dict(),'teacher':teacher.state_dict() if teacher else None,
            'optimizer':optimizer.state_dict(),'scheduler':None,'centers':centers,
            'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,'global_step':global_step,
            'baseline_metrics':baseline_metrics,'rng':E.rng_state()}
    if not latest.exists():

        if stage=='sampc':
            best,baseline_metrics=evaluate(model,val,config,centers,device);best_epoch=-1
            write_json(run_dir/'initial_validation.json',{'epoch':0,'selection_value':best,**baseline_metrics})
        initial=bundle(-1);atomic_save(latest,initial)

        if upstream:retire_stage(ROOT/'checkpoints'/upstream,config['experiment_id'])
        if stage=='sampc':atomic_save(best_path,light_bundle(initial))
        del initial
    elif upstream:

        retire_stage(ROOT/'checkpoints'/upstream,config['experiment_id'])
    full_start=int(config.get('warm_up',10))+int(config.get('ramp_epochs',10)) if stage=='sampc' else 1
    epoch_max=int(config['epochs']);stop_patience=int(config['early_stopping'])
    if stale>=stop_patience:epoch_max=epoch_start
    try:
        for epoch in range(epoch_start,epoch_max):
            started=time.monotonic();model.train()
            frozen=freeze_sampc_stats_and_dropout(model) if stage=='sampc' else {'bn':0,'dropout':0}
            if teacher:teacher.eval()
            if stage=='uicp':batches=E.loader(train,batch,True,limit=batch if smoke else None)
            elif stage=='plcp':batches=E.loader(labeled,batch,True,limit=batch if smoke else None)
            else:batches=E.mixed_batches(labeled,unlabeled,batch,lbs,smoke)
            total=0.;count=0;sums=defaultdict(float);maxima=defaultdict(float)
            for b in batches:
                stop_if_requested();x=b['image'].to(device);y=b['label'].to(device)
                optimizer.zero_grad(set_to_none=True);terms={}
                loss,terms,centers=stage_module.training_step(
                    model,x,y,centers,config,teacher=teacher,epoch=epoch+1)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward()
                if stage=='sampc':
                    grad_norm=clip_gradients(model,float(config.get('grad_clip_norm',1.)))
                    terms['grad_norm_before_clip']=grad_norm
                elif not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError('Nonfinite pretraining gradient')
                optimizer.step();global_step+=1
                if teacher:
                    SAMPC.update_teacher(teacher,model,global_step,config['ema_decay'])
                total+=float(loss.detach());count+=1
                for key,value in terms.items():
                    number=float(value)
                    if not math.isfinite(number):raise FloatingPointError('Nonfinite diagnostic '+key)
                    sums[key]+=number;maxima[key]=max(maxima[key],number)
                if smoke:break
            if stage=='sampc' and not torch.equal(centers,canonical_centers):raise AssertionError('SAMPC centers changed')
            score,val_metrics=evaluate(model,val,config,centers,device)
            if not math.isfinite(score):raise FloatingPointError('Nonfinite validation')
            improved=score<best
            if improved:best=score;best_epoch=epoch;stale=0
            elif stage!='sampc' or epoch+1>full_start:stale+=1
            else:stale=0
            means={k:v/max(count,1) for k,v in sums.items()}
            rec={'epoch':epoch+1,'stage':stage,'train_loss':total/max(count,1),'batches':count,
                'seconds':time.monotonic()-started,'selection_value':score,'best':best,
                'best_epoch':best_epoch+1,'stale':stale,'frozen_bn_modules':frozen['bn'],'disabled_dropout_modules':frozen['dropout'],
                'loss_terms_mean':means,'loss_terms_max':dict(maxima),**val_metrics}
            saved=bundle(epoch);saved['epoch_record']=rec
            atomic_save(latest,saved)
            if improved:atomic_save(best_path,light_bundle(saved))
            del saved
            with (run_dir/'epochs.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n');f.flush();os.fsync(f.fileno())
            print(json.dumps({'experiment':config['experiment_id'],**rec}),flush=True)
            last_epoch=epoch;stop_if_requested()
            if stale>=stop_patience:break
    except (KeyboardInterrupt,SystemExit):
        write_json(run_dir/'interrupted.json',{'resume_from_latest_complete_epoch':True,'next_epoch':last_epoch+2});raise
    stop_if_requested()
    best_saved=torch.load(str(best_path),map_location='cpu',weights_only=False)
    model.load_state_dict(best_saved['model'],strict=True)
    selected_epoch=best_saved['epoch']+1
    metrics={'best_validation_selection':best,'epochs':max(epoch_start,last_epoch+1),'best_epoch':selected_epoch,
        'selected_checkpoint':'PLCP_initialization' if stage=='sampc' and selected_epoch==0 else 'trained_student',
        'global_step':global_step,'smoke_mode':smoke,'gpu':torch.cuda.get_device_name(0),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    del best_saved
    if stage=='sampc' and config.get('evaluate_test',True):
        rows=E.overlap_rows(model,E.build_data(config,'test'),device,config.get('eval_batch_size',2),2 if smoke else None)
        metrics.update(E.aggregate(rows));write_json(run_dir/'test_per_image.json',rows)
    return {'metrics':metrics,'artifacts':[str(best_path),str(run_dir/'epochs.jsonl'),str(run_dir/'provenance.json')],'rows':{}}
