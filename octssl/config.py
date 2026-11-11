CLASS_NAMES = {
    "HCMS": ["NFL","IPL","INL","OPL","ONL","IS","OS","RPE"],
    "OCTA500": ["ILM-IPL","IPL-OPL","OPL-ISOS","ISOS-RPE","RPE-BM"],
}
def stage_config(dataset, protocol, split_file, label_ratio, seed, stage, smoke=False):
    config = dict(dataset=dataset, method="Ours", stage=stage, experiment_id=stage,
        protocol_path=str(protocol), split_file=str(split_file), label_ratio=int(label_ratio),
        seed=int(seed), num_classes=9 if dataset=="HCMS" else 6,
        image_size=[128,1024] if dataset=="HCMS" else [256,256],
        batch_size=16, labeled_bs=2, eval_batch_size=2, optimizer="Adam",
        scheduler="ConstantLR", learning_rate=1e-5 if stage=="sampc" else 1e-4,
        epochs={"uicp":1000,"plcp":600,"sampc":1500}[stage],
        early_stopping=20 if stage=="sampc" else 100, plcp_weight=1.,
        uniform_steps=10000, tau=.05, beta=10., r1=.5, r2=1., r3=.2,
        entropy_threshold=.1, ema_decay=.9, warm_up=10, ramp_epochs=10,
        grad_clip_norm=1., evaluate_test=stage=="sampc",
        revision="stabilization_v4_paper_branch_alignment")
    if stage!="uicp": config["upstream_experiment_id"]={"plcp":"uicp","sampc":"plcp"}[stage]
    if smoke:
        config.update(epochs=1, uniform_steps=10, smoke_mode=True, warm_up=0, ramp_epochs=1)
    return config
