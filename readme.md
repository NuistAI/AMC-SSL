# AMC-SSL

Angular-Margin Contrastive Semi-Supervised Learning for Retinal
OCT Layer Segmentation with UICP, PLCP, and SAMPC.

## Requirements

Python 3.10+ and CUDA-enabled PyTorch for training.

```bash
python -m pip install torch torchvision numpy scipy opencv-python Pillow
```

## Data

Prepare your own data:

- **HCMS:** `image/<sample>.png` and `label/<sample>.png`, both 128 x 1024.
- **OCTA-500:** `OCTA_6mm/OCT/<subject>/1.bmp` through `400.bmp` (640 x 400), and `GroundTruth/GT_Layers/<subject>.mat` containing `Layer` with shape `(6, 400, 400)`.

Create `splits.json` with your subject IDs:

```json
{
  "train": ["s1", "s2", "s3", "s4", "s5"],
  "val": ["s6"],
  "test": ["s7"],
  "labeled": ["s1"]
}
```

Replace the example IDs with your own. The partitions must not share subjects. For HCMS, subject IDs are sample filename prefixes before the first underscore.

## Run

Run these commands from the project directory:

```bash
python prepare_data.py --dataset HCMS --data-root data/HCMS --subject-splits splits.json --label-ratio 20 --output protocol.json
python run.py --dataset HCMS --protocol protocol.json --label-ratio 20 --output outputs/run1
python evaluate.py --checkpoint outputs/run1/checkpoints/sampc/best.pt --protocol protocol.json --device cpu
```

For OCTA-500, use `--dataset OCTA500` and the corresponding data root. Training runs UICP, PLCP, and SAMPC sequentially; evaluation reports per-layer Dice, MDice, and MIOU (%).
