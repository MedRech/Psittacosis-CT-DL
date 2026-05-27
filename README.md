# Psittacosis-CT-DL

Core implementation for a two-stage chest CT deep learning pipeline for psittacosis-assisted diagnosis.

The repository contains:

- `model_seg.py`: 3D U-Net segmentation model for pulmonary lesion masks.
- `model_class.py`: LTA-ResNet classifier with CBAM attention.
- `train.py`: training entry point for segmentation or classification.
- `test.py`: evaluation entry point.
- `dataset.py`: dataset loaders for 3D CT volumes and masks.
- `losses.py`: Dice loss, boundary loss placeholder, and weighted BCE.
- `metrics.py`: segmentation and classification metrics.

## Project Structure

```text
Psittacosis-CT-DL/
├── configs/
│   ├── class.yaml
│   └── seg.yaml
├── dataset.py
├── losses.py
├── metrics.py
├── model_class.py
├── model_seg.py
├── requirements.txt
├── test.py
└── train.py
```

## Data Format

Prepare metadata CSV files with these columns.

For segmentation:

```csv
image,mask
/path/to/ct_001.npy,/path/to/mask_001.npy
```

For classification:

```csv
image,label
/path/to/crop_001.npy,1
/path/to/crop_002.npy,0
```

The example loader expects `.npy` arrays shaped as `D x H x W`. Adapt `dataset.py` if your data are stored as NIfTI, DICOM, or NRRD.

## Usage

Train segmentation:

```bash
python train.py --task seg --config configs/seg.yaml
```

Train classification:

```bash
python train.py --task class --config configs/class.yaml
```

Evaluate:

```bash
python test.py --task seg --config configs/seg.yaml --checkpoint checkpoints/seg_best.pt
python test.py --task class --config configs/class.yaml --checkpoint checkpoints/class_best.pt
```

## Citation Statement

If this repository is used with the manuscript, cite it as:

```text
The core implementation, including training, testing, segmentation model,
and classification model code, is available at:
https://github.com/<your-username>/Psittacosis-CT-DL
```
