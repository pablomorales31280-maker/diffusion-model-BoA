# BoA Diffusion Deblurring

This repository implements a **predict-and-refine diffusion model for image deblurring**.

The model contains two jointly trained networks:

- **PreNet**: a deterministic UNet that produces an initial sharp estimate.
- **DenoiseNet**: a diffusion UNet that models and reconstructs the residual.

The SR3 UNet has been extended with configurable downsampling and upsampling operators.  
The current GoPro configuration uses:

- PreNet: `convstride2` + `pixelshuffle`
- DenoiseNet: `fpdh` + `freqavgup`

The test pipeline also supports an optional **L-infinity PGD attack**.

## Requirements

Python 3.10+ and a CUDA-capable GPU are recommended.

```bash
pip install -r requirement.txt
```

For HPC systems, install a PyTorch build compatible with the CUDA version available on the cluster.

## GoPro dataset

Expected raw dataset structure:

```text
dataset/
├── train/
│   └── <sequence>/
│       ├── blur/*.png
│       └── sharp/*.png
└── test/
    └── <sequence>/
        ├── blur/*.png
        └── sharp/*.png
```

Create the LMDB datasets with:

```bash
python data/create_gopro_lmdb.py \
    --dataset-root dataset \
    --split train \
    --output dataset_lmdb/gopro_train.lmdb \
    --crop-size 128 \
    --overwrite

python data/create_gopro_lmdb.py \
    --dataset-root dataset \
    --split test \
    --output dataset_lmdb/gopro_test.lmdb \
    --crop-size null \
    --overwrite
```

Update the LMDB paths in the JSON configuration files before training or testing.

## Training

Example:

```bash
python train.py \
    -c config/sr_sr3_fpdh_smoke.json \
    -p train \
    -gpu 0
```

Training checkpoints contain both networks:

```text
I20_E1_PreNet_gen.pth
I20_E1_PreNet_opt.pth
I20_E1_DenoiseNet_gen.pth
I20_E1_DenoiseNet_opt.pth
```

## Testing

Example:

```bash
python test.py \
    -c config/sr_sr3_fpdh_smoke_test.json \
    -p val \
    -gpu 0
```

In the test configuration, `resume_state` must be the **checkpoint prefix**, not one individual `.pth` file:

```json
"resume_state": "experiments/<run>/checkpoint/I20_E1"
```

Both `PreNet` and `DenoiseNet` are loaded automatically from this prefix.

Useful test options:

```json
"datasets": {
    "val": {
        "data_len": -1
    }
},
"test": {
    "crop_size": null,
    "save_images": true
},
"attack": {
    "enabled": true,
    "type": "pgd",
    "target": "prenet",
    "norm": "linf",
    "epsilon_255": 8.0,
    "step_size_255": 2.0,
    "steps": 10
}
```

- `data_len: -1` evaluates the complete test dataset.
- `crop_size: null` uses full-resolution images.
- `attack.enabled: false` runs clean evaluation only.
- `attack.enabled: true` additionally evaluates PGD robustness.

Full-resolution inference can require a large amount of GPU memory with the dynamic FPDH architecture.

## Test outputs

Each test run stores its results under:

```text
experiments/<name>_<timestamp>/results/
├── metrics.csv
├── summary.txt
└── images/
```

The reported metrics include clean and attacked PSNR/SSIM, perturbation magnitude, and PSNR drop when PGD is enabled.