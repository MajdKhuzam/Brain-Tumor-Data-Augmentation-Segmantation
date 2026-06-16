# Brain Tumor Data Augmentation — WGAN-GP + U-Net

> A two-stage deep learning pipeline for synthetic brain MRI data augmentation. A WGAN-GP generates realistic brain tumor images (FID 72.4), and a U-Net automatically predicts segmentation masks for each generated image (Dice 0.845, IoU 0.735). Trained the WGAN-GP for 1,500 epochs on 3,000 MRI images augmented to 9,000 samples, and the U-Net for 20 epochs on 3,000 images augmented to 15,000 samples.

---

## Getting Started

### 1 — Clone the repository

```bash
git clone https://github.com/MajdKhuzam/brain-tumor-data-augmentation-segmentation
cd brain-tumor-data-augmentation-segmentation
```

### 2 — Install dependencies

- **Python** 3.10.12

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.10.0+cu128 | U-Net model and training |
| `torchvision` | 0.25.0+cu128 | Image transforms and utilities |
| `torchsummary` | 1.5.1 | Model architecture summary |
| `tensorflow` | 2.19.0 | WGAN-GP model and training |
| `numpy` | 2.2.6 | Array operations |
| `pillow` | 12.1.0 | Image loading and saving |
| `opencv-python` | 4.12.0 | Preprocessing and contour detection |
| `matplotlib` | 3.10.7 | Visualisation and plot saving |
| `tqdm` | 4.67.1 | Training progress bars |

```bash
pip install -r requirements.txt
```

### 3 — Prepare the data

Download each dataset from Kaggle (see [Dataset](#dataset) section) and place the files into the corresponding `data/` folder:

**WGAN-GP:**
```
wgan_gp/
└── data/
    └── *.jpg / *.png    # MRI images (no masks needed)
```

**U-Net:**
```
unet/
└── data/
    ├── images/          # Brain MRI scans
    └── masks/           # Corresponding binary tumor masks (same filenames)
```

> The U-Net preprocessing step automatically crops each image to the bounding box of its largest contour before training.

---

## Dataset

| Model | Dataset | Link |
|---|---|---|
| WGAN-GP | BraTS 2019 Train/Test/Valid | https://www.kaggle.com/datasets/aryanfelix/brats-2019-traintestvalid |
| U-Net | Brain Tumor Segmentation | https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation |

---

## Project Structure

```
project/
│
├── wgan_gp/                  # Generative model
│   ├── config.py             # Paths, hyperparameters, training settings
│   ├── data.py               # Image loading, augmentation, tf.data pipeline
│   ├── losses.py             # WGAN-GP losses and gradient penalty
│   ├── model.py              # Generator and Discriminator architectures
│   ├── train.py              # Per-step training functions and main loop
│   ├── visualize.py          # Sample saving and loss curve plotting
│   ├── evaluation.py         # FID metric computation for generated images
│   └── run_train.py          # Entry point for WGAN-GP training
│
├── unet/                     # Segmentation model
│   ├── config.py             # Paths, hyperparameters, split ratios
│   ├── preprocessing.py      # Bounding-box cropping of raw MRI images
│   ├── dataset.py            # Dataset class, augmentation, DataLoader factory
│   ├── model.py              # U-Net architecture
│   ├── train.py              # Training loop with best-checkpoint tracking
│   ├── utils.py              # Checkpoint I/O, Dice/IoU evaluation, image saving
│   ├── evaluate.py           # Loss curves, test-set evaluation, visualisation
│   └── run_train.py          # Entry point for U-Net training
│
├── inference.py              # Full pipeline: generate images → predict masks
├── inference2.py             # Full pipeline but inference WGAN-GP using .h5 weights
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

```
Random Noise (z)
      │
      ▼
 ┌──────────┐      Synthetic Tumored MRI Images
 │ WGAN-GP  │ ──────────────────────────────────┐
 │ Generator│                                   │
 └──────────┘                                   ▼
                                          ┌──────────┐
                                          │  U-Net   │ ──► Tumor Segmentation Mask
                                          │ (frozen) │
                                          └──────────┘
```

1. **WGAN-GP** is trained on real brain tumor MRI images to learn the data distribution and generate synthetic tumored images.
2. **U-Net** is trained on real images paired with ground-truth tumor masks.
3. **Inference** uses the trained generator to produce new synthetic MRI images, then runs the U-Net to predict a tumor mask for each one — creating augmented (image, mask) pairs ready for downstream training.

---

## Models

### WGAN-GP — Generator & Discriminator (TensorFlow / Keras)

The **Generator** maps a latent vector to a 128×128 grayscale image:

| Layer | Output shape |
|---|---|
| Dense + Reshape | 4 × 4 × 512 |
| ConvTranspose × 5 (stride 2) | 8→16→32→64→128 |
| LayerNormalization + ReLU (×4) | — |
| Conv2DTranspose + Tanh (output) | 128 × 128 × 1 |

The **Discriminator (Critic)** classifies real vs. fake images:

| Layer | Output shape |
|---|---|
| Conv2D × 4 (stride 2) | 64→32→16→8 |
| LeakyReLU(0.2) after each | — |
| Flatten + Dense(1) | scalar score |

> No normalisation layers in the discriminator — required by WGAN-GP.

### U-Net — Segmentation (PyTorch)

Standard encoder–decoder with skip connections:

- **Encoder**: 4 × DoubleConv blocks (32 → 64 → 128 → 256) + MaxPool
- **Bottleneck**: DoubleConv (512 channels)
- **Decoder**: 4 × ConvTranspose2d + DoubleConv blocks with skip concatenation
- **Output**: 1×1 Conv → binary mask logit

Each DoubleConv block applies `Conv2d → BatchNorm2d → ReLU` twice.

---

## Training

### 1 — Train WGAN-GP

```bash
cd wgan_gp
python run_train.py
```

Checkpoints and sample image grids are saved to `output/` every `SAMPLE_EVERY` epochs.

### 2 — Train U-Net

```bash
cd unet
python run_train.py
```

The best checkpoint (by (Dice + IoU) / 2) is saved to `output/best_model.pth.tar`.

---

## Inference

Run the full pipeline to generate synthetic images and predict their masks:

```bash
python inference.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--num-images` | 5 | Number of images to generate |
| `--generator-path` | auto | Path to `.keras` generator model |
| `--unet-path` | auto | Path to U-Net `.pth.tar` checkpoint |
| `--output-dir` | `inference_output/` | Where to save results |
| `--threshold` | 0.5 | Mask binarisation threshold |
| `--seed` | None | Random seed for reproducibility |

Results are saved as paired `generated_XXXX.png` / `mask_XXXX.png` files and visualised in a matplotlib window.

---

## Configuration

### WGAN-GP (`wgan_gp/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `IMG_HEIGHT / IMG_WIDTH` | 128 | Generated image resolution |
| `LATENT_DIM` | 128 | Noise vector size |
| `BATCH_SIZE` | 64 | Training batch size |
| `EPOCHS` | 1500 | Total training epochs |
| `N_CRITIC` | 5 | Discriminator steps per generator step |
| `LAMBDA_GP` | 10 | Gradient penalty weight |
| `LR_G / LR_D` | 1e-4 | Adam learning rates |
| `AUGMENTATION_FACTOR` | 1 | Augmented copies per original image |
| `SAMPLE_EVERY` | 25 | Save image grid every N epochs |

### U-Net (`unet/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | 128 | Input resolution |
| `BATCH_SIZE` | 32 | Training batch size |
| `LEARNING_RATE` | 1e-4 | Adam learning rate |
| `NUM_EPOCHS` | 20 | Total training epochs |
| `NUM_AUG` | 5 | Augmented copies per training image |
| `VAL_SIZE / TEST_SIZE` | 0.15 | Validation and test split ratios |

---

## Outputs

| Path | Description |
|---|---|
| `wgan_gp/output/generator_model_final.keras` | Final trained generator |
| `wgan_gp/output/generated_epoch_XXXX.png` | Sample grids during training |
| `unet/output/best_model.pth.tar` | Best U-Net checkpoint |
| `unet/output/{train,val}_losses.npy` | Loss history arrays |
| `unet/output/{dice,iou}_scores.npy` | Metric history arrays |
| `inference_output/generated_XXXX.png` | Synthetic MRI images |
| `inference_output/mask_XXXX.png` | Predicted tumor masks |

---

## Notes

- The WGAN-GP generator outputs images in [-1, 1] range; these are converted to [0, 255] for display and saved as grayscale PNGs.
- The U-Net expects input in [0, 1] range (no additional normalization).
- Both models use 128×128 grayscale images with a single channel.
- Checkpoints are saved automatically during training; inference auto-detects the latest weights.
