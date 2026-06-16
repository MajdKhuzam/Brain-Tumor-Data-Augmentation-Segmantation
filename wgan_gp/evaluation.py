"""
evaluate.py — FID evaluation for the WGAN-GP brain-tumour generator.

Usage:
    python evaluate.py
    python evaluate.py --model output/generator_model_final.keras
    python evaluate.py --model output/generator_model.h5 --num_samples 2000

The script:
  1. Loads real images from config.IMAGE_DIR.
  2. Loads the saved generator (Keras or H5 format).
  3. Generates an equal number of synthetic images.
  4. Extracts InceptionV3 pool3 features for both sets.
  5. Computes Fréchet Inception Distance (FID).
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from scipy import linalg
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

# ── Project imports ────────────────────────────────────────────────────────────
# Allow running from the project root or from a sibling directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import IMAGE_DIR, OUTPUT_DIR, LATENT_DIM
from data import load_images
from model import build_generator


# ── Constants ──────────────────────────────────────────────────────────────────
INCEPTION_INPUT_SIZE = 299   # InceptionV3 expects 299×299 RGB


# ── Inception feature extractor ────────────────────────────────────────────────

def build_inception_extractor() -> tf.keras.Model:
    """Return InceptionV3 truncated at the global-average-pool layer (2048-d)."""
    base = InceptionV3(include_top=False, pooling="avg", input_shape=(INCEPTION_INPUT_SIZE, INCEPTION_INPUT_SIZE, 3))
    base.trainable = False
    return base


def preprocess_for_inception(images_norm: np.ndarray) -> np.ndarray:
    """
    Convert [-1, 1] grayscale (N, H, W, 1) → InceptionV3-ready (N, 299, 299, 3).

    Steps:
      1. Denormalise to [0, 255] uint8.
      2. Tile single channel → 3 channels (grayscale → pseudo-RGB).
      3. Resize to 299×299.
      4. Apply Inception preprocessing (scales to [-1, 1] internally).
    """
    # Denormalise to uint8
    imgs = ((images_norm + 1.0) / 2.0 * 255.0).astype(np.uint8)

    processed = []
    for img in imgs:
        img = img.squeeze()                                           # (H, W)
        img_rgb = np.stack([img, img, img], axis=-1)                  # (H, W, 3)
        img_resized = tf.image.resize(img_rgb, [INCEPTION_INPUT_SIZE, INCEPTION_INPUT_SIZE]).numpy().astype(np.float32)
        processed.append(img_resized)

    batch = np.stack(processed, axis=0)                               # (N, 299, 299, 3)
    return preprocess_input(batch)                                     # Inception scaling


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(
    extractor: tf.keras.Model,
    images: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Run images through InceptionV3 in batches; return (N, 2048) activations."""
    all_features = []
    n = len(images)
    for start in range(0, n, batch_size):
        batch = images[start : start + batch_size]
        feats = extractor(batch, training=False).numpy()
        all_features.append(feats)
        print(f"  Extracted features: {min(start + batch_size, n)}/{n}", end="\r")
    print()
    return np.concatenate(all_features, axis=0)


# ── FID calculation ────────────────────────────────────────────────────────────

def compute_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
) -> float:
    """
    Fréchet Inception Distance between two sets of feature vectors.

    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2·(Σ_r·Σ_g)^½)

    Lower is better. Typical ranges:
        < 50   : decent quality
        < 20   : good
        < 10   : excellent (hard to achieve without a large dataset)
    """
    mu_r = real_features.mean(axis=0)
    mu_g = fake_features.mean(axis=0)

    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_g

    # Symmetric matrix square root via scipy
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)

    # Numerical clean-up: sqrtm can introduce small imaginary parts
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("sqrtm produced large imaginary components — check your features.")
        covmean = covmean.real

    fid = float(
        diff @ diff
        + np.trace(sigma_r)
        + np.trace(sigma_g)
        - 2.0 * np.trace(covmean)
    )
    return fid


# ── Model loading ──────────────────────────────────────────────────────────────

def load_generator(model_path: str, latent_dim: int) -> tf.keras.Model:
    """
    Build the generator architecture from model.py and load weights from
    a .h5 weights file (or a full .keras / SavedModel if available).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading generator weights from: {model_path}")

    # Build architecture first so we can load bare weights files
    generator = build_generator(latent_dim)

    # Trigger weight creation by running a dummy forward pass
    _ = generator(tf.zeros([1, latent_dim]), training=False)

    generator.load_weights(model_path)
    print("Weights loaded successfully.\n")
    return generator


def find_model(output_dir: str) -> str:
    """
    Auto-detect the latest saved generator in *output_dir*.
    Preference order: final .h5 → checkpoint .h5 → final .keras → checkpoint .keras
    """
    candidates = [
        os.path.join(output_dir, "generator_model_final.h5"),
        os.path.join(output_dir, "generator_model.h5"),
        os.path.join(output_dir, "generator_model_final.keras"),
        os.path.join(output_dir, "generator_model.keras"),
        os.path.join(output_dir, "generator_weights.weights.h5"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No generator model found in '{output_dir}'. "
        "Pass --model <path> explicitly."
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID for the WGAN-GP brain-tumour generator.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the saved generator (.keras or .h5). Auto-detected from OUTPUT_DIR if omitted.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of images to use for FID (default: all real images).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for Inception feature extraction (default: 64).",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=LATENT_DIM,
        help=f"Generator latent dimension (default: {LATENT_DIM} from config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load real images ──────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1/4 — Loading real images")
    print("=" * 60)
    real_images = load_images(IMAGE_DIR)          # (N, H, W, 1) in [-1, 1]

    num_samples = args.num_samples or len(real_images)
    num_samples = min(num_samples, len(real_images))

    # FID is unreliable with very few samples; warn if below threshold
    if num_samples < 100:
        print(
            f"\n⚠ Warning: only {num_samples} samples — FID estimates become "
            "unreliable below ~100 images.\n"
        )

    # Sub-sample if requested
    idx = np.random.choice(len(real_images), size=num_samples, replace=False)
    real_images = real_images[idx]
    print(f"Using {num_samples} real images.\n")

    # ── 2. Load generator and produce fakes ──────────────────────────────────
    print("=" * 60)
    print("Step 2/4 — Generating synthetic images")
    print("=" * 60)
    model_path = args.model or find_model(OUTPUT_DIR)
    generator = load_generator(model_path, latent_dim=args.latent_dim)

    fake_images_list = []
    batch_size = args.batch_size
    generated_so_far = 0
    while generated_so_far < num_samples:
        n_batch = min(batch_size, num_samples - generated_so_far)
        noise = tf.random.normal([n_batch, args.latent_dim])
        batch = generator(noise, training=False).numpy()
        fake_images_list.append(batch)
        generated_so_far += n_batch
        print(f"  Generated: {generated_so_far}/{num_samples}", end="\r")

    print()
    fake_images = np.concatenate(fake_images_list, axis=0)   # (N, H, W, 1)
    print(f"Synthetic images shape: {fake_images.shape}\n")

    # ── 3. Preprocess both sets for InceptionV3 ───────────────────────────────
    print("=" * 60)
    print("Step 3/4 — Extracting InceptionV3 features")
    print("=" * 60)
    extractor = build_inception_extractor()

    print("Processing real images …")
    real_inception = preprocess_for_inception(real_images)
    real_features  = extract_features(extractor, real_inception, batch_size=batch_size)

    print("Processing fake images …")
    fake_inception = preprocess_for_inception(fake_images)
    fake_features  = extract_features(extractor, fake_inception, batch_size=batch_size)

    print(f"Feature shapes — real: {real_features.shape}, fake: {fake_features.shape}\n")

    # ── 4. Compute FID ────────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 4/4 — Computing FID")
    print("=" * 60)
    fid = compute_fid(real_features, fake_features)

    print(f"\n{'─' * 40}")
    print(f"  FID score : {fid:.4f}")
    print(f"  Samples   : {num_samples}")
    print(f"  Model     : {os.path.basename(model_path)}")
    print(f"{'─' * 40}")
    print()

    if fid < 10:
        verdict = "Excellent"
    elif fid < 20:
        verdict = "Good"
    elif fid < 50:
        verdict = "Moderate"
    elif fid < 100:
        verdict = "Poor"
    else:
        verdict = "Very poor — model may need more training"

    print(f"  Quality verdict: {verdict}")
    print()


if __name__ == "__main__":
    main()