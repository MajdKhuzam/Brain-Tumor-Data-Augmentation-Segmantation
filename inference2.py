"""
inference2.py
------------
Generate brain tumor images using the trained WGAN-GP generator,
then segment them using the trained U-Net model.
but this version if for loading WGAN-GP model weights, .h5 file not the .keras file.
"""

import os
import sys
import argparse
import importlib.util
import numpy as np
from PIL import Image

PROJECT_DIR = os.path.dirname(__file__)


# ── Module loading ────────────────────────────────────────────────────────────

def load_module_from_file(module_name, file_path, config_module=None):
    """Load a Python module directly from a file path, bypassing sys.path.

    If config_module is provided, it is temporarily registered as 'config'
    in sys.modules so that bare 'from config import ...' statements resolve
    to the correct config file.
    """
    config_key = 'config'
    old_config = sys.modules.get(config_key)

    if config_module is not None:
        sys.modules[config_key] = config_module

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if old_config is not None:
            sys.modules[config_key] = old_config
        elif config_module is not None:
            sys.modules.pop(config_key, None)

    return module


def load_project_modules():
    """Load all project-specific modules and return (unet_config, unet_model, wgan_config, wgan_model)."""
    unet_dir = os.path.join(PROJECT_DIR, 'unet')
    unet_config = load_module_from_file('unet_config', os.path.join(unet_dir, 'config.py'))
    unet_model = load_module_from_file(
        'unet_model', os.path.join(unet_dir, 'model.py'), config_module=unet_config
    )

    wgan_dir = os.path.join(PROJECT_DIR, 'wgan_gp')
    wgan_config = load_module_from_file('wgan_config', os.path.join(wgan_dir, 'config.py'))
    wgan_model = load_module_from_file(
        'wgan_model', os.path.join(wgan_dir, 'model.py'), config_module=wgan_config
    )

    return unet_config, unet_model, wgan_config, wgan_model


# ── Model loading ─────────────────────────────────────────────────────────────

def load_generator(wgan_config, model_path=None):
    """Load the trained WGAN-GP generator from weights file."""
    import tensorflow as tf
    from tensorflow.keras import layers

    if model_path is None:
        wgan_dir = os.path.join(PROJECT_DIR, 'wgan_gp', 'output')
        model_path = os.path.join(wgan_dir, 'generator_weights.weights.h5')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Generator weights not found at {model_path}")

    # Rebuild architecture (must match training exactly)
    generator = tf.keras.Sequential([
        layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(wgan_config.LATENT_DIM,)),
        layers.Reshape((4, 4, 512)),
        layers.LayerNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(256, (3,3), strides=(2,2), padding='same', use_bias=False),
        layers.LayerNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', use_bias=False),
        layers.LayerNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', use_bias=False),
        layers.LayerNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', use_bias=False),
        layers.LayerNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(wgan_config.IMG_CHANNELS, (3,3), strides=(2,2),
                               padding='same', use_bias=True, activation='tanh'),
    ], name='generator')

    generator.load_weights(model_path)
    print(f"WGAN-GP generator loaded from {model_path}")
    return generator


def load_unet(unet_config, unet_model, model_path=None):
    """Load the trained U-Net segmentation model."""
    import torch

    if model_path is None:
        model_path = unet_config.CHECKPOINT_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"U-Net checkpoint not found at {model_path}")

    model = unet_model.UNET(in_channels=1, out_channels=1).to(unet_config.DEVICE)
    checkpoint = torch.load(model_path, map_location=unet_config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"U-NET loaded from {model_path}")
    return model


# ── Generation & segmentation ─────────────────────────────────────────────────

def generate_images(generator, wgan_config, num_images=1):
    """Generate brain MRI images from random latent vectors.

    Returns a uint8 array of shape (N, H, W, 1).
    """
    import tensorflow as tf

    noise = tf.random.normal([num_images, wgan_config.LATENT_DIM])
    generated = generator(noise, training=False)
    return (generated.numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8)


def preprocess_for_unet(image_np, unet_config):
    """Resize and tensorise a single HxW uint8 array for U-Net input."""
    import torch
    import torchvision.transforms.functional as TF

    if image_np.ndim == 3 and image_np.shape[-1] == 1:
        image_np = image_np.squeeze(-1)

    pil_img = Image.fromarray(image_np).convert('L')
    pil_img = pil_img.resize((unet_config.IMAGE_SIZE, unet_config.IMAGE_SIZE), Image.BILINEAR)
    return TF.to_tensor(pil_img).unsqueeze(0).to(unet_config.DEVICE)


def predict_mask(unet, image_tensor, threshold=0.5):
    """Return a binary HxW float32 segmentation mask."""
    import torch

    with torch.no_grad():
        probs = torch.sigmoid(unet(image_tensor))
        mask = (probs > threshold).float()
    return mask.squeeze().cpu().numpy()


def segment_images(unet, generated_images, unet_config, threshold=0.5):
    """Run U-Net over every generated image.

    Yields (img_np, mask_np, tumor_pct) for each image.
    """
    for img_np in generated_images:
        tensor = preprocess_for_unet(img_np, unet_config)
        mask_np = predict_mask(unet, tensor, threshold=threshold)
        tumor_pct = mask_np.mean() * 100
        yield img_np.squeeze(), mask_np, tumor_pct


# ── Persistence ───────────────────────────────────────────────────────────────

def save_results(results, output_dir):
    """Save generated images and masks to disk.

    Expects results to be a list of (img_np, mask_np, tumor_pct) tuples.
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, (img_np, mask_np, tumor_pct) in enumerate(results):
        img_path = os.path.join(output_dir, f'generated_{idx:04d}.png')
        mask_path = os.path.join(output_dir, f'mask_{idx:04d}.png')

        Image.fromarray(img_np, mode='L').save(img_path)
        Image.fromarray((mask_np * 255).astype(np.uint8), mode='L').save(mask_path)

        print(f"  [{idx+1}/{len(results)}] {img_path}, {mask_path}  (tumor: {tumor_pct:.1f}%)")


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_results(results):
    """Display generated images alongside their predicted masks."""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 5, rows * 5))
    axes = np.array(axes).flatten()

    for idx, (img_np, mask_np, tumor_pct) in enumerate(results):
        axes[idx * 2].imshow(img_np, cmap='gray')
        axes[idx * 2].set_title(f'Generated Image #{idx + 1}')
        axes[idx * 2].axis('off')

        axes[idx * 2 + 1].imshow(mask_np, cmap='gray')
        axes[idx * 2 + 1].set_title(f'Predicted Mask (tumor: {tumor_pct:.1f}%)')
        axes[idx * 2 + 1].axis('off')

    for ax in axes[n * 2:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_inference(
    num_images=5,
    generator_path=None,
    unet_path=None,
    output_dir=None,
    threshold=0.5,
    seed=None,
):
    """Full pipeline: generate images → predict masks → save → visualize."""
    import tensorflow as tf
    import torch

    if seed is not None:
        tf.random.set_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    if output_dir is None:
        output_dir = os.path.join(PROJECT_DIR, 'inference_output')

    print("=" * 60)
    print("Loading project modules...")
    unet_config, unet_model, wgan_config, wgan_model = load_project_modules()

    print("Loading models...")
    generator = load_generator(wgan_config, generator_path)
    unet = load_unet(unet_config, unet_model, unet_path)

    print(f"\nGenerating {num_images} brain tumor images...")
    generated_images = generate_images(generator, wgan_config, num_images)

    print("Predicting segmentation masks...")
    results = list(segment_images(unet, generated_images, unet_config, threshold))

    print("Saving results...")
    save_results(results, output_dir)
    print(f"\nDone. All outputs saved to {output_dir}/")
    print("=" * 60)

    visualize_results(results)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(description='Generate & segment brain tumor images')
    parser.add_argument('--num-images',      type=int,   default=5,   help='Number of images to generate')
    parser.add_argument('--generator-path',  type=str,   default=None, help='Path to generator model')
    parser.add_argument('--unet-path',       type=str,   default=None, help='Path to U-Net checkpoint')
    parser.add_argument('--output-dir',      type=str,   default=None, help='Output directory')
    parser.add_argument('--threshold',       type=float, default=0.5,  help='Segmentation threshold')
    parser.add_argument('--seed',            type=int,   default=None, help='Random seed for reproducibility')
    return parser


def main():
    args = build_parser().parse_args()
    run_inference(
        num_images=args.num_images,
        generator_path=args.generator_path,
        unet_path=args.unet_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
