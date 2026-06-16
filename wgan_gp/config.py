import os
# ── Paths ────────────────────────────────────────────────────────────────────
IMAGE_DIR  = os.path.abspath(os.path.join(__file__, '..', 'data'))
OUTPUT_DIR = os.path.abspath(os.path.join(__file__, '..', 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Image settings ────────────────────────────────────────────────────────────
IMG_HEIGHT   = 128
IMG_WIDTH    = 128
IMG_CHANNELS = 1

# ── Dataset settings ──────────────────────────────────────────────────────────
BATCH_SIZE            = 64
AUGMENTATION_FACTOR   = 1   # Total images created per original (1 original + 4 augmented)

# ── Model settings ────────────────────────────────────────────────────────────
LATENT_DIM = 128

# ── Training settings ─────────────────────────────────────────────────────────
EPOCHS    = 1500
N_CRITIC  = 5
LAMBDA_GP = 10
LR_G      = 1e-4
LR_D      = 1e-4

# ── Logging / checkpointing ───────────────────────────────────────────────────
SAMPLE_EVERY  = 25   # Save image grid every N epochs (and at epoch 1)
N_SEED_IMAGES = 16   # Number of fixed noise images for progress tracking
