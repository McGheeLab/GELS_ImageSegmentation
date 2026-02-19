"""
Granule Segmentation — Image Segmenter
========================================

Opens dialogs to:
  1. Select a trained model (.joblib) from granule_trainer.py
  2. Select one or more raw fluorescence images to segment

Reads tunable postprocessing parameters from segmentation_config.json.
The config used during training is also stored inside the model file
to ensure feature extraction stays consistent.

Requirements:
    pip install numpy scikit-learn scikit-image tifffile matplotlib joblib

Usage:
    python granule_segmenter.py
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict

try:
    import tifffile
    HAS_TIFF = True
except ImportError:
    HAS_TIFF = False

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False

from skimage import io, color, filters, morphology
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk, remove_small_objects, remove_small_holes, closing
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import generic_filter
import joblib
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_CONFIG = {
    "mask_colors": {
        "void":       [0, 0, 255],
        "functional": [255, 165, 0],
        "inert":      [0, 255, 0],
    },
    "features": {
        "gaussian_scales": [1, 3, 7],
        "entropy_radii": [3, 7],
        "include_entropy": True,
        "include_channel_ratios": True,
    },
    "postprocessing": {
        "smooth_radius": 3,
        "min_object_size": 500,
        "min_hole_size": 300,
        "closing_radius": 2,
    },
    "output": {
        "save_comparison_plots": True,
        "output_suffix": "_segmented",
        "dpi": 200,
    },
}

CLASS_NAMES = ["void", "functional", "inert"]


def load_config(config_path: Path) -> Dict:
    """Load config JSON, falling back to defaults for missing keys."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy

    if config_path.exists():
        print(f"Loading config: {config_path}")
        with open(config_path) as f:
            user_cfg = json.load(f)
        for section, values in user_cfg.items():
            if isinstance(values, dict) and section in cfg:
                for k, v in values.items():
                    if k != "comment":
                        cfg[section][k] = v
    else:
        print(f"Config not found at {config_path}, using defaults.")

    return cfg


def build_mask_colors(cfg: Dict) -> Dict:
    mc = cfg["mask_colors"]
    return {
        "void":       {"label": 0, "rgb": tuple(mc["void"])},
        "functional": {"label": 1, "rgb": tuple(mc["functional"])},
        "inert":      {"label": 2, "rgb": tuple(mc["inert"])},
    }


# =============================================================================
# IMAGE I/O
# =============================================================================

def load_image(path: str) -> np.ndarray:
    """Load image, return (H,W,3) uint8 RGB."""
    path = str(path)
    if path.lower().endswith(('.tif', '.tiff')):
        img = tifffile.imread(path) if HAS_TIFF else io.imread(path)
    else:
        img = io.imread(path)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.ndim == 3 and img.shape[0] in (3, 4):
        img = np.moveaxis(img, 0, -1)
        if img.shape[2] == 4:
            img = img[:, :, :3]

    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else \
              (img / img.max() * 255).astype(np.uint8)

    return img


# =============================================================================
# FEATURE EXTRACTION  (must match trainer exactly)
# =============================================================================

def extract_pixel_features(image: np.ndarray, cfg: Dict) -> np.ndarray:
    """
    Extract per-pixel features.  Feature set controlled by cfg["features"].
    MUST match the trainer — the config stored inside the model file is used
    at prediction time to guarantee consistency.
    """
    feat_cfg = cfg["features"]
    scales = feat_cfg["gaussian_scales"]
    entropy_radii = feat_cfg["entropy_radii"]

    img_f = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else \
            image.astype(np.float32)
    if img_f.max() > 1.0:
        img_f /= img_f.max()

    H, W = img_f.shape[:2]
    feats = []

    feats.append(img_f.reshape(H * W, 3))

    hsv = color.rgb2hsv(img_f)
    feats.append(hsv.reshape(H * W, 3))

    lab = color.rgb2lab(img_f)
    lab_n = lab.copy()
    lab_n[:, :, 0] /= 100.0
    lab_n[:, :, 1] = (lab_n[:, :, 1] + 128) / 256.0
    lab_n[:, :, 2] = (lab_n[:, :, 2] + 128) / 256.0
    feats.append(lab_n.reshape(H * W, 3))

    for sigma in scales:
        smoothed = np.stack([filters.gaussian(img_f[:, :, c], sigma=sigma)
                             for c in range(3)], axis=-1)
        feats.append(smoothed.reshape(H * W, 3))
        for c in range(3):
            mu = filters.gaussian(img_f[:, :, c], sigma=sigma)
            mu2 = filters.gaussian(img_f[:, :, c] ** 2, sigma=sigma)
            std = np.sqrt(np.maximum(mu2 - mu ** 2, 0))
            feats.append(std.reshape(H * W, 1))

    gray = color.rgb2gray(img_f)
    feats.append(filters.sobel(gray).reshape(H * W, 1))

    if feat_cfg["include_entropy"]:
        gray_u8 = (gray * 255).astype(np.uint8)
        for r in entropy_radii:
            ent = rank_entropy(gray_u8, disk(r)).astype(np.float32)
            if ent.max() > 0:
                ent /= ent.max()
            feats.append(ent.reshape(H * W, 1))

    if feat_cfg["include_channel_ratios"]:
        r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
        eps = 1e-6
        total = r + g + b + eps
        feats.append((r / total).reshape(H * W, 1))
        feats.append((g / total).reshape(H * W, 1))
        feats.append((b / total).reshape(H * W, 1))
        feats.append((r / (g + eps)).reshape(H * W, 1))
        feats.append(((2 * g - r - b) / total).reshape(H * W, 1))

    feats.append(gray.reshape(H * W, 1))

    return np.hstack(feats).astype(np.float32)


# =============================================================================
# PREDICTION
# =============================================================================

def predict_labels(image: np.ndarray,
                   clf: RandomForestClassifier,
                   cfg: Dict,
                   batch_size: int = 100_000,
                   verbose: bool = True) -> np.ndarray:
    """Run pixel-wise classification, return (H,W) label array."""
    H, W = image.shape[:2]
    if verbose:
        print("    Extracting features...")
    features = extract_pixel_features(image, cfg)

    n = features.shape[0]
    preds = np.zeros(n, dtype=np.int32)

    if verbose:
        print(f"    Classifying {n:,} pixels...")
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        preds[start:end] = clf.predict(features[start:end])

    return preds.reshape(H, W)


# =============================================================================
# POST-PROCESSING
# =============================================================================

def postprocess_mask(labels: np.ndarray,
                     cfg: Dict,
                     verbose: bool = True) -> np.ndarray:
    """
    Clean predicted mask.  All thresholds come from cfg["postprocessing"].
    """
    pp = cfg["postprocessing"]
    smooth_radius = pp["smooth_radius"]
    min_obj = pp["min_object_size"]
    min_hole = pp["min_hole_size"]
    close_r = pp["closing_radius"]

    if verbose:
        print("    Post-processing...")

    cleaned = labels.copy()

    def local_mode(values):
        counts = np.bincount(values.astype(int), minlength=3)
        return np.argmax(counts)

    # Mode filter
    if smooth_radius > 0:
        cleaned = generic_filter(
            cleaned.astype(np.float64), local_mode,
            size=2 * smooth_radius + 1
        ).astype(np.int32)

    # Per-class morphological cleanup
    mask_colors = build_mask_colors(cfg)
    for name in CLASS_NAMES:
        lbl = mask_colors[name]["label"]
        binary = cleaned == lbl

        if min_obj > 0:
            binary = remove_small_objects(binary, max_size=min_obj)
        if min_hole > 0:
            binary = remove_small_holes(binary, max_size=min_hole)
        if close_r > 0:
            binary = closing(binary, disk(close_r))

        cleaned[binary] = lbl

    # Light final smoothing to resolve overlaps from closing
    cleaned = generic_filter(
        cleaned.astype(np.float64), local_mode, size=3
    ).astype(np.int32)

    if verbose:
        for name in CLASS_NAMES:
            lbl = mask_colors[name]["label"]
            pct = np.sum(cleaned == lbl) / cleaned.size * 100
            print(f"      {name:12s}: {pct:5.1f}%")

    return cleaned


# =============================================================================
# OUTPUT
# =============================================================================

def labels_to_rgb(labels: np.ndarray, cfg: Dict) -> np.ndarray:
    mask_colors = build_mask_colors(cfg)
    H, W = labels.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for name in CLASS_NAMES:
        info = mask_colors[name]
        rgb[labels == info["label"]] = info["rgb"]
    return rgb


def save_comparison(image: np.ndarray, labels: np.ndarray, cfg: Dict, save_path: str):
    """Save side-by-side original + mask plot."""
    dpi = cfg["output"]["dpi"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(labels_to_rgb(labels, cfg))
    axes[1].set_title("Segmented Mask")
    axes[1].axis("off")

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=n) for n, c in
              [("Void", "blue"), ("Functional", "orange"), ("Inert", "green")]]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# =============================================================================
# DIALOG HELPERS
# =============================================================================

def dialog_select_model() -> Optional[str]:
    if not HAS_TK:
        print("ERROR: tkinter not available.")
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    messagebox.showinfo("Granule Segmenter",
                        "Select the trained model file (.joblib)\n"
                        "produced by granule_trainer.py.")

    path = filedialog.askopenfilename(
        title="Select trained model",
        filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
    )
    root.destroy()
    return path if path else None


def dialog_select_images() -> List[str]:
    if not HAS_TK:
        return []
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    messagebox.showinfo("Granule Segmenter",
                        "Now select one or more raw images to segment.")

    paths = filedialog.askopenfilenames(
        title="Select images to segment",
        filetypes=[
            ("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return list(paths)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── Explicit settings (edit these directly) ──────────────────────────
    CONFIG_PATH   = Path("segmentation_config.json")  # postprocessing params
    MODEL_PATH    = r"Trained_Models\Model1\granule_model.joblib"   # None = dialog; or set e.g. Path("granule_model.joblib")
    IMAGE_PATHS   = None   # None = dialog; or list of paths e.g. [Path("img.tiff")]
    OUTPUT_DIR    = None   # None = save next to each image; or set a directory
    OUTPUT_SUFFIX = "_2"   # None = read from config; or override e.g. "_segmented"
    # ─────────────────────────────────────────────────────────────────────

    print("=" * 60)
    print("  Granule Segmentation — Image Segmenter")
    print("=" * 60)

    # Load local config (for postprocessing overrides)
    local_cfg = load_config(CONFIG_PATH)

    # ── Load model ──
    model_path = MODEL_PATH or dialog_select_model()
    if not model_path:
        print("No model selected. Exiting.")
        return

    print(f"\nLoading model: {model_path}")
    data = joblib.load(str(model_path))
    clf = data["clf"]
    info = data.get("info", {})

    # The config that was used during training (controls feature extraction)
    # We MUST use this for features so they match the trained model.
    train_cfg = info.get("config_used", DEFAULT_CONFIG)

    # Override postprocessing + output from local config (user can tune these freely)
    train_cfg["postprocessing"] = local_cfg["postprocessing"]
    train_cfg["output"] = local_cfg["output"]
    train_cfg["mask_colors"] = local_cfg["mask_colors"]

    cfg = train_cfg  # combined config

    print(f"  Features:     {info.get('n_features', '?')}")
    print(f"  CV accuracy:  {info.get('cv_accuracy_mean', '?')}")
    if "training_pairs" in info:
        print(f"  Trained on    {len(info['training_pairs'])} image pair(s)")

    pp = cfg["postprocessing"]
    print(f"\n  Postprocessing settings:")
    print(f"    smooth_radius:   {pp['smooth_radius']}")
    print(f"    min_object_size: {pp['min_object_size']}")
    print(f"    min_hole_size:   {pp['min_hole_size']}")
    print(f"    closing_radius:  {pp['closing_radius']}")

    # ── Select images ──
    if IMAGE_PATHS is not None:
        image_paths = [str(p) for p in IMAGE_PATHS]
    else:
        image_paths = dialog_select_images()

    if not image_paths:
        print("No images selected. Exiting.")
        return

    suffix = OUTPUT_SUFFIX or cfg["output"]["output_suffix"]
    save_plots = cfg["output"]["save_comparison_plots"]

    print(f"\nSegmenting {len(image_paths)} image(s)...\n")

    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(image_paths):
        img_path = Path(img_path)
        tag = f"[{idx+1}/{len(image_paths)}]"
        print(f"{tag} {img_path.name}")

        image = load_image(str(img_path))
        print(f"    Size: {image.shape[1]}x{image.shape[0]}")

        raw_labels = predict_labels(image, clf, cfg)
        labels = postprocess_mask(raw_labels, cfg)

        # Output paths
        dest = out_dir or img_path.parent
        stem = img_path.stem
        if stem.lower().endswith("_raw"):
            stem = stem[:-4]

        mask_path = dest / f"{stem}{suffix}.png"
        rgb_mask = labels_to_rgb(labels, cfg)
        io.imsave(str(mask_path), rgb_mask)
        print(f"    Mask -> {mask_path}")

        if save_plots:
            plot_path = dest / f"{stem}{suffix}_comparison.png"
            save_comparison(image, labels, cfg, str(plot_path))
            print(f"    Plot -> {plot_path}")

    print(f"\nDone — {len(image_paths)} image(s) segmented.")


if __name__ == "__main__":
    main()
