"""
Granule Segmentation — Model Trainer
======================================

Opens a dialog to select one or more image pairs (*_raw / *_mask),
extracts pixel features, trains a Random Forest classifier, and
saves the model for use with granule_segmenter.py.

Reads tunable parameters from segmentation_config.json.

File naming convention:
    sample001_raw.tiff   +   sample001_mask.png
    experiment_raw.png   +   experiment_mask.png

Requirements:
    pip install numpy scikit-learn scikit-image tifffile matplotlib joblib

Usage:
    python granule_trainer.py
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import sys
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from datetime import datetime

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
from skimage.morphology import disk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG LOADER
# =============================================================================

DEFAULT_CONFIG = {
    "mask_colors": {
        "void":       [0, 0, 255],
        "functional": [255, 165, 0],
        "inert":      [0, 255, 0],
    },
    "training": {
        "n_samples_per_class_per_image": 30000,
        "n_trees": 200,
        "max_depth": 28,
        "min_samples_leaf": 10,
        "class_weight": "balanced",
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
    cfg = DEFAULT_CONFIG.copy()

    if config_path.exists():
        print(f"Loading config: {config_path}")
        with open(config_path) as f:
            user_cfg = json.load(f)

        # Merge user values into defaults (one level deep)
        for section, values in user_cfg.items():
            if isinstance(values, dict) and section in cfg:
                for k, v in values.items():
                    if k != "comment":
                        cfg[section][k] = v
            elif section in cfg:
                cfg[section] = values
    else:
        print(f"Config not found at {config_path}, using defaults.")

    return cfg


def build_mask_colors(cfg: Dict) -> Dict:
    """Build MASK_COLORS dict from config."""
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
    """Load image from TIFF or common formats, return (H,W,3) uint8 RGB."""
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
# FEATURE EXTRACTION
# =============================================================================

def get_feature_names(cfg: Dict) -> List[str]:
    feat_cfg = cfg["features"]
    scales = feat_cfg["gaussian_scales"]
    entropy_radii = feat_cfg["entropy_radii"]

    names = ["R", "G", "B", "H", "S", "V", "L", "a*", "b*"]
    for s in scales:
        names += [f"R_s{s}", f"G_s{s}", f"B_s{s}"]
        names += [f"Rstd_s{s}", f"Gstd_s{s}", f"Bstd_s{s}"]
    names += ["gradient_mag"]
    if feat_cfg["include_entropy"]:
        for r in entropy_radii:
            names += [f"entropy_s{r}"]
    if feat_cfg["include_channel_ratios"]:
        names += ["r_norm", "g_norm", "b_norm", "R/G_ratio", "green_excess"]
    names += ["intensity"]
    return names


def extract_pixel_features(image: np.ndarray, cfg: Dict) -> np.ndarray:
    """
    Extract per-pixel features across multiple color spaces and scales.
    Feature set is controlled by cfg["features"].
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

    # Raw color spaces
    feats.append(img_f.reshape(H * W, 3))

    hsv = color.rgb2hsv(img_f)
    feats.append(hsv.reshape(H * W, 3))

    lab = color.rgb2lab(img_f)
    lab_n = lab.copy()
    lab_n[:, :, 0] /= 100.0
    lab_n[:, :, 1] = (lab_n[:, :, 1] + 128) / 256.0
    lab_n[:, :, 2] = (lab_n[:, :, 2] + 128) / 256.0
    feats.append(lab_n.reshape(H * W, 3))

    # Multi-scale smoothed + local std
    for sigma in scales:
        smoothed = np.stack([filters.gaussian(img_f[:, :, c], sigma=sigma)
                             for c in range(3)], axis=-1)
        feats.append(smoothed.reshape(H * W, 3))
        for c in range(3):
            mu = filters.gaussian(img_f[:, :, c], sigma=sigma)
            mu2 = filters.gaussian(img_f[:, :, c] ** 2, sigma=sigma)
            std = np.sqrt(np.maximum(mu2 - mu ** 2, 0))
            feats.append(std.reshape(H * W, 1))

    # Gradient magnitude
    gray = color.rgb2gray(img_f)
    feats.append(filters.sobel(gray).reshape(H * W, 1))

    # Entropy
    if feat_cfg["include_entropy"]:
        gray_u8 = (gray * 255).astype(np.uint8)
        for r in entropy_radii:
            ent = rank_entropy(gray_u8, disk(r)).astype(np.float32)
            if ent.max() > 0:
                ent /= ent.max()
            feats.append(ent.reshape(H * W, 1))

    # Channel ratios
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
# MASK PARSING
# =============================================================================

def parse_mask_to_labels(mask: np.ndarray, mask_colors: Dict) -> np.ndarray:
    """Convert RGB mask → label array (nearest colour assignment)."""
    pixels = mask.reshape(-1, 3).astype(np.float32)
    ref = np.array([mask_colors[n]["rgb"] for n in CLASS_NAMES], dtype=np.float32)
    dists = np.sqrt(np.sum((pixels[:, None, :] - ref[None, :, :]) ** 2, axis=2))
    return np.argmin(dists, axis=1).reshape(mask.shape[:2])


# =============================================================================
# FILE PAIR DISCOVERY
# =============================================================================

IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}


def find_pair(path: Path) -> Optional[Tuple[Path, Path]]:
    """Given a selected file, find its raw/mask partner by *_raw / *_mask naming."""
    stem = path.stem.lower()
    parent = path.parent

    if stem.endswith("_raw"):
        prefix = path.stem[:-4]
        for ext in IMAGE_EXTS:
            candidate = parent / f"{prefix}_mask{ext}"
            if candidate.exists():
                return (path, candidate)
    elif stem.endswith("_mask"):
        prefix = path.stem[:-5]
        for ext in IMAGE_EXTS:
            candidate = parent / f"{prefix}_raw{ext}"
            if candidate.exists():
                return (candidate, path)
    return None


def dialog_select_pairs() -> List[Tuple[Path, Path]]:
    """Open a file dialog for the user to select raw and/or mask files."""
    if not HAS_TK:
        print("ERROR: tkinter not available.")
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    messagebox.showinfo(
        "Granule Trainer",
        "Select one or more *_raw or *_mask image files.\n\n"
        "The script will automatically find the matching partner\n"
        "(e.g. selecting sample_raw.tiff finds sample_mask.png)."
    )

    filepaths = filedialog.askopenfilenames(
        title="Select raw / mask image files",
        filetypes=[
            ("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if not filepaths:
        return []

    pairs = {}
    missing = []
    for fp in filepaths:
        p = Path(fp)
        result = find_pair(p)
        if result is None:
            missing.append(p.name)
            continue
        key = str(result[0])
        if key not in pairs:
            pairs[key] = result

    if missing:
        print(f"WARNING — could not find partners for: {missing}")

    return list(pairs.values())


def dialog_save_model() -> Optional[str]:
    """Ask user where to save the model."""
    if not HAS_TK:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    path = filedialog.asksaveasfilename(
        title="Save trained model as...",
        defaultextension=".joblib",
        filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
        initialfile="granule_model.joblib",
    )
    root.destroy()
    return path if path else None


# =============================================================================
# TRAINING
# =============================================================================

def collect_training_data(pairs: List[Tuple[Path, Path]],
                          cfg: Dict,
                          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and sample balanced pixels from every image pair."""
    mask_colors = build_mask_colors(cfg)
    n_per_class = cfg["training"]["n_samples_per_class_per_image"]
    rng = np.random.RandomState(42)
    all_X, all_y = [], []

    for idx, (raw_path, mask_path) in enumerate(pairs):
        tag = f"[{idx+1}/{len(pairs)}]"
        if verbose:
            print(f"\n{tag} Loading  {raw_path.name}")
        image = load_image(str(raw_path))

        if verbose:
            print(f"{tag} Loading  {mask_path.name}")
        mask = load_image(str(mask_path))

        if image.shape[:2] != mask.shape[:2]:
            from skimage.transform import resize
            if verbose:
                print(f"{tag} Resizing mask {mask.shape[:2]} -> {image.shape[:2]}")
            mask = resize(mask, image.shape[:2] + (3,), order=0,
                          preserve_range=True).astype(np.uint8)

        if verbose:
            print(f"{tag} Extracting features ({image.shape[1]}x{image.shape[0]})...")
        features = extract_pixel_features(image, cfg)
        labels = parse_mask_to_labels(mask, mask_colors).ravel()

        if verbose:
            for name in CLASS_NAMES:
                lbl = mask_colors[name]["label"]
                pct = np.sum(labels == lbl) / labels.size * 100
                print(f"        {name:12s}: {pct:5.1f}%")

        for cls in range(len(CLASS_NAMES)):
            idx_cls = np.where(labels == cls)[0]
            n_take = min(n_per_class, len(idx_cls))
            chosen = rng.choice(idx_cls, size=n_take, replace=False)
            all_X.append(features[chosen])
            all_y.append(labels[chosen])

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    shuffle = rng.permutation(len(y))
    return X[shuffle], y[shuffle]


def train_model(X: np.ndarray, y: np.ndarray,
                cfg: Dict,
                verbose: bool = True) -> Tuple[RandomForestClassifier, Dict]:
    """Train and cross-validate a Random Forest classifier."""
    tcfg = cfg["training"]
    n_trees = tcfg["n_trees"]
    max_depth = tcfg["max_depth"]

    if verbose:
        print(f"\nTraining Random Forest  ({n_trees} trees, max_depth={max_depth})")
        print(f"  Training samples: {len(y):,}  "
              f"({', '.join(f'{CLASS_NAMES[c]}={np.sum(y==c):,}' for c in range(len(CLASS_NAMES)))})")

    clf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        min_samples_leaf=tcfg["min_samples_leaf"],
        max_features="sqrt",
        class_weight=tcfg["class_weight"],
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X, y)

    if verbose:
        print("  Cross-validating (3-fold)...")
    cv = cross_val_score(clf, X, y, cv=3, scoring="accuracy")

    info = {
        "created": datetime.now().isoformat(),
        "n_features": int(X.shape[1]),
        "n_training_samples": int(len(y)),
        "config_used": cfg,
        "cv_accuracy_mean": float(np.mean(cv)),
        "cv_accuracy_std": float(np.std(cv)),
        "class_names": CLASS_NAMES,
        "feature_names": get_feature_names(cfg),
    }

    if verbose:
        print(f"  CV accuracy: {info['cv_accuracy_mean']:.4f} "
              f"+/- {info['cv_accuracy_std']:.4f}")

    return clf, info


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_feature_importance(clf: RandomForestClassifier, cfg: Dict, save_path: str):
    names = get_feature_names(cfg)
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(order)), imp[order][::-1], color="steelblue")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Feature importance plot -> {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── Explicit settings (edit these directly) ──────────────────────────
    CONFIG_PATH = Path("segmentation_config.json")   # path to config JSON
    MODEL_OUTPUT = None  # None = dialog prompt; or set e.g. Path("granule_model.joblib")
    # ─────────────────────────────────────────────────────────────────────

    print("=" * 60)
    print("  Granule Segmentation — Model Trainer")
    print("=" * 60)

    # Load config
    cfg = load_config(CONFIG_PATH)

    # Select image pairs
    pairs = dialog_select_pairs()
    if not pairs:
        print("No valid image pairs found. Exiting.")
        return

    print(f"\nFound {len(pairs)} image pair(s):")
    for raw_p, mask_p in pairs:
        print(f"  raw:  {raw_p.name}")
        print(f"  mask: {mask_p.name}")

    # Extract features and train
    X, y = collect_training_data(pairs, cfg)
    clf, info = train_model(X, y, cfg)

    info["training_pairs"] = [(str(r), str(m)) for r, m in pairs]

    # Save model
    if MODEL_OUTPUT is not None:
        model_path = Path(MODEL_OUTPUT)
    else:
        model_path_str = dialog_save_model()
        if not model_path_str:
            model_path = pairs[0][0].parent / "granule_model.joblib"
            print(f"No save location chosen, defaulting to: {model_path}")
        else:
            model_path = Path(model_path_str)

    joblib.dump({"clf": clf, "info": info}, str(model_path))
    print(f"\nModel saved -> {model_path}")

    info_path = model_path.with_suffix(".json")
    with open(info_path, "w") as f:
        # Strip non-serialisable bits from config for JSON dump
        json.dump(info, f, indent=2, default=str)
    print(f"Model info  -> {info_path}")

    feat_path = model_path.with_name(model_path.stem + "_features.png")
    plot_feature_importance(clf, cfg, str(feat_path))

    print("\nDone. Use granule_segmenter.py to segment new images with this model.")


if __name__ == "__main__":
    main()
