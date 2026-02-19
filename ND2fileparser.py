"""
ND2 Microscopy Processor & Mask Generator
==========================================

PySide6 application for:
  1. Parsing .nd2 files with dimension-range selection (T, Z, C, M, L)
  2. Memory-aware loading (hyperstack vs direct export)
  3. Post-processing pipeline (3D CLAHE, color deconvolution, hole closing,
     blob detection, median filtering, thresholding, 3D LBP)
  4. Multi-class mask generation for training segmentation models

Requirements:
    pip install PySide6 nd2 numpy scipy scikit-image psutil matplotlib tifffile

Usage:
    python nd2_processor.py
"""

import sys
import os
import json
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QPushButton, QFileDialog, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QProgressBar, QStatusBar,
    QMessageBox, QSlider, QSplitter, QScrollArea, QFrame, QDialog,
    QDialogButtonBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QRadioButton, QButtonGroup, QToolButton,
    QTextEdit
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QSize
from PySide6.QtGui import QFont, QColor, QPixmap, QImage, QPainter

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

# Optional imports with graceful fallback
try:
    import nd2
    HAS_ND2 = True
except ImportError:
    HAS_ND2 = False
    print("WARNING: 'nd2' package not found. Install with: pip install nd2")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from skimage import (
        exposure, filters, morphology, feature, measure, segmentation, color
    )
    from skimage.filters import threshold_otsu, threshold_li, threshold_yen
    from skimage.morphology import (
        ball, disk, remove_small_holes, remove_small_objects,
        binary_closing, binary_opening, binary_dilation, binary_erosion
    )
    from skimage.feature import blob_log, blob_dog
    from skimage.restoration import denoise_bilateral
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("WARNING: scikit-image not found. Install with: pip install scikit-image")

try:
    from scipy import ndimage as ndi
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import tifffile
    HAS_TIFF = True
except ImportError:
    HAS_TIFF = False


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class ND2Metadata:
    """Metadata extracted from an .nd2 file."""
    filepath: str = ""
    n_timepoints: int = 1
    n_zslices: int = 1
    n_channels: int = 1
    n_multipoints: int = 1
    n_large_images: int = 1  # stitched tile count
    height: int = 0
    width: int = 0
    dtype: np.dtype = np.dtype("uint16")
    pixel_size_um: float = 1.0
    z_step_um: float = 1.0
    channel_names: List[str] = field(default_factory=list)
    voxel_size_um: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    @property
    def bytes_per_pixel(self) -> int:
        return self.dtype.itemsize

    @property
    def single_frame_bytes(self) -> int:
        return self.height * self.width * self.bytes_per_pixel

    def estimate_bytes(self, t_range, z_range, c_range, m_range, l_range) -> int:
        nt = t_range[1] - t_range[0] + 1
        nz = z_range[1] - z_range[0] + 1
        nc = c_range[1] - c_range[0] + 1
        nm = m_range[1] - m_range[0] + 1
        nl = l_range[1] - l_range[0] + 1
        return nt * nz * nc * nm * nl * self.single_frame_bytes


@dataclass
class DimensionRange:
    """A user-selected range for one dimension."""
    name: str
    total: int
    start: int = 0
    stop: int = 0

    @property
    def count(self) -> int:
        return self.stop - self.start + 1


class LoadMode(Enum):
    HYPERSTACK = "hyperstack"
    EXPORT = "export"


@dataclass
class MaskClass:
    """One semantic class in the mask."""
    name: str
    label_value: int
    color: Tuple[int, int, int]
    channel_index: Optional[int] = None  # which C channel drives this


# =============================================================================
# ND2 FILE READER (THREAD-SAFE WORKER)
# =============================================================================

class ND2Reader:
    """Wrapper around the nd2 library."""

    @staticmethod
    def read_metadata(filepath: str) -> ND2Metadata:
        if not HAS_ND2:
            raise ImportError("nd2 package required")

        meta = ND2Metadata(filepath=filepath)

        with nd2.ND2File(filepath) as f:
            meta.dtype = f.dtype
            sizes = f.sizes  # dict like {'T':10, 'Z':20, 'C':3, 'Y':512, 'X':512}

            meta.n_timepoints = sizes.get("T", 1)
            meta.n_zslices = sizes.get("Z", 1)
            meta.n_channels = sizes.get("C", 1)
            meta.n_multipoints = sizes.get("P", sizes.get("M", 1))
            meta.n_large_images = sizes.get("L", sizes.get("S", 1))
            meta.height = sizes.get("Y", 0)
            meta.width = sizes.get("X", 0)

            # Voxel calibration
            try:
                vox = f.voxel_size()
                meta.pixel_size_um = vox.x if hasattr(vox, "x") else 1.0
                meta.z_step_um = vox.z if hasattr(vox, "z") else 1.0
                meta.voxel_size_um = (
                    meta.z_step_um,
                    meta.pixel_size_um,
                    meta.pixel_size_um,
                )
            except Exception:
                pass

            # Channel names
            try:
                meta.channel_names = [
                    ch.channel.name for ch in f.metadata.channels
                ]
            except Exception:
                meta.channel_names = [f"Ch{i}" for i in range(meta.n_channels)]

        return meta

    @staticmethod
    def load_stack(filepath: str, t_range, z_range, c_range,
                   m_range=None, l_range=None,
                   callback=None) -> np.ndarray:
        """
        Load a sub-volume from the nd2 file.
        Returns array shaped (T, Z, C, Y, X).
        """
        if not HAS_ND2:
            raise ImportError("nd2 package required")

        with nd2.ND2File(filepath) as f:
            full = f.to_dask()  # lazy dask array
            sizes = f.sizes
            dim_order = list(sizes.keys())

            # Build slicing tuple in the order of dimensions in the file
            slices = {}
            slices["T"] = slice(t_range[0], t_range[1] + 1)
            slices["Z"] = slice(z_range[0], z_range[1] + 1)
            slices["C"] = slice(c_range[0], c_range[1] + 1)
            if m_range:
                for key in ("P", "M"):
                    if key in sizes:
                        slices[key] = slice(m_range[0], m_range[1] + 1)
            if l_range:
                for key in ("L", "S"):
                    if key in sizes:
                        slices[key] = slice(l_range[0], l_range[1] + 1)

            idx = tuple(slices.get(d, slice(None)) for d in dim_order)
            sub = full[idx]

            if callback:
                callback(50)
            data = np.asarray(sub)
            if callback:
                callback(100)

            # Reshape to 5D (T, Z, C, Y, X) by squeezing/expanding as needed
            # The nd2 library returns in the file's dimension order, so we
            # reorder to our canonical form.
            # For simplicity we handle the common cases:
            target_shape = _reshape_to_5d(data, dim_order, sizes, slices)
            return target_shape

    @staticmethod
    def export_frames(filepath: str, output_dir: str, t_range, z_range,
                      c_range, m_range, l_range, callback=None):
        """Export individual frames as TIFF files with metadata identifiers."""
        if not HAS_TIFF:
            raise ImportError("tifffile required for export")

        os.makedirs(output_dir, exist_ok=True)
        meta = ND2Reader.read_metadata(filepath)

        with nd2.ND2File(filepath) as f:
            total = (
                (t_range[1] - t_range[0] + 1)
                * (z_range[1] - z_range[0] + 1)
                * (c_range[1] - c_range[0] + 1)
                * (m_range[1] - m_range[0] + 1)
                * (l_range[1] - l_range[0] + 1)
            )
            count = 0
            data = np.asarray(f.to_dask())
            sizes = f.sizes
            dim_order = list(sizes.keys())

            for t in range(t_range[0], t_range[1] + 1):
                for m in range(m_range[0], m_range[1] + 1):
                    for l_idx in range(l_range[0], l_range[1] + 1):
                        for z in range(z_range[0], z_range[1] + 1):
                            for c in range(c_range[0], c_range[1] + 1):
                                # Build index
                                idx_map = {"T": t, "Z": z, "C": c, "Y": slice(None), "X": slice(None)}
                                for key in ("P", "M"):
                                    if key in sizes:
                                        idx_map[key] = m
                                for key in ("L", "S"):
                                    if key in sizes:
                                        idx_map[key] = l_idx
                                idx = tuple(idx_map.get(d, slice(None)) for d in dim_order)

                                try:
                                    frame = data[idx]
                                    fname = (
                                        f"T{t:04d}_Z{z:04d}_C{c:02d}"
                                        f"_M{m:03d}_L{l_idx:03d}.tif"
                                    )
                                    tifffile.imwrite(
                                        os.path.join(output_dir, fname),
                                        np.asarray(frame),
                                        metadata={
                                            "T": t, "Z": z, "C": c,
                                            "M": m, "L": l_idx,
                                            "pixel_um": meta.pixel_size_um,
                                            "z_step_um": meta.z_step_um,
                                        },
                                    )
                                except Exception as e:
                                    print(f"  skip frame T{t}Z{z}C{c}M{m}L{l_idx}: {e}")

                                count += 1
                                if callback:
                                    callback(int(100 * count / total))


def _reshape_to_5d(data, dim_order, sizes, slices):
    """Best-effort reshape of nd2 sub-array to (T, Z, C, Y, X)."""
    # If data is already 5-D with matching dims, just transpose
    # Otherwise squeeze singletons and expand to 5-D
    target_dims = ["T", "Z", "C", "Y", "X"]
    # Compute expected size per dim
    expected = {}
    for d in dim_order:
        if d in slices:
            s = slices[d]
            if isinstance(s, slice):
                start = s.start or 0
                stop = s.stop or sizes[d]
                expected[d] = stop - start
            else:
                expected[d] = 1
        else:
            expected[d] = sizes.get(d, 1)

    # For canonical 5D, fill missing with 1
    shape_5d = []
    for d in target_dims:
        if d in expected:
            shape_5d.append(expected[d])
        elif d == "Y":
            shape_5d.append(sizes.get("Y", data.shape[-2] if data.ndim >= 2 else 1))
        elif d == "X":
            shape_5d.append(sizes.get("X", data.shape[-1] if data.ndim >= 1 else 1))
        else:
            shape_5d.append(1)

    try:
        return data.reshape(shape_5d)
    except Exception:
        # Fallback: return with singletons expanded
        while data.ndim < 5:
            data = data[np.newaxis, ...]
        return data


# =============================================================================
# BACKGROUND WORKERS
# =============================================================================

class LoadWorker(QThread):
    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, filepath, t_range, z_range, c_range, m_range, l_range, mode):
        super().__init__()
        self.filepath = filepath
        self.t_range = t_range
        self.z_range = z_range
        self.c_range = c_range
        self.m_range = m_range
        self.l_range = l_range
        self.mode = mode
        self.output_dir = ""

    def run(self):
        try:
            if self.mode == LoadMode.HYPERSTACK:
                data = ND2Reader.load_stack(
                    self.filepath, self.t_range, self.z_range, self.c_range,
                    self.m_range, self.l_range,
                    callback=lambda p: self.progress.emit(p),
                )
                self.finished.emit(data)
            else:
                ND2Reader.export_frames(
                    self.filepath, self.output_dir,
                    self.t_range, self.z_range, self.c_range,
                    self.m_range, self.l_range,
                    callback=lambda p: self.progress.emit(p),
                )
                self.finished.emit(None)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class ProcessWorker(QThread):
    """Run a processing pipeline step in background."""
    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs,
                                callback=lambda p: self.progress.emit(p))
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# =============================================================================
# IMAGE PROCESSING PIPELINE
# =============================================================================

class ImageProcessor:
    """Collection of 3D image processing routines for mask creation."""

    # ---- 3D Adaptive Histogram Equalization (CLAHE) ----
    @staticmethod
    def clahe_3d(volume: np.ndarray, kernel_size: int = 64,
                 clip_limit: float = 0.01, callback=None) -> np.ndarray:
        """Apply CLAHE slice-by-slice (Z dimension) then blend with 3D context."""
        if not HAS_SKIMAGE:
            raise ImportError("scikit-image required")
        out = np.zeros_like(volume, dtype=np.float32)
        nz = volume.shape[0]
        for z in range(nz):
            sl = volume[z].astype(np.float32)
            if sl.max() > sl.min():
                sl_norm = (sl - sl.min()) / (sl.max() - sl.min())
            else:
                sl_norm = sl
            out[z] = exposure.equalize_adapthist(
                sl_norm, kernel_size=min(kernel_size, min(sl.shape)),
                clip_limit=clip_limit
            )
            if callback and z % max(1, nz // 20) == 0:
                callback(int(100 * z / nz))
        if callback:
            callback(100)
        return out

    # ---- Color Deconvolution ----
    @staticmethod
    def color_deconvolution(multichannel: np.ndarray,
                            mixing_matrix: Optional[np.ndarray] = None,
                            callback=None) -> np.ndarray:
        """
        Unmix spectral bleed-through between channels.
        multichannel: (C, Z, Y, X) or (C, Y, X)
        Returns same shape with bleed removed.
        """
        n_ch = multichannel.shape[0]
        if mixing_matrix is None:
            # Default: slight bleed between adjacent channels
            mixing_matrix = np.eye(n_ch) + 0.1 * np.ones((n_ch, n_ch))
            np.fill_diagonal(mixing_matrix, 1.0)

        # Invert mixing matrix
        try:
            unmixing = np.linalg.inv(mixing_matrix)
        except np.linalg.LinAlgError:
            unmixing = np.linalg.pinv(mixing_matrix)

        shape_tail = multichannel.shape[1:]
        flat = multichannel.reshape(n_ch, -1).astype(np.float64)
        unmixed = unmixing @ flat
        unmixed = np.clip(unmixed, 0, None)
        result = unmixed.reshape(n_ch, *shape_tail).astype(multichannel.dtype)
        if callback:
            callback(100)
        return result

    # ---- 3D Median Filter ----
    @staticmethod
    def median_filter_3d(volume: np.ndarray, size: int = 3,
                         callback=None) -> np.ndarray:
        if not HAS_SCIPY:
            raise ImportError("scipy required")
        result = ndi.median_filter(volume, size=size)
        if callback:
            callback(100)
        return result

    # ---- Threshold Methods ----
    @staticmethod
    def threshold(volume: np.ndarray, method: str = "otsu",
                  callback=None) -> np.ndarray:
        if not HAS_SKIMAGE:
            raise ImportError("scikit-image required")

        vol_f = volume.astype(np.float64)
        if vol_f.max() > vol_f.min():
            vol_f = (vol_f - vol_f.min()) / (vol_f.max() - vol_f.min())

        methods = {
            "otsu": threshold_otsu,
            "li": threshold_li,
            "yen": threshold_yen,
        }
        func = methods.get(method, threshold_otsu)
        thresh = func(vol_f)
        binary = vol_f > thresh
        if callback:
            callback(100)
        return binary.astype(np.uint8)

    # ---- Close Holes ----
    @staticmethod
    def close_holes(binary: np.ndarray, min_size_um: float = 10.0,
                    max_size_um: float = 200.0,
                    voxel_size: Tuple[float, float, float] = (1, 1, 1),
                    callback=None) -> np.ndarray:
        """Fill holes within a size range (in microns)."""
        if not HAS_SKIMAGE:
            raise ImportError("scikit-image required")
        voxel_vol = np.prod(voxel_size)
        min_voxels = int((4 / 3 * np.pi * (min_size_um / 2) ** 3) / voxel_vol)
        max_voxels = int((4 / 3 * np.pi * (max_size_um / 2) ** 3) / voxel_vol)
        max_voxels = max(max_voxels, min_voxels + 1)

        filled = remove_small_holes(
            binary.astype(bool), area_threshold=max_voxels
        )
        # Re-open holes smaller than min
        inverted = ~filled
        small_holes = remove_small_objects(inverted, min_size=min_voxels)
        result = ~small_holes
        if callback:
            callback(100)
        return result.astype(np.uint8)

    # ---- Blob Detection ----
    @staticmethod
    def detect_blobs_3d(volume: np.ndarray,
                        min_sigma: float = 5, max_sigma: float = 30,
                        threshold_rel: float = 0.1,
                        callback=None) -> np.ndarray:
        """Detect blob-like structures, returns labeled volume."""
        if not HAS_SKIMAGE:
            raise ImportError("scikit-image required")

        vol_f = volume.astype(np.float64)
        if vol_f.max() > vol_f.min():
            vol_f = (vol_f - vol_f.min()) / (vol_f.max() - vol_f.min())

        # 2D blob detection per slice (3D blob_log is very slow for large volumes)
        label_vol = np.zeros_like(volume, dtype=np.int32)
        current_label = 1
        nz = volume.shape[0]

        for z in range(nz):
            blobs = blob_log(
                vol_f[z], min_sigma=min_sigma, max_sigma=max_sigma,
                threshold=threshold_rel, num_sigma=5
            )
            for y, x, sigma in blobs:
                r = int(sigma * np.sqrt(2))
                yy, xx = np.ogrid[
                    max(0, int(y) - r):min(volume.shape[1], int(y) + r + 1),
                    max(0, int(x) - r):min(volume.shape[2], int(x) + r + 1),
                ]
                mask_local = ((yy - int(y)) ** 2 + (xx - int(x)) ** 2) <= r ** 2
                label_vol[z][
                    max(0, int(y) - r):min(volume.shape[1], int(y) + r + 1),
                    max(0, int(x) - r):min(volume.shape[2], int(x) + r + 1),
                ][mask_local] = current_label
                current_label += 1

            if callback and z % max(1, nz // 20) == 0:
                callback(int(100 * z / nz))
        if callback:
            callback(100)
        return label_vol

    # ---- Morphological Closing (structure-aware) ----
    @staticmethod
    def morphological_close_3d(binary: np.ndarray, radius: int = 3,
                                callback=None) -> np.ndarray:
        if not HAS_SKIMAGE:
            raise ImportError("scikit-image required")
        selem = ball(radius)
        result = binary_closing(binary.astype(bool), selem)
        if callback:
            callback(100)
        return result.astype(np.uint8)

    # ---- 3D Local Binary Pattern (edge/texture) ----
    @staticmethod
    def lbp_3d(volume: np.ndarray, radius: int = 1, callback=None) -> np.ndarray:
        """
        Compute 3D LBP-like texture descriptor.
        Uses 6-connected neighborhood in 3D.
        Returns a texture feature volume useful for boundary tracing.
        """
        vol = volume.astype(np.float32)
        nz, ny, nx = vol.shape
        lbp_vol = np.zeros_like(vol, dtype=np.uint8)

        # 6-connected offsets
        offsets = [
            (-radius, 0, 0), (radius, 0, 0),
            (0, -radius, 0), (0, radius, 0),
            (0, 0, -radius), (0, 0, radius),
        ]
        for z in range(radius, nz - radius):
            for bit, (dz, dy, dx) in enumerate(offsets):
                neighbor = vol[z + dz,
                               max(0, 0):ny,
                               max(0, 0):nx]
                center = vol[z, :ny, :nx]
                # Ensure shapes match
                min_y = min(neighbor.shape[0], center.shape[0])
                min_x = min(neighbor.shape[1], center.shape[1])
                shifted = np.zeros_like(center)
                shifted[:min_y, :min_x] = (
                    vol[z + dz,
                        max(0, dy):max(0, dy) + ny,
                        max(0, dx):max(0, dx) + nx][:min_y, :min_x]
                    if (0 <= dy + ny and 0 <= dx + nx) else 0
                )
                lbp_vol[z] |= ((vol[z] >= shifted) << bit).astype(np.uint8)[:ny, :nx]

            if callback and z % max(1, nz // 20) == 0:
                callback(int(100 * z / nz))

        if callback:
            callback(100)
        return lbp_vol

    # ---- Simplified 3D LBP (faster, slice-based + Z neighbors) ----
    @staticmethod
    def lbp_3d_fast(volume: np.ndarray, radius: int = 1,
                     callback=None) -> np.ndarray:
        """Fast 3D LBP using shifted arrays."""
        vol = volume.astype(np.float32)
        pad = np.pad(vol, radius, mode="reflect")
        nz, ny, nx = vol.shape
        result = np.zeros_like(vol, dtype=np.uint8)

        offsets = [
            (-radius, 0, 0), (radius, 0, 0),
            (0, -radius, 0), (0, radius, 0),
            (0, 0, -radius), (0, 0, radius),
        ]
        r = radius
        center = pad[r:r + nz, r:r + ny, r:r + nx]
        for bit, (dz, dy, dx) in enumerate(offsets):
            neighbor = pad[r + dz:r + dz + nz,
                           r + dy:r + dy + ny,
                           r + dx:r + dx + nx]
            result |= ((center >= neighbor).astype(np.uint8) << bit)
        if callback:
            callback(100)
        return result


# =============================================================================
# MASK BUILDER
# =============================================================================

class MaskBuilder:
    """Builds multi-class volumetric masks from processed channel data."""

    def __init__(self, n_classes: int = 4):
        self.classes: List[MaskClass] = [
            MaskClass("Void", 0, (0, 0, 0)),
            MaskClass("Functional", 1, (255, 0, 0)),
            MaskClass("Inert", 2, (0, 0, 255)),
            MaskClass("Cells", 3, (0, 255, 0)),
        ]

    def build_mask(self, binary_volumes: Dict[str, np.ndarray],
                   priority_order: Optional[List[str]] = None) -> np.ndarray:
        """
        Combine per-class binary masks into a single label volume.
        Later classes in priority_order overwrite earlier ones.
        """
        if not binary_volumes:
            return np.zeros((1, 1, 1), dtype=np.uint8)

        ref = list(binary_volumes.values())[0]
        mask = np.zeros(ref.shape, dtype=np.uint8)

        if priority_order is None:
            priority_order = [c.name for c in self.classes if c.label_value > 0]

        for class_name in priority_order:
            mc = next((c for c in self.classes if c.name == class_name), None)
            if mc and class_name in binary_volumes:
                mask[binary_volumes[class_name] > 0] = mc.label_value

        return mask

    def mask_to_rgb(self, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Convert label mask to RGBA for overlay."""
        h, w = mask.shape[-2], mask.shape[-1]
        rgba = np.zeros((*mask.shape[:-2], h, w, 4), dtype=np.uint8)
        for mc in self.classes:
            region = mask == mc.label_value
            rgba[..., 0][region] = mc.color[0]
            rgba[..., 1][region] = mc.color[1]
            rgba[..., 2][region] = mc.color[2]
            rgba[..., 3][region] = int(255 * alpha) if mc.label_value > 0 else 0
        return rgba

    def export_mask(self, mask: np.ndarray, filepath: str, metadata: dict = None):
        """Save mask volume as TIFF."""
        if HAS_TIFF:
            tifffile.imwrite(filepath, mask.astype(np.uint8),
                             metadata=metadata or {})


# =============================================================================
# FILE IMPORT TAB
# =============================================================================

class FileImportTab(QWidget):
    """Tab for opening .nd2 files, selecting dimension ranges, loading/exporting."""

    data_loaded = Signal(object, object)  # (np.ndarray or None, ND2Metadata)

    def __init__(self):
        super().__init__()
        self.metadata: Optional[ND2Metadata] = None
        self.worker: Optional[LoadWorker] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ---- File selection ----
        file_group = QGroupBox("ND2 File")
        fl = QHBoxLayout(file_group)
        self.file_label = QLabel("No file selected")
        self.file_label.setMinimumWidth(400)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        fl.addWidget(self.file_label, 1)
        fl.addWidget(btn_browse)
        layout.addWidget(file_group)

        # ---- Metadata info ----
        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        self.meta_text.setMaximumHeight(100)
        layout.addWidget(self.meta_text)

        # ---- Dimension ranges ----
        dim_group = QGroupBox("Dimension Ranges")
        dim_layout = QGridLayout(dim_group)

        self.dim_spins: Dict[str, Tuple[QSpinBox, QSpinBox]] = {}
        dims = [
            ("T", "Timelapse"),
            ("Z", "Z-Series"),
            ("C", "Channels"),
            ("M", "Multi-Points"),
            ("L", "Large Images"),
        ]
        for row, (key, label) in enumerate(dims):
            dim_layout.addWidget(QLabel(label), row, 0)
            sp_start = QSpinBox()
            sp_stop = QSpinBox()
            sp_start.setMinimum(0)
            sp_stop.setMinimum(0)
            sp_start.setEnabled(False)
            sp_stop.setEnabled(False)
            sp_start.valueChanged.connect(self._update_memory)
            sp_stop.valueChanged.connect(self._update_memory)
            dim_layout.addWidget(QLabel("Start:"), row, 1)
            dim_layout.addWidget(sp_start, row, 2)
            dim_layout.addWidget(QLabel("Stop:"), row, 3)
            dim_layout.addWidget(sp_stop, row, 4)
            self.dim_spins[key] = (sp_start, sp_stop)

        layout.addWidget(dim_group)

        # ---- Memory estimate ----
        mem_group = QGroupBox("Memory Estimate")
        ml = QHBoxLayout(mem_group)
        self.mem_label = QLabel("Select a file first")
        self.mem_fits_label = QLabel("")
        ml.addWidget(self.mem_label, 1)
        ml.addWidget(self.mem_fits_label)
        layout.addWidget(mem_group)

        # ---- Load mode ----
        mode_group = QGroupBox("Load Mode")
        mode_layout = QHBoxLayout(mode_group)
        self.radio_hyper = QRadioButton("Hyperstack (in memory)")
        self.radio_export = QRadioButton("Direct export (individual TIFFs)")
        self.radio_hyper.setChecked(True)
        mode_layout.addWidget(self.radio_hyper)
        mode_layout.addWidget(self.radio_export)
        layout.addWidget(mode_group)

        # ---- Actions ----
        action_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load / Export")
        self.btn_load.setEnabled(False)
        self.btn_load.clicked.connect(self._start_load)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        action_layout.addWidget(self.btn_load)
        action_layout.addWidget(self.progress, 1)
        layout.addLayout(action_layout)

        layout.addStretch()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ND2 File", "", "ND2 Files (*.nd2);;All Files (*)"
        )
        if not path:
            return
        self.file_label.setText(path)
        try:
            self.metadata = ND2Reader.read_metadata(path)
            self._populate_dims()
            self.btn_load.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read metadata:\n{e}")

    def _populate_dims(self):
        m = self.metadata
        totals = {
            "T": m.n_timepoints,
            "Z": m.n_zslices,
            "C": m.n_channels,
            "M": m.n_multipoints,
            "L": m.n_large_images,
        }
        for key, (sp_start, sp_stop) in self.dim_spins.items():
            n = totals.get(key, 1)
            sp_start.setMaximum(max(0, n - 1))
            sp_stop.setMaximum(max(0, n - 1))
            sp_start.setValue(0)
            sp_stop.setValue(max(0, n - 1))
            sp_start.setEnabled(True)
            sp_stop.setEnabled(True)

        ch_names = ", ".join(m.channel_names) if m.channel_names else "N/A"
        self.meta_text.setPlainText(
            f"File: {Path(m.filepath).name}\n"
            f"Dimensions: T={m.n_timepoints} Z={m.n_zslices} C={m.n_channels} "
            f"M={m.n_multipoints} L={m.n_large_images}\n"
            f"Frame: {m.width}×{m.height} ({m.dtype})  |  "
            f"Pixel: {m.pixel_size_um:.4f} µm  Z-step: {m.z_step_um:.4f} µm\n"
            f"Channels: {ch_names}"
        )
        self._update_memory()

    def _get_ranges(self):
        ranges = {}
        for key, (sp_s, sp_e) in self.dim_spins.items():
            ranges[key] = (sp_s.value(), sp_e.value())
        return ranges

    def _update_memory(self):
        if self.metadata is None:
            return
        r = self._get_ranges()
        est = self.metadata.estimate_bytes(r["T"], r["Z"], r["C"], r["M"], r["L"])
        est_with_overhead = int(est * 1.5)

        gb = est / (1024 ** 3)
        gb_oh = est_with_overhead / (1024 ** 3)
        self.mem_label.setText(
            f"Data: {gb:.2f} GB  |  With 50% overhead: {gb_oh:.2f} GB"
        )

        fits = True
        if HAS_PSUTIL:
            avail = psutil.virtual_memory().available
            fits = est_with_overhead < avail
            avail_gb = avail / (1024 ** 3)
            self.mem_fits_label.setText(
                f"{'✓ Fits' if fits else '✗ Does NOT fit'} "
                f"(Available: {avail_gb:.1f} GB)"
            )
            self.mem_fits_label.setStyleSheet(
                "color: green; font-weight: bold;"
                if fits else "color: red; font-weight: bold;"
            )
        else:
            self.mem_fits_label.setText("(psutil not installed – cannot check)")

        if not fits:
            self.radio_export.setChecked(True)

    def _start_load(self):
        if self.metadata is None:
            return
        r = self._get_ranges()
        mode = LoadMode.HYPERSTACK if self.radio_hyper.isChecked() else LoadMode.EXPORT

        self.worker = LoadWorker(
            self.metadata.filepath,
            r["T"], r["Z"], r["C"], r["M"], r["L"],
            mode,
        )
        if mode == LoadMode.EXPORT:
            out_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
            if not out_dir:
                return
            self.worker.output_dir = out_dir

        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_load_done)
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.btn_load.setEnabled(False)
        self.worker.start()

    def _on_load_done(self, data):
        self.btn_load.setEnabled(True)
        self.progress.setValue(100)
        if data is not None:
            QMessageBox.information(
                self, "Loaded",
                f"Hyperstack loaded: shape {data.shape}, dtype {data.dtype}"
            )
        else:
            QMessageBox.information(self, "Exported", "Frames exported successfully.")
        self.data_loaded.emit(data, self.metadata)


# =============================================================================
# POST-PROCESSING TAB
# =============================================================================

class PostProcessTab(QWidget):
    """Tab for image processing pipeline and mask generation."""

    def __init__(self):
        super().__init__()
        self.data: Optional[np.ndarray] = None       # (T, Z, C, Y, X)
        self.metadata: Optional[ND2Metadata] = None
        self.processed: Dict[str, np.ndarray] = {}    # channel-name -> volume
        self.binary_masks: Dict[str, np.ndarray] = {} # class-name -> binary
        self.combined_mask: Optional[np.ndarray] = None
        self.mask_builder = MaskBuilder()
        self._current_t = 0
        self._current_z = 0
        self._current_c = 0
        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # ---- Left: controls ----
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMaximumWidth(420)
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_scroll.setWidget(ctrl_widget)
        main_layout.addWidget(ctrl_scroll)

        # Navigation
        nav_group = QGroupBox("Slice Navigation")
        nav_layout = QGridLayout(nav_group)
        self.spin_t = QSpinBox(); self.spin_t.setPrefix("T: ")
        self.spin_z = QSpinBox(); self.spin_z.setPrefix("Z: ")
        self.spin_c = QSpinBox(); self.spin_c.setPrefix("C: ")
        for i, sp in enumerate([self.spin_t, self.spin_z, self.spin_c]):
            sp.valueChanged.connect(self._update_view)
            nav_layout.addWidget(sp, 0, i)
        ctrl_layout.addWidget(nav_group)

        # ---- Preprocessing ----
        pre_group = QGroupBox("Preprocessing")
        pre_layout = QVBoxLayout(pre_group)

        # CLAHE
        row_clahe = QHBoxLayout()
        self.chk_clahe = QCheckBox("3D CLAHE")
        self.spin_clahe_kernel = QSpinBox()
        self.spin_clahe_kernel.setRange(8, 256)
        self.spin_clahe_kernel.setValue(64)
        self.spin_clahe_kernel.setPrefix("K:")
        self.spin_clahe_clip = QDoubleSpinBox()
        self.spin_clahe_clip.setRange(0.001, 0.1)
        self.spin_clahe_clip.setValue(0.01)
        self.spin_clahe_clip.setSingleStep(0.005)
        self.spin_clahe_clip.setPrefix("Clip:")
        row_clahe.addWidget(self.chk_clahe)
        row_clahe.addWidget(self.spin_clahe_kernel)
        row_clahe.addWidget(self.spin_clahe_clip)
        pre_layout.addLayout(row_clahe)

        # Color deconvolution
        self.chk_deconv = QCheckBox("Color Deconvolution (unmix channel bleed)")
        pre_layout.addWidget(self.chk_deconv)

        # Median filter
        row_med = QHBoxLayout()
        self.chk_median = QCheckBox("3D Median Filter")
        self.spin_median_sz = QSpinBox()
        self.spin_median_sz.setRange(1, 15)
        self.spin_median_sz.setValue(3)
        self.spin_median_sz.setPrefix("Size:")
        row_med.addWidget(self.chk_median)
        row_med.addWidget(self.spin_median_sz)
        pre_layout.addLayout(row_med)

        # 3D LBP
        row_lbp = QHBoxLayout()
        self.chk_lbp = QCheckBox("3D LBP (boundary texture)")
        self.spin_lbp_r = QSpinBox()
        self.spin_lbp_r.setRange(1, 5)
        self.spin_lbp_r.setValue(1)
        self.spin_lbp_r.setPrefix("R:")
        row_lbp.addWidget(self.chk_lbp)
        row_lbp.addWidget(self.spin_lbp_r)
        pre_layout.addLayout(row_lbp)

        btn_preprocess = QPushButton("▶ Run Preprocessing")
        btn_preprocess.clicked.connect(self._run_preprocessing)
        pre_layout.addWidget(btn_preprocess)

        ctrl_layout.addWidget(pre_group)

        # ---- Segmentation ----
        seg_group = QGroupBox("Segmentation & Masking")
        seg_layout = QVBoxLayout(seg_group)

        # Threshold
        row_thresh = QHBoxLayout()
        self.chk_thresh = QCheckBox("Threshold")
        self.combo_thresh = QComboBox()
        self.combo_thresh.addItems(["otsu", "li", "yen"])
        row_thresh.addWidget(self.chk_thresh)
        row_thresh.addWidget(self.combo_thresh)
        seg_layout.addLayout(row_thresh)

        # Hole closing
        row_holes = QHBoxLayout()
        self.chk_holes = QCheckBox("Close Holes")
        self.spin_hole_min = QDoubleSpinBox()
        self.spin_hole_min.setRange(0.1, 500)
        self.spin_hole_min.setValue(10)
        self.spin_hole_min.setSuffix(" µm")
        self.spin_hole_max = QDoubleSpinBox()
        self.spin_hole_max.setRange(1, 2000)
        self.spin_hole_max.setValue(200)
        self.spin_hole_max.setSuffix(" µm")
        row_holes.addWidget(self.chk_holes)
        row_holes.addWidget(QLabel("Min:"))
        row_holes.addWidget(self.spin_hole_min)
        row_holes.addWidget(QLabel("Max:"))
        row_holes.addWidget(self.spin_hole_max)
        seg_layout.addLayout(row_holes)

        # Morphological closing
        row_morph = QHBoxLayout()
        self.chk_morph_close = QCheckBox("Morpho Close")
        self.spin_morph_r = QSpinBox()
        self.spin_morph_r.setRange(1, 20)
        self.spin_morph_r.setValue(3)
        self.spin_morph_r.setPrefix("R:")
        row_morph.addWidget(self.chk_morph_close)
        row_morph.addWidget(self.spin_morph_r)
        seg_layout.addLayout(row_morph)

        # Blob detection
        row_blob = QHBoxLayout()
        self.chk_blob = QCheckBox("Blob Detection")
        self.spin_blob_min = QDoubleSpinBox()
        self.spin_blob_min.setRange(1, 100)
        self.spin_blob_min.setValue(5)
        self.spin_blob_min.setPrefix("σ_min:")
        self.spin_blob_max = QDoubleSpinBox()
        self.spin_blob_max.setRange(5, 200)
        self.spin_blob_max.setValue(30)
        self.spin_blob_max.setPrefix("σ_max:")
        row_blob.addWidget(self.chk_blob)
        row_blob.addWidget(self.spin_blob_min)
        row_blob.addWidget(self.spin_blob_max)
        seg_layout.addLayout(row_blob)

        btn_segment = QPushButton("▶ Run Segmentation")
        btn_segment.clicked.connect(self._run_segmentation)
        seg_layout.addWidget(btn_segment)

        ctrl_layout.addWidget(seg_group)

        # ---- Mask Class Assignment ----
        class_group = QGroupBox("Mask Class ↔ Channel Assignment")
        class_layout = QGridLayout(class_group)
        class_layout.addWidget(QLabel("Class"), 0, 0)
        class_layout.addWidget(QLabel("Channel"), 0, 1)
        class_layout.addWidget(QLabel("Color"), 0, 2)

        self.class_combos: List[Tuple[QLabel, QComboBox, QLabel]] = []
        for i, mc in enumerate(self.mask_builder.classes):
            if mc.label_value == 0:
                continue  # void is automatic
            lbl = QLabel(mc.name)
            combo = QComboBox()
            combo.addItem("(none)")
            clr_lbl = QLabel("■")
            clr_lbl.setStyleSheet(
                f"color: rgb({mc.color[0]},{mc.color[1]},{mc.color[2]}); "
                f"font-size: 18px;"
            )
            row = i
            class_layout.addWidget(lbl, row, 0)
            class_layout.addWidget(combo, row, 1)
            class_layout.addWidget(clr_lbl, row, 2)
            self.class_combos.append((lbl, combo, clr_lbl))

        ctrl_layout.addWidget(class_group)

        # ---- Granule priors ----
        prior_group = QGroupBox("Granule Shape Priors (for guided segmentation)")
        prior_layout = QFormLayout(prior_group)
        self.spin_major = QDoubleSpinBox()
        self.spin_major.setRange(1, 1000)
        self.spin_major.setValue(100)
        self.spin_major.setSuffix(" µm")
        self.spin_minor = QDoubleSpinBox()
        self.spin_minor.setRange(1, 1000)
        self.spin_minor.setValue(75)
        self.spin_minor.setSuffix(" µm")
        self.combo_shape = QComboBox()
        self.combo_shape.addItems(["Ellipsoid", "Sphere", "Cuboid"])
        prior_layout.addRow("Major axis:", self.spin_major)
        prior_layout.addRow("Minor axis:", self.spin_minor)
        prior_layout.addRow("Shape:", self.combo_shape)
        ctrl_layout.addWidget(prior_group)

        # ---- Build & Export Mask ----
        mask_group = QGroupBox("Build & Export Mask")
        mask_layout = QVBoxLayout(mask_group)
        self.btn_build_mask = QPushButton("▶ Build Combined Mask")
        self.btn_build_mask.clicked.connect(self._build_mask)
        self.btn_export_mask = QPushButton("💾 Export Mask as TIFF")
        self.btn_export_mask.clicked.connect(self._export_mask)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.valueChanged.connect(self._update_view)
        row_alpha = QHBoxLayout()
        row_alpha.addWidget(QLabel("Overlay α:"))
        row_alpha.addWidget(self.alpha_slider)
        mask_layout.addLayout(row_alpha)
        mask_layout.addWidget(self.btn_build_mask)
        mask_layout.addWidget(self.btn_export_mask)
        ctrl_layout.addWidget(mask_group)

        # Progress
        self.progress = QProgressBar()
        ctrl_layout.addWidget(self.progress)
        ctrl_layout.addStretch()

        # ---- Right: visualization ----
        viz_layout = QVBoxLayout()
        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavToolbar(self.canvas, self)
        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas, 1)
        main_layout.addLayout(viz_layout, 1)

    # ---- Public: receive data from import tab ----
    @Slot(object, object)
    def set_data(self, data: Optional[np.ndarray], metadata: Optional[ND2Metadata]):
        self.data = data
        self.metadata = metadata
        self.processed.clear()
        self.binary_masks.clear()
        self.combined_mask = None

        if data is not None and metadata is not None:
            shape = data.shape  # (T, Z, C, Y, X)
            self.spin_t.setMaximum(max(0, shape[0] - 1))
            self.spin_z.setMaximum(max(0, shape[1] - 1))
            self.spin_c.setMaximum(max(0, shape[2] - 1))

            # Update class-channel combos
            names = metadata.channel_names or [
                f"Ch{i}" for i in range(shape[2])
            ]
            for _, combo, _ in self.class_combos:
                combo.clear()
                combo.addItem("(none)")
                for ch_name in names:
                    combo.addItem(ch_name)

            self._update_view()

    # ---- Preprocessing ----
    def _run_preprocessing(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Load data first in the Import tab.")
            return

        self.progress.setValue(0)
        t = self.spin_t.value()
        n_c = self.data.shape[2]

        # Color deconvolution across all channels first
        if self.chk_deconv.isChecked() and n_c > 1:
            self.statusBar_msg("Running color deconvolution…")
            multichannel = self.data[t, :, :, :, :].copy()  # (Z, C, Y, X)
            multichannel = multichannel.transpose(1, 0, 2, 3)  # (C, Z, Y, X)
            multichannel = ImageProcessor.color_deconvolution(
                multichannel.astype(np.float32)
            )
            for c in range(n_c):
                key = f"deconv_ch{c}"
                self.processed[key] = multichannel[c]
            self.progress.setValue(20)

        for c in range(n_c):
            vol = self.data[t, :, c, :, :].copy().astype(np.float32)
            label = f"ch{c}"

            # Use deconvolved if available
            if f"deconv_ch{c}" in self.processed:
                vol = self.processed[f"deconv_ch{c}"]

            if self.chk_clahe.isChecked():
                self.statusBar_msg(f"CLAHE on channel {c}…")
                vol = ImageProcessor.clahe_3d(
                    vol,
                    kernel_size=self.spin_clahe_kernel.value(),
                    clip_limit=self.spin_clahe_clip.value(),
                    callback=lambda p: self.progress.setValue(
                        20 + int(60 * (c + p / 100) / n_c)
                    ),
                )

            if self.chk_median.isChecked():
                self.statusBar_msg(f"Median filter on channel {c}…")
                vol = ImageProcessor.median_filter_3d(
                    vol, size=self.spin_median_sz.value()
                )

            if self.chk_lbp.isChecked():
                self.statusBar_msg(f"3D LBP on channel {c}…")
                lbp = ImageProcessor.lbp_3d_fast(
                    vol, radius=self.spin_lbp_r.value()
                )
                self.processed[f"lbp_{label}"] = lbp

            self.processed[label] = vol

        self.progress.setValue(100)
        self.statusBar_msg("Preprocessing complete.")
        self._update_view()

    # ---- Segmentation ----
    def _run_segmentation(self):
        if not self.processed and self.data is None:
            QMessageBox.warning(self, "No Data", "Run preprocessing or load data first.")
            return

        self.progress.setValue(0)
        n_c = self.data.shape[2] if self.data is not None else len(self.processed)
        voxel = self.metadata.voxel_size_um if self.metadata else (1, 1, 1)

        for c in range(self.data.shape[2] if self.data is not None else 0):
            label = f"ch{c}"
            vol = self.processed.get(label)
            if vol is None and self.data is not None:
                vol = self.data[self.spin_t.value(), :, c, :, :].astype(np.float32)
            if vol is None:
                continue

            binary = vol.copy()

            if self.chk_thresh.isChecked():
                self.statusBar_msg(f"Thresholding channel {c}…")
                binary = ImageProcessor.threshold(
                    binary, method=self.combo_thresh.currentText()
                )

            if self.chk_holes.isChecked():
                self.statusBar_msg(f"Closing holes on channel {c}…")
                binary = ImageProcessor.close_holes(
                    binary,
                    min_size_um=self.spin_hole_min.value(),
                    max_size_um=self.spin_hole_max.value(),
                    voxel_size=voxel,
                )

            if self.chk_morph_close.isChecked():
                self.statusBar_msg(f"Morphological close on channel {c}…")
                binary = ImageProcessor.morphological_close_3d(
                    binary, radius=self.spin_morph_r.value()
                )

            self.binary_masks[f"ch{c}"] = binary.astype(np.uint8)
            self.progress.setValue(int(100 * (c + 1) / max(1, n_c)))

        # Blob detection (on first channel or LBP)
        if self.chk_blob.isChecked():
            vol = self.processed.get("ch0")
            if vol is None and self.data is not None:
                vol = self.data[self.spin_t.value(), :, 0, :, :].astype(np.float32)
            if vol is not None:
                self.statusBar_msg("Blob detection…")
                blobs = ImageProcessor.detect_blobs_3d(
                    vol,
                    min_sigma=self.spin_blob_min.value(),
                    max_sigma=self.spin_blob_max.value(),
                    callback=lambda p: self.progress.setValue(p),
                )
                self.processed["blobs"] = blobs

        self.progress.setValue(100)
        self.statusBar_msg("Segmentation complete.")
        self._update_view()

    # ---- Mask building ----
    def _build_mask(self):
        class_binaries: Dict[str, np.ndarray] = {}

        for (lbl, combo, _), mc in zip(
            self.class_combos,
            [c for c in self.mask_builder.classes if c.label_value > 0],
        ):
            ch_idx_text = combo.currentText()
            if ch_idx_text == "(none)":
                continue
            # Find channel index
            if self.metadata and self.metadata.channel_names:
                try:
                    ch_idx = self.metadata.channel_names.index(ch_idx_text)
                except ValueError:
                    continue
            else:
                try:
                    ch_idx = int(ch_idx_text.replace("Ch", ""))
                except ValueError:
                    continue

            key = f"ch{ch_idx}"
            if key in self.binary_masks:
                class_binaries[mc.name] = self.binary_masks[key]
            elif key in self.processed:
                # Auto-threshold if not segmented yet
                vol = self.processed[key]
                binary = ImageProcessor.threshold(vol, "otsu")
                class_binaries[mc.name] = binary

        if not class_binaries:
            QMessageBox.warning(
                self, "No Classes",
                "Assign channels to mask classes and run segmentation first."
            )
            return

        self.combined_mask = self.mask_builder.build_mask(class_binaries)
        self.statusBar_msg(
            f"Mask built: shape {self.combined_mask.shape}, "
            f"classes: {np.unique(self.combined_mask).tolist()}"
        )
        self._update_view()

    def _export_mask(self):
        if self.combined_mask is None:
            QMessageBox.warning(self, "No Mask", "Build a mask first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Mask", "mask.tif", "TIFF (*.tif *.tiff)"
        )
        if path:
            meta = {}
            if self.metadata:
                meta = {
                    "pixel_size_um": self.metadata.pixel_size_um,
                    "z_step_um": self.metadata.z_step_um,
                    "classes": {
                        mc.name: mc.label_value
                        for mc in self.mask_builder.classes
                    },
                }
            self.mask_builder.export_mask(self.combined_mask, path, meta)
            self.statusBar_msg(f"Mask exported to {path}")

    # ---- Visualization ----
    def _update_view(self):
        self.fig.clear()
        t = self.spin_t.value()
        z = self.spin_z.value()
        c = self.spin_c.value()

        has_processed = bool(self.processed)
        has_mask = self.combined_mask is not None

        n_panels = 1 + int(has_processed) + int(has_mask)
        axes = self.fig.subplots(1, n_panels, squeeze=False)[0]
        panel = 0

        # Raw image
        if self.data is not None:
            raw = self.data[t, z, c, :, :]
            ax = axes[panel]
            ax.imshow(raw, cmap="gray", origin="lower")
            ch_name = ""
            if self.metadata and c < len(self.metadata.channel_names):
                ch_name = f" ({self.metadata.channel_names[c]})"
            ax.set_title(f"Raw T{t} Z{z} C{c}{ch_name}", fontsize=9)
            ax.axis("off")
            panel += 1

        # Processed
        if has_processed:
            key = f"ch{c}"
            lbp_key = f"lbp_ch{c}"
            ax = axes[panel]
            if key in self.processed:
                vol = self.processed[key]
                if z < vol.shape[0]:
                    ax.imshow(vol[z], cmap="gray", origin="lower")
            if lbp_key in self.processed:
                lbp = self.processed[lbp_key]
                if z < lbp.shape[0]:
                    ax.imshow(lbp[z], cmap="hot", alpha=0.3, origin="lower")
            ax.set_title("Processed" + (" + LBP" if lbp_key in self.processed else ""),
                         fontsize=9)
            ax.axis("off")
            panel += 1

        # Mask overlay
        if has_mask:
            ax = axes[panel]
            alpha = self.alpha_slider.value() / 100.0
            if self.data is not None:
                raw = self.data[t, z, c, :, :]
                raw_norm = raw.astype(np.float32)
                if raw_norm.max() > raw_norm.min():
                    raw_norm = (raw_norm - raw_norm.min()) / (raw_norm.max() - raw_norm.min())
                ax.imshow(raw_norm, cmap="gray", origin="lower")

            if z < self.combined_mask.shape[0]:
                mask_slice = self.combined_mask[z]
                rgba = self.mask_builder.mask_to_rgb(
                    mask_slice[np.newaxis], alpha=alpha
                )[0]
                ax.imshow(rgba, origin="lower")

            # Legend
            for mc in self.mask_builder.classes:
                if mc.label_value > 0:
                    ax.plot([], [], "s",
                            color=np.array(mc.color) / 255.0,
                            label=mc.name, markersize=8)
            ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
            ax.set_title("Mask Overlay", fontsize=9)
            ax.axis("off")

        self.fig.tight_layout()
        self.canvas.draw()

    def statusBar_msg(self, msg):
        """Try to set status bar message on parent window."""
        parent = self.window()
        if hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(msg, 5000)


# =============================================================================
# MAIN WINDOW
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ND2 Microscopy Processor & Mask Generator")
        self.setMinimumSize(1200, 800)

        # Status bar
        self.statusBar().showMessage("Ready")

        # Tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.import_tab = FileImportTab()
        self.process_tab = PostProcessTab()

        self.tabs.addTab(self.import_tab, "📂 File Import")
        self.tabs.addTab(self.process_tab, "🔬 Post-Processing & Masks")

        # Connect import → process
        self.import_tab.data_loaded.connect(self._on_data_loaded)

    @Slot(object, object)
    def _on_data_loaded(self, data, metadata):
        self.process_tab.set_data(data, metadata)
        if data is not None:
            self.tabs.setCurrentWidget(self.process_tab)
            self.statusBar().showMessage(
                f"Data loaded: {data.shape} — switch to Post-Processing tab", 5000
            )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark-ish palette for a scientific look
    from PySide6.QtGui import QPalette
    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(220, 220, 220))
    palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.BrightText, QColor(255, 50, 50))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()