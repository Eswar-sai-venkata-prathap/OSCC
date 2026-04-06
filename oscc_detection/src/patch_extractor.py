#!/usr/bin/env python3
"""
============================================================
OSCC Detection Project — Patch Extractor
============================================================
Microscope-Aided Deep Learning Framework for Early Oral
Cancer (OSCC) Detection

This module splits histopathology images into fixed-size
patches so that the downstream CNN can analyse local cellular
patterns independently.

Key components
--------------
- **PatchExtractor**         — extract / reconstruct patches
- **extract_dataset_patches** — batch-extract across all splits
- **visualize_patches**       — save a visual grid of patches

Author : OSCC Detection Team
Date   : 2026
============================================================
"""

# ── standard library ──────────────────────────────────────
import os
import glob
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ── reproducibility ───────────────────────────────────────
SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)

# ── project paths ─────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── constants ─────────────────────────────────────────────
IMAGE_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES: Tuple[str, ...] = ("normal", "oscc")
SPLITS: Tuple[str, ...] = ("train", "val", "test")


# ════════════════════════════════════════════════════════════
#  1. PatchExtractor CLASS
# ════════════════════════════════════════════════════════════
class PatchExtractor:
    """
    Extract fixed-size patches from histopathology images.

    Parameters
    ----------
    patch_size : int
        Height and width of each square patch (default 224).
    min_image_size : int
        If an image's smallest dimension is below this value,
        it is resized up *before* patching (default 448).
    overlap : int
        Number of pixels of overlap between adjacent patches
        (default 0 = non-overlapping).
    """

    def __init__(
        self,
        patch_size: int = 224,
        min_image_size: int = 448,
        overlap: int = 0,
    ) -> None:
        """Initialise the extractor with the given parameters."""
        self.patch_size: int = patch_size
        self.min_image_size: int = min_image_size
        self.overlap: int = overlap
        # Effective stride between patch origins
        self.stride: int = patch_size - overlap

        if self.stride <= 0:
            raise ValueError(
                f"overlap ({overlap}) must be strictly less than "
                f"patch_size ({patch_size})."
            )

    # ────────────────────────────────────────────────────────
    #  Core extraction
    # ────────────────────────────────────────────────────────
    def extract_patches(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Divide an image into non-overlapping (or overlapping)
        square patches.

        Steps
        -----
        1. If either dimension is below ``min_image_size``,
           resize the image up (preserving aspect ratio by
           scaling both sides proportionally).
        2. Iterate over the image in strides of
           ``patch_size - overlap`` and crop patches.
        3. Discard edge patches that would be smaller than
           ``patch_size``.
        4. Normalise each patch to [0, 1] float32.

        Parameters
        ----------
        image : np.ndarray
            Input image, shape ``(H, W, 3)``, dtype uint8
            or float.

        Returns
        -------
        list[np.ndarray]
            List of patches, each of shape
            ``(patch_size, patch_size, 3)``  in float32 [0, 1].
        """
        # ── ensure 3-channel ──────────────────────────────
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h, w = image.shape[:2]

        # ── resize up if too small ────────────────────────
        if h < self.min_image_size or w < self.min_image_size:
            scale = self.min_image_size / min(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            image = cv2.resize(
                image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            h, w = image.shape[:2]

        # ── normalise to [0, 1] if needed ─────────────────
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # ── extract patches ───────────────────────────────
        patches: List[np.ndarray] = []
        ps = self.patch_size

        for y in range(0, h - ps + 1, self.stride):
            for x in range(0, w - ps + 1, self.stride):
                patch = image[y : y + ps, x : x + ps]
                patches.append(patch)

        # ── warn if only one patch ────────────────────────
        if len(patches) == 1:
            print(
                f"  ⚠️  Image ({h}×{w}) produced only 1 patch — "
                f"consider using a smaller patch_size or larger image."
            )

        return patches

    # ────────────────────────────────────────────────────────
    #  File-level extraction
    # ────────────────────────────────────────────────────────
    def extract_from_file(
        self, image_path: str
    ) -> Tuple[List[np.ndarray], Tuple[int, int, int]]:
        """
        Load an image from disk and extract patches.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        tuple[list[np.ndarray], tuple[int, int, int]]
            ``(patches, original_shape)`` where
            ``original_shape`` is ``(H, W, C)`` of the loaded
            image *before* any resizing.

        Raises
        ------
        FileNotFoundError
            If ``image_path`` does not exist.
        ValueError
            If the file cannot be decoded as an image.
        """
        fpath = Path(image_path)
        if not fpath.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image: {image_path}")

        # OpenCV loads as BGR — convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape: Tuple[int, int, int] = image.shape  # (H, W, 3)

        patches = self.extract_patches(image)
        return patches, original_shape

    # ────────────────────────────────────────────────────────
    #  Reconstruct image from patches
    # ────────────────────────────────────────────────────────
    def reconstruct_image(
        self,
        patches: List[np.ndarray],
        original_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """
        Reassemble an image from its patches (reverse of
        ``extract_patches``).

        For overlapping patches, overlapping regions are
        averaged.  The reconstructed image is then resized
        back to ``original_shape[:2]``.

        Parameters
        ----------
        patches : list[np.ndarray]
            Patches produced by ``extract_patches``.
        original_shape : tuple
            ``(H, W, C)`` of the original image (before any
            up-scaling that may have been applied).

        Returns
        -------
        np.ndarray
            Reconstructed image of shape ``(H, W, 3)`` in
            float32 [0, 1].
        """
        orig_h, orig_w = original_shape[:2]
        ps = self.patch_size

        # Determine working canvas size (may be larger than
        # the original if the image was resized up).
        work_h = max(orig_h, self.min_image_size)
        work_w = max(orig_w, self.min_image_size)

        # Compute grid dimensions
        positions = self.get_patch_positions((work_h, work_w, 3))
        if len(positions) != len(patches):
            # Fall back: recalculate from actual patch count
            cols = max(1, (work_w - ps) // self.stride + 1)
            rows = max(1, (work_h - ps) // self.stride + 1)
            # Adjust canvas to fit exact number of patches
            work_h = (rows - 1) * self.stride + ps
            work_w = (cols - 1) * self.stride + ps
            positions = self.get_patch_positions((work_h, work_w, 3))

        canvas = np.zeros((work_h, work_w, 3), dtype=np.float64)
        weight = np.zeros((work_h, work_w, 1), dtype=np.float64)

        for patch, (_, _, y, x) in zip(patches, positions):
            canvas[y : y + ps, x : x + ps] += patch.astype(np.float64)
            weight[y : y + ps, x : x + ps] += 1.0

        # Avoid division by zero
        weight = np.maximum(weight, 1.0)
        canvas /= weight

        # Resize back to original dimensions
        canvas = cv2.resize(
            canvas.astype(np.float32),
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.clip(canvas, 0.0, 1.0)

    # ────────────────────────────────────────────────────────
    #  Patch positions
    # ────────────────────────────────────────────────────────
    def get_patch_positions(
        self, image_shape: Tuple[int, ...]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Compute the ``(row_idx, col_idx, y_start, x_start)``
        for every patch that would be extracted from an image.

        Parameters
        ----------
        image_shape : tuple
            ``(H, W, ...)`` of the (possibly resized) image.

        Returns
        -------
        list[tuple[int, int, int, int]]
            Each entry is
            ``(row_index, col_index, y_start_px, x_start_px)``.
        """
        h, w = image_shape[:2]

        # Apply the same up-scaling logic as extract_patches
        if h < self.min_image_size or w < self.min_image_size:
            scale = self.min_image_size / min(h, w)
            h = int(h * scale)
            w = int(w * scale)

        ps = self.patch_size
        positions: List[Tuple[int, int, int, int]] = []

        row_idx = 0
        for y in range(0, h - ps + 1, self.stride):
            col_idx = 0
            for x in range(0, w - ps + 1, self.stride):
                positions.append((row_idx, col_idx, y, x))
                col_idx += 1
            row_idx += 1

        return positions


# ════════════════════════════════════════════════════════════
#  2. BATCH EXTRACTION ACROSS THE DATASET
# ════════════════════════════════════════════════════════════
def extract_dataset_patches(
    data_dir: str,
    patch_extractor: PatchExtractor,
    save_dir: str = "data/patches/",
) -> Dict[str, Dict[str, int]]:
    """
    Extract patches from every image in ``data/{split}/{class}``
    and save them as ``.npy`` files under ``save_dir``.

    Output layout::

        save_dir/
          train/normal/  train/oscc/
          val/normal/    val/oscc/
          test/normal/   test/oscc/

    Parameters
    ----------
    data_dir : str
        Root data directory (e.g. ``"data"``).
    patch_extractor : PatchExtractor
        Configured extractor instance.
    save_dir : str
        Root directory for saved patch ``.npy`` files.

    Returns
    -------
    dict
        ``{split: {class: total_patches}}``
    """
    data_root = Path(data_dir)
    save_root = Path(save_dir)
    stats: Dict[str, Dict[str, int]] = {}

    for split in SPLITS:
        stats[split] = {}
        for cls_name in CLASS_NAMES:
            cls_dir = data_root / split / cls_name
            out_dir = save_root / split / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)

            if not cls_dir.exists():
                print(f"  ⚠️  Skipping {cls_dir} — not found.")
                stats[split][cls_name] = 0
                continue

            image_files = sorted(
                [
                    f
                    for f in cls_dir.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS
                ]
            )

            total_patches = 0
            desc = f"  {split}/{cls_name}"
            for img_path in tqdm(image_files, desc=desc, unit="img"):
                try:
                    patches, _ = patch_extractor.extract_from_file(
                        str(img_path)
                    )
                except (FileNotFoundError, ValueError) as exc:
                    print(f"    ⚠️  Skipping {img_path.name}: {exc}")
                    continue

                for i, patch in enumerate(patches):
                    npy_name = f"{img_path.stem}_patch_{i:03d}.npy"
                    np.save(str(out_dir / npy_name), patch)

                total_patches += len(patches)

            stats[split][cls_name] = total_patches

    # ── print summary ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("           📊  PATCH EXTRACTION STATISTICS")
    print("=" * 55)
    print(f"{'Split':<10} {'Normal':>10} {'OSCC':>10} {'Total':>10}")
    print("-" * 55)

    grand = 0
    for split in SPLITS:
        n = stats[split].get("normal", 0)
        o = stats[split].get("oscc", 0)
        t = n + o
        grand += t
        print(f"{split:<10} {n:>10,} {o:>10,} {t:>10,}")

    print("-" * 55)
    all_n = sum(stats[s].get("normal", 0) for s in stats)
    all_o = sum(stats[s].get("oscc", 0) for s in stats)
    print(f"{'TOTAL':<10} {all_n:>10,} {all_o:>10,} {grand:>10,}")
    print("=" * 55 + "\n")

    return stats


# ════════════════════════════════════════════════════════════
#  3. VISUALIZATION
# ════════════════════════════════════════════════════════════
def visualize_patches(
    image_path: str,
    patch_extractor: PatchExtractor,
    save_path: str = "outputs/plots/patches.png",
    max_patches: int = 8,
) -> None:
    """
    Visualise the original image alongside its first patches.

    Layout: 1 column for the original + ``max_patches`` columns
    for individual patches, displayed in a single row.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    patch_extractor : PatchExtractor
        Configured extractor instance.
    save_path : str
        Where to save the output figure.
    max_patches : int
        Maximum number of patches to display (default 8).
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        patches, orig_shape = patch_extractor.extract_from_file(image_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  ❌ Cannot visualise patches: {exc}")
        return

    # Load original for display
    orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_img = orig_img.astype(np.float32) / 255.0

    n_show = min(len(patches), max_patches)
    n_cols = 1 + n_show  # original + patches

    fig, axes = plt.subplots(
        1, n_cols, figsize=(3 * n_cols, 3.5), facecolor="#1a1a2e"
    )

    # ── original image ────────────────────────────────────
    axes[0].imshow(orig_img)
    axes[0].set_title(
        f"Original\n{orig_shape[1]}×{orig_shape[0]}",
        fontsize=10,
        color="white",
        fontweight="bold",
    )
    axes[0].axis("off")

    # ── patches ───────────────────────────────────────────
    for i in range(n_show):
        ax = axes[i + 1]
        ax.imshow(np.clip(patches[i], 0, 1))
        ax.set_title(
            f"Patch {i}",
            fontsize=9,
            color="#93c5fd",
            fontweight="bold",
        )
        ax.axis("off")

    plt.suptitle(
        f"Patch Extraction — {len(patches)} patches "
        f"({patch_extractor.patch_size}×{patch_extractor.patch_size})",
        fontsize=13,
        color="white",
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        save_file,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"  💾  Patch visualisation saved → {save_file}")


# ════════════════════════════════════════════════════════════
#  4. MAIN — standalone test
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    extractor = PatchExtractor(patch_size=224, min_image_size=448, overlap=0)

    # ── find a sample image ───────────────────────────────
    data_dir = str(PROJECT_ROOT / "data")

    # Try multiple extensions
    sample_path: Optional[str] = None
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        matches = glob.glob(
            os.path.join(data_dir, "train", "normal", ext)
        )
        if matches:
            sample_path = matches[0]
            break

    if sample_path is None:
        print(
            "❌ No sample image found in data/train/normal/.\n"
            "   Run download_dataset.py first."
        )
    else:
        print(f"\n🔬 Testing patch extraction on: {sample_path}\n")

        patches, shape = extractor.extract_from_file(sample_path)
        print(f"  Image shape       : {shape}")
        print(f"  Patches extracted : {len(patches)}")
        print(f"  Patch shape       : {patches[0].shape}")
        print(f"  Patch dtype       : {patches[0].dtype}")
        print(f"  Pixel range       : [{patches[0].min():.3f}, {patches[0].max():.3f}]")

        # Test positions
        positions = extractor.get_patch_positions(shape)
        print(f"  Patch positions   : {len(positions)}")
        if positions:
            print(f"    first → row={positions[0][0]}, col={positions[0][1]}, "
                  f"y={positions[0][2]}, x={positions[0][3]}")

        # Test reconstruction
        reconstructed = extractor.reconstruct_image(patches, shape)
        print(f"  Reconstructed     : {reconstructed.shape}")

        # Visualise
        vis_path = str(
            PROJECT_ROOT / "outputs" / "plots" / "patches.png"
        )
        visualize_patches(sample_path, extractor, vis_path)
        print("\n✅ Patch extraction test complete.")
