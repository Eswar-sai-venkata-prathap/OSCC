#!/usr/bin/env python3
"""
============================================================
OSCC Detection Project — Preprocessing Module
============================================================
Microscope-Aided Deep Learning Framework for Early Oral
Cancer (OSCC) Detection

This module provides:
  1. OSCCDataGenerator  — A tf.keras.utils.Sequence subclass
     that loads histopathology images in batches with optional
     augmentation (training split only).
  2. get_datasets()     — Returns train/val/test generators
     plus class-weight dictionary.
  3. visualize_samples() — Saves a grid of sample images from
     both classes to disk.

Author : OSCC Detection Team
Date   : 2026
============================================================
"""

# ── standard library ──────────────────────────────────────
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt

# ── reproducibility ───────────────────────────────────────
SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── project paths ─────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── constants ─────────────────────────────────────────────
IMG_SIZE: int = 224
BATCH_SIZE: int = 32
CLASS_MAP: Dict[str, int] = {"normal": 0, "oscc": 1}
NUM_CLASSES: int = len(CLASS_MAP)
IMAGE_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ════════════════════════════════════════════════════════════
#  1. AUGMENTATION LAYER (TRAINING ONLY)
# ════════════════════════════════════════════════════════════
def build_augmentation_model() -> tf.keras.Sequential:
    """
    Build a ``tf.keras.Sequential`` model composed of random
    augmentation layers.  Applied **only** to the training
    split.

    Augmentations
    -------------
    - Random horizontal flip
    - Random vertical flip
    - Random rotation  (±15°  ≈  ±0.042 turns)
    - Random brightness (factor 0.2)
    - Random zoom       (0.9× – 1.1×)

    Returns
    -------
    tf.keras.Sequential
        An augmentation pipeline that accepts and returns
        tensors of shape ``(H, W, 3)`` with values in [0, 1].
    """
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomFlip("vertical"),
            tf.keras.layers.RandomRotation(
                factor=15 / 360,  # ±15 degrees expressed as a fraction of 2π
                fill_mode="reflect",
            ),
            tf.keras.layers.RandomBrightness(
                factor=0.2,       # shift pixel values by up to ±0.2
                value_range=(0.0, 1.0),
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.1, 0.1),  # 0.9× to 1.1× zoom
                width_factor=(-0.1, 0.1),
                fill_mode="reflect",
            ),
        ],
        name="training_augmentation",
    )
    return augmentation


# ════════════════════════════════════════════════════════════
#  2. OSCCDataGenerator (tf.keras.utils.Sequence)
# ════════════════════════════════════════════════════════════
class OSCCDataGenerator(tf.keras.utils.Sequence):
    """
    A Keras-compatible data generator for the OSCC dataset.

    Parameters
    ----------
    data_dir : str or Path
        Root data directory (e.g. ``oscc_detection/data``).
    split : str
        One of ``'train'``, ``'val'``, ``'test'``.
    batch_size : int
        Number of images per batch.
    img_size : int
        Target height/width (images are resized to
        ``img_size × img_size``).
    augment : bool or None
        Whether to apply augmentation.  ``None`` means auto —
        augment only when ``split == 'train'``.

    Attributes
    ----------
    image_paths : list[Path]
        All discovered image file paths.
    labels : np.ndarray
        Integer labels (0 = normal, 1 = oscc).
    one_hot_labels : np.ndarray
        One-hot encoded labels, shape ``(N, 2)``.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        batch_size: int = BATCH_SIZE,
        img_size: int = IMG_SIZE,
        augment: Optional[bool] = None,
    ) -> None:
        """Initialise the generator — scan folders, load paths."""
        super().__init__()

        self.data_dir: Path = Path(data_dir)
        self.split: str = split
        self.batch_size: int = batch_size
        self.img_size: int = img_size

        # Auto-decide augmentation: ON for train, OFF otherwise
        self.augment: bool = (split == "train") if augment is None else augment

        # Build the augmentation pipeline (only used when self.augment is True)
        self._augmenter: tf.keras.Sequential = build_augmentation_model()

        # ── discover image paths and labels ────────────────
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self._scan_directory()

        # Convert to numpy
        self.labels_array: np.ndarray = np.array(self.labels, dtype=np.int32)
        self.one_hot_labels: np.ndarray = tf.keras.utils.to_categorical(
            self.labels_array, num_classes=NUM_CLASSES
        )

        # Shuffle indices for training
        self.indices: np.ndarray = np.arange(len(self.image_paths))
        if self.split == "train":
            np.random.shuffle(self.indices)

    # ── private helpers ────────────────────────────────────
    def _scan_directory(self) -> None:
        """
        Walk through ``data_dir/<split>/normal`` and
        ``data_dir/<split>/oscc`` to collect image paths and
        their corresponding integer labels.

        Raises
        ------
        FileNotFoundError
            If the split directory does not exist.
        """
        split_dir: Path = self.data_dir / self.split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"❌ Split directory not found: {split_dir}\n"
                f"   Run download_dataset.py first to organise the data."
            )

        for cls_name, cls_idx in CLASS_MAP.items():
            cls_dir: Path = split_dir / cls_name

            if not cls_dir.exists():
                print(f"⚠️  Class directory missing: {cls_dir} — skipping.")
                continue

            files: List[Path] = sorted(
                [
                    f
                    for f in cls_dir.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS
                ]
            )

            desc = f"  Scanning {self.split}/{cls_name}"
            for fpath in tqdm(files, desc=desc, unit="img", leave=False):
                self.image_paths.append(fpath)
                self.labels.append(cls_idx)

        if not self.image_paths:
            print(
                f"⚠️  No images found for split '{self.split}' "
                f"in {split_dir}"
            )

        print(
            f"  ✅ {self.split}: {len(self.image_paths):,} images "
            f"({sum(1 for l in self.labels if l == 0)} normal, "
            f"{sum(1 for l in self.labels if l == 1)} oscc)"
        )

    def _load_image(self, path: Path) -> np.ndarray:
        """
        Read an image from disk, resize to ``(img_size, img_size)``,
        and normalise pixel values to [0, 1].

        Parameters
        ----------
        path : Path
            Absolute path to the image file.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(img_size, img_size, 3)``.
        """
        try:
            raw_bytes = tf.io.read_file(str(path))
            img = tf.image.decode_image(raw_bytes, channels=3, expand_animations=False)
            img = tf.image.resize(img, [self.img_size, self.img_size])
            img = tf.cast(img, tf.float32) / 255.0
            return img.numpy()
        except Exception as exc:
            print(f"⚠️  Could not load {path}: {exc}")
            # Return a blank image as fallback
            return np.zeros(
                (self.img_size, self.img_size, 3), dtype=np.float32
            )

    # ── Sequence API ───────────────────────────────────────
    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetch the ``idx``-th batch.

        Parameters
        ----------
        idx : int
            Batch index.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(batch_images, batch_labels)`` where images have
            shape ``(B, img_size, img_size, 3)`` and labels
            have shape ``(B, NUM_CLASSES)`` (one-hot).
        """
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        batch_images: List[np.ndarray] = []
        batch_labels: List[np.ndarray] = []

        for i in batch_indices:
            img = self._load_image(self.image_paths[i])

            # Apply augmentation if enabled
            if self.augment:
                img_tensor = tf.expand_dims(img, axis=0)  # (1, H, W, 3)
                img_tensor = self._augmenter(img_tensor, training=True)
                img = tf.squeeze(img_tensor, axis=0).numpy()
                # Clip to [0, 1] after augmentation
                img = np.clip(img, 0.0, 1.0)

            batch_images.append(img)
            batch_labels.append(self.one_hot_labels[i])

        return np.array(batch_images, dtype=np.float32), np.array(
            batch_labels, dtype=np.float32
        )

    def on_epoch_end(self) -> None:
        """Shuffle indices at the end of each epoch (training only)."""
        if self.split == "train":
            np.random.shuffle(self.indices)

    # ── utility ────────────────────────────────────────────
    def get_class_counts(self) -> Dict[str, int]:
        """
        Return the number of samples per class.

        Returns
        -------
        dict
            ``{'normal': int, 'oscc': int}``
        """
        unique, counts = np.unique(self.labels_array, return_counts=True)
        class_names = {v: k for k, v in CLASS_MAP.items()}
        return {class_names.get(u, str(u)): int(c) for u, c in zip(unique, counts)}


# ════════════════════════════════════════════════════════════
#  3. get_datasets()
# ════════════════════════════════════════════════════════════
def get_datasets(
    data_dir: str,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
) -> Tuple[OSCCDataGenerator, OSCCDataGenerator, OSCCDataGenerator, Dict[int, float]]:
    """
    Create data generators for all three splits and compute
    class weights to handle potential class imbalance.

    Parameters
    ----------
    data_dir : str
        Root data directory containing ``train/``, ``val/``,
        ``test/`` sub-folders.
    batch_size : int
        Batch size for all generators.
    img_size : int
        Target image dimension.

    Returns
    -------
    tuple
        ``(train_gen, val_gen, test_gen, class_weights)``
        where ``class_weights`` is a dict ``{0: w0, 1: w1}``.
    """
    print("\n" + "=" * 55)
    print("  📦  Loading OSCC Dataset Generators")
    print("=" * 55 + "\n")

    train_gen = OSCCDataGenerator(
        data_dir, split="train", batch_size=batch_size, img_size=img_size
    )
    val_gen = OSCCDataGenerator(
        data_dir, split="val", batch_size=batch_size, img_size=img_size
    )
    test_gen = OSCCDataGenerator(
        data_dir, split="test", batch_size=batch_size, img_size=img_size
    )

    # ── compute class weights ──────────────────────────────
    # Inverse frequency weighting: weight_i = N / (n_classes * count_i)
    counts = train_gen.get_class_counts()
    total = sum(counts.values())
    n_classes = len(counts)
    class_weights: Dict[int, float] = {}

    for cls_name, cls_idx in CLASS_MAP.items():
        count = counts.get(cls_name, 1)  # avoid division by zero
        class_weights[cls_idx] = round(total / (n_classes * count), 4)

    print(f"\n  ⚖️  Class weights: {class_weights}")
    print(
        f"      (normal={class_weights.get(0, 'N/A')}, "
        f"oscc={class_weights.get(1, 'N/A')})\n"
    )

    return train_gen, val_gen, test_gen, class_weights


# ════════════════════════════════════════════════════════════
#  4. visualize_samples()
# ════════════════════════════════════════════════════════════
def visualize_samples(
    data_dir: str,
    save_path: str = "outputs/plots/sample_images.png",
    num_samples: int = 9,
) -> None:
    """
    Display a grid of sample images from both classes and
    save the figure to disk.

    Layout: 3 rows of Normal images on top, 3 rows of OSCC
    images on the bottom → 3×6 grid (3 cols × 6 rows), each
    cell showing one image with its class label as the title.

    Parameters
    ----------
    data_dir : str
        Root data directory.
    save_path : str
        File path to save the output figure.
    num_samples : int
        Number of sample images per class (default 9).
    """
    data_root: Path = Path(data_dir)
    save_file: Path = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # ── collect image paths per class from the train split ─
    samples: Dict[str, List[Path]] = {}
    for cls_name in CLASS_MAP:
        cls_dir = data_root / "train" / cls_name
        if not cls_dir.exists():
            print(f"⚠️  Cannot visualise — directory missing: {cls_dir}")
            return
        all_imgs = sorted(
            [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        )
        random.shuffle(all_imgs)
        samples[cls_name] = all_imgs[:num_samples]

    # ── build the grid ─────────────────────────────────────
    n_rows: int = 6  # 3 rows normal + 3 rows oscc
    n_cols: int = 3

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(12, 20), facecolor="#1a1a2e"
    )
    fig.suptitle(
        "OSCC Dataset — Sample Images",
        fontsize=18,
        fontweight="bold",
        color="white",
        y=0.98,
    )

    # ── fill normal images (rows 0–2) ─────────────────────
    for i, img_path in enumerate(samples.get("normal", [])):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        img = _load_display_image(img_path)
        ax.imshow(img)
        ax.set_title("Normal", fontsize=11, color="#4ade80", fontweight="bold")
        ax.axis("off")

    # ── fill oscc images (rows 3–5) ───────────────────────
    for i, img_path in enumerate(samples.get("oscc", [])):
        row, col = divmod(i, n_cols)
        row += 3  # offset to bottom half
        ax = axes[row][col]
        img = _load_display_image(img_path)
        ax.imshow(img)
        ax.set_title("OSCC", fontsize=11, color="#f87171", fontweight="bold")
        ax.axis("off")

    # Turn off any unused axes
    for ax_row in axes:
        for ax in ax_row:
            if not ax.images:
                ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        save_file, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)
    print(f"  💾  Sample grid saved → {save_file}")


def _load_display_image(path: Path) -> np.ndarray:
    """
    Load a single image for display (resize to 224×224,
    normalise to [0, 1]).

    Parameters
    ----------
    path : Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(224, 224, 3)``.
    """
    try:
        raw = tf.io.read_file(str(path))
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return img.numpy()
    except Exception:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)


# ════════════════════════════════════════════════════════════
#  5. MAIN — standalone test
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Resolve data directory relative to project root
    data_path: str = str(PROJECT_ROOT / "data")

    try:
        train_gen, val_gen, test_gen, class_weights = get_datasets(data_path)

        print(f"Train batches : {len(train_gen)}")
        print(f"Val batches   : {len(val_gen)}")
        print(f"Test batches  : {len(test_gen)}")

        # Verify one batch shape
        imgs, lbls = train_gen[0]
        print(f"\nSample batch — images shape : {imgs.shape}")
        print(f"Sample batch — labels shape : {lbls.shape}")
        print(f"Pixel range   : [{imgs.min():.3f}, {imgs.max():.3f}]")

        # Visualise
        vis_path: str = str(PROJECT_ROOT / "outputs" / "plots" / "sample_images.png")
        visualize_samples(data_path, vis_path)
        print("✅ Sample visualization saved.")

    except FileNotFoundError as exc:
        print(f"\n{exc}")
        print("Hint: Run download_dataset.py first.")
        sys.exit(1)
