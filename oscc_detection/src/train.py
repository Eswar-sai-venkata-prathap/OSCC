#!/usr/bin/env python3
"""
OSCC Detection Project — Optimized Training Script
===================================================
SPEED SECRET: Pre-extract CNN features ONCE → save to disk
              → train only GRU + Attention + MLP on features
              → ~5 min training, ~79% accuracy on CPU

Pipeline:
  Step 1: Run EfficientNetB0 on all images ONCE → save .npy files
  Step 2: Train lite model (no CNN) on saved features
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from patch_extractor import PatchExtractor
from cnn_feature_extractor import CNNFeatureExtractor
from model import build_lite_model, build_oscc_model, OSCCLiteModel

SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CLASS_MAP: Dict[str, int] = {"normal": 0, "oscc": 1}
NUM_CLASSES: int = 2
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ════════════════════════════════════════════════════════════
#  FAST FEATURE GENERATOR — loads pre-extracted .npy files
#  No CNN inference during training → very fast
# ════════════════════════════════════════════════════════════
class FeatureSequenceGenerator(tf.keras.utils.Sequence):
    """
    Loads pre-extracted CNN feature files instead of raw images.
    Input to model: (batch, num_patches, feature_dim) — no CNN needed.
    This is why training is fast.
    """

    def __init__(
        self,
        features_dir: str,
        split: str,
        num_patches: int = 4,
        feature_dim: int = 1280,
        batch_size: int = 64,
        augment: bool = False,
    ):
        super().__init__()
        self.split = split
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.augment = augment

        self.feature_paths: List[Path] = []
        self.labels: List[int] = []
        self._scan(features_dir, split)

        self.indices = np.arange(len(self.feature_paths))
        if split == "train":
            np.random.shuffle(self.indices)

    def _scan(self, features_dir: str, split: str) -> None:
        root = Path(features_dir) / split
        if not root.exists():
            raise FileNotFoundError(f"Features dir not found: {root}")
        for cls, idx in CLASS_MAP.items():
            cls_dir = root / cls
            if not cls_dir.exists():
                continue
            files = sorted(cls_dir.glob("*_feat.npy"))
            self.feature_paths.extend(files)
            self.labels.extend([idx] * len(files))
        print(f"  📂 {split}: {len(self.feature_paths)} feature files")

    def __len__(self) -> int:
        return int(np.ceil(len(self.feature_paths) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y = [], []
        for i in batch_idx:
            feat = self._load_features(self.feature_paths[i])
            label = tf.keras.utils.to_categorical(self.labels[i], NUM_CLASSES)
            batch_x.append(feat)
            batch_y.append(label)
        return (
            np.array(batch_x, dtype=np.float32),
            np.array(batch_y, dtype=np.float32),
        )

    def _load_features(self, feat_path: Path) -> np.ndarray:
        try:
            feat = np.load(str(feat_path)).astype(np.float32)
            if feat.ndim == 1:
                feat = feat[np.newaxis, :]
            if feat.shape[0] < self.num_patches:
                repeat = np.tile(feat, (self.num_patches, 1))[:self.num_patches]
                feat = repeat
            elif feat.shape[0] > self.num_patches:
                feat = feat[:self.num_patches]
            if self.augment:
                feat = feat + np.random.normal(0, 0.01, feat.shape).astype(np.float32)
            return feat
        except Exception:
            return np.zeros((self.num_patches, self.feature_dim), dtype=np.float32)

    def on_epoch_end(self) -> None:
        if self.split == "train":
            np.random.shuffle(self.indices)

    def get_class_counts(self) -> Dict[int, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


# ════════════════════════════════════════════════════════════
#  WRAPPER — fixes dict output for model.fit()
# ════════════════════════════════════════════════════════════
class LiteTrainingWrapper(tf.keras.Model):
    """Returns only predictions tensor for model.fit()."""

    def __init__(self, lite_model: OSCCLiteModel, **kwargs):
        super().__init__(**kwargs)
        self.lite_model = lite_model

    def call(self, inputs, training=False):
        return self.lite_model(inputs, training=training)["predictions"]


# ════════════════════════════════════════════════════════════
#  COMPILE
# ════════════════════════════════════════════════════════════
def compile_model(wrapper: LiteTrainingWrapper, lr: float = 0.001) -> LiteTrainingWrapper:
    wrapper.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    print("  ✅ Model compiled.")
    return wrapper


# ════════════════════════════════════════════════════════════
#  CALLBACKS
# ════════════════════════════════════════════════════════════
def get_callbacks(save_dir: str) -> List:
    """No TensorBoard — avoids install errors."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    csv_path = str(PROJECT_ROOT / "outputs" / "training_log.csv")

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=8,
            restore_best_weights=True, mode="max", verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_path / "best_model.weights.h5"),
            monitor="val_auc", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4,
            min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(csv_path, append=True),
    ]


# ════════════════════════════════════════════════════════════
#  PLOT HISTORY
# ════════════════════════════════════════════════════════════
def plot_training_history(history, save_dir: str) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for metric, title, ylabel, filename in [
        ("accuracy", "Training vs Validation Accuracy", "Accuracy", "training_accuracy.png"),
        ("loss",     "Training vs Validation Loss",     "Loss",     "training_loss.png"),
        ("auc",      "AUC Over Epochs",                 "AUC",      "training_auc.png"),
    ]:
        train_vals = history.history.get(metric, [])
        val_vals   = history.history.get(f"val_{metric}", [])
        if not train_vals:
            continue
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#1a1a2e")
        ax.set_facecolor("#16213e")
        epochs = range(1, len(train_vals) + 1)
        ax.plot(epochs, train_vals, "o-", color="#4ade80", linewidth=2, label=f"Train {metric}")
        if val_vals:
            ax.plot(epochs, val_vals, "s-", color="#f87171", linewidth=2, label=f"Val {metric}")
        ax.set_title(title, fontsize=14, color="white", fontweight="bold")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors="white")
        plt.tight_layout()
        plt.savefig(save_path / filename, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  📊 Saved {filename}")


# ════════════════════════════════════════════════════════════
#  MAIN TRAINING PIPELINE
# ════════════════════════════════════════════════════════════
def train_model(
    data_dir: str = "data/",
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    backbone: str = "efficientnet",
    rnn_type: str = "gru",
    attention_type: str = "soft",
    num_patches: int = 4,
    force_recompute: bool = False,
) -> Tuple[OSCCLiteModel, Any]:

    print("\n" + "=" * 60)
    print("  🏋️  OSCC Detection — Optimized Training Pipeline")
    print("=" * 60)

    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"  🖥️  GPU detected: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("  ⚠️  No GPU — training on CPU.")
        print("  💡 Using pre-extracted features for fast CPU training.")

    # Resolve paths
    data_path = str(PROJECT_ROOT / data_dir) if not Path(data_dir).is_absolute() else data_dir
    features_path = str(PROJECT_ROOT / "data" / "features")
    save_dir = str(PROJECT_ROOT / "outputs" / "models")

    # ── STEP 1: Pre-extract CNN features (runs once, then cached) ──
    print("\n" + "=" * 60)
    print("  ⚡ STEP 1: CNN Feature Pre-Extraction")
    print("=" * 60)

    patch_ext = PatchExtractor(patch_size=224, min_image_size=448)
    cnn_extractor = CNNFeatureExtractor(backbone=backbone, trainable=False)

    cnn_extractor.pre_extract_all_features(
        data_dir=data_path,
        save_dir=features_path,
        patch_extractor=patch_ext,
        num_patches=num_patches,
        force_recompute=force_recompute,
    )

    # ── STEP 2: Build fast feature generators ──
    print("\n" + "=" * 60)
    print("  📦 STEP 2: Building Feature Generators")
    print("=" * 60)

    feature_dim = 1280 if backbone == "efficientnet" else 2048

    train_gen = FeatureSequenceGenerator(
        features_path, "train",
        num_patches=num_patches, feature_dim=feature_dim,
        batch_size=batch_size, augment=True,
    )
    val_gen = FeatureSequenceGenerator(
        features_path, "val",
        num_patches=num_patches, feature_dim=feature_dim,
        batch_size=batch_size, augment=False,
    )

    print(f"  📊 Train batches: {len(train_gen)} | Val batches: {len(val_gen)}")
    print(f"  📦 Batch size: {batch_size} | Each step processes {batch_size} images")

    # Class weights
    counts = train_gen.get_class_counts()
    total = sum(counts.values())
    class_weights = {k: round(total / (NUM_CLASSES * v), 4) for k, v in counts.items()}
    print(f"  ⚖️  Class weights: {class_weights}")

    # ── STEP 3: Build lite model (no CNN) ──
    print("\n" + "=" * 60)
    print("  🔨 STEP 3: Building Lite Model (GRU + Attention + MLP)")
    print("=" * 60)

    config = {
        "feature_dim": feature_dim,
        "attention_type": attention_type,
        "attention_units": 128,
        "rnn_type": rnn_type,
        "hidden_dim": 128,
        "dropout_rate": 0.3,
        "num_patches": num_patches,
    }
    lite_model = build_lite_model(config)

    # Wrap for model.fit()
    wrapper = LiteTrainingWrapper(lite_model, name="oscc_wrapper")
    dummy = np.zeros((1, num_patches, feature_dim), dtype=np.float32)
    _ = wrapper(dummy, training=False)

    # Compile
    wrapper = compile_model(wrapper, learning_rate)
    callbacks = get_callbacks(save_dir)

    # ── STEP 4: Train ──
    print("\n" + "=" * 60)
    print(f"  🚀 STEP 4: Training ({epochs} epochs, {len(train_gen)} steps/epoch)")
    print("=" * 60 + "\n")

    history = wrapper.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ══════════════════════════════════════════════════════════
    #  FIX: EarlyStopping restores best weights into wrapper.
    #  Resave from lite_model directly so evaluate.py can load
    #  cleanly — wrapper layer names don't match OSCCLiteModel.
    #  Training accuracy/AUC is NOT affected by this step.
    # ══════════════════════════════════════════════════════════
    best_path = str(Path(save_dir) / "best_model.weights.h5")
    lite_model.save_weights(best_path)
    print(f"  💾 Best lite_model weights (re-saved) → {best_path}")

    # Plots
    plot_training_history(history, str(PROJECT_ROOT / "outputs" / "plots"))

    # Save final lite model weights
    final_path = str(Path(save_dir) / "final_model.weights.h5")
    lite_model.save_weights(final_path)
    print(f"  💾 Lite model weights → {final_path}")

    # Save config
    cfg_path = str(Path(save_dir) / "model_config.json")
    full_config = {**config, "backbone": backbone, "patch_size": 224}
    with open(cfg_path, "w") as f:
        json.dump(full_config, f, indent=2)
    print(f"  📝 Config saved → {cfg_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  📋  Training Summary")
    print("=" * 60)
    final_epoch  = len(history.history.get("loss", []))
    best_val_auc = max(history.history.get("val_auc", [0]))
    best_val_acc = max(history.history.get("val_accuracy", [0]))
    print(f"  Epochs completed : {final_epoch}")
    print(f"  Best val AUC     : {best_val_auc:.4f}")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print("=" * 60 + "\n")

    return lite_model, history


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OSCC Detection Model")
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--backbone",        default="efficientnet",
                        choices=["efficientnet", "resnet"])
    parser.add_argument("--rnn_type",        default="gru",
                        choices=["lstm", "gru"])
    parser.add_argument("--attention",       default="soft",
                        choices=["soft", "multihead"])
    parser.add_argument("--lr",              type=float, default=0.001)
    parser.add_argument("--num_patches",     type=int,   default=4)
    parser.add_argument("--force_recompute", action="store_true",
                        help="Force re-extract CNN features even if cached")
    args = parser.parse_args()

    model, history = train_model(
        data_dir="data/",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        backbone=args.backbone,
        rnn_type=args.rnn_type,
        attention_type=args.attention,
        num_patches=args.num_patches,
        force_recompute=args.force_recompute,
    )
    print("✅ Training complete. Model saved to outputs/models/")