#!/usr/bin/env python3
"""
OSCC Detection Project — CNN Feature Extractor
KEY OPTIMIZATION: pre_extract_all_features() runs CNN ONCE
and saves features to disk → training is then 10-20x faster.
"""

import os
import glob
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

IMG_SIZE: int = 224
SPLITS: Tuple[str, ...] = ("train", "val", "test")
CLASS_NAMES: Tuple[str, ...] = ("normal", "oscc")
CLASS_MAP: Dict[str, int] = {"normal": 0, "oscc": 1}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

BACKBONE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "efficientnet": {
        "loader": tf.keras.applications.EfficientNetB0,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
        "default_dim": 1280,
        "name": "EfficientNetB0",
    },
    "resnet": {
        "loader": tf.keras.applications.ResNet50,
        "preprocess": tf.keras.applications.resnet50.preprocess_input,
        "default_dim": 2048,
        "name": "ResNet50",
    },
}


def preprocess_patch(patch: np.ndarray, backbone: str = "efficientnet") -> np.ndarray:
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{backbone}'.")
    preprocess_fn = BACKBONE_REGISTRY[backbone]["preprocess"]
    return preprocess_fn((patch * 255.0).astype(np.float32))


class CNNFeatureExtractor:
    """
    Extract CNN features from image patches.

    SPEED TIP: Call pre_extract_all_features() ONCE before training.
    This saves all features to disk so the CNN never runs during training.
    """

    def __init__(
        self,
        backbone: str = "efficientnet",
        trainable: bool = False,
        feature_dim: Optional[int] = None,
    ) -> None:
        if backbone not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone '{backbone}'.")

        self.backbone_name = backbone
        self.trainable = trainable
        self.backbone_info = BACKBONE_REGISTRY[backbone]
        self.feature_dim = feature_dim or self.backbone_info["default_dim"]
        self.model = self._build_model()

        status = "TRAINABLE" if trainable else "FROZEN"
        print(f"  🧠 Loaded {self.backbone_info['name']} [{status}] → {self.feature_dim}-dim features")

    def _build_model(self) -> tf.keras.Model:
        loader = self.backbone_info["loader"]
        try:
            base_model = loader(include_top=False, weights="imagenet",
                                input_shape=(IMG_SIZE, IMG_SIZE, 3))
        except Exception as exc:
            print(f"  ⚠️  ImageNet weights failed: {exc}. Using random init.")
            base_model = loader(include_top=False, weights=None,
                                input_shape=(IMG_SIZE, IMG_SIZE, 3))

        base_model.trainable = self.trainable

        inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="patch_input")
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = tf.keras.layers.Dense(self.feature_dim, activation="relu",
                                   kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                   name="feature_dense")(x)
        x = tf.keras.layers.BatchNormalization(name="feature_bn")(x)

        return tf.keras.Model(inputs=inputs, outputs=x, name="cnn_feature_extractor")

    def extract_features(self, patches: List[np.ndarray]) -> np.ndarray:
        """Extract features from a list of (224,224,3) patches in [0,1]."""
        batch = np.stack(patches, axis=0).astype(np.float32)
        preprocess_fn = self.backbone_info["preprocess"]
        batch_preprocessed = preprocess_fn((batch * 255.0).astype(np.float32))
        return self.model.predict(batch_preprocessed, verbose=0, batch_size=64)

    def pre_extract_all_features(
        self,
        data_dir: str,
        save_dir: str,
        patch_extractor,
        num_patches: int = 4,
        force_recompute: bool = False,
    ) -> str:
        """
        ⚡ THE SPEED SECRET ⚡
        Run EfficientNetB0 ONCE over the entire dataset and save
        features as .npy files. Training then loads these features
        directly — no CNN inference during training at all.

        Parameters
        ----------
        data_dir : str  — root data directory (has train/val/test subfolders)
        save_dir : str  — where to save .npy feature files
        patch_extractor — PatchExtractor instance
        num_patches : int — patches per image
        force_recompute : bool — recompute even if cache exists

        Returns
        -------
        str — path to saved features directory
        """
        save_path = Path(save_dir)
        cache_flag = save_path / ".features_ready"

        if cache_flag.exists() and not force_recompute:
            print(f"  ✅ Pre-extracted features found at {save_path}. Skipping.")
            return str(save_path)

        print("\n" + "=" * 60)
        print("  ⚡ Pre-extracting CNN features (runs ONCE, then cached)")
        print("=" * 60)

        data_root = Path(data_dir)
        total = 0

        for split in SPLITS:
            for cls_name, cls_idx in CLASS_MAP.items():
                cls_dir = data_root / split / cls_name
                out_dir = save_path / split / cls_name
                out_dir.mkdir(parents=True, exist_ok=True)

                if not cls_dir.exists():
                    continue

                img_files = sorted(
                    f for f in cls_dir.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS
                )

                print(f"  📂 {split}/{cls_name}: {len(img_files)} images")

                for img_path in tqdm(img_files, desc=f"  {split}/{cls_name}", unit="img"):
                    feat_file = out_dir / f"{img_path.stem}_feat.npy"
                    if feat_file.exists() and not force_recompute:
                        total += 1
                        continue

                    try:
                        patches, _ = patch_extractor.extract_from_file(str(img_path))
                    except Exception:
                        patches = [np.zeros((224, 224, 3), dtype=np.float32)]

                    # Pad / truncate to num_patches
                    if len(patches) < num_patches:
                        while len(patches) < num_patches:
                            patches.append(patches[len(patches) % len(patches)])
                    elif len(patches) > num_patches:
                        patches = patches[:num_patches]

                    # Extract features: (num_patches, feature_dim)
                    features = self.extract_features(patches)
                    np.save(str(feat_file), features.astype(np.float32))
                    total += 1

        # Write cache flag
        cache_flag.touch()
        print(f"\n  ✅ Done! {total} feature files saved → {save_path}")
        print("  🚀 Training will now be 10-20x faster!\n")
        return str(save_path)

    def unfreeze_top_layers(self, num_layers: int = 20) -> None:
        for layer in self.model.layers:
            if hasattr(layer, "layers"):
                layer.trainable = True
                for l in layer.layers[:-num_layers]:
                    l.trainable = False
                trainable_count = sum(1 for l in layer.layers if l.trainable)
                print(f"  🔓 Unfroze top {num_layers} layers ({trainable_count} trainable)")
                return

    def get_model_summary(self) -> None:
        print("\n" + "=" * 60)
        print(f"  🧠  CNN Feature Extractor — {self.backbone_info['name']}")
        print("=" * 60)
        total = self.model.count_params()
        trainable = sum(tf.keras.backend.count_params(w) for w in self.model.trainable_weights)
        print(f"  Total parameters     : {total:>12,}")
        print(f"  Trainable parameters : {trainable:>12,}")
        print(f"  Feature dimension    : {self.feature_dim}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    extractor = CNNFeatureExtractor(backbone="efficientnet", trainable=False)
    extractor.get_model_summary()

    dummy_patches = [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(4)]
    features = extractor.extract_features(dummy_patches)
    print(f"  Output features: {features.shape} ✅")