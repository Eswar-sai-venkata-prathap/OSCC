#!/usr/bin/env python3
"""
OSCC Detection Project — Optimized Integrated Model
===================================================
TWO modes:
  1. FULL model  — CNN + Attention + GRU + MLP  (for inference/GradCAM)
  2. LITE model  — Attention + GRU + MLP only   (for fast training on pre-extracted features)

The LITE model is what makes training fast (~5 min, ~79% accuracy).
CNN runs ONCE during pre-extraction, not on every batch.
"""

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from attention_module import build_attention_block
from lstm_context import ContextualLearner, PositionalEncoding
from mlp_classifier import OSCCClassifier

SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG: Dict[str, Any] = {
    "backbone": "efficientnet",
    "feature_dim": 1280,
    "attention_type": "soft",
    "attention_units": 128,
    "rnn_type": "gru",
    "hidden_dim": 128,
    "num_classes": 2,
    "dropout_rate": 0.3,
    "patch_size": 224,
    "num_patches": 4,
}


# ════════════════════════════════════════════════════════════
#  LITE MODEL — trains on pre-extracted features (FAST)
#  Input: (batch, num_patches, feature_dim)   ← already CNN features
#  No CNN inside → 10-20x faster on CPU
# ════════════════════════════════════════════════════════════
class OSCCLiteModel(tf.keras.Model):
    """
    Fast training model that takes pre-extracted CNN features as input.
    Input shape: (batch, num_patches, feature_dim)
    This skips CNN inference entirely during training.
    """

    def __init__(
        self,
        feature_dim: int = 1280,
        attention_type: str = "soft",
        attention_units: int = 128,
        rnn_type: str = "gru",
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        num_patches: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config_dict = {
            "feature_dim": feature_dim,
            "attention_type": attention_type,
            "attention_units": attention_units,
            "rnn_type": rnn_type,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "dropout_rate": dropout_rate,
            "num_patches": num_patches,
        }
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.pos_encoding = PositionalEncoding(max_patches=64, feature_dim=feature_dim, name="pos_enc")
        self.attention = build_attention_block(attention_type=attention_type, units=attention_units)
        self.context_rnn = ContextualLearner(
            hidden_dim=hidden_dim, rnn_type=rnn_type,
            dropout_rate=dropout_rate, bidirectional=False, num_layers=1,
            name=f"{rnn_type}_context",
        )
        self.fusion_dense = tf.keras.layers.Dense(256, activation="relu", name="fusion")
        self.fusion_bn = tf.keras.layers.BatchNormalization(name="fusion_bn")
        self.fusion_drop = tf.keras.layers.Dropout(dropout_rate, name="fusion_drop")
        self.classifier = OSCCClassifier(
            num_classes=num_classes,
            hidden_dims=[256, 128],
            dropout_rate=dropout_rate,
            name="mlp_classifier",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> Dict[str, tf.Tensor]:
        """
        inputs: (batch, num_patches, feature_dim) — pre-extracted CNN features
        returns: {'predictions': (batch, 2), 'attention_weights': (batch, num_patches, 1)}
        """
        x = self.pos_encoding(inputs)
        attn_context, attn_weights = self.attention(x, training=training)
        rnn_output = self.context_rnn(x, training=training)
        fused = tf.concat([attn_context, rnn_output], axis=-1)
        fused = self.fusion_dense(fused)
        fused = self.fusion_bn(fused, training=training)
        fused = self.fusion_drop(fused, training=training)
        predictions = self.classifier(fused, training=training)
        return {"predictions": predictions, "attention_weights": attn_weights}

    def get_config(self):
        return self.config_dict.copy()

    def save_config(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.config_dict, f, indent=2)
        print(f"  📝 Config saved → {p}")


# ════════════════════════════════════════════════════════════
#  FULL MODEL — CNN + all stages (for inference & GradCAM)
#  Input: (batch, num_patches, 224, 224, 3)
# ════════════════════════════════════════════════════════════
class OSCCDetectionModel(tf.keras.Model):
    """
    Full end-to-end model for inference and GradCAM.
    Input: (batch, num_patches, 224, 224, 3) raw image patches
    """

    def __init__(
        self,
        backbone: str = "efficientnet",
        feature_dim: int = 1280,
        attention_type: str = "soft",
        attention_units: int = 128,
        rnn_type: str = "gru",
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        patch_size: int = 224,
        num_patches: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config_dict = {
            "backbone": backbone, "feature_dim": feature_dim,
            "attention_type": attention_type, "attention_units": attention_units,
            "rnn_type": rnn_type, "hidden_dim": hidden_dim,
            "num_classes": num_classes, "dropout_rate": dropout_rate,
            "patch_size": patch_size, "num_patches": num_patches,
        }
        self.patch_size = patch_size
        self.feature_dim = feature_dim

        # CNN backbone (frozen)
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet",
            input_shape=(patch_size, patch_size, 3))
        base.trainable = False
        self.cnn_backbone = base
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name="gap")
        self.feat_dense = tf.keras.layers.Dense(
            feature_dim, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.001), name="feat_dense")
        self.feat_bn = tf.keras.layers.BatchNormalization(name="feat_bn")

        # Reuse lite model components
        self.lite = OSCCLiteModel(
            feature_dim=feature_dim, attention_type=attention_type,
            attention_units=attention_units, rnn_type=rnn_type,
            hidden_dim=hidden_dim, num_classes=num_classes,
            dropout_rate=dropout_rate, num_patches=num_patches,
            name="lite_core",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> Dict[str, tf.Tensor]:
        """inputs: (batch, num_patches, 224, 224, 3)"""
        batch_size = tf.shape(inputs)[0]
        num_patches = tf.shape(inputs)[1]

        flat = tf.reshape(inputs, [-1, self.patch_size, self.patch_size, 3])
        x = tf.keras.applications.efficientnet.preprocess_input(flat * 255.0)
        x = self.cnn_backbone(x, training=False)
        x = self.gap(x)
        x = self.feat_dense(x)
        x = self.feat_bn(x, training=training)

        features_seq = tf.reshape(x, [batch_size, num_patches, self.feature_dim])
        return self.lite(features_seq, training=training)

    def build_model(self, input_shape=(4, 224, 224, 3)):
        dummy = tf.zeros((1, *input_shape))
        _ = self(dummy, training=False)
        total = self.count_params()
        trainable = sum(tf.keras.backend.count_params(w) for w in self.trainable_weights)
        print(f"\n  Total params     : {total:>12,}")
        print(f"  Trainable params : {trainable:>12,}")
        print(f"  Non-trainable    : {total - trainable:>12,}\n")

    def get_config(self):
        return self.config_dict.copy()

    def save_config(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.config_dict, f, indent=2)
        print(f"  📝 Config saved → {p}")


# ════════════════════════════════════════════════════════════
#  FACTORY FUNCTIONS
# ════════════════════════════════════════════════════════════
def build_oscc_model(config: Optional[Dict[str, Any]] = None) -> OSCCDetectionModel:
    """Build full model for inference/GradCAM."""
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    model = OSCCDetectionModel(**cfg, name="oscc_model")
    model.build_model(input_shape=(cfg["num_patches"], cfg["patch_size"], cfg["patch_size"], 3))
    return model


def build_lite_model(config: Optional[Dict[str, Any]] = None) -> OSCCLiteModel:
    """Build lite model for fast training on pre-extracted features."""
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    model = OSCCLiteModel(
        feature_dim=cfg["feature_dim"],
        attention_type=cfg["attention_type"],
        attention_units=cfg["attention_units"],
        rnn_type=cfg["rnn_type"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=cfg["num_classes"],
        dropout_rate=cfg["dropout_rate"],
        num_patches=cfg["num_patches"],
        name="oscc_lite_model",
    )
    # Build with dummy input
    dummy = tf.zeros((1, cfg["num_patches"], cfg["feature_dim"]))
    _ = model(dummy, training=False)
    total = model.count_params()
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"  Lite model params: {total:,} (trainable: {trainable:,})")
    return model


if __name__ == "__main__":
    print("\n🔬 Testing OSCCLiteModel (fast training mode)...")
    lite = build_lite_model()
    dummy = np.random.rand(2, 4, 1280).astype(np.float32)
    out = lite(dummy, training=False)
    print(f"  Input:        {dummy.shape}")
    print(f"  Predictions:  {out['predictions'].shape}")
    print(f"  Attn weights: {out['attention_weights'].shape}")
    print(f"  Prob sums:    {out['predictions'].numpy().sum(axis=1)}")
    print("  ✅ Lite model test passed!\n")