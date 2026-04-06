#!/usr/bin/env python3
"""
OSCC Detection Project — Attention Module
SoftAttention + MultiHeadPatchAttention + AttentionVisualizer
"""

import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class SoftAttention(tf.keras.layers.Layer):
    """Bahdanau-style soft attention over patch features."""

    def __init__(self, units: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W_attention = tf.keras.layers.Dense(units, activation="tanh", name="attn_W")
        self.v_attention = tf.keras.layers.Dense(1, activation=None, use_bias=False, name="attn_v")

    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        hidden = self.W_attention(inputs)
        score = self.v_attention(hidden)
        attention_weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context, attention_weights

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"units": self.units})
        return config


class MultiHeadPatchAttention(tf.keras.layers.Layer):
    """Multi-head self-attention over patch features."""

    def __init__(self, num_heads: int = 4, key_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.layer_norm = tf.keras.layers.LayerNormalization(name="attn_ln")
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D(name="attn_gap")

    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        attended, scores = self.mha(
            query=inputs, key=inputs, value=inputs,
            return_attention_scores=True, training=training)
        attended = self.layer_norm(attended + inputs, training=training)
        context = self.global_pool(attended)
        return context, scores

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
        return config


class AttentionVisualizer:
    """Map attention weights to spatial heatmap overlays."""

    def __init__(self, patch_extractor: Any) -> None:
        self.patch_extractor = patch_extractor

    def create_attention_map(self, image, attention_weights, original_shape):
        h, w = original_shape[:2]
        ps = self.patch_extractor.patch_size
        positions = self.patch_extractor.get_patch_positions(original_shape)
        work_h = max(h, self.patch_extractor.min_image_size)
        work_w = max(w, self.patch_extractor.min_image_size)
        canvas = np.zeros((work_h, work_w), dtype=np.float64)

        weights = attention_weights.flatten().astype(np.float64)
        w_min, w_max = weights.min(), weights.max()
        if w_max - w_min > 1e-8:
            weights = (weights - w_min) / (w_max - w_min)
        else:
            weights = np.ones_like(weights)

        for i, (_, _, y, x) in enumerate(positions):
            if i < len(weights):
                canvas[y:y + ps, x:x + ps] = weights[i]

        canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)
        canvas_uint8 = (canvas * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(canvas_uint8, cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    def overlay_attention(self, original_image, attention_map, alpha=0.4):
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8) if original_image.max() <= 1.0 else original_image.astype(np.uint8)
        if attention_map.dtype != np.uint8:
            attention_map = attention_map.astype(np.uint8)
        if attention_map.shape[:2] != original_image.shape[:2]:
            attention_map = cv2.resize(attention_map, (original_image.shape[1], original_image.shape[0]))
        return cv2.addWeighted(original_image, 1.0 - alpha, attention_map, alpha, 0)

    def save_attention_visualization(self, original_image, attention_weights, original_shape, save_path, title=""):
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        heatmap = self.create_attention_map(original_image, attention_weights, original_shape)
        overlay = self.overlay_attention(original_image, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#1a1a2e")
        disp_img = original_image
        if disp_img.dtype != np.uint8:
            disp_img = (disp_img * 255).astype(np.uint8) if disp_img.max() <= 1.0 else disp_img.astype(np.uint8)

        for ax, (img, subtitle, color) in zip(axes, [
            (disp_img, "Original Image", "#e2e8f0"),
            (heatmap, "Attention Heatmap", "#fbbf24"),
            (overlay, "Overlay", "#34d399"),
        ]):
            ax.imshow(img)
            ax.set_title(subtitle, fontsize=12, color=color, fontweight="bold")
            ax.axis("off")

        sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Attention Weight", color="white", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        if title:
            fig.suptitle(title, fontsize=14, color="white", fontweight="bold", y=1.02)

        plt.tight_layout()
        plt.savefig(save_file, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  💾 Attention visualisation saved → {save_file}")


def build_attention_block(
    attention_type: str = "soft",
    units: int = 128,
    num_heads: int = 4,
    key_dim: int = 64,
) -> tf.keras.layers.Layer:
    if attention_type == "soft":
        return SoftAttention(units=units, name="soft_attention")
    elif attention_type == "multihead":
        return MultiHeadPatchAttention(num_heads=num_heads, key_dim=key_dim, name="multihead_attention")
    else:
        raise ValueError(f"Unknown attention_type '{attention_type}'. Choose 'soft' or 'multihead'.")


if __name__ == "__main__":
    dummy = np.random.rand(2, 4, 1280).astype(np.float32)

    soft = SoftAttention(units=128)
    ctx, wts = soft(dummy)
    print(f"SoftAttention context: {ctx.shape}, weights: {wts.shape}")
    print(f"Weight sums: {wts.numpy().sum(axis=1).flatten()} ✅")

    mh = MultiHeadPatchAttention(num_heads=4, key_dim=64)
    ctx2, sc2 = mh(dummy)
    print(f"MultiHead context: {ctx2.shape} ✅")