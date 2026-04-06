#!/usr/bin/env python3
"""
============================================================
OSCC Detection Project — Grad-CAM Explainability
============================================================
Microscope-Aided Deep Learning Framework for Early Oral
Cancer (OSCC) Detection
============================================================
"""

import glob
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from patch_extractor import PatchExtractor
from mlp_classifier import get_risk_level, CLASS_NAMES

SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ════════════════════════════════════════════════════════════
#  LOAD FULL MODEL FOR GRAD-CAM
# ════════════════════════════════════════════════════════════
def load_full_model_for_gradcam(
    model_path: str = "outputs/models/best_model.weights.h5",
    config_path: str = "outputs/models/model_config.json",
) -> tf.keras.Model:
    """
    Build OSCCDetectionModel and transfer trained lite weights into it.
    Uses index-based transfer (robust — not affected by name differences).
    """
    from model import build_oscc_model, build_lite_model

    weights_file = Path(model_path)
    if not weights_file.is_absolute():
        weights_file = PROJECT_ROOT / model_path
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_path

    # Load config
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print(f"  📝 Config loaded from {config_file}")
    else:
        print("  ⚠️  Config not found, using defaults.")

    # ── Step 1: Load trained weights into lite model ──
    print("  🔄 Loading trained weights into lite model …")
    lite_model = build_lite_model(config)
    if weights_file.exists():
        lite_model.load_weights(str(weights_file))
        print(f"  ✅ Lite weights loaded ({len(lite_model.weights)} tensors)")
    else:
        print("  ⚠️  Weights file not found — using random init.")

    # ── Step 2: Build full model ──
    print("  🔄 Building full OSCCDetectionModel for GradCAM …")
    full_model = build_oscc_model(config)

    # ── Step 3: Transfer by index — lite_model.weights and
    #    full_model.lite.weights have the same order and shapes ──
    print("  🔄 Transferring weights (index-based) …")

    lite_weights   = lite_model.weights        # trained weights
    full_lite_weights = full_model.lite.weights  # same architecture inside full model

    if len(lite_weights) != len(full_lite_weights):
        print(f"  ⚠️  Weight count mismatch: lite={len(lite_weights)}, full.lite={len(full_lite_weights)}")
        print("      Attempting partial transfer …")

    transferred = 0
    skipped = 0
    for lw, fw in zip(lite_weights, full_lite_weights):
        if lw.shape == fw.shape:
            fw.assign(lw)
            transferred += 1
        else:
            skipped += 1

    print(f"  ✅ Transferred {transferred} tensors | Skipped {skipped} (shape mismatch)")

    if transferred < len(lite_weights) * 0.8:
        print("  ⚠️  Less than 80% of weights transferred — predictions may be inaccurate.")
    else:
        print("  ✅ Weight transfer successful — full model carries trained weights.")

    return full_model


# ════════════════════════════════════════════════════════════
#  1. GRAD-CAM
# ════════════════════════════════════════════════════════════
class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(
        self,
        model: tf.keras.Model,
        layer_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.backbone = self._find_backbone(model)

        if layer_name is None:
            layer_name = self._find_last_conv_layer(self.backbone)

        self.layer_name = layer_name
        print(f"  🔍 Grad-CAM target layer: {layer_name}")

        self.grad_model = tf.keras.Model(
            inputs=self.backbone.input,
            outputs=[
                self.backbone.get_layer(layer_name).output,
                self.backbone.output,
            ],
        )

    @staticmethod
    def _find_backbone(model: tf.keras.Model) -> tf.keras.Model:
        """Find the EfficientNetB0 backbone (has Conv2D layers)."""
        for layer in model.layers:
            if hasattr(layer, "layers"):
                for sl in layer.layers:
                    if isinstance(sl, tf.keras.layers.Conv2D):
                        return layer
        return model

    @staticmethod
    def _find_last_conv_layer(model: tf.keras.Model) -> str:
        """Auto-detect the last Conv2D layer name."""
        last_conv = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
            if hasattr(layer, "layers"):
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        last_conv = sub_layer.name
        if last_conv is None:
            raise ValueError("No Conv2D layer found in the model.")
        return last_conv

    def compute_gradcam(self, image: np.ndarray, class_idx: int = 1) -> np.ndarray:
        """Compute Grad-CAM heatmap for a single patch (224,224,3)."""
        img_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor)
            if predictions.shape[-1] == 2:
                target = predictions[:, class_idx]
            else:
                target = tf.reduce_mean(predictions)

        grads = tape.gradient(target, conv_outputs)
        if grads is None:
            return np.zeros((224, 224), dtype=np.float32)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1).numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        heatmap = cv2.resize(heatmap, (224, 224))
        return heatmap.astype(np.float32)

    def generate_heatmap_overlay(
        self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
    ) -> np.ndarray:
        """Overlay Grad-CAM heatmap on the original image."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        if heatmap_color.shape[:2] != image.shape[:2]:
            heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))

        return cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)


# ════════════════════════════════════════════════════════════
#  2. ATTENTION + GRAD-CAM COMBINED
# ════════════════════════════════════════════════════════════
class AttentionGradCAM:
    """Combines Grad-CAM with attention-weight visualisation."""

    def __init__(self, model: tf.keras.Model, patch_extractor: PatchExtractor) -> None:
        self.model = model
        self.pe = patch_extractor

        try:
            self.gradcam = GradCAM(model)
        except Exception as e:
            print(f"  ⚠️  Grad-CAM init warning: {e}")
            self.gradcam = None

    def visualize_prediction(
        self,
        image_path: str,
        save_path: str,
        true_label: Optional[str] = None,
        num_patches: int = 4,
    ) -> None:
        """Generate a 4-panel explanation figure for one image."""
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            patches, orig_shape = self.pe.extract_from_file(image_path)
        except Exception as e:
            print(f"  ⚠️  Cannot process {image_path}: {e}")
            return

        orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        while len(patches) < num_patches:
            patches.append(patches[len(patches) % len(patches)])
        patches = patches[:num_patches]

        # Full model takes raw patches (1, N, 224, 224, 3)
        patch_array = np.stack(patches, axis=0)[np.newaxis, ...]
        output = self.model(patch_array.astype(np.float32), training=False)
        preds = output["predictions"].numpy()[0]
        attn_weights = output["attention_weights"].numpy()[0].flatten()

        pred_class = "oscc" if np.argmax(preds) == 1 else "normal"
        confidence = float(np.max(preds))
        risk = get_risk_level(float(preds[1]))

        # Grad-CAM on highest-attention patch
        best_patch_idx = int(np.argmax(attn_weights))
        best_patch = patches[best_patch_idx]

        if self.gradcam is not None:
            try:
                heatmap = self.gradcam.compute_gradcam(best_patch, class_idx=1)
                overlay = self.gradcam.generate_heatmap_overlay(best_patch, heatmap)
            except Exception:
                heatmap = np.zeros((224, 224), dtype=np.float32)
                overlay = (best_patch * 255).astype(np.uint8)
        else:
            heatmap = np.zeros((224, 224), dtype=np.float32)
            overlay = (best_patch * 255).astype(np.uint8)

        # Attention map over patches
        positions = self.pe.get_patch_positions(orig_shape)
        ps = self.pe.patch_size
        work_h = max(orig_shape[0], self.pe.min_image_size)
        work_w = max(orig_shape[1], self.pe.min_image_size)
        attn_canvas = np.zeros((work_h, work_w), dtype=np.float64)

        w_norm = attn_weights / (attn_weights.max() + 1e-8)
        for i, (_, _, y, x) in enumerate(positions):
            if i < len(w_norm):
                attn_canvas[y: y + ps, x: x + ps] = w_norm[i]

        attn_canvas = cv2.resize(attn_canvas, (orig_shape[1], orig_shape[0]))
        attn_color = cv2.applyColorMap(
            (attn_canvas * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        attn_color = cv2.cvtColor(attn_color, cv2.COLOR_BGR2RGB)

        orig_display = orig_img.copy()
        if orig_display.max() <= 1.0:
            orig_display = (orig_display * 255).astype(np.uint8)
        elif orig_display.dtype != np.uint8:
            orig_display = orig_display.astype(np.uint8)

        full_overlay = cv2.addWeighted(orig_display, 0.6, attn_color, 0.4, 0)

        # Build figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#1a1a2e")
        panels = [
            (orig_display, "Original",                         "#e2e8f0"),
            (attn_color,   "Attention Map",                    "#fbbf24"),
            (overlay,      f"Grad-CAM (Patch {best_patch_idx})", "#f87171"),
            (full_overlay, "Full Overlay",                     "#34d399"),
        ]

        for ax, (img, subtitle, color) in zip(axes, panels):
            ax.imshow(img)
            ax.set_title(subtitle, fontsize=11, color=color, fontweight="bold")
            ax.axis("off")

        true_str = f"  |  True: {true_label}" if true_label else ""
        fig.suptitle(
            f"Predicted: {pred_class.upper()} | "
            f"Confidence: {confidence:.1%} | "
            f"Risk: {risk}{true_str}",
            fontsize=13, color="white", fontweight="bold", y=1.02,
        )

        plt.tight_layout()
        plt.savefig(save_file, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"    💾 {save_file.name}")


# ════════════════════════════════════════════════════════════
#  3. BATCH GENERATION
# ════════════════════════════════════════════════════════════
def generate_explanations(
    model: tf.keras.Model,
    data_dir: str = "data/test/",
    patch_extractor: Optional[PatchExtractor] = None,
    n_samples: int = 5,
    save_dir: str = "outputs/heatmaps/",
) -> None:
    if patch_extractor is None:
        patch_extractor = PatchExtractor(patch_size=224, min_image_size=448)

    data_root = Path(data_dir) if Path(data_dir).is_absolute() else PROJECT_ROOT / data_dir
    save_root = Path(save_dir) if Path(save_dir).is_absolute() else PROJECT_ROOT / save_dir
    save_root.mkdir(parents=True, exist_ok=True)

    ag = AttentionGradCAM(model, patch_extractor)
    total = 0

    for cls_name in ["normal", "oscc"]:
        cls_dir = data_root / cls_name
        if not cls_dir.exists():
            parent = data_root.parent / "test" / cls_name
            cls_dir = parent if parent.exists() else cls_dir
        if not cls_dir.exists():
            print(f"  ⚠️  {cls_dir} not found, skipping.")
            continue

        images = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        random.shuffle(images)

        print(f"\n  📸 Generating {min(len(images), n_samples)} explanations for {cls_name} …")
        for img_path in tqdm(images[:n_samples], desc=f"    {cls_name}", unit="img"):
            save_name = f"{img_path.stem}_{cls_name}_explanation.png"
            ag.visualize_prediction(str(img_path), str(save_root / save_name), true_label=cls_name)
            total += 1

    print(f"\n  ✅ Generated {total} explanation visualisations → {save_root}")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🔬  Grad-CAM Explainability — Runner")
    print("=" * 55 + "\n")

    model = load_full_model_for_gradcam()

    extractor = PatchExtractor(patch_size=224, min_image_size=448)

    test_dir = str(PROJECT_ROOT / "data" / "test")
    test_images: List[str] = []
    for cls in ["oscc", "normal"]:
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            found = glob.glob(f"{test_dir}/{cls}/{ext}")
            test_images.extend(found[:3])

    if test_images:
        print(f"  Found {len(test_images)} test images.\n")
        ag = AttentionGradCAM(model, extractor)
        for img_path in test_images:
            label = "oscc" if "oscc" in img_path else "normal"
            save_name = Path(img_path).stem + "_explanation.png"
            save_path = str(PROJECT_ROOT / "outputs" / "heatmaps" / save_name)
            ag.visualize_prediction(img_path, save_path, true_label=label)

        print(f"\n  ✅ Saved {len(test_images)} explanation visualisations.")
    else:
        print("  ❌ No test images found.")