#!/usr/bin/env python3
"""
============================================================
OSCC Detection Project — Evaluation Script
============================================================
Microscope-Aided Deep Learning Framework for Early Oral
Cancer (OSCC) Detection
============================================================
"""

# ── standard library ──────────────────────────────────────
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# ── third-party ───────────────────────────────────────────
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── local imports ─────────────────────────────────────────
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from model import build_lite_model
from train import FeatureSequenceGenerator

# ── reproducibility ───────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── project root ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES = ["normal", "oscc"]


# ==========================================================
# LOAD MODEL
# ==========================================================
def load_model_for_eval(
    model_path="outputs/models/best_model.weights.h5",
    config_path="outputs/models/model_config.json",
):
    weights_file = Path(model_path)
    if not weights_file.is_absolute():
        weights_file = PROJECT_ROOT / model_path

    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_path

    # Load config
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)
        print(f"📝 Config loaded from {config_file}")
    else:
        print("⚠️ Config not found, using defaults")
        config = None

    # Build model
    model = build_lite_model(config)

    # Build model graph before loading weights
    dummy_input = tf.zeros((1, 4, 1280))
    _ = model(dummy_input, training=False)
    print("✅ Model graph initialized")

    # Load weights  ← FIX: removed by_name=True fallback (incompatible with .weights.h5)
    if weights_file.exists():
        model.load_weights(str(weights_file))
        print(f"✅ Weights loaded from {weights_file}")
    else:
        print("⚠️ Weights file not found")

    return model


# ==========================================================
# EVALUATION
# ==========================================================
def evaluate_model(model, test_gen, class_names=None, save_dir="outputs"):

    if class_names is None:
        class_names = CLASS_NAMES

    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    y_true = []
    y_pred = []
    y_scores = []

    print("⏳ Running inference...")

    for i in tqdm(range(len(test_gen))):
        x_batch, y_batch = test_gen[i]

        outputs = model(x_batch, training=False)
        preds = outputs["predictions"].numpy()

        true_labels = np.argmax(y_batch, axis=1)
        pred_labels = np.argmax(preds, axis=1)

        y_true.extend(true_labels)
        y_pred.extend(pred_labels)
        y_scores.extend(preds[:, 1])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    cm = confusion_matrix(y_true, y_pred)

    print("\n========== RESULTS ==========")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("AUC:", auc_score)

    plot_confusion_matrix(cm, class_names, save_root / "plots/confusion_matrix.png")
    plot_roc_curve(y_true, y_scores, save_root / "plots/roc_curve.png")

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc_score,
    }

    save_metrics_report(metrics, save_root / "metrics.csv")

    return metrics


# ==========================================================
# CONFUSION MATRIX
# ==========================================================
def plot_confusion_matrix(cm, class_names, save_path):

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print("Saved confusion matrix:", save_path)


# ==========================================================
# ROC CURVE
# ==========================================================
def plot_roc_curve(y_true, y_scores, save_path):

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print("Saved ROC curve:", save_path)


# ==========================================================
# SAVE METRICS
# ==========================================================
def save_metrics_report(metrics, save_path):

    df = pd.DataFrame([metrics])
    df.to_csv(save_path, index=False)

    print("Saved metrics:", save_path)


# ==========================================================
# RUN PIPELINE
# ==========================================================
def run_evaluation():

    print("\n========== OSCC Evaluation ==========\n")

    model = load_model_for_eval()

    features_path = str(PROJECT_ROOT / "data/features")

    test_gen = FeatureSequenceGenerator(
        features_path,
        "test",
        num_patches=4,
        feature_dim=1280,
        batch_size=64,
        augment=False,
    )

    print("Test batches:", len(test_gen))

    evaluate_model(model, test_gen)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    run_evaluation()