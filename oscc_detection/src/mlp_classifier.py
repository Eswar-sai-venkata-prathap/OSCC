#!/usr/bin/env python3
"""
OSCC Detection Project — MLP Classifier
Final classification head: Dense → BN → Dropout → softmax
"""

import random
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf

SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES: Dict[int, str] = {0: "normal", 1: "oscc"}


class OSCCClassifier(tf.keras.layers.Layer):
    """
    MLP classification head.
    Dense(hidden1, relu) → BN → Dropout
    Dense(hidden2, relu) → BN → Dropout
    Dense(num_classes, softmax)
    """

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.4,
        l2_reg: float = 0.001,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if hidden_dims is None:
            hidden_dims = [256, 128]   # smaller = faster on CPU

        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        reg = tf.keras.regularizers.l2(l2_reg)

        self.dense1 = tf.keras.layers.Dense(
            hidden_dims[0], activation="relu",
            kernel_regularizer=reg, name="cls_dense1")
        self.bn1 = tf.keras.layers.BatchNormalization(name="cls_bn1")
        self.drop1 = tf.keras.layers.Dropout(dropout_rate, name="cls_drop1")

        self.dense2 = tf.keras.layers.Dense(
            hidden_dims[1], activation="relu",
            kernel_regularizer=reg, name="cls_dense2")
        self.bn2 = tf.keras.layers.BatchNormalization(name="cls_bn2")
        self.drop2 = tf.keras.layers.Dropout(
            max(dropout_rate - 0.1, 0.1), name="cls_drop2")

        self.output_layer = tf.keras.layers.Dense(
            num_classes, activation="softmax", name="cls_output")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)
        return self.output_layer(x)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg,
        })
        return config


def get_risk_level(probability: float) -> str:
    if probability < 0.4:
        return "Low Risk"
    elif probability < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"


def predict_with_risk(model_output: np.ndarray) -> List[Dict[str, Any]]:
    results = []
    for probs in model_output:
        cls_idx = int(np.argmax(probs))
        oscc_prob = float(probs[1])
        results.append({
            "predicted_class": CLASS_NAMES[cls_idx],
            "confidence": float(np.max(probs)),
            "oscc_probability": oscc_prob,
            "risk_level": get_risk_level(oscc_prob),
        })
    return results


if __name__ == "__main__":
    classifier = OSCCClassifier(num_classes=2)
    dummy_input = np.random.rand(4, 512).astype(np.float32)
    probs = classifier(dummy_input, training=False)
    print(f"Output shape: {probs.shape}")
    print(f"Prob sums:    {probs.numpy().sum(axis=1)}")
    print("✅ OSCCClassifier test passed.")