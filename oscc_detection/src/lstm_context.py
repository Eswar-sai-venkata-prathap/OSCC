#!/usr/bin/env python3
"""
OSCC Detection Project — LSTM / GRU Contextual Learning
Optimized for CPU: single-layer GRU (no bidirectional).
"""

import random
from typing import Any, Dict

import numpy as np
import tensorflow as tf

SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class PositionalEncoding(tf.keras.layers.Layer):
    """Sinusoidal positional encoding for patch sequences."""

    def __init__(self, max_patches: int = 64, feature_dim: int = 1280, **kwargs):
        super().__init__(**kwargs)
        self.max_patches = max_patches
        self.feature_dim = feature_dim
        self.pos_encoding = self._build_encoding(max_patches, feature_dim)

    @staticmethod
    def _build_encoding(max_len: int, d_model: int) -> tf.Tensor:
        positions = np.arange(max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000.0, (2 * (dims // 2)) / np.float32(d_model))
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        return tf.constant(angles[np.newaxis, :, :], dtype=tf.float32)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :self.feature_dim]

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"max_patches": self.max_patches, "feature_dim": self.feature_dim})
        return config


class ContextualLearner(tf.keras.layers.Layer):
    """
    Stacked RNN for sequential patch context.
    Default: single-layer GRU (fastest on CPU).
    Set bidirectional=True / num_layers=2 for more accuracy (slower).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        rnn_type: str = "gru",
        dropout_rate: float = 0.3,
        bidirectional: bool = False,   # False = 2x faster on CPU
        num_layers: int = 1,           # 1 layer = faster
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        RNNClass = tf.keras.layers.GRU if self.rnn_type == "gru" else tf.keras.layers.LSTM

        self.rnn_layers = []
        self.norm_layers = []
        self.drop_layers = []

        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)
            rnn = RNNClass(
                units=hidden_dim,
                return_sequences=return_sequences,
                name=f"{self.rnn_type}_{i}",
            )
            if bidirectional:
                rnn = tf.keras.layers.Bidirectional(rnn, name=f"bi_{self.rnn_type}_{i}")

            self.rnn_layers.append(rnn)
            self.norm_layers.append(tf.keras.layers.LayerNormalization(name=f"ln_{i}"))
            self.drop_layers.append(tf.keras.layers.Dropout(dropout_rate, name=f"drop_{i}"))

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = inputs
        for rnn, norm, drop in zip(self.rnn_layers, self.norm_layers, self.drop_layers):
            x = rnn(x, training=training)
            x = norm(x, training=training)
            x = drop(x, training=training)
        return x

    @property
    def output_dim(self) -> int:
        return self.hidden_dim * (2 if self.bidirectional else 1)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "rnn_type": self.rnn_type,
            "dropout_rate": self.dropout_rate,
            "bidirectional": self.bidirectional,
            "num_layers": self.num_layers,
        })
        return config


def build_context_module(
    rnn_type: str = "gru",
    hidden_dim: int = 128,
    dropout_rate: float = 0.3,
    bidirectional: bool = False,
    num_layers: int = 1,
) -> ContextualLearner:
    return ContextualLearner(
        hidden_dim=hidden_dim,
        rnn_type=rnn_type,
        dropout_rate=dropout_rate,
        bidirectional=bidirectional,
        num_layers=num_layers,
        name=f"{rnn_type}_context",
    )


if __name__ == "__main__":
    dummy = np.random.rand(2, 4, 1280).astype(np.float32)

    pe = PositionalEncoding(max_patches=64, feature_dim=1280)
    enc = pe(dummy)
    print(f"PositionalEncoding output: {enc.shape} ✅")

    for rnn_type in ["gru", "lstm"]:
        learner = ContextualLearner(hidden_dim=128, rnn_type=rnn_type, num_layers=1)
        out = learner(dummy, training=False)
        print(f"{rnn_type.upper()} output: {out.shape} ✅")