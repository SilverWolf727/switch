# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of TSMixer (fixed for TF2.x / Keras)."""

import tensorflow as tf
from tensorflow.keras import layers


def res_block(inputs, norm_type, activation, dropout, ff_dim):
    """Residual block of TSMixer."""

    # choose normalization
    if norm_type == 'L':
        norm_layer = layers.LayerNormalization
        norm_args = {'axis': [-2, -1]}
    else:
        norm_layer = layers.BatchNormalization
        norm_args = {'axis': -1}

    # --- Temporal Linear ---
    x = norm_layer(**norm_args)(inputs)
    x = layers.Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1]))(x)  # [B, C, L]
    x = layers.Dense(x.shape[-1], activation=activation)(x)
    x = layers.Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1]))(x)  # [B, L, C]
    x = layers.Dropout(dropout)(x)
    res = layers.Add()([x, inputs])

    # --- Feature Linear ---
    x = norm_layer(**norm_args)(res)
    x = layers.Dense(ff_dim, activation=activation)(x)  # [B, L, FF]
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)  # [B, L, C]
    x = layers.Dropout(dropout)(x)

    return layers.Add()([x, res])


def build_model(
    input_shape,
    pred_len,
    norm_type,
    activation,
    n_block,
    dropout,
    ff_dim,
    target_slice,
):
    """Build TSMixer model."""

    inputs = tf.keras.Input(shape=input_shape)  # [B, L, C]
    x = inputs

    for _ in range(n_block):
        x = res_block(x, norm_type, activation, dropout, ff_dim)

    if target_slice:
        # use Lambda to ensure graph compatibility
        x = layers.Lambda(lambda t: t[:, :, target_slice])(x)

    x = layers.Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1]))(x)  # [B, C, L]
    x = layers.Dense(pred_len)(x)  # [B, C, pred_len]
    outputs = layers.Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1]))(x)  # [B, pred_len, C]

    return tf.keras.Model(inputs, outputs)
