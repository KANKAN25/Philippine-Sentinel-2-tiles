"""Shared model definitions for the experiments."""
import keras
from keras import layers


def small_cnn(input_shape, n_classes, dropout=0.3):
    """Small CNN baseline (E1).

    Expects raw pixel values in 0-255 (float32); rescaling to 0-1 happens inside
    the model so the same preprocessing travels with the saved .keras file.
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="small_cnn")
