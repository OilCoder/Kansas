"""Implements custom TensorFlow metrics for masked classification and regression evaluation. Provides specialized loss functions and accuracy measures that handle unknown class indices for geological formation prediction tasks."""

# IMPORTANT: Initialize GPU environment BEFORE importing TensorFlow
import src.utils.initialize_gpu

import tensorflow as tf
from tensorflow.keras.metrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    RootMeanSquaredError,
    SparseCategoricalAccuracy
)
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K

# ------------------------------------------------------------------------------
# Definimos un unknown_index global que debe ser coherente con el valor obtenido
# en la normalización (por ejemplo, si las clases conocidas tienen valores de 0 a 11,
# unknown_index debe ser 12). Asegúrate de actualizar este valor si es necesario.
# ------------------------------------------------------------------------------
# UNKNOWN_INDEX = 12

# ------------------------------
# Loss personalizada para ignorar UNKNOWN
# ------------------------------
def masked_sparse_categorical_crossentropy(y_true, y_pred, unknown_index):
    # Utilizamos el unknown_index global en lugar de calcularlo dinámicamente.
    mask = tf.not_equal(y_true, unknown_index)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.where(mask, loss, tf.zeros_like(loss))

def create_masked_sparse_categorical_crossentropy(unknown_index):
    """
    Factory function to create a masked sparse categorical crossentropy loss
    with the unknown_index parameter bound.
    """
    def loss_fn(y_true, y_pred):
        return masked_sparse_categorical_crossentropy(y_true, y_pred, unknown_index)
    
    # Set the function name for better debugging
    loss_fn.__name__ = f'masked_sparse_categorical_crossentropy_unknown_{unknown_index}'
    return loss_fn

# Actualizamos los custom objects usando la función con su nombre original.
get_custom_objects().update({
    "masked_sparse_categorical_crossentropy": masked_sparse_categorical_crossentropy
})

# ------------------------------
# Métricas personalizadas
# ------------------------------
class MaskedSparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, unknown_index, name='masked_sparse_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.unknown_index = unknown_index
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_pred = tf.reshape(y_pred, [-1])
        mask = tf.not_equal(y_true, self.unknown_index)
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        # Opcional: para depuración, puedes habilitar:
        # tf.print("MaskedSparseCategoricalAccuracy - y_true_masked:", y_true_masked, " y_pred_masked:", y_pred_masked)
        matches = tf.cast(tf.equal(y_true_masked, y_pred_masked), tf.float32)
        self.total.assign_add(tf.reduce_sum(matches))
        self.count.assign_add(tf.cast(tf.size(matches), tf.float32))

    def result(self):
        return self.total / (self.count + K.epsilon())

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

class MaskedTopKAccuracy(tf.keras.metrics.Metric):
    def __init__(self, unknown_index, k=3, name='masked_top_k_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.unknown_index = unknown_index
        self.k = k
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        mask = tf.not_equal(y_true, self.unknown_index)
        mask = tf.reshape(mask, [-1])
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.cast(tf.boolean_mask(y_pred, mask, axis=0), tf.float32)
        top_k = tf.keras.metrics.top_k_categorical_accuracy(
            tf.one_hot(y_true_masked, depth=tf.shape(y_pred)[-1]), y_pred_masked, k=self.k
        )
        # Opcional: para depuración
        # tf.print("MaskedTopKAccuracy - top_k:", top_k)
        self.total.assign_add(tf.reduce_sum(top_k))
        self.count.assign_add(tf.cast(tf.size(top_k), tf.float32))

    def result(self):
        return self.total / (self.count + K.epsilon())

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

# ------------------------------
# Métricas listas para usar
# ------------------------------
def get_regression_metrics():
    return [
        MeanAbsoluteError(name='mae'),
        MeanSquaredError(name='mse'),
        RootMeanSquaredError(name='rmse')
    ]

def get_classification_metrics(num_classes, unknown_index):
    return [
        SparseCategoricalAccuracy(name='sparse_acc'),
        MaskedSparseCategoricalAccuracy(unknown_index=unknown_index, name='masked_sparse_acc'),
        MaskedTopKAccuracy(unknown_index=unknown_index, k=3, name='masked_top_k_acc')
    ]
