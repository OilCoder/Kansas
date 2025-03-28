import tensorflow as tf
from tensorflow.keras.metrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    RootMeanSquaredError,
    SparseCategoricalAccuracy
)
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K

# ------------------------------
# Loss personalizada para ignorar UNKNOWN
# ------------------------------
def masked_sparse_categorical_crossentropy(y_true, y_pred):
    unknown_index = tf.reduce_max(y_true) + 1
    mask = tf.not_equal(y_true, unknown_index)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.where(mask, loss, tf.zeros_like(loss))

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
