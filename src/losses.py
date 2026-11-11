"""Losses and a water-IoU metric that ignore no-data (-1) label pixels.

Shapes: y_true and y_pred are (batch, H, W, 1). y_true holds -1 / 0 / 1; y_pred is the
sigmoid probability of water. Everything is computed in float32, also under mixed precision.

    bce_loss      binary cross-entropy averaged over valid pixels
    dice_loss     1 - (2*sum(p*y) + eps) / (sum(p) + sum(y) + eps), sums over valid pixels of the batch
    bce_dice_loss BCE + Dice
    WaterIoU      Keras metric, TP / (TP + FP + FN) accumulated over the epoch (for EarlyStopping)
"""
import tensorflow as tf

NO_DATA = -1
EPS = 1e-6


def _prepare(y_true, y_pred):
    """mask = (label != -1); then the -1 labels are set to 0 so they cannot leak into any sum."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(tf.not_equal(y_true, NO_DATA), tf.float32)
    y_true = y_true * mask  # -1 -> 0 (0 and 1 stay as they are)
    return y_true, y_pred, mask


def bce_loss(y_true, y_pred):
    y, p, mask = _prepare(y_true, y_pred)
    p = tf.clip_by_value(p, EPS, 1.0 - EPS)
    bce = -(y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
    return tf.reduce_sum(bce * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def dice_loss(y_true, y_pred):
    y, p, mask = _prepare(y_true, y_pred)
    p = p * mask  # predictions on no-data pixels do not count
    intersection = tf.reduce_sum(p * y)
    return 1.0 - (2.0 * intersection + EPS) / (tf.reduce_sum(p) + tf.reduce_sum(y) + EPS)


def bce_dice_loss(y_true, y_pred):
    return bce_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


class WaterIoU(tf.keras.metrics.Metric):
    """Water IoU over valid pixels, with TP/FP/FN summed over all batches before dividing once."""

    def __init__(self, threshold=0.5, name="water_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_variable(shape=(), initializer="zeros", name="tp")
        self.fp = self.add_variable(shape=(), initializer="zeros", name="fp")
        self.fn = self.add_variable(shape=(), initializer="zeros", name="fn")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y, p, mask = _prepare(y_true, y_pred)
        pred = tf.cast(p >= self.threshold, tf.float32) * mask
        self.tp.assign_add(tf.reduce_sum(pred * y))
        self.fp.assign_add(tf.reduce_sum(pred * (1.0 - y)))
        self.fn.assign_add(tf.reduce_sum((1.0 - pred) * y * mask))

    def result(self):
        return tf.math.divide_no_nan(self.tp, self.tp + self.fp + self.fn)

    def reset_state(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(0.0)

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}
