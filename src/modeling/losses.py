def categorical_focal_loss(gamma: float = 2.0, alpha: float = 1.0):
    import tensorflow as tf
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0)
        return tf.reduce_sum(-alpha * y_true * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred), axis=-1)
    return loss

