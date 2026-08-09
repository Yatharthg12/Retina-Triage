from __future__ import annotations

import numpy as np

def qwk_callback(validation_data):
    import tensorflow as tf
    from sklearn.metrics import cohen_kappa_score
    class QWKCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            y_true, y_pred = [], []
            for images, labels in validation_data:
                y_true.extend(np.argmax(labels.numpy(), axis=1))
                y_pred.extend(np.argmax(self.model.predict(images, verbose=0), axis=1))
            (logs or {})["val_qwk"] = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return QWKCallback()

