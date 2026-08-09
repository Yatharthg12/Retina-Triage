from __future__ import annotations

import cv2
import numpy as np

class GradCAMError(RuntimeError):
    pass

def find_last_conv_layer(model):
    layers = getattr(model, "layers", None)
    if not layers:
        raise GradCAMError("The model does not expose a compatible Keras layer graph.")
    for layer in reversed(layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except (AttributeError, TypeError):
            continue
        if hasattr(layer, "layers"):
            for nested in reversed(layer.layers):
                try:
                    if len(nested.output.shape) == 4:
                        return nested.name
                except (AttributeError, TypeError):
                    continue
    raise GradCAMError("No compatible four-dimensional convolutional feature layer was found.")

def make_gradcam_heatmap(model, model_input, class_index=None, layer_name=None):
    import tensorflow as tf
    layer_name = layer_name or find_last_conv_layer(model)
    try:
        layer = model.get_layer(layer_name)
    except ValueError:
        # EfficientNet layers can be exposed directly in a Functional model; report cleanly otherwise.
        raise GradCAMError(f"Grad-CAM layer is not accessible: {layer_name}")
    inputs = tf.convert_to_tensor(model_input)
    if hasattr(layer, "layers") and len(layer.output.shape) == 4:
        # Keep a nested backbone explicit and replay the already-trained head so
        # gradients remain connected under Keras 3.
        layer_index = model.layers.index(layer)
        with tf.GradientTape() as tape:
            features = layer(inputs, training=False)
            predictions = features
            for head_layer in model.layers[layer_index + 1:]:
                predictions = head_layer(predictions, training=False)
            index = int(class_index) if class_index is not None else tf.argmax(predictions[0])
            score = predictions[:, index]
    else:
        grad_model = tf.keras.Model(model.inputs, [layer.output, model.output])
        with tf.GradientTape() as tape:
            features, predictions = grad_model(inputs, training=False)
            index = int(class_index) if class_index is not None else tf.argmax(predictions[0])
            score = predictions[:, index]
    grads = tape.gradient(score, features)
    if grads is None:
        raise GradCAMError("Gradients could not be calculated for the selected class.")
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(features[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    return (heatmap / tf.maximum(maximum, 1e-8)).numpy()

def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = .42):
    heat = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
    color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip(rgb * (1 - alpha) + color * alpha, 0, 255))
