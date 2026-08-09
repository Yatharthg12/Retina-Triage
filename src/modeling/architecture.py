from __future__ import annotations

def build_model(image_size: int = 224, weights: str | None = "imagenet"):
    import tensorflow as tf
    inputs = tf.keras.Input((image_size, image_size, 3), name="retina")
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=(image_size, image_size, 3)
    )
    backbone.trainable = False
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dropout(0.40, name="dropout_1")(x)
    x = tf.keras.layers.Dense(256, activation="swish", name="feature_dense")(x)
    x = tf.keras.layers.Dropout(0.30, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax", name="severity")(x)
    return tf.keras.Model(inputs, outputs, name="retina_triage_efficientnetb0")
