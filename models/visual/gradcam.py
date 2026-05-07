"""
models/visual/gradcam.py
Grad-CAM attention overlay for the ResNet50V2 emotion classifier.
Used by app/utils/visualizer.py to generate saliency heatmaps.
"""
import numpy as np
import tensorflow as tf
import cv2


def get_gradcam_heatmap(
    model: tf.keras.Model,
    image: np.ndarray,
    pred_index: int | None = None,
    last_conv_layer_name: str = "conv5_block3_2_conv",
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for the given image.

    Parameters
    ----------
    model               : compiled Keras model (ResNet50V2)
    image               : float32 ndarray (224, 224, 3) in [0, 1]
    pred_index          : class index to explain; defaults to argmax
    last_conv_layer_name: name of the final convolutional layer in ResNet50V2

    Returns
    -------
    heatmap : float32 ndarray (H, W) in [0, 1], upsampled to image spatial size
    """
    # Build a model that outputs (conv activations, final logits)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    img_array = np.expand_dims(image, axis=0)  # (1, 224, 224, 3)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)         # (1, h, w, c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))       # (c,)

    conv_outputs = conv_outputs[0]                             # (h, w, c)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]     # (h, w, 1)
    heatmap = tf.squeeze(heatmap)                              # (h, w)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Upsample to original image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    return heatmap.astype(np.float32)


def overlay_gradcam(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Superimpose a Grad-CAM heatmap onto the original image.

    Parameters
    ----------
    image   : uint8 RGB ndarray (H, W, 3)
    heatmap : float32 ndarray   (H, W)  in [0, 1]
    alpha   : blend weight for the heatmap overlay

    Returns
    -------
    overlaid uint8 RGB ndarray (H, W, 3)
    """
    heat_uint8 = np.uint8(255 * heatmap)
    heat_coloured = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_coloured = cv2.cvtColor(heat_coloured, cv2.COLOR_BGR2RGB)

    if image.dtype != np.uint8:
        base = np.uint8(image * 255)
    else:
        base = image.copy()

    overlaid = cv2.addWeighted(base, 1 - alpha, heat_coloured, alpha, 0)
    return overlaid
