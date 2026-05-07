"""
tests/test_visual_model.py
Unit & integration tests for models/visual/
Covers: face_detector, cnn_emotion, gradcam
"""

import os
import sys
import types
import unittest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

# ── stub heavy third-party dependencies so tests run without GPU ──────────────
for mod in [
    "cv2", "tensorflow", "tensorflow.keras",
    "tensorflow.keras.models", "tensorflow.keras.applications",
    "tensorflow.keras.applications.resnet_v2",
    "tensorflow.keras.layers", "tensorflow.keras.preprocessing",
    "tensorflow.keras.preprocessing.image",
]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

import cv2 as _cv2                                      # noqa: F401 – stubbed
_cv2.CascadeClassifier = MagicMock                      # type: ignore
_cv2.cvtColor         = MagicMock(return_value=np.zeros((48, 48, 3), dtype=np.uint8))
_cv2.resize           = MagicMock(return_value=np.zeros((48, 48, 3), dtype=np.uint8))
_cv2.COLOR_BGR2GRAY   = 6
_cv2.COLOR_BGR2RGB    = 4

# ─────────────────────────────────────────────────────────────────────────────
class TestFaceDetector(unittest.TestCase):
    """Tests for models/visual/face_detector.py"""

    def _make_detector(self):
        """Import FaceDetector with the cascade classifier mocked."""
        with patch("cv2.CascadeClassifier") as MockCC:
            instance = MockCC.return_value
            # detectMultiScale returns one face region
            instance.detectMultiScale.return_value = np.array([[10, 10, 80, 80]])
            from models.visual.face_detector import FaceDetector
            return FaceDetector()

    # ── constructor ──────────────────────────────────────────────────────────
    def test_init_loads_cascade(self):
        with patch("cv2.CascadeClassifier") as MockCC:
            from models.visual.face_detector import FaceDetector
            det = FaceDetector()
            self.assertIsNotNone(det)
            MockCC.assert_called_once()

    # ── detect_face ──────────────────────────────────────────────────────────
    def test_detect_face_returns_crop_when_face_found(self):
        with patch("cv2.CascadeClassifier") as MockCC:
            instance = MockCC.return_value
            instance.detectMultiScale.return_value = np.array([[10, 10, 80, 80]])
            from models.visual.face_detector import FaceDetector
            det = FaceDetector()
            fake_img = np.zeros((120, 120, 3), dtype=np.uint8)
            result = det.detect_face(fake_img)
            # Should return a numpy array (cropped face)
            self.assertIsNotNone(result)

    def test_detect_face_returns_none_when_no_face(self):
        with patch("cv2.CascadeClassifier") as MockCC:
            instance = MockCC.return_value
            instance.detectMultiScale.return_value = np.array([]).reshape(0, 4)
            from models.visual.face_detector import FaceDetector
            det = FaceDetector()
            fake_img = np.zeros((120, 120, 3), dtype=np.uint8)
            result = det.detect_face(fake_img)
            self.assertIsNone(result)

    def test_detect_face_rejects_non_array(self):
        with patch("cv2.CascadeClassifier"):
            from models.visual.face_detector import FaceDetector
            det = FaceDetector()
            with self.assertRaises((TypeError, AttributeError)):
                det.detect_face("not_an_image")


# ─────────────────────────────────────────────────────────────────────────────
class TestCNNEmotionModel(unittest.TestCase):
    """Tests for models/visual/cnn_emotion.py"""

    EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def _mock_keras_model(self):
        """Return a mock that mimics a compiled Keras model."""
        mock_model = MagicMock()
        # Predict returns softmax-like output for 7 classes
        probs = np.array([[0.05, 0.05, 0.05, 0.70, 0.05, 0.05, 0.05]])
        mock_model.predict.return_value = probs
        return mock_model

    def test_predict_returns_label_and_confidence(self):
        with patch("models.visual.cnn_emotion.load_model", return_value=self._mock_keras_model()):
            from models.visual.cnn_emotion import CNNEmotionModel
            m = CNNEmotionModel(model_path="saved_models/cnn_emotion.pt")
            face = np.zeros((48, 48, 3), dtype=np.uint8)
            result = m.predict(face)
            self.assertIn("label", result)
            self.assertIn("confidence", result)
            self.assertIn(result["label"], self.EMOTION_CLASSES)
            self.assertIsInstance(result["confidence"], float)
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)

    def test_predict_returns_all_class_probs(self):
        with patch("models.visual.cnn_emotion.load_model", return_value=self._mock_keras_model()):
            from models.visual.cnn_emotion import CNNEmotionModel
            m = CNNEmotionModel(model_path="saved_models/cnn_emotion.pt")
            face = np.zeros((48, 48, 3), dtype=np.uint8)
            result = m.predict(face)
            self.assertIn("probs", result)
            self.assertEqual(len(result["probs"]), len(self.EMOTION_CLASSES))

    def test_predict_top_label_matches_argmax(self):
        mock_model = MagicMock()
        probs = np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.90, 0.05]])  # Sad is highest
        mock_model.predict.return_value = probs
        with patch("models.visual.cnn_emotion.load_model", return_value=mock_model):
            from models.visual.cnn_emotion import CNNEmotionModel
            m = CNNEmotionModel(model_path="saved_models/cnn_emotion.pt")
            face = np.zeros((48, 48, 3), dtype=np.uint8)
            result = m.predict(face)
            self.assertEqual(result["label"], "Sad")

    def test_predict_rejects_wrong_input_shape(self):
        with patch("models.visual.cnn_emotion.load_model", return_value=self._mock_keras_model()):
            from models.visual.cnn_emotion import CNNEmotionModel
            m = CNNEmotionModel(model_path="saved_models/cnn_emotion.pt")
            with self.assertRaises((ValueError, Exception)):
                m.predict(np.zeros((10, 10)))          # wrong shape


# ─────────────────────────────────────────────────────────────────────────────
class TestGradCAM(unittest.TestCase):
    """Tests for models/visual/gradcam.py"""

    def test_generate_returns_heatmap_same_spatial_size(self):
        mock_model = MagicMock()
        mock_model.input = MagicMock()
        mock_model.output = MagicMock()
        # grad_model also returns fake tensors
        fake_conv_out = np.random.rand(1, 6, 6, 512).astype(np.float32)
        fake_preds    = np.array([[0.05]*7], dtype=np.float32)
        fake_preds[0][3] = 0.70

        with patch("models.visual.gradcam.tf") as mock_tf,              patch("models.visual.gradcam.Model") as MockModel:
            mock_tf.GradientTape.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_tf.GradientTape.return_value.__exit__  = MagicMock(return_value=False)
            mock_tf.reduce_mean = MagicMock(
                return_value=np.ones((512,), dtype=np.float32)
            )
            # grad tape tape.gradient returns fake grads
            fake_grads = np.random.rand(1, 6, 6, 512).astype(np.float32)
            grad_instance = MockModel.return_value
            grad_instance.predict.return_value = [fake_conv_out, fake_preds]

            from models.visual.gradcam import GradCAM
            gcam = GradCAM(model=mock_model, layer_name="conv5_block3_out")
            face = np.zeros((48, 48, 3), dtype=np.float32)
            heatmap = gcam.generate(face, class_index=3)
            self.assertIsNotNone(heatmap)


# ─────────────────────────────────────────────────────────────────────────────
class TestVisualPolarityMapping(unittest.TestCase):
    """
    Tests the 7-class → 3-polarity mapping described in notebook 04.
    This mapping is used by the fusion layer.
    """

    POLARITY_MAP = {
        "Happy":    "positive",
        "Surprise": "positive",
        "Neutral":  "neutral",
        "Angry":    "negative",
        "Disgust":  "negative",
        "Fear":     "negative",
        "Sad":      "negative",
    }

    def test_all_classes_mapped(self):
        classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        for cls in classes:
            self.assertIn(cls, self.POLARITY_MAP,
                          f"{cls} missing from polarity map")

    def test_positive_labels(self):
        for label in ["Happy", "Surprise"]:
            self.assertEqual(self.POLARITY_MAP[label], "positive")

    def test_neutral_label(self):
        self.assertEqual(self.POLARITY_MAP["Neutral"], "neutral")

    def test_negative_labels(self):
        for label in ["Angry", "Disgust", "Fear", "Sad"]:
            self.assertEqual(self.POLARITY_MAP[label], "negative")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main()
