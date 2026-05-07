"""
tests/test_fusion.py
Unit & integration tests for models/fusion/
Covers: fusion_layer (weighted-average), mismatch_detector
"""

import unittest
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Helpers – synthetic model outputs
# ─────────────────────────────────────────────────────────────────────────────
def _visual(label, confidence):
    return {"label": label, "confidence": confidence,
            "probs": {l: 0.1 for l in
                      ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]}}


def _text(label, confidence):
    return {"label": label, "confidence": confidence,
            "probs": {l: 0.33 for l in ["negative","neutral","positive"]}}


# ─────────────────────────────────────────────────────────────────────────────
class TestMismatchDetector(unittest.TestCase):
    """
    Tests for models/fusion/mismatch_detector.py
    Canonical polarity map (from notebook 04):
      Happy / Surprise   → positive
      Neutral            → neutral
      Angry/Disgust/Fear/Sad → negative
    Mismatch severity:
      MATCH         — polarities agree
      SOFT_MISMATCH — at least one modality below confidence threshold
      HARD_MISMATCH — both confident AND polarities oppose
    """

    CONF_THRESHOLD = 0.50

    def _detect(self, visual_pred, visual_conf, text_pred, text_conf):
        from models.fusion.mismatch_detector import detect_sentiment_mismatch
        return detect_sentiment_mismatch(
            visual_pred=visual_pred,
            text_pred=text_pred,
            visual_conf=visual_conf,
            text_conf=text_conf,
            conf_threshold=self.CONF_THRESHOLD,
        )

    # ── MATCH cases ──────────────────────────────────────────────────────────
    def test_match_happy_positive(self):
        r = self._detect("Happy", 0.80, "positive", 0.81)
        self.assertEqual(r["severity"], "MATCH")

    def test_match_neutral_neutral(self):
        r = self._detect("Neutral", 0.72, "neutral", 0.75)
        self.assertEqual(r["severity"], "MATCH")

    def test_match_sad_negative(self):
        r = self._detect("Sad", 0.85, "negative", 0.78)
        self.assertEqual(r["severity"], "MATCH")

    # ── HARD_MISMATCH cases ──────────────────────────────────────────────────
    def test_hard_mismatch_sad_positive_both_confident(self):
        """Assignment brief scenario: sad face + positive text (both confident)."""
        r = self._detect("Sad", 0.68, "positive", 0.81)
        self.assertEqual(r["severity"], "HARD_MISMATCH")

    def test_hard_mismatch_fear_neutral_both_confident(self):
        r = self._detect("Fear", 0.55, "neutral", 0.60)
        self.assertEqual(r["severity"], "HARD_MISMATCH")

    def test_hard_mismatch_angry_positive(self):
        r = self._detect("Angry", 0.90, "positive", 0.88)
        self.assertEqual(r["severity"], "HARD_MISMATCH")

    # ── SOFT_MISMATCH cases ──────────────────────────────────────────────────
    def test_soft_mismatch_low_visual_confidence(self):
        """Visual below threshold → SOFT even if polarities oppose."""
        r = self._detect("Sad", 0.35, "positive", 0.82)
        self.assertEqual(r["severity"], "SOFT_MISMATCH")

    def test_soft_mismatch_low_text_confidence(self):
        r = self._detect("Angry", 0.80, "positive", 0.40)
        self.assertEqual(r["severity"], "SOFT_MISMATCH")

    def test_soft_mismatch_both_below_threshold(self):
        r = self._detect("Disgust", 0.30, "positive", 0.45)
        self.assertEqual(r["severity"], "SOFT_MISMATCH")

    # ── output structure ──────────────────────────────────────────────────────
    def test_result_contains_required_keys(self):
        r = self._detect("Happy", 0.80, "positive", 0.85)
        for key in ("severity", "badge_colour", "visual_polarity",
                    "text_pred", "polarity_match", "both_confident"):
            self.assertIn(key, r, f"Missing key: {key}")

    def test_badge_colour_green_for_match(self):
        r = self._detect("Happy", 0.80, "positive", 0.81)
        self.assertEqual(r["badge_colour"], "#4CAF50")

    def test_badge_colour_red_for_hard_mismatch(self):
        r = self._detect("Sad", 0.70, "positive", 0.80)
        self.assertEqual(r["badge_colour"], "#EF5350")

    def test_badge_colour_amber_for_soft_mismatch(self):
        r = self._detect("Sad", 0.35, "positive", 0.82)
        self.assertEqual(r["badge_colour"], "#FFA726")


# ─────────────────────────────────────────────────────────────────────────────
class TestFusionLayer(unittest.TestCase):
    """Tests for models/fusion/fusion_layer.py"""

    def test_fuse_returns_combined_state(self):
        from models.fusion.fusion_layer import fuse
        visual = _visual("Happy", 0.85)
        text   = _text("positive", 0.96)
        result = fuse(visual_output=visual, text_output=text)
        self.assertIn("fused_label", result)
        self.assertIn("fused_confidence", result)
        self.assertIn("mismatch", result)

    def test_fuse_confidence_in_unit_range(self):
        from models.fusion.fusion_layer import fuse
        result = fuse(visual_output=_visual("Sad", 0.70),
                      text_output=_text("negative", 0.80))
        self.assertGreaterEqual(result["fused_confidence"], 0.0)
        self.assertLessEqual(result["fused_confidence"], 1.0)

    def test_fuse_agreement_no_mismatch(self):
        from models.fusion.fusion_layer import fuse
        result = fuse(visual_output=_visual("Happy", 0.90),
                      text_output=_text("positive", 0.95))
        self.assertEqual(result["mismatch"]["severity"], "MATCH")

    def test_fuse_disagreement_detects_hard_mismatch(self):
        from models.fusion.fusion_layer import fuse
        # Sad face (→ negative), positive text — both confident
        result = fuse(visual_output=_visual("Sad", 0.80),
                      text_output=_text("positive", 0.88))
        self.assertEqual(result["mismatch"]["severity"], "HARD_MISMATCH")

    def test_fuse_weighted_average_label(self):
        """
        With equal weights the fused polarity should be driven by the
        higher-confidence modality when they agree.
        """
        from models.fusion.fusion_layer import fuse
        result = fuse(visual_output=_visual("Neutral", 0.90),
                      text_output=_text("neutral", 0.85))
        self.assertIn(result["fused_label"], ["negative", "neutral", "positive"])

    def test_fuse_uses_custom_weights(self):
        """Passing explicit alpha weights must not raise."""
        from models.fusion.fusion_layer import fuse
        result = fuse(visual_output=_visual("Happy", 0.80),
                      text_output=_text("positive", 0.90),
                      visual_weight=0.4,
                      text_weight=0.6)
        self.assertIn("fused_label", result)


# ─────────────────────────────────────────────────────────────────────────────
class TestVisualPolarityBridge(unittest.TestCase):
    """
    Tests for get_visual_polarity() in mismatch_detector.
    Ensures the 7-class → 3-polarity bridge is correct.
    """

    def _get_polarity(self, visual_label: str) -> str:
        from models.fusion.mismatch_detector import get_visual_polarity
        return get_visual_polarity(visual_label)

    def test_happy_maps_to_positive(self):
        self.assertEqual(self._get_polarity("Happy"), "positive")

    def test_surprise_maps_to_positive(self):
        self.assertEqual(self._get_polarity("Surprise"), "positive")

    def test_neutral_maps_to_neutral(self):
        self.assertEqual(self._get_polarity("Neutral"), "neutral")

    def test_angry_maps_to_negative(self):
        self.assertEqual(self._get_polarity("Angry"), "negative")

    def test_disgust_maps_to_negative(self):
        self.assertEqual(self._get_polarity("Disgust"), "negative")

    def test_fear_maps_to_negative(self):
        self.assertEqual(self._get_polarity("Fear"), "negative")

    def test_sad_maps_to_negative(self):
        self.assertEqual(self._get_polarity("Sad"), "negative")

    def test_unknown_label_defaults_to_neutral(self):
        self.assertEqual(self._get_polarity("Unknown"), "neutral")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main()
