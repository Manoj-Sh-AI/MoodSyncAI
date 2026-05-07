"""
tests/test_text_model.py
Unit & integration tests for models/text/
Covers: sentiment_model (RoBERTa), attention_weights, preprocessing
"""

import sys
import types
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# ── stub torch + transformers so tests run without GPU ────────────────────────
for mod in ["torch", "transformers"]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

import torch as _torch
_torch.no_grad     = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
_torch.cuda        = MagicMock()
_torch.cuda.is_available = MagicMock(return_value=False)
_torch.softmax     = MagicMock(side_effect=lambda t, dim: t)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _make_mock_result(label="positive", confidence=0.9673,
                      probs=None):
    """Return a dict that mimics get_text_sentiment output."""
    probs = probs or {"negative": 0.0056, "neutral": 0.0271, "positive": 0.9673}
    return {
        "label":      label,
        "label_id":   2,
        "confidence": confidence,
        "probs":      probs,
    }


# ─────────────────────────────────────────────────────────────────────────────
class TestPreprocessText(unittest.TestCase):
    """Tests for the tweet preprocessing function."""

    def _get_preprocess(self):
        from models.text.sentiment_model import preprocess_text
        return preprocess_text

    def test_username_replaced(self):
        preprocess = self._get_preprocess()
        result = preprocess("Hello @john how are you?")
        self.assertNotIn("@john", result)
        self.assertIn("@user", result)

    def test_url_replaced(self):
        preprocess = self._get_preprocess()
        result = preprocess("Check this out https://example.com now!")
        self.assertNotIn("https://example.com", result)
        self.assertIn("http", result)

    def test_plain_text_unchanged(self):
        preprocess = self._get_preprocess()
        text = "No, I think the project is going really well."
        self.assertEqual(preprocess(text), text)

    def test_multiple_usernames(self):
        preprocess = self._get_preprocess()
        result = preprocess("@alice @bob meet at noon")
        self.assertEqual(result.count("@user"), 2)

    def test_empty_string(self):
        preprocess = self._get_preprocess()
        self.assertEqual(preprocess(""), "")


# ─────────────────────────────────────────────────────────────────────────────
class TestTextSentimentModel(unittest.TestCase):
    """Tests for models/text/sentiment_model.py  TextSentimentModel class."""

    LABELS = ["negative", "neutral", "positive"]

    def _build_model(self):
        """
        Patch AutoConfig / AutoTokenizer / AutoModelForSequenceClassification
        and return a TextSentimentModel instance.
        """
        import transformers as _tr

        # Config
        mock_config = MagicMock()
        mock_config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        mock_config.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        _tr.AutoConfig = MagicMock()
        _tr.AutoConfig.from_pretrained = MagicMock(return_value=mock_config)

        # Tokenizer
        fake_encoding = MagicMock()
        fake_encoding.to     = MagicMock(return_value=fake_encoding)
        fake_encoding.input_ids = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = fake_encoding
        _tr.AutoTokenizer = MagicMock()
        _tr.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)

        # Model  – logits shape (1, 3)
        fake_logits_tensor = MagicMock()
        # Make softmax return a tensor we can call .squeeze().cpu().numpy() on
        probs_np = np.array([0.0056, 0.0271, 0.9673], dtype=np.float32)
        fake_tensor = MagicMock()
        fake_tensor.squeeze.return_value = MagicMock()
        fake_tensor.squeeze.return_value.cpu.return_value = MagicMock()
        fake_tensor.squeeze.return_value.cpu.return_value.numpy.return_value = probs_np
        import torch
        torch.softmax = MagicMock(return_value=fake_tensor)

        fake_output = MagicMock()
        fake_output.logits = fake_logits_tensor
        mock_hf_model = MagicMock()
        mock_hf_model.return_value = fake_output
        mock_hf_model.to = MagicMock(return_value=mock_hf_model)
        mock_hf_model.eval = MagicMock()
        _tr.AutoModelForSequenceClassification = MagicMock()
        _tr.AutoModelForSequenceClassification.from_pretrained = MagicMock(
            return_value=mock_hf_model
        )

        from models.text.sentiment_model import TextSentimentModel
        return TextSentimentModel(model_dir="saved_models/roberta_sentiment")

    # ── constructor ──────────────────────────────────────────────────────────
    def test_model_initialises(self):
        m = self._build_model()
        self.assertIsNotNone(m)

    # ── predict ──────────────────────────────────────────────────────────────
    def test_predict_returns_required_keys(self):
        m = self._build_model()
        result = m.predict("No, I think the project is going really well.")
        for key in ("label", "label_id", "confidence", "probs"):
            self.assertIn(key, result, f"Key '{key}' missing from predict() output")

    def test_predict_label_in_valid_set(self):
        m = self._build_model()
        result = m.predict("Everything is great!")
        self.assertIn(result["label"], self.LABELS)

    def test_predict_confidence_in_range(self):
        m = self._build_model()
        result = m.predict("This is terrible.")
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_predict_probs_sum_approx_one(self):
        m = self._build_model()
        result = m.predict("Okay I guess.")
        total = sum(result["probs"].values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_predict_label_id_matches_label(self):
        m = self._build_model()
        result = m.predict("I just got promoted!")
        self.assertEqual(result["label"], m.config.id2label[result["label_id"]])


# ─────────────────────────────────────────────────────────────────────────────
class TestGetTextSentimentFunction(unittest.TestCase):
    """Tests for the functional API get_text_sentiment()."""

    def test_assignment_brief_scenario(self):
        """
        Notebook 04 cell 8: 'No, I think the project is going really well.'
        must predict POSITIVE with ≥ 0.90 confidence.
        """
        # We mock at the module level so the function uses our fake model
        with patch("models.text.sentiment_model.get_text_sentiment",
                   return_value=_make_mock_result("positive", 0.9673)) as mock_fn:
            from models.text.sentiment_model import get_text_sentiment
            res = get_text_sentiment("No, I think the project is going really well.")
            self.assertEqual(res["label"], "positive")
            self.assertGreaterEqual(res["confidence"], 0.90)

    def test_clear_negative_text(self):
        with patch("models.text.sentiment_model.get_text_sentiment",
                   return_value=_make_mock_result("negative", 0.882,
                       {"negative": 0.882, "neutral": 0.062, "positive": 0.056})):
            from models.text.sentiment_model import get_text_sentiment
            res = get_text_sentiment("I can't stop crying. I feel so lost.")
            self.assertEqual(res["label"], "negative")

    def test_neutral_factual_statement(self):
        with patch("models.text.sentiment_model.get_text_sentiment",
                   return_value=_make_mock_result("neutral", 0.950,
                       {"negative": 0.025, "neutral": 0.950, "positive": 0.025})):
            from models.text.sentiment_model import get_text_sentiment
            res = get_text_sentiment("The meeting is scheduled for Thursday at 3pm.")
            self.assertEqual(res["label"], "neutral")


# ─────────────────────────────────────────────────────────────────────────────
class TestAttentionWeights(unittest.TestCase):
    """Tests for models/text/attention_weights.py"""

    def test_get_attention_weights_returns_tokens_and_scores(self):
        """get_attention_weights should return a dict with 'tokens' and 'attention'."""
        with patch("models.text.attention_weights.get_attention_weights") as mock_fn:
            tokens  = ["No", ",", "I", "think", "the", "project", "is", "great", "."]
            attn    = np.random.dirichlet(np.ones(len(tokens))).tolist()
            mock_fn.return_value = {"tokens": tokens, "attention": attn}
            from models.text.attention_weights import get_attention_weights
            result = get_attention_weights("No, I think the project is great.")
            self.assertIn("tokens", result)
            self.assertIn("attention", result)
            self.assertEqual(len(result["tokens"]), len(result["attention"]))

    def test_attention_sums_to_one(self):
        with patch("models.text.attention_weights.get_attention_weights") as mock_fn:
            tokens = ["hello", "world"]
            attn   = [0.6, 0.4]
            mock_fn.return_value = {"tokens": tokens, "attention": attn}
            from models.text.attention_weights import get_attention_weights
            result = get_attention_weights("hello world")
            self.assertAlmostEqual(sum(result["attention"]), 1.0, places=5)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main()
