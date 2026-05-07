"""
tests/test_generator.py
Unit tests for models/generative/summary_generator.py
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ── stub heavy deps ───────────────────────────────────────────────────────────
for mod in ["torch", "transformers"]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
MATCH_STATE = {
    "fused_label":      "positive",
    "fused_confidence": 0.88,
    "mismatch": {
        "severity":       "MATCH",
        "badge_colour":   "#4CAF50",
        "visual_polarity":"positive",
        "text_pred":      "positive",
        "polarity_match": True,
        "both_confident": True,
    },
    "visual_output": {"label": "Happy",    "confidence": 0.85},
    "text_output":   {"label": "positive", "confidence": 0.91},
}

HARD_MISMATCH_STATE = {
    "fused_label":      "positive",
    "fused_confidence": 0.75,
    "mismatch": {
        "severity":       "HARD_MISMATCH",
        "badge_colour":   "#EF5350",
        "visual_polarity":"negative",
        "text_pred":      "positive",
        "polarity_match": False,
        "both_confident": True,
    },
    "visual_output": {"label": "Sad",      "confidence": 0.68},
    "text_output":   {"label": "positive", "confidence": 0.97},
}

SOFT_MISMATCH_STATE = {
    "fused_label":      "neutral",
    "fused_confidence": 0.55,
    "mismatch": {
        "severity":       "SOFT_MISMATCH",
        "badge_colour":   "#FFA726",
        "visual_polarity":"negative",
        "text_pred":      "positive",
        "polarity_match": False,
        "both_confident": False,
    },
    "visual_output": {"label": "Sad",      "confidence": 0.38},
    "text_output":   {"label": "positive", "confidence": 0.82},
}


# ─────────────────────────────────────────────────────────────────────────────
class TestSummaryGenerator(unittest.TestCase):
    """Tests for models/generative/summary_generator.py"""

    def _build_generator(self, generated_text="I feel great today!"):
        """Return a SummaryGenerator with the LLM pipeline mocked."""
        import transformers as _tr
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": generated_text}]
        _tr.pipeline = MagicMock(return_value=mock_pipeline)

        from models.generative.summary_generator import SummaryGenerator
        return SummaryGenerator()

    # ── constructor ──────────────────────────────────────────────────────────
    def test_init_creates_instance(self):
        gen = self._build_generator()
        self.assertIsNotNone(gen)

    # ── generate ─────────────────────────────────────────────────────────────
    def test_generate_returns_string(self):
        gen = self._build_generator()
        summary = gen.generate(fused_state=MATCH_STATE)
        self.assertIsInstance(summary, str)

    def test_generate_nonempty_for_match(self):
        gen = self._build_generator("The person appears genuinely happy.")
        summary = gen.generate(fused_state=MATCH_STATE)
        self.assertTrue(len(summary) > 0)

    def test_generate_nonempty_for_hard_mismatch(self):
        gen = self._build_generator("There is a clear emotional mismatch.")
        summary = gen.generate(fused_state=HARD_MISMATCH_STATE)
        self.assertTrue(len(summary) > 0)

    def test_generate_nonempty_for_soft_mismatch(self):
        gen = self._build_generator("The emotional signals are ambiguous.")
        summary = gen.generate(fused_state=SOFT_MISMATCH_STATE)
        self.assertTrue(len(summary) > 0)

    # ── prompt building ──────────────────────────────────────────────────────
    def test_build_prompt_contains_severity(self):
        from models.generative.summary_generator import SummaryGenerator
        gen = self._build_generator()
        prompt = gen.build_prompt(fused_state=HARD_MISMATCH_STATE)
        self.assertIn("HARD_MISMATCH", prompt)

    def test_build_prompt_contains_visual_label(self):
        from models.generative.summary_generator import SummaryGenerator
        gen = self._build_generator()
        prompt = gen.build_prompt(fused_state=MATCH_STATE)
        self.assertIn("Happy", prompt)

    def test_build_prompt_contains_text_label(self):
        from models.generative.summary_generator import SummaryGenerator
        gen = self._build_generator()
        prompt = gen.build_prompt(fused_state=MATCH_STATE)
        self.assertIn("positive", prompt)

    # ── edge cases ───────────────────────────────────────────────────────────
    def test_generate_with_none_fused_state_raises(self):
        gen = self._build_generator()
        with self.assertRaises((TypeError, KeyError, AttributeError)):
            gen.generate(fused_state=None)

    def test_generate_max_length_respected(self):
        """SummaryGenerator should accept an optional max_length kwarg."""
        gen = self._build_generator("Short.")
        summary = gen.generate(fused_state=MATCH_STATE, max_new_tokens=60)
        self.assertIsInstance(summary, str)


# ─────────────────────────────────────────────────────────────────────────────
class TestBuildPromptFormat(unittest.TestCase):
    """Validate the prompt template used to condition the LLM."""

    def _build(self, state):
        gen = _make_mock_generator()
        return gen.build_prompt(fused_state=state)

    def test_prompt_is_non_empty_string(self):
        gen = _make_mock_generator()
        prompt = gen.build_prompt(MATCH_STATE)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 10)

    def test_mismatch_severity_in_prompt(self):
        gen = _make_mock_generator()
        for state in [MATCH_STATE, HARD_MISMATCH_STATE, SOFT_MISMATCH_STATE]:
            prompt = gen.build_prompt(state)
            self.assertIn(state["mismatch"]["severity"], prompt)


def _make_mock_generator():
    import transformers as _tr
    mock_pipeline = MagicMock(return_value=[{"generated_text": "stub"}])
    _tr.pipeline = MagicMock(return_value=mock_pipeline)
    from models.generative.summary_generator import SummaryGenerator
    return SummaryGenerator()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main()
