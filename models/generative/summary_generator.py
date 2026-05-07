"""
models/generative/summary_generator.py
GPT-2 based summary generator for MoodSyncAI fusion output.
Generates a natural-language interpretation of the detected emotion state
and any mismatch between face and text modalities.
"""
from __future__ import annotations

from transformers import pipeline, Pipeline

_GENERATOR: Pipeline | None = None
_MODEL_ID   = "gpt2"
_MAX_NEW     = 80
_MIN_NEW     = 20


def _get_generator() -> Pipeline:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = pipeline("text-generation", model=_MODEL_ID)
    return _GENERATOR


def build_prompt(fusion_result: dict) -> str:
    """
    Construct a concise natural-language prompt from the fusion result dict.
    """
    v = fusion_result.get("visual", {})
    t = fusion_result.get("text",   {})
    f = fusion_result.get("fusion", {})

    emotion    = v.get("emotion",    "Unknown")
    v_conf     = v.get("confidence", 0.0)
    text_label = t.get("label",      "unknown")
    t_conf     = t.get("confidence", 0.0)
    severity   = f.get("severity",   "MATCH")

    if severity == "MATCH":
        prompt = (
            f"A person appears {emotion.lower()} (confidence {v_conf:.0%}) "
            f"and their words sound {text_label} ({t_conf:.0%}). "
            "Overall emotional state summary:"
        )
    elif severity == "HARD_MISMATCH":
        prompt = (
            f"A person appears {emotion.lower()} (confidence {v_conf:.0%}) "
            f"but their words sound {text_label} ({t_conf:.0%}). "
            "This is a hard mismatch. Possible explanation:"
        )
    else:  # SOFT_MISMATCH
        prompt = (
            f"A person appears {emotion.lower()} (confidence {v_conf:.0%}) "
            f"while their words are {text_label} ({t_conf:.0%}). "
            "The signals are mixed. Possible interpretation:"
        )
    return prompt


def generate_summary(
    fusion_result: dict,
    max_new_tokens: int = _MAX_NEW,
    min_new_tokens: int = _MIN_NEW,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
) -> str:
    """
    Generate a natural-language summary from the MoodSyncAI fusion output.

    Parameters
    ----------
    fusion_result        : output of mismatch_detector.run_fusion()
    max_new_tokens       : maximum tokens to generate
    min_new_tokens       : minimum tokens to generate
    temperature          : sampling temperature (higher = more creative)
    top_p                : nucleus sampling probability
    num_return_sequences : number of candidate summaries (returns the first)

    Returns
    -------
    str  – the generated summary text (prompt stripped)
    """
    generator = _get_generator()
    prompt    = build_prompt(fusion_result)

    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id,
    )

    full_text = outputs[0]["generated_text"]
    # Strip the prompt, keep only the generated continuation
    summary = full_text[len(prompt):].strip()
    return summary
