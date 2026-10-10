"""Released Taraneh checkpoints.

Hugging Face ids are the intended publication names. Until those repos
are public, pass a local checkpoint directory to the scripts instead.
"""

from __future__ import annotations

MODELS = {
    "Taraneh-WBase": {
        "family": "whisper",
        "base": "openai/whisper-base",
        "hf": "areffarhadi/Taraneh-WBase",
    },
    "Taraneh-WMedium": {
        "family": "whisper",
        "base": "openai/whisper-medium",
        "hf": "areffarhadi/Taraneh-WMedium",
    },
    "Taraneh-WLarge": {
        "family": "whisper",
        "base": "openai/whisper-large-v3",
        "hf": "areffarhadi/Taraneh-WLarge",
    },
    "Taraneh-Q0.6B": {
        "family": "qwen",
        "base": "Qwen/Qwen3-ASR-0.6B",
        "hf": "areffarhadi/Taraneh-Q0.6B",
    },
    "Taraneh-Q1.7B": {
        "family": "qwen",
        "base": "Qwen/Qwen3-ASR-1.7B",
        "hf": "areffarhadi/Taraneh-Q1.7B",
    },
}


def resolve_model(name_or_path: str) -> dict:
    """Return family, checkpoint id, and display name for a Taraneh alias or a path."""
    key = name_or_path.strip()
    folded = {k.lower(): k for k in MODELS}
    canonical = folded.get(key.lower())
    if canonical:
        spec = dict(MODELS[canonical])
        spec["name"] = canonical
        spec["checkpoint"] = spec["hf"]
        return spec
    # Local directory or an arbitrary Hugging Face repo id.
    family = "qwen" if "qwen" in key.lower() or key.lower().startswith("taraneh-q") else "whisper"
    return {
        "name": key,
        "family": family,
        "base": None,
        "hf": key,
        "checkpoint": key,
    }
