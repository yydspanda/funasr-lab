"""Conservative content normalization for Chinese-centric ASR evaluation."""

from __future__ import annotations

import unicodedata


NORMALIZER_VERSION = "zh-content-v0.1"


def normalize_content(text: str) -> str:
    """Normalize presentation differences without changing lexical content.

    The v0.1 contract applies Unicode NFKC, lowercases Latin text, and removes
    whitespace and Unicode punctuation. It intentionally does not convert
    Traditional Chinese to Simplified Chinese or rewrite numeric expressions.
    """

    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )
