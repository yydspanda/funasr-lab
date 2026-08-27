"""Versioned, deterministic content scoring for the downstream ASR lab.

CER uses the existing ``zh-content-v0.1`` normalizer and scores its resulting
Unicode code points.  MER must instead tokenize the raw text directly so that
English word boundaries are still available.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Sequence, TypeVar

from eval.normalizers import NORMALIZER_VERSION
from eval.normalizers import normalize_content


ALIGNMENT_VERSION = "levenshtein-diagonal-deletion-insertion-v1"
MER_TOKENIZER_VERSION = "zh-en-mixed-v0.1"


@dataclass(frozen=True)
class EditCounts:
    """Levenshtein substitution, deletion, and insertion counts."""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def total(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def __add__(self, other: "EditCounts") -> "EditCounts":
        return EditCounts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
        )


@dataclass(frozen=True)
class ScoringResult:
    """Edit components and denominator for one CER or MER comparison.

    ``error_rate`` is ``None`` when the reference has no scoring units.  The
    edit components remain meaningful in that case (for example, a non-empty
    hypothesis consists entirely of insertions), while ``None`` avoids
    inventing a finite rate or emitting a non-finite JSON number.
    """

    counts: EditCounts
    reference_units: int

    @property
    def error_rate(self) -> float | None:
        if self.reference_units == 0:
            return None
        return self.counts.total / self.reference_units


Unit = TypeVar("Unit")


def sequence_edit_counts(
    reference: Sequence[Unit], hypothesis: Sequence[Unit]
) -> EditCounts:
    """Return deterministic Levenshtein components for arbitrary sequences.

    When minimum-cost alignments tie, each dynamic-programming cell prefers a
    diagonal operation, then deletion, then insertion.  The rule makes S/D/I
    accounting stable even when the total edit distance alone is ambiguous.
    """

    previous = [EditCounts(insertions=index) for index in range(len(hypothesis) + 1)]
    for ref_index, ref_unit in enumerate(reference, start=1):
        current = [EditCounts(deletions=ref_index)]
        for hyp_index, hyp_unit in enumerate(hypothesis, start=1):
            diagonal = previous[hyp_index - 1]
            if ref_unit != hyp_unit:
                diagonal = diagonal + EditCounts(substitutions=1)
            deletion = previous[hyp_index] + EditCounts(deletions=1)
            insertion = current[hyp_index - 1] + EditCounts(insertions=1)
            candidates = (diagonal, deletion, insertion)
            current.append(
                min(
                    enumerate(candidates),
                    key=lambda entry: (entry[1].total, entry[0]),
                )[1]
            )
        previous = current
    return previous[-1]


def cer_units(text: str) -> tuple[str, ...]:
    """Return ``zh-content-v0.1`` normalized Unicode code-point units."""

    return tuple(normalize_content(text))


def cer_components(reference: str, hypothesis: str) -> EditCounts:
    """Score two ``zh-content-v0.1`` character strings.

    This low-level compatibility API intentionally does not normalize its
    inputs.  Existing baseline callers already pass their ``content`` views;
    new raw-text callers should use :func:`cer_score` or :func:`cer_units`.
    """

    return sequence_edit_counts(reference, hypothesis)


def cer_score(reference: str, hypothesis: str) -> ScoringResult:
    """Return content CER components, denominator, and optional rate."""

    reference_units = cer_units(reference)
    hypothesis_units = cer_units(hypothesis)
    return ScoringResult(
        counts=sequence_edit_counts(reference_units, hypothesis_units),
        reference_units=len(reference_units),
    )


def _is_han(character: str) -> bool:
    name = unicodedata.name(character, "")
    return name.startswith("CJK UNIFIED IDEOGRAPH-") or name.startswith(
        "CJK COMPATIBILITY IDEOGRAPH-"
    )


def _is_latin_or_digit(character: str) -> bool:
    if character.isdigit():
        return True
    return (
        unicodedata.category(character).startswith("L")
        and "LATIN" in unicodedata.name(character, "")
    )


def mixed_units(text: str) -> tuple[str, ...]:
    """Tokenize raw Chinese-English text using ``zh-en-mixed-v0.1``.

    The versioned contract is:

    * apply Unicode NFKC and full Unicode case folding;
    * score every Han ideograph as one unit;
    * group each contiguous run of Unicode Latin letters and digits as one
      word unit;
    * discard whitespace and Unicode punctuation, treating both as boundaries;
    * preserve every other code point (symbols, emoji, and other scripts) as a
      singleton unit and as a boundary between Latin/digit runs.

    The conservative final rule prevents unsupported content from disappearing
    silently.  Call this function on raw text, not on ``normalize_content``
    output, because that Chinese-centric normalizer removes English word
    boundaries.
    """

    normalized = unicodedata.normalize("NFKC", text).casefold()
    units: list[str] = []
    latin_or_digit_run: list[str] = []

    def flush_run() -> None:
        if latin_or_digit_run:
            units.append("".join(latin_or_digit_run))
            latin_or_digit_run.clear()

    for character in normalized:
        if _is_han(character):
            flush_run()
            units.append(character)
            continue
        if _is_latin_or_digit(character):
            latin_or_digit_run.append(character)
            continue

        flush_run()
        if character.isspace() or unicodedata.category(character).startswith("P"):
            continue
        units.append(character)

    flush_run()
    return tuple(units)


def mer_components(reference: str, hypothesis: str) -> EditCounts:
    """Return mixed error-rate components over raw Chinese-English text."""

    return sequence_edit_counts(mixed_units(reference), mixed_units(hypothesis))


def mer_score(reference: str, hypothesis: str) -> ScoringResult:
    """Return MER components, denominator, and optional rate."""

    reference_units = mixed_units(reference)
    hypothesis_units = mixed_units(hypothesis)
    return ScoringResult(
        counts=sequence_edit_counts(reference_units, hypothesis_units),
        reference_units=len(reference_units),
    )


__all__ = [
    "ALIGNMENT_VERSION",
    "MER_TOKENIZER_VERSION",
    "NORMALIZER_VERSION",
    "EditCounts",
    "ScoringResult",
    "cer_components",
    "cer_score",
    "cer_units",
    "mer_components",
    "mer_score",
    "mixed_units",
    "sequence_edit_counts",
]
