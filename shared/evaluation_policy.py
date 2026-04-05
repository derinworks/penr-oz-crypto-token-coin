"""Deterministic AI Evaluation Policy and Normalization Rules.

This module provides a rule-based evaluation policy for turning AI work
output into economic value. It converts raw scores into a discrete,
explicitly-defined set of normalized scores, ensuring that identical
inputs always produce identical outputs.

Score Normalization
-------------------
Raw scores (any float) are converted to normalized scores as follows:

1. **Clamping** – raw scores are clamped to ``[0, raw_score_max]``
   (default 100).
2. **Scaling** – the clamped value is divided by ``raw_score_max``
   to produce a continuous value in [0.0, 1.0].
3. **Discretization** – the continuous value is mapped to the nearest
   value in the ``ALLOWED_SCORES`` set via rounding to the closest
   discrete level.

Allowed Discrete Score Set
--------------------------
The canonical set of output scores is::

    {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}

These eleven levels represent the only valid normalized scores that the
system may produce or accept.

Determinism Guarantee
---------------------
Given the same ``raw_score``, ``raw_score_max``, and ``POLICY_VERSION``,
the ``normalize`` function will always return the same result. No
randomness, floating-point tolerance heuristics, or external state are
involved.
"""

from math import floor, isnan
from typing import List, Tuple

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------

POLICY_VERSION: str = "1.0.0"
"""Semantic version of the evaluation policy.  Any change to the
normalization logic MUST bump this version."""

ALLOWED_SCORES: Tuple[float, ...] = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)
"""Explicit discrete score set: 0.0, 0.1, … , 1.0."""

DEFAULT_RAW_SCORE_MAX: float = 100.0
"""Default upper bound for raw scores before normalization."""


# ---------------------------------------------------------------------------
# Core normalization function
# ---------------------------------------------------------------------------


def normalize(raw_score: float, raw_score_max: float = DEFAULT_RAW_SCORE_MAX) -> float:
    """Convert a raw score to the nearest allowed discrete score.

    Parameters
    ----------
    raw_score:
        The raw evaluation score (any finite float).
    raw_score_max:
        Upper bound of the raw score range.  Must be > 0.

    Returns
    -------
    float
        A value guaranteed to be in :data:`ALLOWED_SCORES`.

    Raises
    ------
    ValueError
        If *raw_score_max* is not positive.
    """
    if not (raw_score_max > 0):
        raise ValueError(f"raw_score_max must be positive, got {raw_score_max}")
    if isnan(raw_score):
        raise ValueError("raw_score cannot be NaN")

    # 1. Clamp
    clamped = max(0.0, min(raw_score, raw_score_max))

    # 2. Scale to [0, 1]
    continuous = clamped / raw_score_max

    # 3. Discretize – round half-up to nearest tenth using integer arithmetic
    #    to avoid Banker's Rounding and floating-point edge cases.
    nearest_index = floor(continuous * 10 + 0.5)
    return ALLOWED_SCORES[nearest_index]


# ---------------------------------------------------------------------------
# Evaluation result model
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel, frozen=True):
    """Immutable record produced by :func:`evaluate`."""

    raw_score: float
    normalized_score: float = Field(ge=0.0, le=1.0)
    raw_score_max: float = Field(gt=0)
    policy_version: str = POLICY_VERSION

    @field_validator("normalized_score")
    @classmethod
    def must_be_allowed_score(cls, v: float) -> float:
        if v not in ALLOWED_SCORES:
            raise ValueError(
                f"normalized_score must be one of {ALLOWED_SCORES}, got {v}"
            )
        return v


# ---------------------------------------------------------------------------
# High-level evaluate helper
# ---------------------------------------------------------------------------


def evaluate(
    raw_score: float,
    raw_score_max: float = DEFAULT_RAW_SCORE_MAX,
) -> EvaluationResult:
    """Evaluate a raw score and return a full :class:`EvaluationResult`.

    This is the primary entry-point for service consumers.

    Parameters
    ----------
    raw_score:
        The raw evaluation score.
    raw_score_max:
        Upper bound of the raw score range.  Must be > 0.

    Returns
    -------
    EvaluationResult
        Contains both raw and normalized scores plus policy metadata.
    """
    normalized = normalize(raw_score, raw_score_max)
    return EvaluationResult(
        raw_score=raw_score,
        normalized_score=normalized,
        raw_score_max=raw_score_max,
    )


def allowed_scores() -> List[float]:
    """Return the allowed discrete score set as a sorted list."""
    return list(ALLOWED_SCORES)
