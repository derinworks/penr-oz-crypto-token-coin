"""Deterministic Reward Engine mapping AI work evaluations to token amounts.

This module contains the business logic that connects evaluated AI work to
token issuance. It consumes :class:`~shared.models.ai_work.AIWorkEvaluation`
data and produces :class:`~shared.models.ai_work.RewardDecision` output that
downstream services (e.g. the transaction service) can act on.

Reward Formula
--------------
The reward amount is computed with a single, explicit formula::

    reward_amount = normalized_score * max_reward   (if score >= min_score)
    reward_amount = 0.0                             (if score <  min_score)

where:

- ``normalized_score`` is the evaluation's score and MUST be one of the
  discrete levels defined by ``shared.evaluation_policy.ALLOWED_SCORES``.
- ``max_reward`` is the fixed maximum token reward per task.
- ``min_score`` is the minimum normalized score required before any reward
  is issued. Evaluations scoring below it are rejected with a zero reward.

The product is rounded to :data:`REWARD_PRECISION` decimal places so that
binary floating-point artifacts never leak into token amounts.

A decision is *approved* only when the resulting amount is strictly
positive. This keeps the engine consistent with the transaction service,
which rejects transfers of a non-positive amount, and covers the cases
where a zero score or a very small ``max_reward`` rounds the reward down
to zero.

Determinism Guarantee
---------------------
Reward amounts derive exclusively from the evaluation's normalized score and
the :class:`RewardPolicy` in effect. Identical inputs always produce
identical reward amounts, and the generated ``decision_id`` is derived from
the ``evaluation_id`` so re-running the engine over the same evaluation
yields the same decision identity. No randomness, market data, or external
state is involved.
"""

from pydantic import BaseModel, Field

from .evaluation_policy import ALLOWED_SCORES
from .models.ai_work import AIWorkEvaluation, AIWorkSubmission, RewardDecision

# ---------------------------------------------------------------------------
# Engine constants
# ---------------------------------------------------------------------------

REWARD_ENGINE_VERSION: str = "1.0.0"
"""Semantic version of the reward engine. Any change to the reward formula
MUST bump this version."""

DEFAULT_MAX_REWARD: float = 50.0
"""Default fixed maximum token reward for a fully scored (1.0) task."""

DEFAULT_MIN_SCORE: float = 0.1
"""Default minimum normalized score required before any reward is issued."""

REWARD_PRECISION: int = 8
"""Decimal places the computed reward amount is rounded to."""


# ---------------------------------------------------------------------------
# Reward policy model
# ---------------------------------------------------------------------------


class RewardPolicy(BaseModel, frozen=True):
    """Immutable configuration for the reward formula.

    Attributes
    ----------
    max_reward:
        Fixed maximum token reward per task. Must be a finite number > 0;
        ``inf`` and ``NaN`` are rejected so that a policy can never mint an
        unbounded token amount or one that fails JSON serialization.
    min_score:
        Minimum normalized score (inclusive) required for a reward to be
        issued. Must lie in ``[0.0, 1.0]``.
    """

    max_reward: float = Field(default=DEFAULT_MAX_REWARD, gt=0, allow_inf_nan=False)
    min_score: float = Field(default=DEFAULT_MIN_SCORE, ge=0.0, le=1.0)


DEFAULT_REWARD_POLICY = RewardPolicy()
"""The default policy used when none is supplied explicitly."""


# ---------------------------------------------------------------------------
# Core reward computation
# ---------------------------------------------------------------------------


def compute_reward(
    normalized_score: float, policy: RewardPolicy = DEFAULT_REWARD_POLICY
) -> float:
    """Compute the token reward for a normalized evaluation score.

    Parameters
    ----------
    normalized_score:
        The evaluation's normalized score. Must be a member of
        ``shared.evaluation_policy.ALLOWED_SCORES``.
    policy:
        The :class:`RewardPolicy` to apply.

    Returns
    -------
    float
        ``round(normalized_score * policy.max_reward, REWARD_PRECISION)``
        when the score meets ``policy.min_score``; otherwise ``0.0``.

    Raises
    ------
    ValueError
        If *normalized_score* is not in the allowed discrete score set.
    """
    if normalized_score not in ALLOWED_SCORES:
        raise ValueError(
            f"normalized_score must be one of {ALLOWED_SCORES}, "
            f"got {normalized_score}"
        )
    if normalized_score < policy.min_score:
        return 0.0
    return round(normalized_score * policy.max_reward, REWARD_PRECISION)


# ---------------------------------------------------------------------------
# High-level decision helper
# ---------------------------------------------------------------------------


def decide_reward(
    evaluation: AIWorkEvaluation,
    submission: AIWorkSubmission,
    policy: RewardPolicy = DEFAULT_REWARD_POLICY,
) -> RewardDecision:
    """Turn an evaluation of a submission into a :class:`RewardDecision`.

    Parameters
    ----------
    evaluation:
        The evaluation of the submitted AI work. Its ``normalized_score``
        must be a member of ``shared.evaluation_policy.ALLOWED_SCORES``.
    submission:
        The submission the evaluation refers to; supplies ``task_id`` and
        ``worker_address`` for the decision.
    policy:
        The :class:`RewardPolicy` to apply.

    Returns
    -------
    RewardDecision
        Approved with ``reward_amount = normalized_score * max_reward`` when
        the score meets ``policy.min_score`` and that amount is strictly
        positive; otherwise rejected with a zero reward. The ``reason``
        field records the applied formula inputs or why no reward was
        issued, and the decision's ``decision_id`` is
        ``"reward-<evaluation_id>"``.

    Raises
    ------
    ValueError
        If the evaluation does not reference *submission* (mismatched
        ``submission_id``) or its score is not an allowed discrete score.
    """
    if evaluation.submission_id != submission.submission_id:
        raise ValueError(
            "evaluation.submission_id "
            f"{evaluation.submission_id!r} does not match "
            f"submission.submission_id {submission.submission_id!r}"
        )

    reward_amount = compute_reward(evaluation.normalized_score, policy)
    # A decision is only approved when it carries a payable amount: the
    # transaction service rejects non-positive amounts, so an "approved"
    # zero-value decision would be internally inconsistent.
    approved = reward_amount > 0

    if approved:
        reason = (
            f"normalized_score {evaluation.normalized_score} >= "
            f"min_score {policy.min_score}; reward = "
            f"{evaluation.normalized_score} * {policy.max_reward}"
        )
    elif evaluation.normalized_score < policy.min_score:
        reason = (
            f"normalized_score {evaluation.normalized_score} below "
            f"min_score {policy.min_score}; no reward issued"
        )
    else:
        reason = (
            f"normalized_score {evaluation.normalized_score} yields a "
            "non-positive reward amount; no reward issued"
        )

    return RewardDecision(
        decision_id=f"reward-{evaluation.evaluation_id}",
        evaluation_id=evaluation.evaluation_id,
        task_id=submission.task_id,
        worker_address=submission.worker_address,
        reward_amount=reward_amount,
        approved=approved,
        reason=reason,
    )
