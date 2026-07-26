"""Unit tests for the deterministic reward engine."""

import pytest
from pydantic import ValidationError

from shared.evaluation_policy import ALLOWED_SCORES, evaluate
from shared.models.ai_work import AIWorkEvaluation, AIWorkSubmission, RewardDecision
from shared.reward_engine import (
    DEFAULT_MAX_REWARD,
    DEFAULT_MIN_SCORE,
    DEFAULT_REWARD_POLICY,
    REWARD_ENGINE_VERSION,
    RewardPolicy,
    compute_reward,
    decide_reward,
)


def make_submission(**overrides) -> AIWorkSubmission:
    defaults = {
        "submission_id": "sub-1",
        "task_id": "task-1",
        "worker_address": "WORKER_ADDRESS",
        "result": "42",
    }
    defaults.update(overrides)
    return AIWorkSubmission(**defaults)


def make_evaluation(normalized_score: float, **overrides) -> AIWorkEvaluation:
    defaults = {
        "evaluation_id": "eval-1",
        "submission_id": "sub-1",
        "evaluator_address": "EVALUATOR_ADDRESS",
        "raw_score": normalized_score * 100,
        "normalized_score": normalized_score,
    }
    defaults.update(overrides)
    return AIWorkEvaluation(**defaults)


class TestRewardPolicy:
    """Verify RewardPolicy defaults and constraints."""

    def test_defaults(self):
        policy = RewardPolicy()
        assert policy.max_reward == DEFAULT_MAX_REWARD
        assert policy.min_score == DEFAULT_MIN_SCORE

    def test_default_policy_instance(self):
        assert DEFAULT_REWARD_POLICY == RewardPolicy()

    def test_max_reward_must_be_positive(self):
        with pytest.raises(ValidationError):
            RewardPolicy(max_reward=0.0)
        with pytest.raises(ValidationError):
            RewardPolicy(max_reward=-1.0)

    def test_min_score_bounds(self):
        with pytest.raises(ValidationError):
            RewardPolicy(min_score=-0.1)
        with pytest.raises(ValidationError):
            RewardPolicy(min_score=1.1)

    def test_immutability(self):
        policy = RewardPolicy()
        with pytest.raises(ValidationError):
            policy.max_reward = 100.0


class TestComputeReward:
    """Verify the deterministic reward formula."""

    def test_full_score_gets_max_reward(self):
        assert compute_reward(1.0) == DEFAULT_MAX_REWARD

    def test_zero_score_gets_no_reward(self):
        assert compute_reward(0.0) == 0.0

    def test_proportional_to_score(self):
        for score in ALLOWED_SCORES:
            if score >= DEFAULT_MIN_SCORE:
                assert compute_reward(score) == pytest.approx(
                    score * DEFAULT_MAX_REWARD
                )

    def test_below_threshold_gets_zero(self):
        policy = RewardPolicy(min_score=0.5)
        assert compute_reward(0.4, policy) == 0.0
        assert compute_reward(0.0, policy) == 0.0

    def test_at_threshold_gets_reward(self):
        policy = RewardPolicy(min_score=0.5)
        assert compute_reward(0.5, policy) == 25.0

    def test_custom_max_reward(self):
        policy = RewardPolicy(max_reward=10.0)
        assert compute_reward(0.7, policy) == 7.0
        assert compute_reward(1.0, policy) == 10.0

    def test_zero_threshold_rewards_everything(self):
        policy = RewardPolicy(min_score=0.0)
        assert compute_reward(0.0, policy) == 0.0
        assert compute_reward(0.1, policy) == 5.0

    def test_no_float_artifacts(self):
        """Rewards are rounded so binary float noise never leaks out."""
        policy = RewardPolicy(max_reward=0.3, min_score=0.0)
        assert compute_reward(0.3, policy) == 0.09

    def test_rejects_score_outside_allowed_set(self):
        with pytest.raises(ValueError, match="must be one of"):
            compute_reward(0.55)
        with pytest.raises(ValueError, match="must be one of"):
            compute_reward(-0.1)
        with pytest.raises(ValueError, match="must be one of"):
            compute_reward(1.5)

    def test_determinism_repeated_calls(self):
        for _ in range(100):
            assert compute_reward(0.7) == compute_reward(0.7)


class TestDecideReward:
    """Verify the high-level decide_reward helper."""

    def test_returns_reward_decision(self):
        decision = decide_reward(make_evaluation(0.8), make_submission())
        assert isinstance(decision, RewardDecision)

    def test_approved_with_proportional_reward(self):
        decision = decide_reward(make_evaluation(0.8), make_submission())
        assert decision.approved is True
        assert decision.reward_amount == 40.0

    def test_rejected_below_threshold(self):
        decision = decide_reward(make_evaluation(0.0), make_submission())
        assert decision.approved is False
        assert decision.reward_amount == 0.0
        assert "below" in decision.reason

    def test_reason_records_formula_inputs(self):
        decision = decide_reward(make_evaluation(0.8), make_submission())
        assert "0.8" in decision.reason
        assert str(DEFAULT_MAX_REWARD) in decision.reason

    def test_decision_links_evaluation_and_submission(self):
        decision = decide_reward(make_evaluation(0.8), make_submission())
        assert decision.evaluation_id == "eval-1"
        assert decision.task_id == "task-1"
        assert decision.worker_address == "WORKER_ADDRESS"

    def test_decision_id_is_deterministic(self):
        first = decide_reward(make_evaluation(0.8), make_submission())
        second = decide_reward(make_evaluation(0.8), make_submission())
        assert first.decision_id == second.decision_id == "reward-eval-1"

    def test_custom_policy_applied(self):
        policy = RewardPolicy(max_reward=100.0, min_score=0.9)
        approved = decide_reward(make_evaluation(0.9), make_submission(), policy)
        rejected = decide_reward(make_evaluation(0.8), make_submission(), policy)
        assert approved.approved is True
        assert approved.reward_amount == 90.0
        assert rejected.approved is False
        assert rejected.reward_amount == 0.0

    def test_mismatched_submission_rejected(self):
        evaluation = make_evaluation(0.8, submission_id="sub-2")
        with pytest.raises(ValueError, match="does not match"):
            decide_reward(evaluation, make_submission())

    def test_score_outside_allowed_set_rejected(self):
        with pytest.raises(ValueError, match="must be one of"):
            decide_reward(make_evaluation(0.55), make_submission())

    def test_decision_feeds_transaction_flow(self):
        """An approved decision carries everything a transaction needs."""
        decision = decide_reward(make_evaluation(1.0), make_submission())
        assert decision.worker_address
        assert decision.reward_amount > 0
        assert decision.approved is True


class TestEvaluationPolicyIntegration:
    """Verify the raw score -> normalized score -> reward pipeline."""

    def test_raw_score_to_reward(self):
        result = evaluate(85.0)  # normalizes to 0.9
        evaluation = make_evaluation(result.normalized_score, raw_score=85.0)
        decision = decide_reward(evaluation, make_submission())
        assert decision.approved is True
        assert decision.reward_amount == 45.0

    def test_every_allowed_score_produces_valid_decision(self):
        for score in ALLOWED_SCORES:
            decision = decide_reward(make_evaluation(score), make_submission())
            assert decision.reward_amount >= 0
            assert decision.approved is (score >= DEFAULT_MIN_SCORE)


class TestEngineVersion:
    """Verify reward engine version metadata."""

    def test_version_is_string(self):
        assert isinstance(REWARD_ENGINE_VERSION, str)

    def test_version_semver_format(self):
        parts = REWARD_ENGINE_VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()
