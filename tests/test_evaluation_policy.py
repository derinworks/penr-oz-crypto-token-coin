"""Unit tests for the deterministic AI evaluation policy."""

import math

import pytest
from pydantic import ValidationError

from shared.evaluation_policy import (
    ALLOWED_SCORES,
    DEFAULT_RAW_SCORE_MAX,
    POLICY_VERSION,
    EvaluationResult,
    allowed_scores,
    evaluate,
    normalize,
)


class TestAllowedScores:
    """Verify the allowed discrete score set."""

    def test_eleven_levels(self):
        assert len(ALLOWED_SCORES) == 11

    def test_range_zero_to_one(self):
        assert ALLOWED_SCORES[0] == 0.0
        assert ALLOWED_SCORES[-1] == 1.0

    def test_step_size(self):
        for i in range(len(ALLOWED_SCORES)):
            assert ALLOWED_SCORES[i] == pytest.approx(i * 0.1)

    def test_allowed_scores_helper(self):
        assert allowed_scores() == list(ALLOWED_SCORES)


class TestNormalize:
    """Verify deterministic normalization behaviour."""

    def test_zero_raw(self):
        assert normalize(0.0) == 0.0

    def test_max_raw(self):
        assert normalize(100.0) == 1.0

    def test_midpoint(self):
        assert normalize(50.0) == 0.5

    def test_rounds_to_nearest_tenth(self):
        # 74 / 100 = 0.74 → rounds to 0.7
        assert normalize(74.0) == 0.7
        # 75 / 100 = 0.75 → rounds half-up to 0.8
        assert normalize(75.0) == 0.8
        # 85 / 100 = 0.85 → rounds half-up to 0.9
        assert normalize(85.0) == 0.9

    def test_clamps_below_zero(self):
        assert normalize(-10.0) == 0.0

    def test_clamps_above_max(self):
        assert normalize(150.0) == 1.0

    def test_custom_raw_score_max(self):
        assert normalize(5.0, raw_score_max=10.0) == 0.5
        assert normalize(10.0, raw_score_max=10.0) == 1.0
        assert normalize(0.0, raw_score_max=10.0) == 0.0

    def test_invalid_raw_score_max(self):
        with pytest.raises(ValueError, match="positive"):
            normalize(50.0, raw_score_max=0.0)
        with pytest.raises(ValueError, match="positive"):
            normalize(50.0, raw_score_max=-5.0)

    def test_nan_raw_score_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            normalize(float("nan"))

    def test_nan_raw_score_max_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            normalize(50.0, raw_score_max=float("nan"))

    def test_inf_raw_score_clamped(self):
        assert normalize(math.inf) == 1.0
        assert normalize(-math.inf) == 0.0

    def test_result_always_in_allowed_set(self):
        """Sweep many raw values and confirm every output is allowed."""
        for raw in range(0, 101):
            result = normalize(float(raw))
            assert result in ALLOWED_SCORES, f"raw={raw} gave {result}"

    def test_determinism_repeated_calls(self):
        """Identical inputs must always yield identical outputs."""
        for _ in range(100):
            assert normalize(73.0) == normalize(73.0)
            assert normalize(0.0) == normalize(0.0)
            assert normalize(100.0) == normalize(100.0)


class TestEvaluate:
    """Verify the high-level evaluate helper."""

    def test_returns_evaluation_result(self):
        result = evaluate(80.0)
        assert isinstance(result, EvaluationResult)

    def test_raw_score_preserved(self):
        result = evaluate(42.0)
        assert result.raw_score == 42.0

    def test_normalized_score_matches_normalize(self):
        for raw in [0.0, 25.0, 50.0, 75.0, 100.0]:
            result = evaluate(raw)
            assert result.normalized_score == normalize(raw)

    def test_policy_version_recorded(self):
        result = evaluate(50.0)
        assert result.policy_version == POLICY_VERSION

    def test_raw_score_max_recorded(self):
        result = evaluate(50.0)
        assert result.raw_score_max == DEFAULT_RAW_SCORE_MAX

    def test_custom_raw_score_max(self):
        result = evaluate(7.0, raw_score_max=10.0)
        assert result.raw_score_max == 10.0
        assert result.normalized_score == 0.7

    def test_serialization_roundtrip(self):
        result = evaluate(65.0)
        json_str = result.model_dump_json()
        restored = EvaluationResult.model_validate_json(json_str)
        assert restored == result


class TestEvaluationResultModel:
    """Verify EvaluationResult Pydantic constraints."""

    def test_normalized_score_bounds(self):
        with pytest.raises(ValidationError):
            EvaluationResult(
                raw_score=50.0,
                normalized_score=1.5,
                raw_score_max=100.0,
                policy_version="1.0.0",
            )
        with pytest.raises(ValidationError):
            EvaluationResult(
                raw_score=50.0,
                normalized_score=-0.1,
                raw_score_max=100.0,
                policy_version="1.0.0",
            )

    def test_raw_score_max_must_be_positive(self):
        with pytest.raises(ValidationError):
            EvaluationResult(
                raw_score=50.0,
                normalized_score=0.5,
                raw_score_max=0.0,
                policy_version="1.0.0",
            )

    def test_normalized_score_must_be_allowed(self):
        with pytest.raises(ValidationError, match="must be one of"):
            EvaluationResult(
                raw_score=50.0,
                normalized_score=0.55,
                raw_score_max=100.0,
                policy_version="1.0.0",
            )

    def test_immutability(self):
        result = evaluate(50.0)
        with pytest.raises(ValidationError):
            result.normalized_score = 0.9

    def test_default_policy_version(self):
        result = EvaluationResult(
            raw_score=50.0,
            normalized_score=0.5,
            raw_score_max=100.0,
        )
        assert result.policy_version == POLICY_VERSION


class TestPolicyVersion:
    """Verify policy version metadata."""

    def test_version_is_string(self):
        assert isinstance(POLICY_VERSION, str)

    def test_version_semver_format(self):
        parts = POLICY_VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()
