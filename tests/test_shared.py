"""Unit tests for shared models and constants."""

import pytest
from pydantic import ValidationError

from shared.constants import DIFFICULTY_PREFIX, MINING_REWARD
from shared.models.ai_work import (
    AIWorkEvaluation,
    AIWorkSubmission,
    AIWorkTask,
    RewardDecision,
)
from shared.models.block import Block
from shared.models.transaction import Transaction
from shared.models.wallet import Wallet


class TestTransactionFields:
    """Ensure transaction fields meet requirements."""

    def test_valid_transaction(self):
        tx = Transaction(sender="alice", receiver="bob", amount=10.0)
        assert tx.sender == "alice"
        assert tx.receiver == "bob"
        assert tx.amount == 10.0
        assert isinstance(tx.timestamp, float)

    def test_timestamp_auto_set(self):
        tx = Transaction(sender="alice", receiver="bob", amount=5.0)
        assert tx.timestamp > 0

    def test_custom_timestamp(self):
        tx = Transaction(
            sender="alice", receiver="bob", amount=1.0, timestamp=1234567890.0
        )
        assert tx.timestamp == 1234567890.0

    def test_sender_required(self):
        with pytest.raises(ValidationError):
            Transaction(receiver="bob", amount=10.0)

    def test_receiver_required(self):
        with pytest.raises(ValidationError):
            Transaction(sender="alice", amount=10.0)

    def test_amount_required(self):
        with pytest.raises(ValidationError):
            Transaction(sender="alice", receiver="bob")

    def test_amount_is_float(self):
        tx = Transaction(sender="alice", receiver="bob", amount=7)
        assert isinstance(tx.amount, float)

    def test_fractional_amount(self):
        tx = Transaction(sender="alice", receiver="bob", amount=0.001)
        assert tx.amount == pytest.approx(0.001)


class TestBlockHashFormat:
    """Confirm block hashes follow the DIFFICULTY_PREFIX rule."""

    def _make_block(self, hash_value: str) -> Block:
        return Block(
            index=1,
            timestamp=1000000.0,
            transactions=[],
            previous_hash="0" * 64,
            nonce=42,
            hash=hash_value,
        )

    def test_valid_hash_starts_with_prefix(self):
        valid_hash = DIFFICULTY_PREFIX + "a" * (64 - len(DIFFICULTY_PREFIX))
        block = self._make_block(valid_hash)
        assert block.hash.startswith(DIFFICULTY_PREFIX)

    def test_invalid_hash_does_not_start_with_prefix(self):
        invalid_hash = "1234" + "a" * 60
        block = self._make_block(invalid_hash)
        assert not block.hash.startswith(DIFFICULTY_PREFIX)

    def test_block_fields(self):
        tx = Transaction(sender="alice", receiver="bob", amount=5.0)
        hash_value = DIFFICULTY_PREFIX + "b" * (64 - len(DIFFICULTY_PREFIX))
        block = Block(
            index=0,
            timestamp=999999.0,
            transactions=[tx],
            previous_hash="0" * 64,
            nonce=0,
            hash=hash_value,
        )
        assert block.index == 0
        assert len(block.transactions) == 1
        assert block.nonce == 0


class TestWalletAddressFormat:
    """Verify wallet addresses conform to expected formats."""

    def test_valid_wallet(self):
        address = "a" * 64
        wallet = Wallet(address=address)
        assert wallet.address == address

    def test_address_length_enforced(self):
        with pytest.raises(ValidationError):
            Wallet(address="a" * 63)
        with pytest.raises(ValidationError):
            Wallet(address="a" * 65)

    def test_address_hex_chars_enforced(self):
        with pytest.raises(ValidationError):
            Wallet(address="g" * 64)  # 'g' is not a hex character
        with pytest.raises(ValidationError):
            Wallet(address="A" * 64)  # uppercase not allowed

    def test_default_balance_zero(self):
        wallet = Wallet(address="b" * 64)
        assert wallet.balance == 0.0

    def test_custom_balance(self):
        wallet = Wallet(address="c" * 64, balance=100.0)
        assert wallet.balance == 100.0

    def test_created_at_auto_set(self):
        wallet = Wallet(address="d" * 64)
        assert isinstance(wallet.created_at, float)
        assert wallet.created_at > 0

    def test_address_required(self):
        with pytest.raises(ValidationError):
            Wallet()


class TestConstants:
    """Verify constants have expected values."""

    def test_mining_reward(self):
        assert MINING_REWARD == 50.0

    def test_difficulty_prefix_length(self):
        assert len(DIFFICULTY_PREFIX) == 4

    def test_difficulty_prefix_all_zeros(self):
        assert all(c == "0" for c in DIFFICULTY_PREFIX)


class TestAIWorkTask:
    """Verify AIWorkTask model fields and validation."""

    def test_valid_task(self):
        task = AIWorkTask(
            task_id="task-001",
            description="Summarize a document",
            requester_address="a" * 64,
            reward_amount=10.0,
        )
        assert task.task_id == "task-001"
        assert task.description == "Summarize a document"
        assert task.reward_amount == 10.0
        assert isinstance(task.created_at, float)
        assert task.metadata == {}

    def test_task_id_required_nonempty(self):
        with pytest.raises(ValidationError):
            AIWorkTask(
                task_id="   ",
                description="desc",
                requester_address="a" * 64,
                reward_amount=5.0,
            )

    def test_reward_amount_must_be_positive(self):
        with pytest.raises(ValidationError):
            AIWorkTask(
                task_id="task-002",
                description="desc",
                requester_address="a" * 64,
                reward_amount=0.0,
            )
        with pytest.raises(ValidationError):
            AIWorkTask(
                task_id="task-002",
                description="desc",
                requester_address="a" * 64,
                reward_amount=-1.0,
            )

    def test_metadata_roundtrip(self):
        task = AIWorkTask(
            task_id="task-003",
            description="desc",
            requester_address="b" * 64,
            reward_amount=1.0,
            metadata={"priority": "high"},
        )
        data = task.model_dump()
        restored = AIWorkTask(**data)
        assert restored.metadata == {"priority": "high"}
        assert restored == task

    def test_serialization_deserialization(self):
        task = AIWorkTask(
            task_id="task-004",
            description="Test task",
            requester_address="c" * 64,
            reward_amount=25.0,
        )
        json_str = task.model_dump_json()
        restored = AIWorkTask.model_validate_json(json_str)
        assert restored.task_id == task.task_id
        assert restored.reward_amount == task.reward_amount


class TestAIWorkSubmission:
    """Verify AIWorkSubmission model fields and validation."""

    def test_valid_submission(self):
        sub = AIWorkSubmission(
            submission_id="sub-001",
            task_id="task-001",
            worker_address="d" * 64,
            result="The summary is ...",
        )
        assert sub.submission_id == "sub-001"
        assert sub.task_id == "task-001"
        assert sub.result == "The summary is ..."
        assert isinstance(sub.submitted_at, float)
        assert sub.trace_metadata == {}

    def test_submission_id_required_nonempty(self):
        with pytest.raises(ValidationError):
            AIWorkSubmission(
                submission_id="",
                task_id="task-001",
                worker_address="d" * 64,
                result="result",
            )

    def test_task_id_required_nonempty(self):
        with pytest.raises(ValidationError):
            AIWorkSubmission(
                submission_id="sub-002",
                task_id="  ",
                worker_address="d" * 64,
                result="result",
            )

    def test_trace_metadata_roundtrip(self):
        sub = AIWorkSubmission(
            submission_id="sub-003",
            task_id="task-002",
            worker_address="e" * 64,
            result="done",
            trace_metadata={"model": "gpt-4", "latency_ms": "120"},
        )
        data = sub.model_dump()
        restored = AIWorkSubmission(**data)
        assert restored.trace_metadata == {"model": "gpt-4", "latency_ms": "120"}

    def test_serialization_deserialization(self):
        sub = AIWorkSubmission(
            submission_id="sub-004",
            task_id="task-003",
            worker_address="f" * 64,
            result="output",
        )
        json_str = sub.model_dump_json()
        restored = AIWorkSubmission.model_validate_json(json_str)
        assert restored.submission_id == sub.submission_id
        assert restored.task_id == sub.task_id


class TestAIWorkEvaluation:
    """Verify AIWorkEvaluation model fields and validation."""

    def test_valid_evaluation(self):
        ev = AIWorkEvaluation(
            evaluation_id="eval-001",
            submission_id="sub-001",
            evaluator_address="a" * 64,
            raw_score=85.0,
            normalized_score=0.85,
        )
        assert ev.evaluation_id == "eval-001"
        assert ev.raw_score == 85.0
        assert ev.normalized_score == 0.85
        assert ev.comments == ""
        assert isinstance(ev.evaluated_at, float)

    def test_evaluation_id_required_nonempty(self):
        with pytest.raises(ValidationError):
            AIWorkEvaluation(
                evaluation_id="",
                submission_id="sub-001",
                evaluator_address="a" * 64,
                raw_score=50.0,
                normalized_score=0.5,
            )

    def test_normalized_score_bounds(self):
        with pytest.raises(ValidationError):
            AIWorkEvaluation(
                evaluation_id="eval-002",
                submission_id="sub-001",
                evaluator_address="a" * 64,
                raw_score=50.0,
                normalized_score=1.5,
            )
        with pytest.raises(ValidationError):
            AIWorkEvaluation(
                evaluation_id="eval-003",
                submission_id="sub-001",
                evaluator_address="a" * 64,
                raw_score=50.0,
                normalized_score=-0.1,
            )

    def test_normalized_score_edge_values(self):
        ev_zero = AIWorkEvaluation(
            evaluation_id="eval-004",
            submission_id="sub-001",
            evaluator_address="a" * 64,
            raw_score=0.0,
            normalized_score=0.0,
        )
        assert ev_zero.normalized_score == 0.0

        ev_one = AIWorkEvaluation(
            evaluation_id="eval-005",
            submission_id="sub-001",
            evaluator_address="a" * 64,
            raw_score=100.0,
            normalized_score=1.0,
        )
        assert ev_one.normalized_score == 1.0

    def test_serialization_deserialization(self):
        ev = AIWorkEvaluation(
            evaluation_id="eval-006",
            submission_id="sub-002",
            evaluator_address="b" * 64,
            raw_score=72.5,
            normalized_score=0.725,
            comments="Good work",
        )
        json_str = ev.model_dump_json()
        restored = AIWorkEvaluation.model_validate_json(json_str)
        assert restored.evaluation_id == ev.evaluation_id
        assert restored.comments == "Good work"


class TestRewardDecision:
    """Verify RewardDecision model fields and validation."""

    def test_valid_approved_decision(self):
        dec = RewardDecision(
            decision_id="dec-001",
            evaluation_id="eval-001",
            task_id="task-001",
            worker_address="a" * 64,
            reward_amount=10.0,
            approved=True,
            reason="High quality submission",
        )
        assert dec.decision_id == "dec-001"
        assert dec.approved is True
        assert dec.reward_amount == 10.0
        assert dec.reason == "High quality submission"
        assert isinstance(dec.decided_at, float)

    def test_valid_rejected_decision(self):
        dec = RewardDecision(
            decision_id="dec-002",
            evaluation_id="eval-002",
            task_id="task-002",
            worker_address="b" * 64,
            reward_amount=0.0,
            approved=False,
        )
        assert dec.approved is False
        assert dec.reward_amount == 0.0
        assert dec.reason is None

    def test_decision_id_required_nonempty(self):
        with pytest.raises(ValidationError):
            RewardDecision(
                decision_id="  ",
                evaluation_id="eval-001",
                task_id="task-001",
                worker_address="a" * 64,
                reward_amount=5.0,
                approved=True,
            )

    def test_reward_amount_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            RewardDecision(
                decision_id="dec-003",
                evaluation_id="eval-001",
                task_id="task-001",
                worker_address="a" * 64,
                reward_amount=-1.0,
                approved=False,
            )

    def test_serialization_deserialization(self):
        dec = RewardDecision(
            decision_id="dec-004",
            evaluation_id="eval-003",
            task_id="task-003",
            worker_address="c" * 64,
            reward_amount=15.0,
            approved=True,
            reason="Approved by evaluator",
        )
        json_str = dec.model_dump_json()
        restored = RewardDecision.model_validate_json(json_str)
        assert restored.decision_id == dec.decision_id
        assert restored.reason == "Approved by evaluator"
        assert restored.approved is True
