"""Unit tests for shared models and constants."""

import pytest
from pydantic import ValidationError

from shared.constants import DIFFICULTY_PREFIX, MINING_REWARD
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
