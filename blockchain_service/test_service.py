import hashlib
import json
import time

import pytest
from fastapi.testclient import TestClient

from blockchain_service.main import app, blockchain
from shared.constants import DIFFICULTY_PREFIX

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_blockchain():
    """Reset blockchain before and after each test"""
    # Reset to genesis block
    blockchain.chain = []
    blockchain._create_genesis_block()
    yield
    # Clean up after test
    blockchain.chain = []
    blockchain._create_genesis_block()


def calculate_valid_hash(index, timestamp, transactions, previous_hash, nonce):
    """Helper function to calculate a valid hash for testing"""
    block_dict = {
        "index": index,
        "timestamp": timestamp,
        "transactions": transactions,
        "previous_hash": previous_hash,
        "nonce": nonce,
    }
    block_string = json.dumps(block_dict, sort_keys=True)
    return hashlib.sha256(block_string.encode()).hexdigest()


def mine_block(index, timestamp, transactions, previous_hash):
    """Helper function to mine a valid block for testing"""
    nonce = 0
    while True:
        hash_value = calculate_valid_hash(
            index, timestamp, transactions, previous_hash, nonce
        )
        if hash_value.startswith(DIFFICULTY_PREFIX):
            return nonce, hash_value
        nonce += 1
        if nonce > 1000000:
            raise Exception("Mining took too long, check difficulty")


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_blockchain_initial():
    """Test getting blockchain with only genesis block"""
    response = client.get("/blockchain")
    assert response.status_code == 200
    data = response.json()
    assert "chain" in data
    assert "length" in data
    assert data["length"] == 1
    assert data["chain"][0]["index"] == 0
    assert data["chain"][0]["previous_hash"] == "0"


def test_validate_blockchain_genesis():
    """Test validating blockchain with only genesis block"""
    response = client.get("/blockchain/validate")
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["chain_length"] == 1
    assert "valid" in data["message"].lower()


def test_add_valid_block():
    """Test adding a valid block to the blockchain"""
    # Get current blockchain state
    response = client.get("/blockchain")
    chain = response.json()["chain"]
    previous_block = chain[-1]

    # Mine a valid block
    timestamp = time.time()
    transactions = [
        {"sender": "Alice", "receiver": "Bob", "amount": 10.0, "timestamp": timestamp}
    ]
    nonce, hash_value = mine_block(
        index=previous_block["index"] + 1,
        timestamp=timestamp,
        transactions=transactions,
        previous_hash=previous_block["hash"],
    )

    # Add the block
    payload = {
        "index": previous_block["index"] + 1,
        "timestamp": timestamp,
        "transactions": transactions,
        "previous_hash": previous_block["hash"],
        "nonce": nonce,
        "hash": hash_value,
    }
    response = client.post("/blockchain/add-block", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Block added successfully"
    assert data["chain_length"] == 2

    # Verify blockchain is still valid
    response = client.get("/blockchain/validate")
    assert response.json()["valid"] is True


def test_add_block_invalid_hash():
    """Test rejection of block with invalid hash (doesn't meet difficulty)"""
    response = client.get("/blockchain")
    chain = response.json()["chain"]
    previous_block = chain[-1]

    timestamp = time.time()
    transactions = []

    # Create a hash that doesn't meet difficulty requirements
    invalid_hash = "1234567890abcdef" * 4

    payload = {
        "index": previous_block["index"] + 1,
        "timestamp": timestamp,
        "transactions": transactions,
        "previous_hash": previous_block["hash"],
        "nonce": 0,
        "hash": invalid_hash,
    }
    response = client.post("/blockchain/add-block", json=payload)
    assert response.status_code == 400
    assert "validation" in response.json()["detail"].lower()


def test_add_block_wrong_previous_hash():
    """Test rejection of block with incorrect previous_hash"""
    response = client.get("/blockchain")
    chain = response.json()["chain"]
    previous_block = chain[-1]

    timestamp = time.time()
    transactions = []

    # Mine block with wrong previous hash
    wrong_previous_hash = "0" * 64
    nonce, hash_value = mine_block(
        index=previous_block["index"] + 1,
        timestamp=timestamp,
        transactions=transactions,
        previous_hash=wrong_previous_hash,
    )

    payload = {
        "index": previous_block["index"] + 1,
        "timestamp": timestamp,
        "transactions": transactions,
        "previous_hash": wrong_previous_hash,
        "nonce": nonce,
        "hash": hash_value,
    }
    response = client.post("/blockchain/add-block", json=payload)
    assert response.status_code == 400


def test_add_block_wrong_index():
    """Test rejection of block with incorrect index"""
    response = client.get("/blockchain")
    chain = response.json()["chain"]
    previous_block = chain[-1]

    timestamp = time.time()
    transactions = []

    # Mine block with wrong index
    wrong_index = previous_block["index"] + 5
    nonce, hash_value = mine_block(
        index=wrong_index,
        timestamp=timestamp,
        transactions=transactions,
        previous_hash=previous_block["hash"],
    )

    payload = {
        "index": wrong_index,
        "timestamp": timestamp,
        "transactions": transactions,
        "previous_hash": previous_block["hash"],
        "nonce": nonce,
        "hash": hash_value,
    }
    response = client.post("/blockchain/add-block", json=payload)
    assert response.status_code == 400


def test_add_block_tampered_hash():
    """Test rejection when hash doesn't match block contents"""
    response = client.get("/blockchain")
    chain = response.json()["chain"]
    previous_block = chain[-1]

    timestamp = time.time()
    transactions = []

    # Mine a valid block
    nonce, hash_value = mine_block(
        index=previous_block["index"] + 1,
        timestamp=timestamp,
        transactions=transactions,
        previous_hash=previous_block["hash"],
    )

    # Tamper with timestamp but keep the same hash
    payload = {
        "index": previous_block["index"] + 1,
        "timestamp": timestamp + 100,
        "transactions": transactions,
        "previous_hash": previous_block["hash"],
        "nonce": nonce,
        "hash": hash_value,
    }
    response = client.post("/blockchain/add-block", json=payload)
    assert response.status_code == 400


def test_multiple_blocks():
    """Test adding multiple valid blocks"""
    for i in range(3):
        # Get current state
        response = client.get("/blockchain")
        chain = response.json()["chain"]
        previous_block = chain[-1]

        # Mine and add block
        timestamp = time.time()
        transactions = [
            {
                "sender": f"Sender{i}",
                "receiver": f"Receiver{i}",
                "amount": float(i + 1),
                "timestamp": timestamp,
            }
        ]
        nonce, hash_value = mine_block(
            index=previous_block["index"] + 1,
            timestamp=timestamp,
            transactions=transactions,
            previous_hash=previous_block["hash"],
        )

        payload = {
            "index": previous_block["index"] + 1,
            "timestamp": timestamp,
            "transactions": transactions,
            "previous_hash": previous_block["hash"],
            "nonce": nonce,
            "hash": hash_value,
        }
        response = client.post("/blockchain/add-block", json=payload)
        assert response.status_code == 200

    # Verify final state
    response = client.get("/blockchain")
    assert response.json()["length"] == 4

    # Verify chain is still valid
    response = client.get("/blockchain/validate")
    assert response.json()["valid"] is True


def test_blockchain_validation_after_tampering():
    """Test that validation detects tampering"""
    # Add a valid block first
    response = client.get("/blockchain")
    chain = response.json()["chain"]
    previous_block = chain[-1]

    timestamp = time.time()
    transactions = []
    nonce, hash_value = mine_block(
        index=previous_block["index"] + 1,
        timestamp=timestamp,
        transactions=transactions,
        previous_hash=previous_block["hash"],
    )

    payload = {
        "index": previous_block["index"] + 1,
        "timestamp": timestamp,
        "transactions": transactions,
        "previous_hash": previous_block["hash"],
        "nonce": nonce,
        "hash": hash_value,
    }
    response = client.post("/blockchain/add-block", json=payload)
    assert response.status_code == 200

    # Tamper with the blockchain directly
    blockchain.chain[1].nonce = 999999

    # Validation should detect the tampering
    response = client.get("/blockchain/validate")
    assert response.json()["valid"] is False
    assert "invalid" in response.json()["message"].lower()
