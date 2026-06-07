import pytest
from fastapi.testclient import TestClient
from transaction_service.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_transactions():
    client.post("/transaction/clear")
    yield
    client.post("/transaction/clear")


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_send_transaction_success():
    payload = {"sender": "Alice", "receiver": "Bob", "amount": 10.0}
    response = client.post("/transaction/send", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "pending"}

    # verify it's in pending
    response = client.get("/transaction/pending")
    assert response.status_code == 200
    data = response.json()
    assert "transactions" in data
    txs = data["transactions"]
    assert len(txs) == 1
    assert txs[0]["sender"] == "Alice"
    assert txs[0]["receiver"] == "Bob"
    assert txs[0]["amount"] == 10.0


def test_send_transaction_invalid_amount():
    payload = {"sender": "Alice", "receiver": "Bob", "amount": -5.0}
    response = client.post("/transaction/send", json=payload)
    assert response.status_code == 400

    # verify not added
    response = client.get("/transaction/pending")
    assert len(response.json()["transactions"]) == 0


def test_send_transaction_same_sender_receiver():
    payload = {"sender": "Alice", "receiver": "Alice", "amount": 10.0}
    response = client.post("/transaction/send", json=payload)
    assert response.status_code == 400


def test_send_transaction_empty_address():
    payload = {"sender": "", "receiver": "Bob", "amount": 10.0}
    response = client.post("/transaction/send", json=payload)
    assert response.status_code == 400


def test_clear_transactions():
    payload = {"sender": "Alice", "receiver": "Bob", "amount": 10.0}
    client.post("/transaction/send", json=payload)

    response = client.post("/transaction/clear")
    assert response.status_code == 200

    response = client.get("/transaction/pending")
    assert response.json()["transactions"] == []


def test_get_pending_transactions_pagination():
    """Test pagination behavior on GET /transaction/pending (issue #17)."""
    # Seed 5 transactions
    for i in range(5):
        payload = {
            "sender": f"S{i}",
            "receiver": f"R{i}",
            "amount": float(i + 1),
        }
        client.post("/transaction/send", json=payload)

    # Default call (no params) -> returns all + metadata
    response = client.get("/transaction/pending")
    assert response.status_code == 200
    data = response.json()
    assert len(data["transactions"]) == 5
    assert data["total"] == 5
    assert data["skip"] == 0
    assert data["limit"] is None

    # limit=2
    response = client.get("/transaction/pending?limit=2")
    data = response.json()
    assert len(data["transactions"]) == 2
    assert data["total"] == 5
    assert data["skip"] == 0
    assert data["limit"] == 2

    # skip=2, limit=2
    response = client.get("/transaction/pending?skip=2&limit=2")
    data = response.json()
    assert len(data["transactions"]) == 2
    assert data["transactions"][0]["sender"] == "S2"
    assert data["skip"] == 2
    assert data["limit"] == 2
