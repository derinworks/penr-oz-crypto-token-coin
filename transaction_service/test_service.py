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
    txs = response.json()
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
    assert len(response.json()) == 0


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
    assert response.json() == []
