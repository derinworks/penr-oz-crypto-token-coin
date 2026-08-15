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


def test_send_transaction_overflow_amount_rejected():
    """Regression test: amount 1e400 silently overflows to +inf during JSON
    parsing. Before the fix, this was accepted (inf > 0), stored, and then
    crashed GET /transaction/pending with an unhandled
    ``ValueError: Out of range float values are not JSON compliant`` when
    Starlette tried to serialize the infinite amount. It must now be
    rejected up front with a clean 400, and must never reach the pending
    pool.
    """
    response = client.post(
        "/transaction/send",
        content='{"sender": "Eve", "receiver": "Mallory", "amount": 1e400}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Amount must be a finite number"}

    # The bad transaction must not have been stored, and this call itself
    # must not raise / return a 5xx.
    response = client.get("/transaction/pending")
    assert response.status_code == 200
    assert response.json()["transactions"] == []


def test_send_transaction_nan_amount_rejected():
    """NaN previously evaded the `amount <= 0` check entirely (NaN
    comparisons are always False) and was silently accepted, causing the
    same downstream crash as the overflow case.
    """
    response = client.post(
        "/transaction/send",
        content='{"sender": "Eve", "receiver": "Mallory", "amount": NaN}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Amount must be a finite number"}

    response = client.get("/transaction/pending")
    assert response.status_code == 200
    assert response.json()["transactions"] == []


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
