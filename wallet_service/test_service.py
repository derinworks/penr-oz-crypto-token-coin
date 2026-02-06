import pytest
from fastapi.testclient import TestClient
from wallet_service.main import app, wallets

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_wallets():
    wallets.clear()
    yield
    wallets.clear()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_wallet():
    response = client.post("/wallet/create")
    assert response.status_code == 201
    data = response.json()
    assert "address" in data
    assert len(data["address"]) == 64  # SHA-256 hex digest length


def test_create_wallet_unique_addresses():
    response1 = client.post("/wallet/create")
    response2 = client.post("/wallet/create")
    assert response1.json()["address"] != response2.json()["address"]


def test_get_wallet():
    create_response = client.post("/wallet/create")
    address = create_response.json()["address"]

    response = client.get(f"/wallet/{address}")
    assert response.status_code == 200
    wallet = response.json()
    assert wallet["address"] == address
    assert wallet["balance"] == 0.0
    assert "created_at" in wallet


def test_get_wallet_not_found():
    response = client.get("/wallet/nonexistent")
    assert response.status_code == 404


def test_create_multiple_wallets():
    addresses = []
    for _ in range(3):
        response = client.post("/wallet/create")
        assert response.status_code == 201
        addresses.append(response.json()["address"])

    # All addresses should be unique
    assert len(set(addresses)) == 3

    # All wallets should be retrievable
    for address in addresses:
        response = client.get(f"/wallet/{address}")
        assert response.status_code == 200
        assert response.json()["address"] == address
