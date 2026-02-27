# penr-oz-crypto-token-coin

## Project overview

This repository is a Python monorepo skeleton for a mini cryptocurrency project built with
FastAPI-based microservices. It contains shared contracts and service entrypoints only,
with no business logic yet.

## Services

- Wallet service
- Transaction service
- Blockchain service
- Miner service

## Shared contracts

The `shared/` package contains locked contracts (constants and Pydantic models) that are
imported by each service.

## Running Tests

### Unit tests

Unit tests run against each service in isolation and do not require running services:

```bash
poetry run pytest -m "not integration"
```

### Integration tests

The `tests/` directory contains an end-to-end integration test that validates the full
Wallet → Transaction → Miner → Blockchain flow using real HTTP calls. All four services
must be running before you execute these tests.

1. Start each service on its own port using `uvicorn`:

   ```bash
   poetry run uvicorn wallet_service.main:app --port 8000 &
   poetry run uvicorn transaction_service.main:app --port 8001 &
   poetry run uvicorn blockchain_service.main:app --port 8002 &
   poetry run uvicorn miner_service.main:app --port 8003 &
   ```

2. If needed, override service URLs through environment variables (defaults shown):

   ```bash
   export WALLET_SERVICE_URL=http://localhost:8000
   export TRANSACTION_SERVICE_URL=http://localhost:8001
   export BLOCKCHAIN_SERVICE_URL=http://localhost:8002
   export MINER_SERVICE_URL=http://localhost:8003
   ```

3. Run the integration tests:

   ```bash
   poetry run pytest -m integration
   ```

The tests are marked with `@pytest.mark.integration` so they are excluded from the
default CI pipeline and only run when services are available.
