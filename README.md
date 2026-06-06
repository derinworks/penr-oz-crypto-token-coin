# penr-oz-crypto-token-coin

## Project overview

This repository implements **PENR-OZ**, a proof-of-concept (POC) for an AI Work Token cryptocurrency. It demonstrates a system where AI agents or systems perform "work" (computational tasks, evaluations, etc.), which is deterministically evaluated and rewarded with tokens in a cryptocurrency-like manner.

The project is structured as a Python monorepo with FastAPI-based microservices. It currently contains shared contracts and service entrypoints as a skeleton, with foundations for AI work token mechanisms being integrated. No production business logic is implemented yet.

## Why AI Work Token

Traditional cryptocurrencies reward mining or staking. This POC explores rewarding verifiable AI-generated work or evaluations through token issuance. Key ideas:
- Deterministic evaluation of AI outputs or tasks.
- Normalization of work value across different AI models or agents.
- Token rewards tied to verified contributions.

## Current Architecture

### Services
- **Wallet service**: Manages user/agent wallets and token balances.
- **Transaction service**: Handles token transfers and transaction validation.
- **Blockchain service**: Maintains the ledger of blocks and state (simplified for POC).
- **Miner service**: Performs "mining" which in this context includes AI work evaluation and block proposal.

### Shared contracts
The `shared/` package contains locked contracts (constants and Pydantic models) that are imported by each service, ensuring consistency across the system.

## AI Work Flow (Conceptual)

1. AI agents submit work (e.g., evaluations, generations) via wallet/transaction interfaces.
2. Miner service evaluates the work deterministically.
3. Valid work is normalized and rewarded with tokens.
4. Transactions are processed and recorded on the blockchain.

## Determinism and Reproducibility

All evaluations use fixed seeds, deterministic algorithms, and reproducible environments to ensure fair token rewards. This is crucial for the system.

## POC Scope and Non-Goals

- **In scope**: Basic service skeleton, shared models, integration tests for core flow, documentation of AI work token vision.
- **Not in scope**: Full blockchain consensus, real AI models integration (yet), production security, scalability, economic modeling.
- This is a POC to explore concepts; do not use for real value or production.

## Running with Docker Compose

The base `docker-compose.yml` intentionally does **not** publish service ports to the host. This keeps services internal to the Docker network by default.

To expose services on localhost for manual testing:

1. Copy the example override file:

   ```bash
   cp docker-compose.override.yaml.example docker-compose.override.yaml
   ```

2. Start the stack (Compose automatically loads both files):

   ```bash
   docker compose up --build
   ```

3. Access services on localhost:
   - Wallet: `http://127.0.0.1:8000`
   - Transaction: `http://127.0.0.1:8001`
   - Blockchain: `http://127.0.0.1:8002`
   - Miner: `http://127.0.0.1:8003`

4. Stop services:

   ```bash
   docker compose down
   ```

## Running Tests

### Unit tests

Unit tests run against each service in isolation and do not require running services:

```bash
poetry run pytest -m "not integration"
```

### Integration tests

The `tests/` directory contains an end-to-end integration test that validates the full Wallet → Transaction → Miner → Blockchain flow using real HTTP calls. All four services must be running before you execute these tests.

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

The tests are marked with `@pytest.mark.integration` so they are excluded from the default CI pipeline and only run when services are available.

## Future Directions

- Integrate actual AI evaluation logic in miner service.
- Implement deterministic work normalization.
- Expand token economics.
- Add more services for AI task orchestration.

This keeps the repository coherent as it evolves toward the AI work token vision.