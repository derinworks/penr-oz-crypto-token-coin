# PENR-OZ AI Work Token Cryptocurrency POC

> **Warning**: This is a POC to explore concepts; do not use for real value or production.

## Project overview

This repository is a Python monorepo skeleton for a mini cryptocurrency project built with
FastAPI-based microservices. It contains shared contracts and service entrypoints only,
with no business logic yet. It also serves as the foundation for experimenting with **AI Work Token** concepts.

## Why AI Work Token?

Traditional Proof-of-Work is energy-intensive and not particularly useful. An AI Work Token turns computational effort into productive, verifiable contributions (bug fixes, features, documentation, research, etc.) that advance a project.

## AI Work Flow (Conceptual)

1. **Task Assignment** — Issues or tasks are created
2. **Agent Work** — AI agents (like this one) propose changes via PRs
3. **Verification** — Other agents or humans review/evaluate the quality
4. **Token Minting** — Successful work mints PENR-OZ tokens to the contributor
5. **Staking & Governance** — Token holders participate in project direction

## Determinism and Reproducibility

For fair token rewards, work evaluation must be deterministic. The POC explores using fixed seeds, containerized environments, and reproducible builds.

## Services

- Wallet service
- Transaction service
- Blockchain service
- Miner service

## POC Scope and Non-Goals (AI Work Token)

- **In Scope**: Basic monorepo structure, Docker setup, service skeletons, AI agent experiments
- **Non-Goals**: Production readiness, real economic value, complex AI evaluation models (this is a starting point)

## Running with Docker Compose

The base `docker-compose.yml` intentionally does **not** publish service ports to the
host. This keeps services internal to the Docker network by default.

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

## Future Directions (AI Work Token)

- Advanced AI agent orchestration
- On-chain work verification
- Decentralized review mechanisms

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

(See the main branch for complete instructions.)
