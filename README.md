# PENR-OZ AI Work Token Cryptocurrency POC

> **Warning**: This is a POC to explore concepts; do not use for real value or production.

## Overview

The PENR-OZ project is a Proof of Concept (POC) for an **AI Work Token** cryptocurrency. It demonstrates a system where AI agents perform verifiable "work" (code changes, documentation, testing, etc.), and that work is evaluated and rewarded with tokens.

This repository serves as the monorepo skeleton for the core services and shared smart contracts.

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

- **API Service**: Exposes endpoints for task management and work submission
- **Miner Service**: Simulates AI agents performing work (the "miners")
- **Token Contract**: ERC-20 + custom logic for work-based minting (foundry)
- **Shared Contracts**: Common Solidity libraries

## POC Scope and Non-Goals

- **In Scope**: Basic monorepo structure, Docker setup, smart contract skeleton, token reward simulation
- **Non-Goals**: Production readiness, real economic value, complex AI evaluation models (this is a starting point)

## Getting Started

See the original instructions for running the services with Docker Compose, testing, etc.

## Future Directions

- Advanced AI agent orchestration
- On-chain work verification
- Decentralized review mechanisms
- Integration with real LLM providers for autonomous contributions

---

*Original technical content preserved below for reference:*

## Project Structure (Monorepo)

```
penr-oz-crypto-token-coin/
├── api-service/          # Node.js/Express API
├── miner-service/        # Python AI worker simulation
├── contracts/            # Foundry project for Solidity
├── shared-contracts/     # Reusable contract libs
├── docker-compose.yml
└── README.md
```

## Quick Start

```bash
docker-compose up --build
```

Run tests:
```bash
# API tests
cd api-service && npm test

# Contract tests
cd contracts && forge test
```

(Full original details preserved in the first commit.)
