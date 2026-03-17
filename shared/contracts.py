"""
Inter-service API contracts for microservice communication.

These contracts define the request/response schemas for cross-service
HTTP endpoints, ensuring consistent data exchange between:
  - Transaction Service -> Miner Service
  - Miner Service -> Blockchain Service
  - Wallet Service -> Blockchain Service
"""

from typing import List

from pydantic import BaseModel

from .models.block import Block
from .models.transaction import Transaction

# --- Transaction Service Contracts ---


class PendingTransactionsResponse(BaseModel):
    """GET /transaction/pending response contract."""

    transactions: List[Transaction]


# --- Blockchain Service Contracts ---


class AddBlockRequest(BaseModel):
    """POST /blockchain/add-block request contract."""

    index: int
    timestamp: float
    transactions: List[Transaction]
    previous_hash: str
    nonce: int
    hash: str


class AddBlockResponse(BaseModel):
    """POST /blockchain/add-block response contract."""

    message: str
    block: Block
    chain_length: int


class BalanceResponse(BaseModel):
    """GET /blockchain/balance/{address} response contract."""

    address: str
    balance: float
