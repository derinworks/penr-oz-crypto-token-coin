import hashlib
import json
import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.constants import DIFFICULTY_PREFIX
from shared.models.block import Block
from shared.models.transaction import Transaction


app = FastAPI()


class Blockchain:
    def __init__(self):
        self.chain: List[Block] = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        """Create the first block in the blockchain"""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[],
            previous_hash="0",
            nonce=0,
            hash="",
        )
        genesis_block.hash = self._calculate_hash(genesis_block)
        self.chain.append(genesis_block)

    def _calculate_hash(self, block: Block) -> str:
        """Calculate SHA-256 hash of a block"""
        block_dict = {
            "index": block.index,
            "timestamp": block.timestamp,
            "transactions": [t.model_dump() for t in block.transactions],
            "previous_hash": block.previous_hash,
            "nonce": block.nonce,
        }
        block_string = json.dumps(block_dict, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def _is_valid_hash(self, hash_value: str) -> bool:
        """Check if hash meets difficulty requirements"""
        return hash_value.startswith(DIFFICULTY_PREFIX)

    def _is_valid_block(self, block: Block, previous_block: Block) -> bool:
        """Validate a single block against the previous block"""
        # Check if block index is sequential
        if block.index != previous_block.index + 1:
            return False

        # Check if previous_hash matches
        if block.previous_hash != previous_block.hash:
            return False

        # Verify the block's hash is correct
        calculated_hash = self._calculate_hash(block)
        if block.hash != calculated_hash:
            return False

        # Verify hash meets difficulty requirements
        if not self._is_valid_hash(block.hash):
            return False

        return True

    def is_chain_valid(self) -> bool:
        """Validate the entire blockchain"""
        # A blockchain with only genesis block is valid
        if len(self.chain) <= 1:
            return True

        # Validate each block against the previous one
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if not self._is_valid_block(current_block, previous_block):
                return False

        return True

    def add_block(self, block: Block) -> bool:
        """Add a new block to the blockchain with validation"""
        if len(self.chain) == 0:
            raise ValueError("Genesis block missing")

        previous_block = self.chain[-1]

        # Validate the new block
        if not self._is_valid_block(block, previous_block):
            return False

        self.chain.append(block)
        return True

    def get_chain(self) -> List[Block]:
        """Return the entire blockchain"""
        return self.chain


# Global blockchain instance
blockchain = Blockchain()


class AddBlockRequest(BaseModel):
    index: int
    timestamp: float
    transactions: List[Transaction]
    previous_hash: str
    nonce: int
    hash: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/blockchain")
def get_blockchain():
    """Get the entire blockchain"""
    return {"chain": blockchain.get_chain(), "length": len(blockchain.chain)}


@app.post("/blockchain/add-block")
def add_block(request: AddBlockRequest):
    """Add a new block to the blockchain"""
    # Convert request to Block model
    block = Block(
        index=request.index,
        timestamp=request.timestamp,
        transactions=request.transactions,
        previous_hash=request.previous_hash,
        nonce=request.nonce,
        hash=request.hash,
    )

    # Validate and add the block
    try:
        if blockchain.add_block(block):
            return {
                "message": "Block added successfully",
                "block": block,
                "chain_length": len(blockchain.chain),
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid block: failed validation checks",
            )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/blockchain/validate")
def validate_blockchain():
    """Validate the entire blockchain"""
    is_valid = blockchain.is_chain_valid()
    return {
        "valid": is_valid,
        "chain_length": len(blockchain.chain),
        "message": "Blockchain is valid" if is_valid else "Blockchain is invalid",
    }
