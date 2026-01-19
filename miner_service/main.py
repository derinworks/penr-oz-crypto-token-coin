import asyncio
import hashlib
import json
import logging
import os
import time
from typing import List

import httpx
from fastapi import FastAPI, HTTPException

from shared.constants import DIFFICULTY_PREFIX, MINING_REWARD
from shared.models.block import Block
from shared.models.transaction import Transaction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Environment variables for service URLs
TRANSACTION_SERVICE_URL = os.getenv(
    "TRANSACTION_SERVICE_URL", "http://transaction:8000"
)
BLOCKCHAIN_SERVICE_URL = os.getenv("BLOCKCHAIN_SERVICE_URL", "http://blockchain:8000")
MINER_ADDRESS = os.getenv("MINER_ADDRESS", "MINER_REWARD_ADDRESS")

# Mining statistics
mining_stats = {
    "total_blocks_mined": 0,
    "total_attempts": 0,
    "total_mining_time": 0.0,
    "last_block_hash": None,
    "last_block_index": None,
}


def calculate_hash(block_data: dict) -> str:
    """Calculate SHA-256 hash of block data"""
    block_string = json.dumps(block_data, sort_keys=True)
    return hashlib.sha256(block_string.encode()).hexdigest()


def proof_of_work(
    index: int, timestamp: float, transactions: List[Transaction], previous_hash: str
) -> tuple[int, str, int]:
    """
    Perform Proof-of-Work to find a valid nonce.

    Returns:
        tuple: (nonce, hash, attempts)
    """
    nonce = 0
    attempts = 0

    logger.info(f"Starting Proof-of-Work for block {index}...")

    while True:
        block_data = {
            "index": index,
            "timestamp": timestamp,
            "transactions": [t.model_dump() for t in transactions],
            "previous_hash": previous_hash,
            "nonce": nonce,
        }

        hash_value = calculate_hash(block_data)
        attempts += 1

        # Log progress every 10000 attempts
        if attempts % 10000 == 0:
            logger.info(f"Mining attempt {attempts}, current nonce: {nonce}")

        # Check if hash meets difficulty requirement
        if hash_value.startswith(DIFFICULTY_PREFIX):
            logger.info(
                f"Valid hash found! Nonce: {nonce}, Hash: {hash_value}, "
                f"Attempts: {attempts}"
            )
            return nonce, hash_value, attempts

        nonce += 1


async def get_pending_transactions() -> List[Transaction]:
    """Retrieve pending transactions from Transaction service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{TRANSACTION_SERVICE_URL}/transaction/pending", timeout=10.0
            )
            response.raise_for_status()
            transactions_data = response.json()
            return [Transaction(**t) for t in transactions_data]
    except httpx.HTTPError as e:
        logger.error(f"Failed to retrieve pending transactions: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Transaction service unavailable: {str(e)}",
        )


async def get_latest_block() -> Block:
    """Retrieve the latest block from Blockchain service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BLOCKCHAIN_SERVICE_URL}/blockchain", timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            chain = data["chain"]

            if not chain:
                raise HTTPException(
                    status_code=500, detail="Blockchain is empty (no genesis block)"
                )

            # Get the last block
            latest_block_data = chain[-1]
            return Block(**latest_block_data)
    except httpx.HTTPError as e:
        logger.error(f"Failed to retrieve blockchain: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Blockchain service unavailable: {str(e)}",
        )


async def submit_block(block: Block):
    """
    Submit mined block to Blockchain service.

    Raises:
        HTTPException(503): If blockchain service is unavailable or returns 5xx
        HTTPException(400): If block is rejected due to validation (4xx response)
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BLOCKCHAIN_SERVICE_URL}/blockchain/add-block",
                json=block.model_dump(),
                timeout=10.0,
            )
            response.raise_for_status()
            logger.info(f"Block {block.index} submitted successfully")
    except httpx.TimeoutException as e:
        logger.error(f"Blockchain service timeout: {e}")
        raise HTTPException(
            status_code=503,
            detail="Blockchain service timeout - please retry",
        )
    except httpx.HTTPStatusError as e:
        # HTTP error with response
        if e.response.status_code >= 500:
            # Server error from blockchain service
            logger.error(f"Blockchain service error: {e.response.status_code}")
            raise HTTPException(
                status_code=503,
                detail=f"Blockchain service error: {e.response.text}",
            )
        else:
            # 4xx error - validation failed
            logger.error(f"Block rejected: {e.response.status_code}")
            logger.error(f"Response: {e.response.text}")
            raise HTTPException(
                status_code=400,
                detail=f"Block rejected by blockchain: {e.response.text}",
            )
    except httpx.RequestError as e:
        # Network errors, connection failures
        logger.error(f"Failed to connect to blockchain service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Blockchain service unavailable - please retry",
        )


async def remove_mined_transactions(transactions: List[Transaction]) -> bool:
    """
    Remove specific transactions from Transaction service.
    Only removes transactions that were included in the mined block,
    preserving any new transactions that arrived during mining.

    Retries with exponential backoff to ensure transaction pool consistency.

    Returns:
        bool: True if removal succeeded, False if all retries failed
    """
    max_retries = 3
    base_delay = 1.0  # seconds

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{TRANSACTION_SERVICE_URL}/transaction/remove",
                    json=[tx.model_dump() for tx in transactions],
                    timeout=10.0,
                )
                response.raise_for_status()
                result = response.json()
                logger.info(
                    f"Removed {result.get('count', 0)} mined transactions "
                    f"from pending pool"
                )
                return True  # Success

        except httpx.HTTPError as e:
            logger.error(
                f"Failed to remove mined transactions (attempt {attempt + 1}/"
                f"{max_retries}): {e}"
            )

            if attempt < max_retries - 1:
                # Wait before retrying (exponential backoff)
                delay = base_delay * (2**attempt)
                logger.info(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                # All retries exhausted
                logger.error(
                    "All retry attempts exhausted. Transaction pool may be "
                    "inconsistent - manual cleanup may be required."
                )
                return False  # Cleanup failed


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/mine")
async def mine_block():
    """
    Mine a new block with pending transactions.

    Workflow:
    1. Retrieve pending transactions
    2. Get latest block from blockchain
    3. Add coinbase/miner reward transaction
    4. Perform Proof-of-Work
    5. Submit block to blockchain
    6. Clear pending transactions on success
    """
    start_time = time.time()
    logger.info("=== Starting mining process ===")

    # Step 1: Retrieve pending transactions
    pending_txs = await get_pending_transactions()
    logger.info(f"Retrieved {len(pending_txs)} pending transactions")

    # Step 2: Get latest block
    latest_block = await get_latest_block()
    logger.info(
        f"Latest block: index={latest_block.index}, hash={latest_block.hash[:16]}..."
    )

    # Step 3: Create coinbase/miner reward transaction
    coinbase_tx = Transaction(
        sender="COINBASE",
        receiver=MINER_ADDRESS,
        amount=MINING_REWARD,
        timestamp=time.time(),
    )
    logger.info(
        f"Created coinbase transaction: {MINING_REWARD} coins to {MINER_ADDRESS}"
    )

    # Combine coinbase with pending transactions
    all_transactions = [coinbase_tx] + pending_txs

    # Step 4: Construct candidate block and perform Proof-of-Work
    new_index = latest_block.index + 1
    timestamp = time.time()

    # Run PoW in background thread to prevent blocking the event loop
    # This allows other endpoints (/health, /miner/stats) to remain responsive
    nonce, hash_value, attempts = await asyncio.to_thread(
        proof_of_work,
        index=new_index,
        timestamp=timestamp,
        transactions=all_transactions,
        previous_hash=latest_block.hash,
    )

    # Create the mined block
    mined_block = Block(
        index=new_index,
        timestamp=timestamp,
        transactions=all_transactions,
        previous_hash=latest_block.hash,
        nonce=nonce,
        hash=hash_value,
    )

    # Step 5: Submit block to blockchain
    # Raises HTTPException(503) for service errors or
    # HTTPException(400) for validation errors
    await submit_block(mined_block)

    # Step 6: Remove only the transactions that were included in this block
    # (excluding coinbase, which wasn't in the pending pool)
    cleanup_success = await remove_mined_transactions(pending_txs)

    # Update mining statistics
    mining_duration = time.time() - start_time
    mining_stats["total_blocks_mined"] += 1
    mining_stats["total_attempts"] += attempts
    mining_stats["total_mining_time"] += mining_duration
    mining_stats["last_block_hash"] = hash_value
    mining_stats["last_block_index"] = new_index

    logger.info(f"=== Mining complete in {mining_duration:.2f}s ===")

    # Build response - always return success since block was mined and submitted
    response = {
        "status": "success",
        "message": "Block mined successfully",
        "block_index": new_index,
        "block_hash": hash_value,
        "nonce": nonce,
        "attempts": attempts,
        "duration_seconds": round(mining_duration, 2),
        "transactions_count": len(all_transactions),
    }

    # Add warning if transaction cleanup failed
    if not cleanup_success:
        response["warning"] = (
            "Block mined successfully but failed to update transaction pool. "
            "Manual cleanup may be required to prevent duplicate transactions."
        )
        logger.warning(
            "Mining succeeded but transaction cleanup failed - "
            "returning success with warning"
        )

    return response


@app.get("/miner/stats")
def get_mining_stats():
    """Get mining statistics"""
    avg_attempts = (
        mining_stats["total_attempts"] / mining_stats["total_blocks_mined"]
        if mining_stats["total_blocks_mined"] > 0
        else 0
    )
    avg_time = (
        mining_stats["total_mining_time"] / mining_stats["total_blocks_mined"]
        if mining_stats["total_blocks_mined"] > 0
        else 0
    )

    return {
        "total_blocks_mined": mining_stats["total_blocks_mined"],
        "total_attempts": mining_stats["total_attempts"],
        "total_mining_time_seconds": round(mining_stats["total_mining_time"], 2),
        "average_attempts_per_block": round(avg_attempts, 2),
        "average_time_per_block_seconds": round(avg_time, 2),
        "last_block_index": mining_stats["last_block_index"],
        "last_block_hash": mining_stats["last_block_hash"],
        "miner_address": MINER_ADDRESS,
        "mining_reward": MINING_REWARD,
        "difficulty_prefix": DIFFICULTY_PREFIX,
    }
