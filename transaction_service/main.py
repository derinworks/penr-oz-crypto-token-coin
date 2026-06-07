from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query

from shared.contracts import PendingTransactionsResponse
from shared.models.transaction import Transaction

app = FastAPI()

pending_transactions: List[Transaction] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transaction/send")
def send_transaction(transaction: Transaction):
    if transaction.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than 0")

    if transaction.sender == transaction.receiver:
        raise HTTPException(
            status_code=400, detail="Sender and receiver must be different"
        )

    if not transaction.sender or not transaction.receiver:
        raise HTTPException(
            status_code=400, detail="Sender and receiver cannot be empty"
        )

    pending_transactions.append(transaction)
    return {"status": "pending"}


@app.get("/transaction/pending", response_model=PendingTransactionsResponse)
def get_pending_transactions(
    skip: Optional[int] = Query(0, ge=0, description="Number of pending transactions to skip for pagination"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Maximum number of pending transactions to return (default 100)"),
):
    """Return pending transactions, optionally paginated.

    This is an optional enhancement per issue #17. Pagination parameters are
    optional; defaults provide backward-compatible behavior for existing callers.
    """
    total = len(pending_transactions)
    start = skip
    end = start + limit if limit is not None else None
    paginated_transactions = pending_transactions[start:end]

    return PendingTransactionsResponse(
        transactions=paginated_transactions,
        total=total,
        skip=skip,
        limit=limit,
    )


@app.post("/transaction/clear")
def clear_transactions():
    pending_transactions.clear()
    return {"status": "cleared"}


@app.post("/transaction/remove")
def remove_transactions(transactions: List[Transaction]):
    """
    Remove specific transactions from the pending list.
    This prevents clearing transactions that arrived after mining started.
    """
    removed_count = 0

    for tx_to_remove in transactions:
        # Find and remove matching transactions
        for i, pending_tx in enumerate(pending_transactions):
            if (
                pending_tx.sender == tx_to_remove.sender
                and pending_tx.receiver == tx_to_remove.receiver
                and pending_tx.amount == tx_to_remove.amount
                and pending_tx.timestamp == tx_to_remove.timestamp
            ):
                pending_transactions.pop(i)
                removed_count += 1
                break  # Move to next transaction to remove

    return {"status": "removed", "count": removed_count}
