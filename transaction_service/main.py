from typing import List

from fastapi import FastAPI, HTTPException

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


@app.get("/transaction/pending")
def get_pending_transactions():
    return pending_transactions


@app.post("/transaction/clear")
def clear_transactions():
    pending_transactions.clear()
    return {"status": "cleared"}
