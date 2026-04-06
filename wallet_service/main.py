import hashlib
import uuid
from typing import Dict

from fastapi import FastAPI, HTTPException

from shared.models.wallet import Wallet

app = FastAPI()

wallets: Dict[str, Wallet] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/wallet/create", status_code=201)
def create_wallet():
    address = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    wallet = Wallet(address=address)
    wallets[address] = wallet
    return {"address": address}


@app.get("/wallet/{address}")
def get_wallet(address: str):
    if address not in wallets:
        raise HTTPException(status_code=404, detail="Wallet not found")
    return wallets[address]


@app.get("/wallet/{address}/balance")
def get_wallet_balance(address: str):
    wallet = wallets.get(address)
    if not wallet:
        raise HTTPException(status_code=404, detail="Wallet not found")
    return {"address": address, "balance": wallet.balance}
