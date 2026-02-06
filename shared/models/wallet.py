from time import time

from pydantic import BaseModel, Field


class Wallet(BaseModel):
    address: str
    balance: float = 0.0
    created_at: float = Field(default_factory=time)
