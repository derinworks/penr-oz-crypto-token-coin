import re
from time import time

from pydantic import BaseModel, Field, field_validator


class Wallet(BaseModel):
    address: str
    balance: float = 0.0
    created_at: float = Field(default_factory=time)

    @field_validator("address")
    @classmethod
    def address_must_be_sha256_hex(cls, v: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError("address must be a 64-character lowercase hex string")
        return v
