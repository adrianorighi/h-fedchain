from dataclasses import dataclass
import time
import pickle


@dataclass(frozen=True)
class Certificate:
    node_id: str
    public_key: bytes
    issuer: str
    issued_at: float
    expires_at: float
    signature: bytes

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def serialize(self) -> bytes:
        return pickle.dumps(self)
