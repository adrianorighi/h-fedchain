from dataclasses import dataclass


@dataclass
class KeyMaterial:
    node_id: str
    sk: bytes
    vk: bytes


@dataclass
class SnarkProof:
    proof_bytes: bytes
    public_inputs: dict


@dataclass
class StarkProof:
    proof_bytes: bytes
    public_inputs: dict
