import hashlib
import numpy as np
from typing import Optional
from hfc_types.messages import Gradient, GradientWithProof
from hfc_types.crypto import SnarkProof
from dataset.model import MLP
from core.pki import generate_keypair
from core.pki.certificate import Certificate
from zkp.snark import SnarkProver


class EdgeWorker:
    def __init__(
        self,
        device_id: str,
        indices: list[int],
        all_data: np.ndarray,
        all_labels: np.ndarray,
        is_adversarial: bool = False,
        attack_type: str = "label_flip",
        input_dim: int = 12000,
        hidden_dim: int = 64,
        num_classes: int = 5,
        use_snark: bool = False,
        certificate: Optional[Certificate] = None,
        ca_vk: Optional[bytes] = None,
        use_encryption: bool = False,
    ):
        self.device_id = device_id
        self.indices = indices
        self.is_adversarial = is_adversarial
        self.attack_type = attack_type
        self.num_classes = num_classes
        self.use_snark = use_snark
        self.certificate = certificate
        self.ca_vk = ca_vk
        self.use_encryption = use_encryption
        self.store = None
        if use_encryption:
            from dataset.encrypted_store import EncryptedLocalStore
            self.store = EncryptedLocalStore(device_id)

        self.local_data = all_data[indices]
        self.local_labels = all_labels[indices].copy()

        self.X = self.local_data.reshape(len(indices), -1).astype(np.float64)
        self.y = self.local_labels.copy()

        self.model = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

        self.sk, self.vk = generate_keypair()
        if self.use_snark:
            self.snark_prover = SnarkProver()

    def _apply_attack(self, round_num: int) -> np.ndarray:
        if self.attack_type == "label_flip":
            if (round_num // 5) % 2 == 1:
                return (self.y + 1) % self.num_classes
        return self.y

    def train_round(
        self, global_weights: dict, round_num: int = 0
    ) -> GradientWithProof:
        y_effective = self._apply_attack(round_num) if self.is_adversarial else self.y

        if self.is_adversarial and self.attack_type == "sign_flip":
            grad = self.model.compute_gradient(self.X, y_effective, global_weights)
            grad = -grad
        else:
            grad = self.model.compute_gradient(self.X, y_effective, global_weights)

        from core.pki import generate_keypair, sign as pki_sign

        msg = f"{self.device_id}:{round_num}".encode()
        if self.is_adversarial:
            fake_sk, _ = generate_keypair()
            gradient_signature = pki_sign(fake_sk, msg)
        else:
            gradient_signature = pki_sign(self.sk, msg)

        gradient = Gradient(
            node_id=self.device_id,
            round=round_num,
            data=grad.tolist(),
            signature=gradient_signature,
        )

        snark_proof: Optional[SnarkProof] = None
        if self.use_snark and hasattr(self, 'snark_prover'):
            model_hash = hashlib.sha256(
                str(sorted(global_weights.items())).encode()
            ).digest()
            import asyncio
            snark_proof = asyncio.run(
                self.snark_prover.generate_proof(gradient, model_hash, self.sk)
            )

        return GradientWithProof(gradient=gradient, snark_proof=snark_proof)

    def save_local_data(self, X, y, path: str):
        import pickle
        data = pickle.dumps({"X": X, "y": y})
        if self.store:
            self.store.save_data(data, path)
        else:
            with open(path, 'wb') as f:
                f.write(data)

    def load_local_data(self, path: str):
        import pickle
        import os as _os
        if not _os.path.exists(path):
            raise FileNotFoundError(f"Dados não encontrados: {path}")
        if self.store:
            data = self.store.load_data(path)
        else:
            with open(path, 'rb') as f:
                data = f.read()
        loaded = pickle.loads(data)
        return loaded["X"], loaded["y"]

    @property
    def num_samples(self) -> int:
        return len(self.indices)
