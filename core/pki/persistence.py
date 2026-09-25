"""Persistent node keypair storage shared by fog entrypoint and edge runner."""

import os
import pickle


def load_or_create_keypair(key_path: str) -> tuple[bytes, bytes]:
    """Load a persisted node keypair, creating it on first boot.

    The secret key MUST survive container restarts: peers pin our vk on first
    key exchange, so a fresh key every boot would permanently poison their
    vk_map (and QC verification would fail). Edge devices need the same
    guarantee so their registered vk stays stable across restarts.
    """
    if os.path.exists(key_path):
        with open(key_path, "rb") as fh:
            loaded = pickle.load(fh)
        if not (isinstance(loaded, (tuple, list)) and len(loaded) == 2
                and all(isinstance(k, bytes) for k in loaded)):
            raise ValueError(
                f"Invalid key file {key_path}: expected 2 bytes objects, got "
                f"{type(loaded).__name__}"
            )
        return loaded[0], loaded[1]

    from core.pki import generate_keypair

    sk, vk = generate_keypair()
    dirname = os.path.dirname(key_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(key_path, "wb") as fh:
        pickle.dump((sk, vk), fh)
    try:
        os.chmod(key_path, 0o600)
    except OSError as e:
        print(f"Warning: could not chmod 0600 on {key_path}: {e}", flush=True)
    return sk, vk
