import hashlib
import json
import secrets
from typing import Any
from hfc_types.crypto import SnarkProof
from hfc_types.messages import Gradient
from py_ecc import bn128

curve_order = bn128.curve_order


def _random_scalar():
    return secrets.randbelow(curve_order)


def _mod(a):
    return a % curve_order


def _hash_to_scalar(*args):
    h = hashlib.sha256()
    for a in args:
        h.update(str(a).encode())
    return int.from_bytes(h.digest(), 'big') % curve_order


def _scalar_inv(a):
    return pow(a % curve_order, -1, curve_order)


class QAP:
    def __init__(self):
        self.num_vars = 4
        self.num_public = 3
        self.num_private = 1
        self.num_constraints = 1

        # Constraint: x1 * x3 = x2
        #   gradient_hash * witness = output
        self.u = [
            [0], [1], [0], [0]
        ]
        self.v = [
            [0], [0], [0], [1]
        ]
        self.w = [
            [0], [0], [1], [0]
        ]


_cached_crs: "CRS | None" = None
_cached_vk: dict | None = None
_cached_pk: dict | None = None


def _get_global_crs():
    global _cached_crs, _cached_vk, _cached_pk
    if _cached_crs is None:
        qap = QAP()
        _cached_crs = CRS(qap)
        _cached_pk = _cached_crs.get_proving_key()
        _cached_vk = _cached_crs.get_verification_key()
    return _cached_crs, _cached_pk, _cached_vk


class CRS:
    def __init__(self, qap: QAP):
        self.qap = qap
        seed = hashlib.sha256(b"H-FedChain SNARK CRS v1").digest()
        self.alpha = int.from_bytes(seed[0:32], 'big') % curve_order
        self.beta = int.from_bytes(seed[16:48], 'big') % curve_order
        self.gamma = int.from_bytes(seed[8:40], 'big') % curve_order
        self.delta = int.from_bytes(seed[24:56], 'big') % curve_order
        tau_seed = hashlib.sha256(b"H-FedChain SNARK tau v1").digest()
        self.tau = int.from_bytes(tau_seed, 'big') % curve_order

    def get_proving_key(self):
        qap = self.qap
        gamma_inv = _scalar_inv(self.gamma)
        delta_inv = _scalar_inv(self.delta)
        pk = {
            "alpha_g1": bn128.multiply(bn128.G1, self.alpha),
            "beta_g1": bn128.multiply(bn128.G1, self.beta),
            "beta_g2": bn128.multiply(bn128.G2, self.beta),
            "delta_g1": bn128.multiply(bn128.G1, self.delta),
            "delta_g2": bn128.multiply(bn128.G2, self.delta),
            "private_terms": [],
            "vk_public_terms": [],
            "alpha": self.alpha,
            "delta": self.delta,
        }
        for i in range(qap.num_public):
            ui = qap.u[i]
            vi = qap.v[i]
            wi = qap.w[i]
            ui_tau = _poly_eval(ui, self.tau, curve_order)
            vi_tau = _poly_eval(vi, self.tau, curve_order)
            wi_tau = _poly_eval(wi, self.tau, curve_order)
            coeff = _mod(self.beta * ui_tau + self.alpha * vi_tau + wi_tau)
            coeff = _mod(coeff * gamma_inv)
            pk["vk_public_terms"].append(bn128.multiply(bn128.G1, coeff))
        for i in range(qap.num_public, qap.num_vars):
            ui = qap.u[i]
            vi = qap.v[i]
            wi = qap.w[i]
            ui_tau = _poly_eval(ui, self.tau, curve_order)
            vi_tau = _poly_eval(vi, self.tau, curve_order)
            wi_tau = _poly_eval(wi, self.tau, curve_order)
            coeff = _mod(self.beta * ui_tau + self.alpha * vi_tau + wi_tau)
            coeff = _mod(coeff * delta_inv)
            pk["private_terms"].append(bn128.multiply(bn128.G1, coeff))
        return pk

    def get_verification_key(self):
        qap = self.qap
        vk = {
            "alpha_g1": bn128.multiply(bn128.G1, self.alpha),
            "beta_g2": bn128.multiply(bn128.G2, self.beta),
            "gamma_g2": bn128.multiply(bn128.G2, self.gamma),
            "delta_g2": bn128.multiply(bn128.G2, self.delta),
            "gamma_beta_abc": [],
        }
        gamma_inv = _scalar_inv(self.gamma)
        for i in range(qap.num_public):
            ui = qap.u[i]
            vi = qap.v[i]
            wi = qap.w[i]
            ui_tau = _poly_eval(ui, self.tau, curve_order)
            vi_tau = _poly_eval(vi, self.tau, curve_order)
            wi_tau = _poly_eval(wi, self.tau, curve_order)
            coeff = _mod(self.beta * ui_tau + self.alpha * vi_tau + wi_tau)
            coeff = _mod(coeff * gamma_inv)
            vk["gamma_beta_abc"].append(bn128.multiply(bn128.G1, coeff))
        return vk


def _poly_eval(coeffs, x, mod):
    result = 0
    for c in reversed(coeffs):
        result = (result * x + c) % mod
    return result


class SnarkProver:
    async def generate_proof(
        self, gradient: Gradient, model_hash: bytes, sk: bytes
    ) -> SnarkProof:
        crs, pk, _ = _get_global_crs()

        gradient_hash = _hash_to_scalar(gradient.data, model_hash)
        model_hash_scalar = int.from_bytes(model_hash[:8], 'big') % curve_order if model_hash else 0
        norm_bound = int.from_bytes(hashlib.sha256(str(gradient.data).encode()).digest()[:4], 'big') % curve_order

        # Private witnesses
        w = int.from_bytes(sk[:8], 'big') % curve_order
        output = _mod(gradient_hash * w)

        r = _random_scalar()
        s = _random_scalar()
        rs = _mod(r * s)

        a_scalar = _mod(crs.alpha + gradient_hash + r * crs.delta)
        b_scalar = _mod(crs.beta + w + s * crs.delta)

        A = bn128.multiply(bn128.G1, a_scalar)
        B = bn128.multiply(bn128.G2, b_scalar)

        B_in_G1 = bn128.multiply(bn128.G1, b_scalar)

        C = bn128.multiply(pk["private_terms"][0], w)
        C = bn128.add(C, bn128.multiply(A, s))
        C = bn128.add(C, bn128.multiply(B_in_G1, r))
        rs_delta = bn128.multiply(pk["delta_g1"], rs)
        C = bn128.add(C, bn128.neg(rs_delta))

        def point_to_dict(pt):
            if pt is None or bn128.is_inf(pt):
                return {"inf": True}
            return {"x": str(pt[0].n), "y": str(pt[1].n), "inf": False}

        def g2_point_to_dict(pt):
            if pt is None or bn128.is_inf(pt):
                return {"inf": True}
            return {
                "x": [str(pt[0].coeffs[0]), str(pt[0].coeffs[1])],
                "y": [str(pt[1].coeffs[0]), str(pt[1].coeffs[1])],
                "inf": False,
            }

        proof_data = {
            "A": point_to_dict(A),
            "B": g2_point_to_dict(B),
            "C": point_to_dict(C),
            "public": ["1", str(gradient_hash), str(output)],
        }
        proof_bytes = json.dumps(proof_data, sort_keys=True, default=str).encode()
        import numpy as np
        gradient_norm = float(np.linalg.norm(gradient.data))
        public_inputs = {
            "gradient_hash": hashlib.sha256(
                str(gradient.data).encode()
            ).hexdigest(),
            "gradient_norm": gradient_norm,
            "model_hash": model_hash.hex() if model_hash else "",
        }
        return SnarkProof(proof_bytes=proof_bytes, public_inputs=public_inputs)


class SnarkVerifier:
    async def verify(
        self, proof: SnarkProof, model_hash: bytes, vk_bytes: bytes
    ) -> bool:
        try:
            data = json.loads(proof.proof_bytes.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False

        _, _, vk = _get_global_crs()

        def dict_to_g1(d):
            if d.get("inf", False):
                return None
            return (bn128.FQ(int(d["x"])), bn128.FQ(int(d["y"])))

        def dict_to_g2(d):
            if d.get("inf", False):
                return None
            return (
                bn128.FQ2([int(d["x"][0]), int(d["x"][1])]),
                bn128.FQ2([int(d["y"][0]), int(d["y"][1])]),
            )

        try:
            A = dict_to_g1(data["A"])
            B = dict_to_g2(data["B"])
            C = dict_to_g1(data["C"])
        except (KeyError, ValueError, TypeError):
            return False

        if A is not None and not bn128.is_on_curve(A, bn128.b):
            return False
        if B is not None and not bn128.is_on_curve(B, bn128.b2):
            return False
        if C is not None and not bn128.is_on_curve(C, bn128.b):
            return False

        public_inputs = data.get("public", ["0", "0", "0"])

        lhs = bn128.pairing(B, A)

        rhs1 = bn128.pairing(vk["beta_g2"], vk["alpha_g1"])

        gamma_g2 = vk["gamma_g2"]
        vk_sum = None
        for i, pi_str in enumerate(public_inputs):
            pi = int(pi_str) % curve_order
            term = bn128.multiply(vk["gamma_beta_abc"][i], pi)
            vk_sum = bn128.add(vk_sum, term)
        rhs2 = bn128.pairing(gamma_g2, vk_sum)

        rhs3 = bn128.pairing(vk["delta_g2"], C)

        final_rhs = bn128.final_exponentiate(rhs1 * rhs2 * rhs3)
        final_lhs = bn128.final_exponentiate(lhs)

        return final_lhs == final_rhs
