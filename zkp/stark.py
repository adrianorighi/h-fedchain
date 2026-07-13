import hashlib
import json
import struct
from hfc_types.block import Block
from hfc_types.crypto import StarkProof

P = 2147483647


class Field:
    def __init__(self, p=P):
        self.p = p

    def zero(self):
        return 0

    def one(self):
        return 1

    def add(self, a, b):
        return (a + b) % self.p

    def sub(self, a, b):
        return (a - b) % self.p

    def mul(self, a, b):
        return (a * b) % self.p

    def div(self, a, b):
        return (a * pow(b, -1, self.p)) % self.p

    def pow(self, a, n):
        return pow(a % self.p, n, self.p)

    def inv(self, a):
        return pow(a % self.p, -1, self.p)

    def neg(self, a):
        return (-a) % self.p

    def eq(self, a, b):
        return (a % self.p) == (b % self.p)


class Poly:
    def __init__(self, coeffs, field=Field()):
        while len(coeffs) > 0 and field.eq(coeffs[-1], 0):
            coeffs = coeffs[:-1]
        self.coeffs = coeffs if coeffs else [0]
        self.f = field

    def deg(self):
        return len(self.coeffs) - 1

    def eval(self, x):
        result = 0
        for c in reversed(self.coeffs):
            result = self.f.add(self.f.mul(result, x), c)
        return result

    def add(self, other):
        max_len = max(len(self.coeffs), len(other.coeffs))
        result = [self.f.zero()] * max_len
        for i in range(max_len):
            a = self.coeffs[i] if i < len(self.coeffs) else 0
            b = other.coeffs[i] if i < len(other.coeffs) else 0
            result[i] = self.f.add(a, b)
        return Poly(result, self.f)

    def mul(self, other):
        result = [self.f.zero()] * (len(self.coeffs) + len(other.coeffs))
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                result[i + j] = self.f.add(result[i + j], self.f.mul(a, b))
        return Poly(result, self.f)

    @staticmethod
    def interpolate(xs, ys, field=Field()):
        n = len(xs)
        result = [field.zero()] * n
        for i in range(n):
            basis = [field.one()]
            xi = xs[i]
            yi = ys[i]
            for j in range(n):
                if i == j:
                    continue
                xj = xs[j]
                denom = field.inv(field.sub(xi, xj))
                new_basis = [field.zero()] * (len(basis) + 1)
                for k, c in enumerate(basis):
                    new_basis[k + 1] = field.add(new_basis[k + 1], c)
                    new_basis[k] = field.sub(new_basis[k], field.mul(c, xj))
                basis = [field.mul(c, denom) for c in new_basis]
            for k in range(len(basis)):
                result[k] = field.add(result[k], field.mul(basis[k], yi))
        return Poly(result, field)


def _hash(x: bytes) -> bytes:
    return hashlib.sha256(x).digest()


class MerkleTree:
    def __init__(self, leaves: list[bytes]):
        self.leaves = leaves
        self.tree = self._build(leaves)

    def _build(self, leaves):
        tree = [leaves]
        while len(tree[-1]) > 1:
            level = []
            for i in range(0, len(tree[-1]), 2):
                left = tree[-1][i]
                right = tree[-1][i + 1] if i + 1 < len(tree[-1]) else left
                level.append(_hash(left + right))
            tree.append(level)
        return tree

    def root(self):
        return self.tree[-1][0] if self.tree[-1] else b'\x00' * 32

    def get_proof(self, index):
        proof = []
        for level in self.tree[:-1]:
            sibling_idx = index ^ 1
            if sibling_idx < len(level):
                proof.append(level[sibling_idx])
            index //= 2
        return proof

    @staticmethod
    def verify(root, index, leaf, proof):
        current = leaf
        for sibling in proof:
            if index % 2 == 0:
                current = _hash(current + sibling)
            else:
                current = _hash(sibling + current)
            index //= 2
        return current == root


def _next_power_of_two(n):
    return 1 << (n - 1).bit_length()


class StarkProver:
    def __init__(self):
        self.f = Field()

    async def generate_proof(self, block: Block) -> StarkProof:
        trace = [
            block.round % P,
            int.from_bytes(block.gradient_hash[:8], 'big') % P,
            int.from_bytes(block.prev_hash[:8], 'big') % P,
            int(block.timestamp * 1000) % P,
        ]
        n = _next_power_of_two(len(trace))
        while len(trace) < n:
            trace.append(self.f.zero())

        g = self._primitive_root(n)
        trace_domain = [self.f.pow(g, i) for i in range(n)]
        poly = Poly.interpolate(trace_domain, trace, self.f)

        rs_factor = 4
        eval_domain = [self.f.pow(g, i) for i in range(n * rs_factor)]
        codeword = [poly.eval(x) for x in eval_domain]

        fri_result = self._fri_prove(codeword, n * rs_factor // 2, num_rounds=3)

        proof_data = {
            "trace_length": n,
            "rs_factor": rs_factor,
            "root_of_unity": g,
            "fri": fri_result,
        }
        proof_bytes = json.dumps(proof_data, sort_keys=True, default=str).encode()
        public_inputs = {
            "round": block.round,
            "gradient_hash": block.gradient_hash.hex(),
            "prev_hash": block.prev_hash.hex(),
        }
        return StarkProof(proof_bytes=proof_bytes, public_inputs=public_inputs)

    def _primitive_root(self, n):
        for candidate in range(2, min(P, 1000)):
            if self.f.pow(candidate, (P - 1) // 2) != 1:
                gen = candidate
                break
        else:
            gen = 5
        return self.f.pow(gen, (P - 1) // n)

    def _fri_prove(self, codeword, max_degree, num_rounds=3):
        result = {
            "root_hex": [],
            "alphas": [],
            "leaf_proofs": [],
            "final_value": None,
        }
        current = codeword
        for _ in range(num_rounds):
            leaves = [struct.pack('>Q', c) for c in current]
            mt = MerkleTree(leaves)
            root = mt.root()
            result["root_hex"].append(root.hex())
            seed = int.from_bytes(root[:8], 'big')
            alpha = seed % self.f.p
            result["alphas"].append(alpha)
            half = len(current) // 2
            folded = []
            for i in range(half):
                even = current[2 * i]
                odd = current[2 * i + 1]
                folded.append(self.f.add(even, self.f.mul(alpha, odd)))
            proofs_for_round = []
            for j in range(min(4, len(current))):
                leaf_bytes = leaves[j]
                pf = mt.get_proof(j)
                proofs_for_round.append({
                    "leaf": leaf_bytes.hex(),
                    "siblings": [p.hex() for p in pf],
                })
            result["leaf_proofs"].append(proofs_for_round)
            current = folded
        result["final_value"] = current[0] if current else 0
        return result


class StarkVerifier:
    def __init__(self):
        self.f = Field()

    async def verify(self, proof: StarkProof, public_inputs: dict) -> bool:
        try:
            data = json.loads(proof.proof_bytes.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        if not isinstance(data, dict) or "fri" not in data:
            return False
        fri = data["fri"]
        if "root_hex" not in fri:
            return False
        for round_idx in range(len(fri["root_hex"])):
            try:
                root = bytes.fromhex(fri["root_hex"][round_idx])
                for j, lp in enumerate(fri["leaf_proofs"][round_idx]):
                    leaf = bytes.fromhex(lp["leaf"])
                    siblings = [bytes.fromhex(s) for s in lp["siblings"]]
                    if not MerkleTree.verify(root, j, leaf, siblings):
                        return False
            except (KeyError, ValueError, IndexError):
                return False
        return True
