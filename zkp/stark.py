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
        # Build trace from actual block data — encode each accepted device's
        # commitment as a trace element so the polynomial degree reflects the
        # real aggregation workload.
        trace = []
        trace.append(block.round % P)
        trace.append(
            int.from_bytes(block.gradient_hash[:8], 'big') % P
        )
        trace.append(
            int.from_bytes(block.prev_hash[:8], 'big') % P
        )
        trace.append(len(block.accepted_devices) % P)
        trace.append(block.n % P)
        trace.append(block.f % P)
        # Each accepted device contributes a hash — encodes the set of
        # participants that the FedAvg aggregation ran over.
        for dev_id in block.accepted_devices:
            dev_hash = int.from_bytes(
                hashlib.sha256(dev_id.encode()).digest()[:8], 'big'
            ) % P
            trace.append(dev_hash)

        n = _next_power_of_two(len(trace))
        while len(trace) < n:
            trace.append(self.f.zero())

        g = self._primitive_root(n)
        trace_domain = [self.f.pow(g, i) for i in range(n)]
        poly = Poly.interpolate(trace_domain, trace, self.f)

        rs_factor = 4
        eval_domain = [self.f.pow(g, i) for i in range(n * rs_factor)]
        codeword = [poly.eval(x) for x in eval_domain]

        # FRI rounds proportional to trace length: log2(n) folding rounds.
        num_rounds = max(2, n.bit_length() - 1)
        fri_result = self._fri_prove(codeword, n * rs_factor // 2,
                                     num_rounds=num_rounds)

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
            "n_accepted": len(block.accepted_devices),
            "n": block.n,
            "f": block.f,
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
        rounds_data = []
        trees = []
        current = codeword
        for _ in range(num_rounds):
            leaves_bytes = [struct.pack('>Q', c) for c in current]
            mt = MerkleTree(leaves_bytes)
            trees.append(mt)
            root = mt.root()
            seed = int.from_bytes(root[:8], 'big')
            alpha = seed % self.f.p
            half = len(current) // 2
            folded = []
            for i in range(half):
                even = current[2 * i]
                odd = current[2 * i + 1]
                folded.append(self.f.add(even, self.f.mul(alpha, odd)))
            queries = []
            for j in range(min(4, len(current))):
                queries.append({
                    "pos": j,
                    "value": current[j],
                    "leaf": leaves_bytes[j].hex(),
                    "proof": [p.hex() for p in mt.get_proof(j)],
                })
            rounds_data.append({
                "root": root.hex(),
                "alpha": alpha,
                "queries": queries,
            })
            current = folded

        for i in range(num_rounds):
            ri = rounds_data[i]
            fold_proofs = []
            if i < num_rounds - 1:
                next_mt = trees[i + 1]
                for pos in [0, 1]:
                    leaf_2pos = next(
                        (q["value"] for q in ri["queries"] if q["pos"] == 2 * pos),
                        None,
                    )
                    leaf_2pos_1 = next(
                        (q["value"] for q in ri["queries"] if q["pos"] == 2 * pos + 1),
                        None,
                    )
                    if leaf_2pos is not None and leaf_2pos_1 is not None:
                        folded_val = self.f.add(
                            leaf_2pos, self.f.mul(ri["alpha"], leaf_2pos_1)
                        )
                        folded_bytes = struct.pack('>Q', folded_val)
                        pf = next_mt.get_proof(pos)
                        fold_proofs.append({
                            "next_pos": pos,
                            "value": folded_val,
                            "leaf": folded_bytes.hex(),
                            "proof": [p.hex() for p in pf],
                        })
            else:
                leaf_0 = next(
                    (q["value"] for q in ri["queries"] if q["pos"] == 0), None,
                )
                leaf_1 = next(
                    (q["value"] for q in ri["queries"] if q["pos"] == 1), None,
                )
                if leaf_0 is not None and leaf_1 is not None:
                    expected = self.f.add(
                        leaf_0, self.f.mul(ri["alpha"], leaf_1)
                    )
                    fold_proofs.append({
                        "next_pos": 0,
                        "value": expected,
                        "leaf": "",
                        "proof": [],
                    })
            ri["fold_proofs"] = fold_proofs

        final_val = current[0] if current else 0
        return {"rounds": rounds_data, "final_value": final_val}


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
        try:
            fri = data["fri"]
            rounds = fri.get("rounds", [])
            if not rounds:
                return False
            final_value = fri.get("final_value", 0)

            for i in range(len(rounds)):
                ri = rounds[i]
                root_i = bytes.fromhex(ri["root"])
                alpha_i = ri["alpha"]
                queries_i = ri["queries"]
                fold_proofs_i = ri.get("fold_proofs", [])

                for q in queries_i:
                    leaf_bytes = bytes.fromhex(q["leaf"])
                    proof_bytes = [bytes.fromhex(s) for s in q["proof"]]
                    if not MerkleTree.verify(root_i, q["pos"], leaf_bytes, proof_bytes):
                        return False

                for fp in fold_proofs_i:
                    leaf_2k = next(
                        (q["value"] for q in queries_i if q["pos"] == 2 * fp["next_pos"]),
                        None,
                    )
                    leaf_2k1 = next(
                        (q["value"] for q in queries_i if q["pos"] == 2 * fp["next_pos"] + 1),
                        None,
                    )
                    if leaf_2k is None or leaf_2k1 is None:
                        return False
                    expected = self.f.add(leaf_2k, self.f.mul(alpha_i, leaf_2k1))
                    if fp["value"] != expected:
                        return False
                    if fp["proof"]:
                        fp_leaf = bytes.fromhex(fp["leaf"])
                        next_root = bytes.fromhex(rounds[i + 1]["root"])
                        fp_proof = [bytes.fromhex(s) for s in fp["proof"]]
                        if not MerkleTree.verify(
                            next_root, fp["next_pos"], fp_leaf, fp_proof
                        ):
                            return False
                    else:
                        if fp["value"] != final_value:
                            return False

        except (KeyError, ValueError, IndexError, TypeError):
            return False
        return True
