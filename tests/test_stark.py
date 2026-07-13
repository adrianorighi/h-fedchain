import pytest
from zkp.stark import Field, Poly, MerkleTree, StarkProver, StarkVerifier, P
from hfc_types.block import Block


class TestField:
    def test_add_sub(self):
        f = Field()
        assert f.add(5, 7) == 12
        assert f.sub(5, 7) == (5 - 7) % P

    def test_mul_div(self):
        f = Field()
        assert f.mul(3, 5) == 15
        assert f.div(15, 5) == 3
        assert f.div(7, 2) == 7 * pow(2, -1, P) % P

    def test_inv(self):
        f = Field()
        a = 42
        assert f.mul(a, f.inv(a)) == 1

    def test_neg(self):
        f = Field()
        assert f.neg(5) == P - 5
        assert f.neg(f.neg(5)) == 5

    def test_pow(self):
        f = Field()
        assert f.pow(2, 10) == 1024
        assert f.pow(2, P - 1) == 1


class TestPoly:
    def test_eval_constant(self):
        p = Poly([5])
        assert p.eval(10) == 5

    def test_eval_linear(self):
        p = Poly([1, 2])
        assert p.eval(3) == 7

    def test_interpolate(self):
        xs = [1, 2, 3]
        ys = [2, 4, 6]
        p = Poly.interpolate(xs, ys)
        for x, y in zip(xs, ys):
            assert p.eval(x) == y

    def test_add_poly(self):
        a = Poly([1, 2])
        b = Poly([3, 4])
        c = a.add(b)
        assert c.eval(5) == (1 + 10) + (3 + 20)

    def test_mul_poly(self):
        a = Poly([1, 1])
        b = Poly([1, 1])
        c = a.mul(b)
        assert c.eval(2) == 9


class TestMerkleTree:
    def test_root_consistency(self):
        leaves = [b'a', b'b', b'c', b'd']
        mt = MerkleTree(leaves)
        root = mt.root()
        for i, leaf in enumerate(leaves):
            proof = mt.get_proof(i)
            assert MerkleTree.verify(root, i, leaf, proof)

    def test_wrong_leaf_fails(self):
        leaves = [b'a', b'b', b'c', b'd']
        mt = MerkleTree(leaves)
        root = mt.root()
        proof = mt.get_proof(0)
        assert not MerkleTree.verify(root, 0, b'e', proof)


class TestStarkProverVerifier:
    @pytest.mark.asyncio
    async def test_prove_and_verify(self):
        block = Block(
            round=1,
            gradient_hash=b'\x01' * 32,
            qc_commit=None,
            stark_proof=None,
            accepted_devices=["d0"],
            rejected_devices=[],
            timestamp=1000.0,
            prev_hash=b'\x00' * 32,
        )
        prover = StarkProver()
        verifier = StarkVerifier()
        proof = await prover.generate_proof(block)
        result = await verifier.verify(proof, proof.public_inputs)
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_rejects_tampered_proof(self):
        block = Block(
            round=1,
            gradient_hash=b'\x01' * 32,
            qc_commit=None,
            stark_proof=None,
            accepted_devices=["d0"],
            rejected_devices=[],
            timestamp=1000.0,
            prev_hash=b'\x00' * 32,
        )
        prover = StarkProver()
        verifier = StarkVerifier()
        proof = await prover.generate_proof(block)
        tampered_bytes = proof.proof_bytes.replace(b'"root_hex"', b'"root_Xhex"')
        proof.proof_bytes = tampered_bytes
        result = await verifier.verify(proof, proof.public_inputs)
        assert result is False
