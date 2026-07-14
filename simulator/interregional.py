from typing import Optional
from hfc_types.block import RegionalOutput, GlobalOutput
from zkp.stark import StarkProver, StarkVerifier


class InterRegionalManager:
    def __init__(self, stark_verifier: Optional[StarkVerifier] = None,
                 stark_prover: Optional[StarkProver] = None):
        self.stark_verifier = stark_verifier or StarkVerifier()
        self.stark_prover = stark_prover or StarkProver()

    async def process(self, outputs: list[RegionalOutput]) -> GlobalOutput:
        active = []
        for out in outputs:
            if out.stark_proof is not None and not await self.stark_verifier.verify(
                out.stark_proof, out.stark_proof.public_inputs,
            ):
                continue
            active.append(out)

        if not active:
            raise RuntimeError("No active clusters after STARK verification")

        total_devices = sum(o.n_devices for o in active)
        delta_w_inter = active[0].delta_w
        pi_inter = None

        return GlobalOutput(
            delta_w_inter=delta_w_inter,
            pi_inter=pi_inter,
            n_active_clusters=len(active),
            round_num=outputs[0].round_num,
        )
