"""FastAPI microservice wrapping a FogNode for containerized deployment.

Environment variables:
  NODE_ID  — unique fog node identifier (default: n0)
  F        — Byzantine fault tolerance parameter (default: 1)
"""
import os
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from hfc_types.messages import Gradient, AggregateGradient
from core.multikrum.aggregator import MultiKrum

app = FastAPI(title="H-FedChain Fog Node")

NODE_ID = os.environ.get("NODE_ID", "n0")
F = int(os.environ.get("F", "1"))

multikrum = MultiKrum()


class ConsensusRequest(BaseModel):
    round: int
    gradients: list[list[float]]
    device_ids: list[str]


class ConsensusResponse(BaseModel):
    node_id: str
    round: int
    accepted_devices: list[str]
    rejected_devices: list[str]
    latency: float


@app.post("/consensus", response_model=ConsensusResponse)
async def consensus(req: ConsensusRequest):
    if len(req.gradients) < 2 * F + 1:
        raise HTTPException(status_code=400, detail="Not enough gradients")
    t0 = time.time()
    np_grads = [np.array(g) for g in req.gradients]
    selected = multikrum.select(np_grads, F)
    accepted = [req.device_ids[i] for i in selected]
    rejected = [
        req.device_ids[i]
        for i in range(len(req.device_ids))
        if i not in selected
    ]
    latency = time.time() - t0
    return ConsensusResponse(
        node_id=NODE_ID,
        round=req.round,
        accepted_devices=accepted,
        rejected_devices=rejected,
        latency=latency,
    )


@app.get("/health")
async def health():
    return {"node_id": NODE_ID, "status": "ok"}
