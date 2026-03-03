#!/usr/bin/env python3
"""
Example: CLI config file for qgate.

Save this as config.json, then run:

    qgate validate config.json
    qgate run config.json --adapter mock --seed 42
    qgate run config.json --adapter mock --output results.jsonl
"""
import json
from qgate.config import GateConfig, FusionConfig, DynamicThresholdConfig

config = GateConfig(
    n_subsystems=4,
    n_cycles=2,
    shots=1000,
    variant="score_fusion",
    k_fraction=0.9,
    fusion=FusionConfig(alpha=0.5, threshold=0.65),
    dynamic_threshold=DynamicThresholdConfig(enabled=False),
    metadata={"experiment": "demo", "run_id": 1},
)

print(config.model_dump_json(indent=2))
