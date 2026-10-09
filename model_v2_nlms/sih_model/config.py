"""Model-side pipeline configuration (JSON). Every stage can be switched without code changes.

{
  "experts": [                                   # order = expert index k seen by the gate
    {"name": "E0", "source": "pretrained", "enabled": true},
    {"name": "E1", "source": "<export dir or .pt>", "enabled": true},
    {"name": "E2", "source": "<export dir or .pt>", "enabled": true}
  ],
  "gate": {"enabled": true, "checkpoint": null,   # null -> untrained (zero-init = uniform weights)
           "use_impulse": false,                  # true -> gate gets causal impulse features (gate_v2+); false -> zeros
           "mode": "learned"},                    # "rule" -> [1 - flag, flag] over exactly 2 experts, no checkpoint
  "canceller": {"enabled": true, "mu": 0.5, ...}  # ResidualCanceller keyword arguments
}
Relative paths are resolved against the config file's directory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 48000


@dataclass
class ExpertSpec:
    name: str
    source: str                 # "pretrained" | path to an init_df export dir | path to a .pt state_dict/checkpoint
    enabled: bool = True


@dataclass
class GateSpec:
    enabled: bool = True
    checkpoint: str | None = None    # FusionGate checkpoint ({"gate": state_dict}); must match the number of enabled experts
    use_impulse: bool = False        # feed sih_model/impulse.py features to the gate; must match how the gate was trained
    mode: str = "learned"            # "learned" = FusionGate (checkpoint); "rule" = [1 - flag, flag] over 2 experts, no checkpoint


@dataclass
class CancellerSpec:
    enabled: bool = True
    params: dict = field(default_factory=dict)   # ResidualCanceller(**params); defaults are the tested values


@dataclass
class ModelConfig:
    experts: list[ExpertSpec]
    gate: GateSpec = field(default_factory=GateSpec)
    canceller: CancellerSpec = field(default_factory=CancellerSpec)

    @property
    def active_experts(self) -> list[ExpertSpec]:
        return [e for e in self.experts if e.enabled]

    def validate(self) -> None:
        if not self.active_experts:
            raise ValueError("at least one expert must be enabled")
        names = [e.name for e in self.experts]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate expert names: {names}")
        if self.gate.mode not in ("learned", "rule"):
            raise ValueError(f"gate.mode must be 'learned' or 'rule', got {self.gate.mode!r}")
        if self.gate.enabled and self.gate.mode == "rule":
            if len(self.active_experts) != 2:
                raise ValueError(f"gate mode 'rule' needs exactly 2 enabled experts, got {len(self.active_experts)}")
            if self.gate.checkpoint:
                raise ValueError("gate mode 'rule' takes no checkpoint")
            if not self.gate.use_impulse:
                raise ValueError("gate mode 'rule' needs use_impulse: true (the rule is driven by the impulse flag)")

    @staticmethod
    def load(path: str | Path) -> "ModelConfig":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))

        def resolve(p: str | None) -> str | None:
            if p is None or p == "pretrained":
                return p
            q = Path(p)
            return str(q if q.is_absolute() else (path.parent / q).resolve())

        canc = dict(raw.get("canceller", {}))
        gate = raw.get("gate", {})
        cfg = ModelConfig(
            experts=[ExpertSpec(e["name"], resolve(e["source"]), e.get("enabled", True)) for e in raw["experts"]],
            gate=GateSpec(gate.get("enabled", True), resolve(gate.get("checkpoint")), bool(gate.get("use_impulse", False)),
                          gate.get("mode", "learned")),
            canceller=CancellerSpec(canc.pop("enabled", True), canc),
        )
        cfg.validate()
        return cfg
