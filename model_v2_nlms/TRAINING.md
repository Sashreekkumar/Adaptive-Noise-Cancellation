# How the model-side checkpoints were produced

Runtime pipeline (STFT -> impulse detector -> E1 + E2 -> impulse-gated fusion -> residual canceller): see `README.md` and `SIH_MODEL.md`.
This file covers the code that produced the detached checkpoints listed in `checkpoints_manifest.json`
(`python tools/verify_checkpoints.py` checks a local copy in `model_v2_nlms/checkpoints/`).

## Code
| path | purpose |
|---|---|
| `sih_train/dfn_finetune.py` | DeepFilterNet3 fine-tuning: DFN3's own loss (multi-resolution spectral + local SNR), frozen BatchNorm, deterministic data order, bit-exact resume, exports loadable by `init_df` |
| `sih_train/data.py` | manifest-relative audio access (folders or the dataset ZIPs) and noisy/clean pair validation |
| `sih_train/evaluation.py` | SNR / STOI / PESQ-WB (and optional PESQ-NB) metrics used everywhere |
| `scripts/build_training_manifests.py` | train / val / test manifests with roles and audio checks |
| `scripts/finetune_dfn3.py` | CLI for fine-tuning (E1, E2) |
| `scripts/train_gate.py` | fusion-gate training for two frozen experts (`--e0`, `--e1`; `--tier-c-weight` repeats impulsive rows) |
| `scripts/eval_b0_b1.py`, `scripts/eval_gate.py` | frozen-set evaluation of single models, uniform fusion, the learned gate and the impulse rule |
Extra dependencies for training/evaluation: `pesq`, `pystoi`, `soundfile`. Default paths in the scripts point to the
author's machine (`D:\SIH26052\...`); pass your own `--manifest`, `--zip`/`--root`, `--run-dir`.

## Models
| checkpoint | how | status on the frozen evaluation pairs (4 transient + 8 continuous) |
|---|---|---|
| E0 | pretrained DeepFilterNet3 | dSNR 11.47 dB, STOI 0.874, PESQ 3.194 |
| E1 `e1_ft_v1` | `finetune_dfn3.py`, pretrained -> 2000 steps, batch 16, 3 s, lr 5e-5, warm-up 100, clip 1.0, frozen BN, seed 26052, 17,303 train_fast rows (clipped rows excluded) | dSNR 12.92 dB, STOI 0.881, PESQ 3.239 |
| E2 `e2_impulse_e2a` | same trainer, started from E1, tier-C rows only (gunfire / explosion / artillery, 2,983 rows), 1000 steps | did not meet its acceptance criteria vs E1: +0.11 dB on impulse + 0-100 ms windows (needed 1 dB), -0.057 PESQ on continuous noise |
| gate `gate_v1_e0_e1.pt` | `train_gate.py`, E0 + E1 frozen, 2000 steps on 1,127 held-out val_pool pairs | E0+E1+gate: dSNR 12.85 dB, STOI 0.882, PESQ 3.254 (not better than E1 alone) |
| gate `gate_v2_e1_e2.pt` | `train_gate.py --e0 E1 --e1 E2 --tier-c-weight 2`, impulse features on, 500 steps on CPU on 1,127 val_pool pairs (265 tier C, weighted x2) | E1+E2+gate: dSNR 13.19 dB, STOI 0.884, PESQ 3.313; the impulse rule (no training) scores dSNR 13.17 dB, STOI 0.885, PESQ 3.335, so the rule is the recommended gate. Full tables: `../docs/results/c3_gate_v2_e1_e2.md` |

Not yet done: a gate for three experts (the deprecated `frozen_v1.json` runs an untrained, uniform 3-expert gate);
a benchmark of the complete pipeline including the residual canceller with a real reference microphone.
