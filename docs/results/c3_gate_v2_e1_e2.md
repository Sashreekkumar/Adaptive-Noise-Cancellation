# c3_gate_v2_e1_e2: dual-expert fusion on the frozen evaluation pairs

All numbers on this page are copied from the evaluation outputs (`summary.md`, `per_file.csv`) written by
`scripts/eval_gate.py --tag c3_gate_v2_e1_e2`. Group tables are the script's own means; the per-file table prints the
CSV values to the same number of decimals the script uses. Nothing is filtered.

## Setup

- **Files:** the 12 frozen evaluation pairs (role `eval_frozen`, never used for training or tuning): 8 continuous-noise
  (engine, siren, vehicle, wind at two nominal SNRs) and 4 transient (gunfire, explosion at two nominal SNRs). 48 kHz mono.
- **Experts:** E1 = `e1_ft_v1` (DeepFilterNet3 fine-tuned, 2000 steps), E2 = `e2_impulse_e2a` (E1 fine-tuned on
  tier-C impulsive noise, 1000 steps). Both frozen; one shared STFT. No reference microphone, so the residual canceller
  is not in this evaluation path.
- **Systems:**

  | system | what it is | name in `per_file.csv` |
  |---|---|---|
  | B0 | E1 alone | `B0_pretrained` |
  | B1 | E2 alone | `B1_ft_v1` |
  | B2 | uniform 50/50 fusion of E1 and E2 in every band | `B2_uniform` |
  | B3 | learned FusionGate `gate_v2_e1_e2.pt` (500 steps on CPU, best = step 500, validation loss 0.5999 -> 0.5986) | `B3_gate` |
  | B4 | impulse rule: E2 in all 32 bands on frames where the causal impulse flag is 1, E1 elsewhere (gate mode `rule`) | `B4_rule` |

  The CSV system names and the `w_e0` / `w_e1` columns are positional (first / second expert passed to the script);
  in this run the first expert is E1 and the second is E2.
- **Metrics** (`sih_train/evaluation.py`, `scripts/eval_b0_b1.py`): out SNR and dSNR (output minus input SNR) in dB
  against the clean file; STOI; PESQ-WB (16 kHz resample) and PESQ-NB (8 kHz resample); SI-SDR and its improvement
  over the noisy input; mean band weight per expert.

## Overall (12 files)

| group | system | n | out SNR | dSNR | STOI | PESQ-WB | PESQ-NB | SI-SDR | dSI-SDR | w_E1 | w_E2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | B0 (E1 alone) | 12 | 20.06 | 12.92 | 0.8814 | 3.239 | 3.673 | 20.46 | 13.30 | - | - |
| ALL | B1 (E2 alone) | 12 | 20.47 | 13.32 | 0.8834 | 3.293 | 3.719 | 20.90 | 13.74 | - | - |
| ALL | B2 uniform | 12 | 20.33 | 13.18 | 0.8835 | 3.286 | 3.715 | 20.76 | 13.60 | 0.500 | 0.500 |
| ALL | B3 learned gate | 12 | 20.33 | 13.19 | 0.8842 | 3.313 | 3.724 | 20.77 | 13.61 | 0.746 | 0.254 |
| ALL | B4 impulse rule | 12 | 20.32 | 13.17 | 0.8848 | 3.335 | 3.726 | 20.75 | 13.59 | 0.788 | 0.212 |

## By suite

| group | system | n | out SNR | dSNR | STOI | PESQ-WB | PESQ-NB | SI-SDR | dSI-SDR | w_E1 | w_E2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| continuous | B0 (E1 alone) | 8 | 19.57 | 11.12 | 0.8715 | 3.243 | 3.695 | 19.81 | 11.36 | - | - |
| continuous | B1 (E2 alone) | 8 | 19.67 | 11.22 | 0.8701 | 3.186 | 3.687 | 19.86 | 11.41 | - | - |
| continuous | B2 uniform | 8 | 19.67 | 11.22 | 0.8712 | 3.217 | 3.702 | 19.89 | 11.44 | 0.500 | 0.500 |
| continuous | B3 learned gate | 8 | 19.66 | 11.22 | 0.8717 | 3.250 | 3.702 | 19.89 | 11.44 | 0.785 | 0.215 |
| continuous | B4 impulse rule | 8 | 19.59 | 11.14 | 0.8718 | 3.246 | 3.695 | 19.79 | 11.35 | 0.890 | 0.110 |
| transient | B0 (E1 alone) | 4 | 21.05 | 16.50 | 0.9012 | 3.231 | 3.629 | 21.77 | 17.19 | - | - |
| transient | B1 (E2 alone) | 4 | 22.07 | 17.52 | 0.9101 | 3.507 | 3.781 | 22.98 | 18.40 | - | - |
| transient | B2 uniform | 4 | 21.64 | 17.10 | 0.9080 | 3.423 | 3.740 | 22.50 | 17.92 | 0.500 | 0.500 |
| transient | B3 learned gate | 4 | 21.67 | 17.12 | 0.9093 | 3.439 | 3.767 | 22.52 | 17.94 | 0.669 | 0.331 |
| transient | B4 impulse rule | 4 | 21.78 | 17.23 | 0.9107 | 3.513 | 3.789 | 22.65 | 18.08 | 0.586 | 0.414 |

## B4 per file

Targets: out SNR >= 15 dB, STOI >= 0.85, PESQ-WB >= 2.5.

| file | suite | in SNR | out SNR | dSNR | STOI | PESQ-WB | PESQ-NB | SI-SDR | w_E2 | meets all targets |
|---|---|---|---|---|---|---|---|---|---|---|
| val_gunfire_-5_0000 | transient | -1.68 | 27.10 | 28.78 | 0.9379 | 3.648 | 4.048 | 29.46 | 0.506 | yes |
| val_gunfire_+10_0000 | transient | 11.88 | 26.27 | 14.39 | 0.8583 | 4.337 | 4.510 | 27.51 | 0.160 | yes |
| val_explosion_-5_0000 | transient | -3.56 | 11.86 | 15.41 | 0.8800 | 2.452 | 2.795 | 11.64 | 0.453 | no: out SNR, PESQ-WB |
| val_explosion_+10_0000 | transient | 11.54 | 21.89 | 10.35 | 0.9665 | 3.617 | 3.802 | 22.02 | 0.539 | yes |
| val_engine_+10_fileid0 | continuous | 7.76 | 20.24 | 12.49 | 0.9121 | 3.335 | 3.738 | 20.54 | 0.013 | yes |
| val_engine_-5_fileid0 | continuous | 6.21 | 19.86 | 13.65 | 0.9075 | 3.128 | 3.767 | 20.00 | 0.072 | yes |
| val_siren_+10_fileid0 | continuous | 12.11 | 27.16 | 15.05 | 0.8231 | 3.947 | 4.156 | 27.89 | 0.223 | no: STOI |
| val_siren_-5_fileid0 | continuous | -0.13 | 17.41 | 17.54 | 0.8235 | 3.037 | 3.604 | 17.50 | 0.005 | no: STOI |
| val_vehicle_+10_fileid0 | continuous | 17.65 | 20.31 | 2.66 | 0.9223 | 3.600 | 3.909 | 20.74 | 0.017 | yes |
| val_vehicle_-5_fileid0 | continuous | 2.65 | 12.72 | 10.07 | 0.8706 | 2.256 | 2.891 | 12.64 | 0.000 | no: out SNR, PESQ-WB |
| val_wind_+10_fileid0 | continuous | 16.88 | 22.91 | 6.03 | 0.8260 | 3.537 | 3.882 | 22.95 | 0.185 | no: STOI |
| val_wind_-5_fileid0 | continuous | 4.42 | 16.09 | 11.67 | 0.8894 | 3.126 | 3.610 | 16.09 | 0.366 | yes |

**7 of 12 files meet all targets (out SNR >= 15 dB, STOI >= 0.85, PESQ-WB >= 2.5).** The other 5:

- `val_explosion_-5_0000`: out SNR 11.86 dB and PESQ-WB 2.452. The input is at -3.56 dB; B4 improves on E1 alone (9.11 dB, PESQ-WB 1.622) but not enough to reach either target.
- `val_vehicle_-5_fileid0`: out SNR 12.72 dB and PESQ-WB 2.256. The rule gives E2 a mean weight of 0.000 on this file, so B4 is the same as E1 alone (12.72 dB, 2.256); E2 alone is not better (12.57 dB, 2.224).
- `val_siren_+10_fileid0`: STOI 0.8231 (out SNR 27.16 dB and PESQ-WB 3.947 pass). Every system is at 0.8231-0.8268 on this file, so the limit is the experts, not the fusion.
- `val_siren_-5_fileid0`: STOI 0.8235 (out SNR 17.41 dB and PESQ-WB 3.037 pass). Every system is at 0.8235-0.8248.
- `val_wind_+10_fileid0`: STOI 0.8260 (out SNR 22.91 dB and PESQ-WB 3.537 pass). Every system is at 0.8177-0.8261.

## Expert difference

| suite | n files | mean abs(mask_E1 - mask_E2) | same, impulse frames (n files) | SI-SDR E2 vs E1 (dB) |
|---|---|---|---|---|
| transient | 4 | 0.0266 | 0.0242 (4) | 31.47 |
| continuous | 8 | 0.0248 | 0.0195 (7) | 33.15 |

The two experts produce nearly the same output (their outputs are 31-33 dB SI-SDR apart), which bounds what any
fusion of them can gain.

## Caveats

- **Small evaluation set.** 12 files (4 transient). Differences of a few hundredths between systems are within noise;
  none of these comparisons has a confidence interval.
- **The transient gain of B4 over B0 is dominated by one file.** On `val_explosion_-5_0000` the PESQ-WB gap is +0.830
  and the STOI gap +0.0320; on the other three transient files the PESQ-WB gaps are
  +0.006, +0.238, +0.057 and the STOI gaps +0.0006, +0.0032, +0.0019.
- **The learned gate (B3) did not beat the rule (B4).** B3 is within 0.01 dB dSNR of uniform fusion (B2) overall and
  below B4 on STOI, PESQ-WB and PESQ-NB overall and on the transient suite. The recommended configuration is therefore
  the rule (`configs/e1_e2_rule_v2.json`), not the learned gate.
- **E2 alone (B1) has the highest dSNR** overall and on the transient suite; B4 is ahead of it on STOI and PESQ there,
  by small margins. On the continuous suite B1 has the lowest PESQ-WB of the five systems.
- **The impulse flag is not specific to impulses.** B4 gives E2 a mean weight of 0.110 on the continuous suite, where
  there are no impulsive events.
- The residual canceller and the `audio_cleaner` pre/post-processing and MP3 export are not part of these numbers.

## Reproduce

```
cd model_v2_nlms
python scripts/eval_gate.py --e0 checkpoints/e1_ft_v1 --e1 checkpoints/e2_impulse_e2a \
    --gate checkpoints/gate_v2_e1_e2.pt --tag <new tag> --output-root <dir>
```
The frozen pairs and the default output root are paths on the author's machine (`SUITES` in `scripts/eval_b0_b1.py`).
