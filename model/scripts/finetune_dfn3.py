"""CLI for DeepFilterNet3 fine-tuning (sih_train/dfn_finetune.py). Resumes automatically from run_dir.

Windows dry run (dev_train, local ZIPs):
  python scripts/finetune_dfn3.py --run-dir D:\\SIH26052\\experiments\\finetuned_dfn3\\dryrun --train-role dev_train
      --zip train_fast_test-...zip --zip val-...-001.zip --zip val-...-002.zip --steps 40 --batch-size 4 --val-max-files 6
Colab (train_fast on Drive, after build_training_manifests.py --check-audio there):
  python scripts/finetune_dfn3.py --run-dir /content/drive/MyDrive/SIH26052_runs/ft_v1
      --root /content/SIH26052_DATASET --manifest .../train_manifest.csv --val-manifest .../val_manifest.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sih_train.dfn_finetune import TrainConfig, Trainer


def main() -> None:
    defaults = TrainConfig(manifest="", val_manifest="", run_dir="")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--manifest", default=r"D:\SIH26052\manifests\train_manifest.csv")
    p.add_argument("--val-manifest", default=r"D:\SIH26052\manifests\val_manifest.csv")
    p.add_argument("--root", action="append", default=[])
    p.add_argument("--zip", action="append", default=[])
    p.add_argument("--train-role", default=defaults.train_role, choices=["train", "dev_train"])
    p.add_argument("--val-max-files", type=int, default=defaults.val_max_files)
    for name in ("steps", "batch_size", "warmup_steps", "seed", "val_every", "checkpoint_every", "max_nan_steps"):
        p.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("segment_sec", "lr", "weight_decay", "grad_clip"):
        p.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    p.add_argument("--bn-mode", default=defaults.bn_mode, choices=["frozen", "train"])
    p.add_argument("--init-model-dir", default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--row-filter", default=defaults.row_filter, choices=["use_for_training", "metadata_no_clip"])
    p.add_argument("--no-validate", action="store_true", help="no validation passes; export_best = final model")
    a = p.parse_args()
    manifest = a.manifest
    if a.train_role == "dev_train" and manifest.endswith("train_manifest.csv") and "dev_" not in Path(manifest).name:
        manifest = str(Path(manifest).with_name("dev_train_manifest.csv"))
    cfg = TrainConfig(manifest=manifest, val_manifest=a.val_manifest, run_dir=a.run_dir, roots=a.root, zips=a.zip,
                      train_role=a.train_role, val_max_files=a.val_max_files, steps=a.steps, batch_size=a.batch_size,
                      segment_sec=a.segment_sec, lr=a.lr, weight_decay=a.weight_decay, warmup_steps=a.warmup_steps,
                      grad_clip=a.grad_clip, bn_mode=a.bn_mode, seed=a.seed, val_every=a.val_every,
                      checkpoint_every=a.checkpoint_every, max_nan_steps=a.max_nan_steps, init_model_dir=a.init_model_dir,
                      row_filter=a.row_filter, validate=not a.no_validate)
    Trainer(cfg).run(resume=not a.no_resume)


if __name__ == "__main__":
    main()
