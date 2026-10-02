"""Run one registered paper method on one held-out target and random seed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from rul_tta.protocol_v1_config import DOMAINS, FROZEN_FIXED_EPOCHS, REPORT_SEEDS, ProtocolV1Config
from rul_tta.run_condition_processing_controls import execute as run_condition_control
from rul_tta.run_mdmt_unified import execute as run_mdmt
from rul_tta.run_p3_mechanism_ablation import execute as run_hccr_or_erm
from rul_tta.run_source_only_baselines import execute as run_baseline


METHODS = {
    "hccr": ("main", "raw_compensated"),
    "erm": ("main", "raw_no_condition"),
    "coral": ("baseline", "slow_coral"),
    "groupdro": ("baseline", "slow_group_dro"),
    "vrex": ("baseline", "slow_vrex"),
    "ms_dsn_style": ("baseline", "slow_ms_dsn"),
    "mdmt": ("mdmt", "mdmt"),
    "settings_input": ("condition", "settings_input"),
    "cluster_mean": ("condition", "healthy_cluster_mean"),
    "cluster_mean_scale": ("condition", "healthy_cluster_zscore"),
    "rafd_ocmm_style": ("condition", "rafd_ocmm_style"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--target", choices=DOMAINS, required=True)
    parser.add_argument("--seed", type=int, choices=REPORT_SEEDS, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper_runs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.raw_dir.is_dir():
        raise FileNotFoundError(args.raw_dir)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    family, variant = METHODS[args.method]
    config = ProtocolV1Config(
        target_domain=args.target,
        raw_dir=args.raw_dir.resolve(),
        output_dir=(args.output_dir / family).resolve(),
        seed=args.seed,
        device=args.device,
        epochs=FROZEN_FIXED_EPOCHS,
    )
    if family == "main":
        audit = run_hccr_or_erm(config, variant, FROZEN_FIXED_EPOCHS, args.force_train)
    elif family == "baseline":
        audit = run_baseline(config, variant, args.force_train, representation_mode="raw_only")
    elif family == "mdmt":
        audit = run_mdmt(config, args.force_train)
    else:
        audit = run_condition_control(config, variant, force_train=args.force_train)
    print(json.dumps({"method": args.method, "target": args.target, "seed": args.seed, "metrics": audit["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
