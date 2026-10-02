from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rul_tta.protocol_v1_config import FROZEN_FIXED_EPOCHS, REPORT_SEEDS
from rul_tta.protocol_v1_models import build_conditioncomp_rul_net


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_source_hashes() -> None:
    manifest = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["source_sha256"].items():
        actual = hashlib.sha256((ROOT / "src/rul_tta" / filename).read_bytes()).hexdigest()
        assert actual == expected, filename


def test_main_model_uses_complete_residual_window() -> None:
    torch.manual_seed(7)
    model = build_conditioncomp_rul_net(sensors=14, window_size=30, representation_mode="raw_only")
    input_window = torch.randn(2, 30, 14)
    output = model(input_window)
    assert output["prediction"].shape == (2,)
    assert output["temporal_features"].shape == (2, 128)
    assert output["sensor_features"].shape == (2, 128)
    assert torch.equal(output["model_input"], input_window)


def test_reported_run_grid_and_macro() -> None:
    runs = pd.read_csv(ROOT / "paper_metrics/main_runs.csv")
    assert len(runs) == 7 * 4 * 5
    assert set(runs.seed) == set(REPORT_SEEDS)
    assert FROZEN_FIXED_EPOCHS == 4
    assert not runs.duplicated(["method", "target", "seed"]).any()
    target_means = runs.groupby(["method", "target"])[["rmse", "phm08"]].mean()
    macro = target_means.groupby("method").mean()
    assert np.isclose(macro.loc["HCCR-Net", "rmse"], 22.143229574574978)
    assert np.isclose(macro.loc["HCCR-Net", "phm08"], 4338.263057273569)


def test_condition_control_grid_uses_same_hccr_runs() -> None:
    main = pd.read_csv(ROOT / "paper_metrics/main_runs.csv")
    controls = pd.read_csv(ROOT / "paper_metrics/condition_controls_runs.csv")
    assert len(controls) == 5 * 4 * 5
    left = main.loc[main.method.eq("HCCR-Net"), ["target", "seed", "rmse", "phm08"]]
    right = controls.loc[controls.method.eq("ConditionComp"), ["target", "seed", "rmse", "phm08"]]
    joined = left.merge(right, on=["target", "seed"], validate="one_to_one", suffixes=("_main", "_control"))
    assert len(joined) == 20
    assert np.allclose(joined.rmse_main, joined.rmse_control, rtol=0, atol=1e-8)
    assert np.allclose(joined.phm08_main, joined.phm08_control, rtol=0, atol=1e-6)
