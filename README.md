# HCCR-Net: reproducibility code

This standalone folder accompanies the manuscript *Healthy-Reference Continuous Condition-Response Separation for Cross-Subset Turbofan Remaining Useful Life Prediction*. It contains the source-only C-MAPSS implementation, the six main comparison methods, four condition-processing controls, the reference and architecture analyses, and the source-calibration interaction probe. No historical TTA, fixed-low-pass, selective-scale or XJTU experiment entry points, data, figures or tests are shipped. A few dormant names remain inside the hash-verified training core so that its initialization and checkpoint behavior are unchanged.

## Data and environment

Obtain the public NASA C-MAPSS files `train_FD001.txt`–`train_FD004.txt`, `test_FD001.txt`–`test_FD004.txt`, and `RUL_FD001.txt`–`RUL_FD004.txt`. Place them in a directory of your choice. Raw data are not redistributed here. From this folder, install the package and check inputs:

```powershell
python -m pip install -e '.[test]'
python scripts/verify_inputs.py --raw-dir D:\path\to\CMAPSSData
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest -q -p no:cacheprovider
```

The recorded run used Python 3.10.20, PyTorch `2.12.0.dev20260408+cu128`, NumPy 2.2.6, pandas 2.3.3, CUDA 12.8, deterministic PyTorch operations, one CPU thread and disabled TF32. The general package dependencies allow installation elsewhere, but exact numerical replay is only expected under a matching runtime and data checksum. The hashes and runtime details are in `provenance.json`.

## Reproduce a single run

```powershell
python scripts/run_experiment.py --method hccr --target FD001 --seed 7 --raw-dir D:\path\to\CMAPSSData
```

`--method` accepts `hccr`, `erm`, `coral`, `groupdro`, `vrex`, `ms_dsn_style`, `mdmt`, `settings_input`, `cluster_mean`, `cluster_mean_scale`, and `rafd_ocmm_style`. The target is one of FD001–FD004; reported seeds are 7, 17, 27, 37 and 47. Run every method–target–seed combination to reconstruct the full main and condition-processing comparisons. Each run writes its checkpoint, source-fitted preprocessing, predictions saved before target labels are opened, final-engine predictions, metrics, and audit to `results/paper_runs/`. Outputs are local and excluded from Git.

For the complete 11-method, four-target, five-seed grid in PowerShell, set `$rawDir` to your C-MAPSS directory and run:

```powershell
$rawDir = 'D:\path\to\CMAPSSData'
$methods = 'hccr','erm','coral','groupdro','vrex','ms_dsn_style','mdmt','settings_input','cluster_mean','cluster_mean_scale','rafd_ocmm_style'
foreach ($method in $methods) {
  foreach ($target in 'FD001','FD002','FD003','FD004') {
    foreach ($seed in 7,17,27,37,47) {
      python scripts/run_experiment.py --method $method --target $target --seed $seed --raw-dir $rawDir
      if ($LASTEXITCODE -ne 0) { throw "Run failed: $method $target $seed" }
    }
  }
}
```

The first seven methods form the main comparison. HCCR-Net and ERM share the predictor and differ in preprocessing. The remaining four methods form the separate, same-version condition-processing comparison. MDMT retains its published BiLSTM-based training schedule rather than the four-epoch shared-backbone budget. The main and condition-control HCCR-Net rows must not be counted as two independent runs: they refer to the same frozen-version method and use the same target–seed predictions.

After all runs complete, recompute metrics directly from the saved final-engine predictions:

```powershell
python scripts/summarise_runs.py --run-root results/paper_runs --output-dir results/summary --require-complete
```

The summary script computes each target mean first and then weights the four targets equally. `paper_metrics/` contains the corresponding lightweight, path-free run-level metrics and published summaries. The four-target HCCR-Net macro averages are RMSE 22.1432296 and PHM08 4338.2631 for the frozen-version comparison. These data are included for result checking, not as a replacement for the raw dataset or training scripts.

## Reproduce the paper analyses

After all 20 HCCR-Net and 20 ERM runs are present:

```powershell
python scripts/mechanisms/run_conditioncomp_mechanism_validation.py --raw-dir D:\path\to\CMAPSSData --stage all
python scripts/mechanisms/run_structural_ablation.py --raw-dir D:\path\to\CMAPSSData
python scripts/mechanisms/run_source_condition_interaction_probe.py --raw-dir D:\path\to\CMAPSSData
```

The first command covers near-healthy/all-life/matched-random references, source-calibration probes, trajectory ordering and response-surface resampling. Its stress stage uses the already-trained HCCR-Net and ERM checkpoints. The architecture command independently trains dual-branch, temporal-only, sensor-only and single-receptive-field variants. The interaction probe fits the conditioner from source-fitting engines and evaluates probes on disjoint source-calibration engines; it does not open target test files or labels.

## Layout and provenance

- `src/rul_tta/`: 15 selected, byte-identical modules from the deterministic-v2 frozen training snapshot. Some internal historical identifiers remain to preserve checkpoint and numerical compatibility; no retired experiment has a release entry point.
- `scripts/`: portable paper-only execution, aggregation and input verification.
- `scripts/mechanisms/`: paper reference, architecture and interaction analyses, adapted to use this standalone directory.
- `paper_metrics/`: current-paper numerical summaries, with machine-specific source paths removed from run-level CSVs.
- `tests/`: minimal release smoke checks, not the exploratory project test suite.
- `provenance.json`: frozen-source and raw-data SHA-256 values plus the recorded runtime.

No historical result directories, model candidates, raw C-MAPSS files, manuscript artwork or unrelated tests were copied into this package. The original research repository remains unchanged by this packaging operation, except for its package registration and catalogue entry.
