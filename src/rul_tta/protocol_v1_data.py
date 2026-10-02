from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data import (
    SourceData,
    SourceStatistics,
    TargetStream,
    add_train_rul,
    apply_condition_response,
    condition_basis,
    load_test_domain,
    load_train_domain,
    make_source_windows,
    make_target_stream,
    normalize,
    normalize_settings,
    fit_source_statistics,
    sensor_columns,
    small_normal_equations,
    solve_small_linear_system,
)
from .protocol_v1_config import DOMAINS, ProtocolV1Config


@dataclass(frozen=True)
class ProtocolFoldData:
    fit: SourceData
    calibration: SourceData
    target: TargetStream
    statistics: SourceStatistics
    active_setting_dimensions: np.ndarray
    fit_engine_ids: tuple[int, ...]
    calibration_engine_ids: tuple[int, ...]


def _engine_length_strata(frame: pd.DataFrame) -> pd.Series:
    lengths = frame.groupby("unit_id").size().sort_index()
    ranks = lengths.rank(method="first")
    strata = pd.qcut(ranks, q=min(4, len(ranks)), labels=False, duplicates="drop")
    return pd.Series(strata.to_numpy(dtype=np.int64), index=lengths.index)


def split_fit_calibration_engines(
    frame: pd.DataFrame,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic engine-level, trajectory-length-stratified split."""
    if not 0.0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be in (0, 0.5).")
    rng = np.random.default_rng(seed)
    strata = _engine_length_strata(frame)
    calibration: list[int] = []
    for value in sorted(strata.unique()):
        engines = strata.index[strata == value].to_numpy(dtype=np.int64)
        engines = engines.copy()
        rng.shuffle(engines)
        count = max(1, int(round(len(engines) * calibration_fraction)))
        if count >= len(engines) and len(engines) > 1:
            count = len(engines) - 1
        calibration.extend(engines[:count].tolist())
    all_engines = np.sort(frame["unit_id"].unique().astype(np.int64))
    calibration_array = np.sort(np.asarray(calibration, dtype=np.int64))
    fit_array = all_engines[~np.isin(all_engines, calibration_array)]
    if not len(fit_array) or not len(calibration_array):
        raise ValueError("Engine split produced an empty partition.")
    return fit_array, calibration_array


def _subset_engines(frame: pd.DataFrame, engines: np.ndarray) -> pd.DataFrame:
    return frame.loc[frame["unit_id"].isin(engines)].copy()


def _combine_source_parts(parts: list[tuple[np.ndarray, ...]]) -> SourceData:
    raw_rul = np.concatenate([part[2] for part in parts])
    return SourceData(
        windows=np.concatenate([part[0] for part in parts]),
        rul=np.concatenate([part[1] for part in parts]),
        stage=np.zeros(len(raw_rul), dtype=np.int64),
        domain=np.concatenate([part[3] for part in parts]),
        engine=np.concatenate([part[4] for part in parts]),
        settings=np.concatenate([part[5] for part in parts]),
        cycle=np.concatenate([part[6] for part in parts]),
    )


def active_setting_dimensions(frames: Iterable[pd.DataFrame]) -> np.ndarray:
    settings = pd.concat(
        [frame.loc[:, ["setting1", "setting2", "setting3"]] for frame in frames],
        ignore_index=True,
    ).to_numpy(dtype=np.float64)
    std = settings.std(axis=0, ddof=1)
    spread = settings.max(axis=0) - settings.min(axis=0)
    return np.flatnonzero((std > 1e-8) & (spread > 1e-8)).astype(np.int64)


def assert_fixed_health_definition(frames: Iterable[pd.DataFrame]) -> None:
    count = sum(int((frame["raw_rul"] >= 125.0).sum()) for frame in frames)
    if count == 0:
        raise ValueError("Protocol v1.0 requires source health samples with raw RUL >= 125.")


def fit_frozen_condition_statistics(
    frames: Iterable[pd.DataFrame],
    feature_columns: tuple[str, ...],
) -> SourceStatistics:
    """Protocol-only conditioner with the exact raw-RUL>=125 health definition."""
    frames = list(frames)
    all_settings = pd.concat(
        [frame.loc[:, ["setting1", "setting2", "setting3"]] for frame in frames],
        ignore_index=True,
    ).to_numpy(dtype=np.float32)
    setting_mean = all_settings.mean(axis=0).astype(np.float32)
    setting_std = all_settings.std(axis=0, ddof=1).astype(np.float32)
    setting_std = np.where(setting_std > 1e-8, setting_std, 1.0).astype(np.float32)

    healthy = pd.concat(
        [frame.loc[frame["raw_rul"] >= 125.0] for frame in frames],
        ignore_index=True,
    )
    if healthy.empty:
        raise ValueError("No source fitting samples satisfy raw RUL >= 125.")
    healthy_settings = healthy.loc[
        :, ["setting1", "setting2", "setting3"]
    ].to_numpy(dtype=np.float32)
    healthy_sensors = healthy.loc[:, feature_columns].to_numpy(dtype=np.float32)
    design = condition_basis(healthy_settings, setting_mean, setting_std)
    regularizer = np.eye(10, dtype=np.float64) * 1e-3
    regularizer[0, 0] = 0.0
    gram, cross = small_normal_equations(
        design, healthy_sensors.astype(np.float64)
    )
    coefficients = solve_small_linear_system(
        gram + regularizer, cross
    ).astype(np.float32)

    all_sensors = pd.concat(
        [frame.loc[:, feature_columns] for frame in frames],
        ignore_index=True,
    ).to_numpy(dtype=np.float32)
    residual = all_sensors - apply_condition_response(
        condition_basis(all_settings, setting_mean, setting_std), coefficients
    )
    residual_mean = residual.mean(axis=0).astype(np.float32)
    residual_std = residual.std(axis=0, ddof=1).astype(np.float32)
    residual_std = np.where(residual_std > 1e-8, residual_std, 1.0).astype(np.float32)
    return SourceStatistics(
        preprocessing="continuous_comp",
        mean=residual_mean,
        std=residual_std,
        feature_columns=feature_columns,
        setting_mean=setting_mean,
        setting_std=setting_std,
        condition_coefficients=coefficients,
        regime_centers=np.empty((0, 3), dtype=np.float32),
        regime_mean=np.empty((0, len(feature_columns)), dtype=np.float32),
        regime_std=np.empty((0, len(feature_columns)), dtype=np.float32),
    )


def prepare_protocol_fold(
    config: ProtocolV1Config,
    preprocessing: str = "continuous_comp",
) -> ProtocolFoldData:
    """Prepare one outer fold without opening target RUL labels."""
    config.validate()
    if preprocessing not in {"continuous_comp", "global_zscore"}:
        raise ValueError(
            "Protocol ablation preprocessing must be continuous_comp or global_zscore."
        )
    sources = tuple(domain for domain in DOMAINS if domain != config.target_domain)
    fit_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    fit_engine_ids: list[int] = []
    calibration_engine_ids: list[int] = []
    for domain_id, domain in enumerate(sources):
        frame = add_train_rul(load_train_domain(config.raw_dir, domain), config.rul_cap)
        fit_units, calibration_units = split_fit_calibration_engines(
            frame, config.calibration_fraction, config.split_seed + domain_id
        )
        fit_frame = _subset_engines(frame, fit_units)
        calibration_frame = _subset_engines(frame, calibration_units)
        fit_frames.append(fit_frame)
        calibration_frames.append(calibration_frame)
        fit_engine_ids.extend((domain_id * 10_000 + fit_units).tolist())
        calibration_engine_ids.extend((domain_id * 10_000 + calibration_units).tolist())

    features = sensor_columns()
    if preprocessing == "continuous_comp":
        assert_fixed_health_definition(fit_frames)
        statistics = fit_frozen_condition_statistics(fit_frames, features)
    else:
        statistics = fit_source_statistics(
            fit_frames,
            features,
            preprocessing="global_zscore",
            seed=config.split_seed,
        )
    active = active_setting_dimensions(fit_frames)

    fit_parts: list[tuple[np.ndarray, ...]] = []
    calibration_parts: list[tuple[np.ndarray, ...]] = []
    for domain_id, (fit_frame, calibration_frame) in enumerate(zip(fit_frames, calibration_frames)):
        fit_normalized = normalize_settings(normalize(fit_frame, statistics), statistics)
        calibration_normalized = normalize_settings(
            normalize(calibration_frame, statistics), statistics
        )
        fit_parts.append(make_source_windows(fit_normalized, features, config.window_size, domain_id))
        calibration_parts.append(
            make_source_windows(calibration_normalized, features, config.window_size, domain_id)
        )

    target_frame = normalize(load_test_domain(config.raw_dir, config.target_domain), statistics)
    target_frame = normalize_settings(target_frame, statistics)
    target = make_target_stream(target_frame, features, config.window_size)
    return ProtocolFoldData(
        fit=_combine_source_parts(fit_parts),
        calibration=_combine_source_parts(calibration_parts),
        target=target,
        statistics=statistics,
        active_setting_dimensions=active,
        fit_engine_ids=tuple(sorted(fit_engine_ids)),
        calibration_engine_ids=tuple(sorted(calibration_engine_ids)),
    )


def subset_source(source: SourceData, indices: np.ndarray) -> SourceData:
    return SourceData(
        windows=source.windows[indices],
        rul=source.rul[indices],
        stage=source.stage[indices],
        domain=source.domain[indices],
        engine=source.engine[indices],
        settings=source.settings[indices],
        cycle=source.cycle[indices],
    )
