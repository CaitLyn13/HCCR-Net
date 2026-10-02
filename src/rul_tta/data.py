from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

DROP_SENSORS = (1, 5, 6, 10, 16, 18, 19)
NORMALIZATION_CLIP = 5.0
RAW_COLUMNS = ["unit_id", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]


@dataclass(frozen=True)
class SourceStatistics:
    preprocessing: str
    mean: np.ndarray
    std: np.ndarray
    feature_columns: tuple[str, ...]
    setting_mean: np.ndarray
    setting_std: np.ndarray
    condition_coefficients: np.ndarray
    regime_centers: np.ndarray
    regime_mean: np.ndarray
    regime_std: np.ndarray
    stage_lower: float = 0.0
    stage_upper: float = 0.0


@dataclass(frozen=True)
class SourceData:
    windows: np.ndarray
    rul: np.ndarray
    stage: np.ndarray
    domain: np.ndarray
    engine: np.ndarray
    settings: np.ndarray
    cycle: np.ndarray


@dataclass(frozen=True)
class TargetStream:
    windows: np.ndarray
    settings: np.ndarray
    engine: np.ndarray
    cycle: np.ndarray
    is_final: np.ndarray


def sensor_columns() -> tuple[str, ...]:
    return tuple(f"s{i}" for i in range(1, 22) if i not in DROP_SENSORS)


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, : len(RAW_COLUMNS)]
    frame.columns = RAW_COLUMNS
    return frame


def load_train_domain(raw_dir: Path, domain: str) -> pd.DataFrame:
    return _load_frame(raw_dir / f"train_{domain}.txt")


def load_test_domain(raw_dir: Path, domain: str) -> pd.DataFrame:
    """Load target covariates without opening the target RUL file."""
    return _load_frame(raw_dir / f"test_{domain}.txt")


def load_target_rul(raw_dir: Path, domain: str, expected_engines: int | None = None) -> np.ndarray:
    path = raw_dir / f"RUL_{domain}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    rul = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=np.float32)
    if expected_engines is not None and expected_engines != len(rul):
        raise ValueError(f"{domain}: test engines ({expected_engines}) and RUL rows ({len(rul)}) do not match.")
    return rul


def load_raw_domain(raw_dir: Path, domain: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Compatibility helper for explicit offline data inspection."""
    train = load_train_domain(raw_dir, domain)
    test = load_test_domain(raw_dir, domain)
    return train, test, load_target_rul(raw_dir, domain, test["unit_id"].nunique())


def add_train_rul(frame: pd.DataFrame, cap: float) -> pd.DataFrame:
    out = frame.copy()
    maximum = out.groupby("unit_id")["cycle"].transform("max")
    out["raw_rul"] = (maximum - out["cycle"]).clip(lower=0.0).astype(np.float32)
    out["rul"] = out["raw_rul"].clip(upper=cap).astype(np.float32)
    return out


def _kmeans(values: np.ndarray, clusters: int, seed: int, iterations: int = 100) -> np.ndarray:
    """Small deterministic K-means used to avoid target-fitted preprocessing."""
    rng = np.random.default_rng(seed)
    unique = np.unique(values, axis=0)
    if len(unique) < clusters:
        raise ValueError(f"Only {len(unique)} unique operating settings for {clusters} regimes")
    centers = unique[rng.choice(len(unique), clusters, replace=False)].astype(np.float64)
    for _ in range(iterations):
        distance = np.square(values[:, None, :] - centers[None, :, :]).sum(axis=2)
        labels = distance.argmin(axis=1)
        updated = centers.copy()
        for cluster in range(clusters):
            selected = values[labels == cluster]
            if len(selected):
                updated[cluster] = selected.mean(axis=0)
        if np.allclose(updated, centers, atol=1e-7, rtol=0.0):
            break
        centers = updated
    order = np.lexsort(tuple(centers[:, index] for index in reversed(range(centers.shape[1]))))
    return centers[order].astype(np.float32)


def fit_source_statistics(
    source_frames: Iterable[pd.DataFrame],
    feature_columns: tuple[str, ...],
    preprocessing: str = "continuous_comp",
    regime_clusters: int = 6,
    seed: int = 7,
) -> SourceStatistics:
    if preprocessing not in {"global_zscore", "regime_norm", "continuous_comp"}:
        raise ValueError(f"Unknown preprocessing: {preprocessing}")
    frames = list(source_frames)
    settings = pd.concat([frame.loc[:, ["setting1", "setting2", "setting3"]] for frame in frames], ignore_index=True)
    setting_mean = settings.mean(axis=0).to_numpy(dtype=np.float32)
    setting_std = settings.std(axis=0).replace(0, 1.0).to_numpy(dtype=np.float32)
    all_settings = settings.to_numpy(dtype=np.float32)
    normalized_settings = np.clip(
        (all_settings - setting_mean) / setting_std,
        -NORMALIZATION_CLIP,
        NORMALIZATION_CLIP,
    ).astype(np.float32)
    all_sensors = pd.concat([frame.loc[:, feature_columns] for frame in frames], ignore_index=True).to_numpy(
        dtype=np.float32
    )
    global_mean = all_sensors.mean(axis=0).astype(np.float32)
    global_std = all_sensors.std(axis=0, ddof=1).astype(np.float32)
    global_std = np.where(global_std > 1e-8, global_std, 1.0).astype(np.float32)

    coefficients = np.zeros((10, len(feature_columns)), dtype=np.float32)
    centers = np.empty((0, 3), dtype=np.float32)
    regime_mean = np.empty((0, len(feature_columns)), dtype=np.float32)
    regime_std = np.empty((0, len(feature_columns)), dtype=np.float32)
    mean, std = global_mean, global_std

    if preprocessing == "global_zscore":
        return SourceStatistics(
            preprocessing, mean, std, feature_columns, setting_mean, setting_std,
            coefficients, centers, regime_mean, regime_std
        )

    if preprocessing == "regime_norm":
        centers = _kmeans(normalized_settings, regime_clusters, seed)
        labels = np.square(normalized_settings[:, None, :] - centers[None, :, :]).sum(axis=2).argmin(axis=1)
        regime_means, regime_stds = [], []
        for cluster in range(regime_clusters):
            selected = all_sensors[labels == cluster]
            if not len(selected):
                regime_means.append(global_mean)
                regime_stds.append(global_std)
                continue
            cluster_mean = selected.mean(axis=0).astype(np.float32)
            cluster_std = selected.std(axis=0, ddof=1).astype(np.float32) if len(selected) > 1 else global_std
            regime_means.append(cluster_mean)
            regime_stds.append(np.where(cluster_std > 1e-8, cluster_std, global_std))
        regime_mean = np.asarray(regime_means, dtype=np.float32)
        regime_std = np.asarray(regime_stds, dtype=np.float32)
        return SourceStatistics(
            preprocessing, mean, std, feature_columns, setting_mean, setting_std,
            coefficients, centers, regime_mean, regime_std
        )

    plateau = pd.concat(
        [frame.loc[frame["rul"] >= frame["rul"].max(), :] for frame in frames],
        ignore_index=True,
    )
    plateau_settings = plateau.loc[:, ["setting1", "setting2", "setting3"]].to_numpy(dtype=np.float32)
    plateau_sensors = plateau.loc[:, feature_columns].to_numpy(dtype=np.float32)
    design = condition_basis(plateau_settings, setting_mean, setting_std)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * 1e-3
    regularizer[0, 0] = 0.0
    gram, cross = small_normal_equations(design, plateau_sensors.astype(np.float64))
    coefficients = solve_small_linear_system(gram + regularizer, cross).astype(np.float32)
    residual = all_sensors - apply_condition_response(
        condition_basis(all_settings, setting_mean, setting_std), coefficients
    )
    mean = residual.mean(axis=0).astype(np.float32)
    std = residual.std(axis=0, ddof=1).astype(np.float32)
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
    return SourceStatistics(
        preprocessing, mean, std, feature_columns, setting_mean, setting_std,
        coefficients, centers, regime_mean, regime_std
    )


def solve_small_linear_system(matrix: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    """Pivoted Gauss-Jordan solve without Windows LAPACK dependencies."""
    a = np.asarray(matrix, dtype=np.float64).copy()
    b = np.asarray(right_hand_side, dtype=np.float64).copy()
    size = a.shape[0]
    for column in range(size):
        pivot = column + int(np.argmax(np.abs(a[column:, column])))
        if abs(a[pivot, column]) < 1e-12:
            raise ValueError("Condition-response normal equation is singular")
        if pivot != column:
            a[[column, pivot]] = a[[pivot, column]]
            b[[column, pivot]] = b[[pivot, column]]
        scale = a[column, column]
        a[column] /= scale
        b[column] /= scale
        for row in range(size):
            if row == column:
                continue
            factor = a[row, column]
            if factor == 0.0:
                continue
            a[row] -= factor * a[column]
            b[row] -= factor * b[column]
    return b


def small_normal_equations(design: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = design.shape[1]
    outputs = targets.shape[1]
    gram = np.empty((features, features), dtype=np.float64)
    cross = np.empty((features, outputs), dtype=np.float64)
    for row in range(features):
        for column in range(features):
            gram[row, column] = np.sum(design[:, row] * design[:, column], dtype=np.float64)
        for output in range(outputs):
            cross[row, output] = np.sum(design[:, row] * targets[:, output], dtype=np.float64)
    return gram, cross


def apply_condition_response(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    output = np.zeros((len(design), coefficients.shape[1]), dtype=np.float64)
    for feature in range(design.shape[1]):
        output += design[:, feature, None] * coefficients[feature][None, :]
    return output


def condition_basis(settings: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = np.clip((settings - mean) / std, -NORMALIZATION_CLIP, NORMALIZATION_CLIP).astype(np.float64)
    s1, s2, s3 = normalized.T
    return np.column_stack(
        [
            np.ones(len(normalized)),
            s1,
            s2,
            s3,
            s1 * s1,
            s2 * s2,
            s3 * s3,
            s1 * s2,
            s1 * s3,
            s2 * s3,
        ]
    )


def normalize(frame: pd.DataFrame, statistics: SourceStatistics) -> pd.DataFrame:
    out = frame.copy()
    features = out.loc[:, statistics.feature_columns].to_numpy(dtype=np.float32)
    settings = out.loc[:, ["setting1", "setting2", "setting3"]].to_numpy(dtype=np.float32)
    if statistics.preprocessing == "global_zscore":
        normalized = (features - statistics.mean) / statistics.std
    elif statistics.preprocessing == "regime_norm":
        normalized_settings = np.clip(
            (settings - statistics.setting_mean) / statistics.setting_std,
            -NORMALIZATION_CLIP,
            NORMALIZATION_CLIP,
        )
        labels = np.square(
            normalized_settings[:, None, :] - statistics.regime_centers[None, :, :]
        ).sum(axis=2).argmin(axis=1)
        normalized = (
            features - statistics.regime_mean[labels]
        ) / statistics.regime_std[labels]
    elif statistics.preprocessing == "continuous_comp":
        condition = apply_condition_response(
            condition_basis(settings, statistics.setting_mean, statistics.setting_std),
            statistics.condition_coefficients,
        )
        residual = features - condition
        normalized = (residual - statistics.mean) / statistics.std
    else:
        raise ValueError(f"Unknown preprocessing: {statistics.preprocessing}")
    normalized = np.clip(normalized, -NORMALIZATION_CLIP, NORMALIZATION_CLIP)
    # Assigning floats into integer sensor columns is deprecated in pandas 2.3.
    for index, column in enumerate(statistics.feature_columns):
        out[column] = normalized[:, index].astype(np.float32)
    return out


def normalize_settings(frame: pd.DataFrame, statistics: SourceStatistics) -> pd.DataFrame:
    out = frame.copy()
    columns = ["setting1", "setting2", "setting3"]
    values = out.loc[:, columns].to_numpy(dtype=np.float32)
    normalized = np.clip(
        (values - statistics.setting_mean) / statistics.setting_std,
        -NORMALIZATION_CLIP,
        NORMALIZATION_CLIP,
    )
    for index, column in enumerate(columns):
        out[column] = normalized[:, index].astype(np.float32)
    return out


def stage_from_rul(rul: np.ndarray, lower_quantile: float, upper_quantile: float) -> np.ndarray:
    """Source-only, fold-specific quantile proxy: early, middle, late."""
    return np.where(rul > upper_quantile, 0, np.where(rul >= lower_quantile, 1, 2)).astype(np.int64)


def _left_pad(values: np.ndarray, size: int) -> np.ndarray:
    if len(values) >= size:
        return values[-size:]
    return np.concatenate([np.repeat(values[:1], size - len(values), axis=0), values], axis=0)


def make_source_windows(frame: pd.DataFrame, feature_columns: tuple[str, ...], window_size: int, domain_id: int):
    x, y, raw_y, domain, engine, settings, cycle = [], [], [], [], [], [], []
    for unit_id, part in frame.groupby("unit_id"):
        part = part.sort_values("cycle")
        values = part.loc[:, feature_columns].to_numpy(dtype=np.float32)
        rul = part["rul"].to_numpy(dtype=np.float32)
        raw_rul = part["raw_rul"].to_numpy(dtype=np.float32)
        operating = part.loc[:, ["setting1", "setting2", "setting3"]].to_numpy(dtype=np.float32)
        cycles = part["cycle"].to_numpy(dtype=np.int64)
        # Match deployment exactly: the first cycles are left-padded instead of
        # silently dropping the first ``window_size - 1`` source observations.
        for end in range(len(part)):
            x.append(_left_pad(values[: end + 1], window_size))
            y.append(rul[end])
            raw_y.append(raw_rul[end])
            domain.append(domain_id)
            engine.append(domain_id * 10000 + int(unit_id))
            settings.append(operating[end])
            cycle.append(cycles[end])
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(raw_y, dtype=np.float32),
        np.asarray(domain, dtype=np.int64),
        np.asarray(engine, dtype=np.int64),
        np.asarray(settings, dtype=np.float32),
        np.asarray(cycle, dtype=np.int64),
    )


def make_target_stream(frame: pd.DataFrame, feature_columns: tuple[str, ...], window_size: int) -> TargetStream:
    x, settings, engine, cycle, final = [], [], [], [], []
    unit_ids = sorted(frame["unit_id"].unique())
    for unit_id in unit_ids:
        part = frame.loc[frame["unit_id"] == unit_id].sort_values("cycle")
        values = part.loc[:, feature_columns].to_numpy(dtype=np.float32)
        cycles = part["cycle"].to_numpy(dtype=np.int64)
        for end in range(len(part)):
            x.append(_left_pad(values[: end + 1], window_size))
            settings.append(part.iloc[end][["setting1", "setting2", "setting3"]].to_numpy(dtype=np.float32))
            engine.append(int(unit_id))
            cycle.append(int(cycles[end]))
            final.append(end == len(part) - 1)
    return TargetStream(
        windows=np.asarray(x, dtype=np.float32),
        settings=np.asarray(settings, dtype=np.float32),
        engine=np.asarray(engine, dtype=np.int64),
        cycle=np.asarray(cycle, dtype=np.int64),
        is_final=np.asarray(final, dtype=bool),
    )


def prepare_fold(
    raw_dir: Path,
    source_domains: tuple[str, ...],
    target_domain: str,
    window_size: int,
    rul_cap: float,
    preprocessing: str = "continuous_comp",
    regime_clusters: int = 6,
    seed: int = 7,
):
    feature_columns = sensor_columns()
    source_frames = []
    for domain in source_domains:
        train = load_train_domain(raw_dir, domain)
        source_frames.append(add_train_rul(train, rul_cap))
    statistics = fit_source_statistics(
        source_frames, feature_columns, preprocessing, regime_clusters, seed
    )
    parts = []
    for domain_id, frame in enumerate(source_frames):
        normalized = normalize_settings(normalize(frame, statistics), statistics)
        parts.append(make_source_windows(normalized, feature_columns, window_size, domain_id))
    source_rul = np.concatenate([part[1] for part in parts])
    source_raw_rul = np.concatenate([part[2] for part in parts])
    # Quantiles are learned from this fold's source labels only.  They are a
    # balanced stage proxy, not claimed physical health boundaries.
    lower, upper = np.quantile(source_raw_rul, [1.0 / 3.0, 2.0 / 3.0])
    statistics = replace(statistics, stage_lower=float(lower), stage_upper=float(upper))
    source = SourceData(
        windows=np.concatenate([part[0] for part in parts]),
        rul=source_rul,
        stage=stage_from_rul(source_raw_rul, float(lower), float(upper)),
        domain=np.concatenate([part[3] for part in parts]),
        engine=np.concatenate([part[4] for part in parts]),
        settings=np.concatenate([part[5] for part in parts]),
        cycle=np.concatenate([part[6] for part in parts]),
    )
    target_test = load_test_domain(raw_dir, target_domain)
    normalized_target = normalize(target_test, statistics)
    normalized_target = normalize_settings(normalized_target, statistics)
    target = make_target_stream(normalized_target, feature_columns, window_size)
    return source, target, statistics


def split_source_engines(source: SourceData, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    validation_engines = []
    for domain in np.unique(source.domain):
        engines = np.unique(source.engine[source.domain == domain])
        count = max(1, round(len(engines) * fraction))
        validation_engines.extend(rng.choice(engines, count, replace=False))
    valid = np.isin(source.engine, np.asarray(validation_engines))
    return np.flatnonzero(~valid), np.flatnonzero(valid)


def build_balanced_loader(source: SourceData, indices: np.ndarray, batch_size: int) -> DataLoader:
    domains = source.domain[indices]
    counts = np.bincount(domains, minlength=int(domains.max()) + 1)
    weights = np.asarray([1.0 / counts[value] for value in domains], dtype=np.float64)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(indices), replacement=True)
    dataset = TensorDataset(
        torch.from_numpy(source.windows[indices]),
        torch.from_numpy(source.rul[indices]),
        torch.from_numpy(source.stage[indices]),
        torch.from_numpy(source.domain[indices]),
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
