from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import SourceData
from .metrics import phm08_score
from .protocol_v1_config import PROTOCOL_V1_NAME, PROTOCOL_V1_VERSION, ProtocolV1Config
from .protocol_v1_data import subset_source
from .protocol_v1_models import ConditionCompRULNet, build_conditioncomp_rul_net


class _SourceWindowDataset(Dataset):
    def __init__(self, source: SourceData):
        self.windows = torch.from_numpy(source.windows)
        self.rul = torch.from_numpy(source.rul).to(dtype=torch.float32)
        self.domain = torch.from_numpy(source.domain).to(dtype=torch.long)

    def __len__(self) -> int:
        return len(self.rul)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "window": self.windows[index],
            "rul": self.rul[index],
            "domain": self.domain[index],
        }


def domain_balanced_loader(
    source: SourceData,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = _SourceWindowDataset(source)
    if not shuffle:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )
    domains, counts = np.unique(source.domain, return_counts=True)
    inverse = {int(domain): 1.0 / int(count) for domain, count in zip(domains, counts)}
    weights = torch.as_tensor(
        [inverse[int(domain)] for domain in source.domain], dtype=torch.double
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(source.rul),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=pin_memory,
    )


def split_training_validation_engines(
    source: SourceData,
    validation_fraction: float,
    seed: int,
) -> tuple[SourceData, SourceData]:
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    validation_indices: list[np.ndarray] = []
    for domain in sorted(np.unique(source.domain)):
        domain_engines = np.unique(source.engine[source.domain == domain]).copy()
        rng.shuffle(domain_engines)
        count = max(1, int(round(len(domain_engines) * validation_fraction)))
        count = min(count, max(1, len(domain_engines) - 1))
        validation_engines = domain_engines[:count]
        validation_indices.append(
            np.flatnonzero(
                (source.domain == domain) & np.isin(source.engine, validation_engines)
            )
        )
        train_indices.append(
            np.flatnonzero(
                (source.domain == domain) & ~np.isin(source.engine, validation_engines)
            )
        )
    return (
        subset_source(source, np.sort(np.concatenate(train_indices))),
        subset_source(source, np.sort(np.concatenate(validation_indices))),
    )


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    training_mse: float
    by_domain: dict[int, dict[str, float]]
    mean_normalized_risk: float = float("nan")
    worst_normalized_risk: float = float("nan")
    signed_bias: float = float("nan")
    epoch_seconds: float = float("nan")
    samples_per_second: float = float("nan")


@dataclass(frozen=True)
class P3TrainingResult:
    model: ConditionCompRULNet
    selected_epoch: int
    history: tuple[EpochMetrics, ...]
    checkpoint_path: Path | None


@torch.no_grad()
def _evaluate_source_macro(
    model: ConditionCompRULNet,
    source: SourceData,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, dict[str, float]], float]:
    model.eval()
    predictions: list[torch.Tensor] = []
    loader = domain_balanced_loader(
        source,
        batch_size,
        seed=0,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    for batch in loader:
        output = model(
            batch["window"].to(device, non_blocking=device.type == "cuda")
        )["prediction"]
        predictions.append(output)
    predicted = torch.cat(predictions).cpu().numpy()
    by_domain: dict[int, dict[str, float]] = {}
    all_bias: list[float] = []
    for domain in sorted(np.unique(source.domain)):
        engine_rmse: list[float] = []
        engine_score: list[float] = []
        engine_bias: list[float] = []
        for engine in np.unique(source.engine[source.domain == domain]):
            mask = (source.domain == domain) & (source.engine == engine)
            error = predicted[mask] - source.rul[mask]
            engine_rmse.append(float(np.sqrt(np.mean(error**2))))
            engine_score.append(
                phm08_score(source.rul[mask], predicted[mask]) / max(1, int(mask.sum()))
            )
            engine_bias.append(float(error.mean()))
        by_domain[int(domain)] = {
            "rmse_macro": float(np.mean(engine_rmse)),
            "score_macro": float(np.mean(engine_score)),
            "signed_bias": float(np.mean(engine_bias)),
            "engines": float(len(engine_rmse)),
        }
        all_bias.extend(engine_bias)
    return by_domain, float(np.mean(all_bias))


def _select_constrained_mean(history: list[EpochMetrics]) -> tuple[int, list[EpochMetrics]]:
    domains = sorted(history[0].by_domain)
    minimum_rmse = {
        domain: min(item.by_domain[domain]["rmse_macro"] for item in history)
        for domain in domains
    }
    minimum_score = {
        domain: min(item.by_domain[domain]["score_macro"] for item in history)
        for domain in domains
    }
    enriched: list[EpochMetrics] = []
    for item in history:
        normalized = [
            max(
                item.by_domain[domain]["rmse_macro"] / max(minimum_rmse[domain], 1e-12),
                item.by_domain[domain]["score_macro"] / max(minimum_score[domain], 1e-12),
            )
            for domain in domains
        ]
        enriched.append(
            EpochMetrics(
                epoch=item.epoch,
                training_mse=item.training_mse,
                by_domain=item.by_domain,
                mean_normalized_risk=float(np.mean(normalized)),
                worst_normalized_risk=float(np.max(normalized)),
                signed_bias=item.signed_bias,
                epoch_seconds=item.epoch_seconds,
                samples_per_second=item.samples_per_second,
            )
        )
    best_worst = min(item.worst_normalized_risk for item in enriched)
    feasible = [
        item for item in enriched if item.worst_normalized_risk <= 1.05 * best_worst
    ]
    selected = min(
        feasible,
        key=lambda item: (
            item.mean_normalized_risk + 0.05 * abs(item.signed_bias) / 125.0,
            item.epoch,
        ),
    )
    return selected.epoch, enriched


def _train_epochs(
    model: ConditionCompRULNet,
    source: SourceData,
    config: ProtocolV1Config,
    epochs: int,
    validation: SourceData | None,
    epoch_callback: Callable[[EpochMetrics], None] | None = None,
) -> tuple[list[EpochMetrics], dict[int, dict[str, torch.Tensor]]]:
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    history: list[EpochMetrics] = []
    states: dict[int, dict[str, torch.Tensor]] = {}
    loader = domain_balanced_loader(
        source,
        config.batch_size,
        config.seed,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        model.train()
        total_loss = torch.zeros((), device=device)
        total_samples = 0
        for batch in loader:
            non_blocking = device.type == "cuda"
            windows = batch["window"].to(device, non_blocking=non_blocking)
            targets = batch["rul"].to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(windows)["prediction"]
            loss = criterion(prediction, targets)
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.detach() * len(targets)
            total_samples += len(targets)
        if validation is not None:
            by_domain, bias = _evaluate_source_macro(
                model, validation, config.batch_size, device
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_seconds = time.perf_counter() - epoch_started
            epoch_metrics = EpochMetrics(
                epoch,
                float((total_loss / max(1, total_samples)).cpu()),
                by_domain,
                signed_bias=bias,
                epoch_seconds=epoch_seconds,
                samples_per_second=total_samples / max(epoch_seconds, 1e-12),
            )
            history.append(epoch_metrics)
            states[epoch] = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            if epoch_callback is not None:
                epoch_callback(epoch_metrics)
    return history, states


def develop_p3(
    source: SourceData,
    config: ProtocolV1Config,
    checkpoint_path: Path | None = None,
) -> P3TrainingResult:
    """Development-only checkpoint selection using source fitting engines."""
    training, validation = split_training_validation_engines(
        source, config.validation_fraction, config.split_seed
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = build_conditioncomp_rul_net(
        sensors=source.windows.shape[2],
        window_size=config.window_size,
        retained_bins=config.slow_bins,
        representation_mode="condition_residual",
    )
    history, states = _train_epochs(
        model, training, config, config.epochs, validation=validation
    )
    selected_epoch, history = _select_constrained_mean(history)
    model.load_state_dict(states[selected_epoch])
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "protocol": PROTOCOL_V1_NAME,
                "protocol_version": PROTOCOL_V1_VERSION,
                "method": "P3",
                "config": {**asdict(config), "raw_dir": str(config.raw_dir), "output_dir": str(config.output_dir)},
                "selected_epoch": selected_epoch,
                "model_state": states[selected_epoch],
                "history": [asdict(item) for item in history],
            },
            checkpoint_path,
        )
    return P3TrainingResult(model, selected_epoch, tuple(history), checkpoint_path)


def train_p3_with_validation(
    training: SourceData,
    validation: SourceData,
    config: ProtocolV1Config,
    checkpoint_path: Path | None = None,
    epoch_callback: Callable[[EpochMetrics], None] | None = None,
    representation_mode: str = "condition_residual",
) -> P3TrainingResult:
    """Train on explicit inner-source engines; never use pseudo-target labels for epoch selection."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = build_conditioncomp_rul_net(
        sensors=training.windows.shape[2],
        window_size=config.window_size,
        retained_bins=config.slow_bins,
        representation_mode=representation_mode,
    )
    history, states = _train_epochs(
        model,
        training,
        config,
        config.epochs,
        validation=validation,
        epoch_callback=epoch_callback,
    )
    selected_epoch, history = _select_constrained_mean(history)
    model.load_state_dict(states[selected_epoch])
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "protocol": PROTOCOL_V1_NAME,
                "protocol_version": PROTOCOL_V1_VERSION,
                "method": "P3-stage1-development",
                "representation_mode": representation_mode,
                "config": {
                    **asdict(config),
                    "raw_dir": str(config.raw_dir),
                    "output_dir": str(config.output_dir),
                },
                "selected_epoch": selected_epoch,
                "model_state": states[selected_epoch],
                "history": [asdict(item) for item in history],
            },
            checkpoint_path,
        )
    return P3TrainingResult(model, selected_epoch, tuple(history), checkpoint_path)


def train_final_p3(
    source: SourceData,
    config: ProtocolV1Config,
    fixed_epochs: int,
    checkpoint_path: Path | None = None,
    representation_mode: str = "condition_residual",
) -> P3TrainingResult:
    """Retrain once on all source-fitting engines for the frozen epoch count."""
    if fixed_epochs <= 0:
        raise ValueError("fixed_epochs must be positive.")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = build_conditioncomp_rul_net(
        sensors=source.windows.shape[2],
        window_size=config.window_size,
        retained_bins=config.slow_bins,
        representation_mode=representation_mode,
    )
    _train_epochs(model, source, config, fixed_epochs, validation=None)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "protocol": PROTOCOL_V1_NAME,
                "protocol_version": PROTOCOL_V1_VERSION,
                "method": "P3",
                "representation_mode": representation_mode,
                "config": {**asdict(config), "raw_dir": str(config.raw_dir), "output_dir": str(config.output_dir)},
                "selected_epoch": fixed_epochs,
                "model_state": model.state_dict(),
                "history": [],
            },
            checkpoint_path,
        )
    return P3TrainingResult(model, fixed_epochs, (), checkpoint_path)


@torch.no_grad()
def extract_source_features(
    model: ConditionCompRULNet,
    source: SourceData,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    temporal: list[np.ndarray] = []
    sensor: list[np.ndarray] = []
    for batch in domain_balanced_loader(source, batch_size, seed=0, shuffle=False):
        encoded = model.encode(batch["window"].to(device))
        temporal.append(encoded["temporal_features"].cpu().numpy())
        sensor.append(encoded["sensor_features"].cpu().numpy())
    return np.concatenate(temporal), np.concatenate(sensor)
