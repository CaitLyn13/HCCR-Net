from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROTOCOL_V1_NAME = "Protocol v1.0 - Frozen"
PROTOCOL_V1_VERSION = 1
DOMAINS = ("FD001", "FD002", "FD003", "FD004")
DEVELOPMENT_SEEDS = (3, 13, 23)
REPORT_SEEDS = (7, 17, 27, 37, 47)
SPLIT_SEED = 2026
FROZEN_FIXED_EPOCHS = 4
PROTOCOL_METHODS = ("P3", "P4", "P5", "A1", "A2", "A3")
CORE_FORMAL_METHODS = ("P3", "P4")
FAST_ADAPTER_METHODS = ("P5", "A1", "A2", "A3")


@dataclass(frozen=True)
class ProtocolV1Config:
    target_domain: str
    raw_dir: Path
    output_dir: Path
    seed: int = 3
    device: str = "cuda"
    window_size: int = 30
    rul_cap: float = 125.0
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    validation_fraction: float = 0.15
    calibration_fraction: float = 0.15
    split_seed: int = SPLIT_SEED
    slow_bins: int = 5
    fourier_probability: float = 0.5
    fourier_lambda_low: float = 0.2
    fourier_lambda_high: float = 0.8
    fourier_rul_tolerance: float = 5.0
    condition_max_clusters: int = 6
    condition_gmm_samples: int = 50_000
    prototype_min_samples: int = 32
    prototype_rul_width: float = 10.0
    prototype_topk: int = 16
    setting_distance_weight: float = 0.1
    retrieval_temperature: float = 0.1
    fast_rank: int = 4
    fast_micro_batch_size: int = 8
    fast_update_stride: int = 4
    fast_minimum_batch_size: int = 4
    tta_inference_batch_size: int = 512
    fast_learning_rate: float = 1e-4
    prototype_loss_weight: float = 1.0
    consistency_loss_weight: float = 0.2
    trust_loss_weight: float = 1e-3
    distance_quantile: float = 0.95
    variance_quantile: float = 0.90
    memory_max_weight: float = 0.3
    point_mask_probability: float = 0.05
    augmentation_noise_std: float = 0.01
    gradient_clip: float = 1.0
    calibration_bootstrap_repetitions: int = 1_000

    def validate(self) -> None:
        if self.window_size != 30:
            raise ValueError("Protocol v1.0 fixes window_size=30.")
        if self.rul_cap != 125.0:
            raise ValueError("Protocol v1.0 fixes rul_cap=125.")
        if self.slow_bins not in {3, 5, 7}:
            raise ValueError("slow_bins must be one of 3, 5, 7.")
        if self.prototype_topk not in {4, 8, 16}:
            raise ValueError("prototype_topk must be one of 4, 8, 16.")
        if self.distance_quantile not in {0.90, 0.95}:
            raise ValueError("distance_quantile must be 0.90 or 0.95.")
        if self.variance_quantile not in {0.85, 0.90, 0.95}:
            raise ValueError("variance_quantile must be 0.85, 0.90, or 0.95.")
        if not 0.0 <= self.fourier_probability <= 1.0:
            raise ValueError("fourier_probability must be in [0, 1].")
        if not 0.0 <= self.fourier_lambda_low <= self.fourier_lambda_high <= 1.0:
            raise ValueError("Invalid Fourier lambda interval.")
        if self.fast_rank <= 0:
            raise ValueError("fast_rank must be positive.")
        if self.fast_minimum_batch_size < 2:
            raise ValueError("fast_minimum_batch_size must be at least 2.")
        if self.fast_micro_batch_size < self.fast_minimum_batch_size:
            raise ValueError(
                "fast_micro_batch_size must be >= fast_minimum_batch_size."
            )
        if not 1 <= self.fast_update_stride <= self.fast_micro_batch_size:
            raise ValueError(
                "fast_update_stride must be in [1, fast_micro_batch_size]."
            )
        if self.tta_inference_batch_size <= 0:
            raise ValueError("tta_inference_batch_size must be positive.")
