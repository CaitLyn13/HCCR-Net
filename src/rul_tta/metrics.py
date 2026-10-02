from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    exponent = np.where(error < 0, -error / 13.0, error / 10.0)
    # Very poor calibration models can produce extreme predictions.  Clipping
    # only prevents numerical overflow; exp(50) is already an overwhelming
    # penalty and preserves the ranking of all practically relevant errors.
    return float(np.expm1(np.clip(exponent, 0.0, 50.0)).sum(dtype=np.float64))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "phm08": phm08_score(y_true, y_pred),
        "signed_bias": float(np.mean(y_pred - y_true)),
    }
