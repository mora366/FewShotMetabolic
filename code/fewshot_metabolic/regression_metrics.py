from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr


@dataclass(frozen=True)
class RegressionMetrics:
    rmse: float
    mae: float
    pearson_r: float
    r_squared: float


@dataclass(frozen=True)
class IntervalMetrics:
    mean_width: float
    coverage: float


@dataclass(frozen=True)
class BlandAltmanMetrics:
    mean_bias: float
    lower_limit: float
    upper_limit: float
    proportional_bias_r: float
    proportional_bias_p: float


def _regression_arrays(
    target: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    target_array = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction_array = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target_array.shape != prediction_array.shape or target_array.size < 2:
        raise ValueError("regression arrays require matching nontrivial shapes")
    if np.any(~np.isfinite(target_array)) or np.any(~np.isfinite(prediction_array)):
        raise ValueError("regression arrays must be finite")
    return target_array, prediction_array


def root_mean_squared_error(target: np.ndarray, prediction: np.ndarray) -> float:
    target_array, prediction_array = _regression_arrays(target, prediction)
    return float(np.sqrt(np.mean(np.square(target_array - prediction_array))))


def mean_absolute_error(target: np.ndarray, prediction: np.ndarray) -> float:
    target_array, prediction_array = _regression_arrays(target, prediction)
    return float(np.mean(np.abs(target_array - prediction_array)))


def pearson_correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    target_array, prediction_array = _regression_arrays(target, prediction)
    if np.std(target_array) == 0.0 or np.std(prediction_array) == 0.0:
        return float("nan")
    return float(pearsonr(target_array, prediction_array).statistic)


def r_squared_score(target: np.ndarray, prediction: np.ndarray) -> float:
    target_array, prediction_array = _regression_arrays(target, prediction)
    residual = np.sum(np.square(target_array - prediction_array))
    centered = np.sum(np.square(target_array - np.mean(target_array)))
    if centered == 0.0:
        return float("nan")
    return float(1.0 - residual / centered)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> RegressionMetrics:
    return RegressionMetrics(
        rmse=root_mean_squared_error(target, prediction),
        mae=mean_absolute_error(target, prediction),
        pearson_r=pearson_correlation(target, prediction),
        r_squared=r_squared_score(target, prediction),
    )


def interval_metrics(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> IntervalMetrics:
    target_array = np.asarray(target, dtype=np.float64).reshape(-1)
    lower_array = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper_array = np.asarray(upper, dtype=np.float64).reshape(-1)
    if (
        target_array.shape != lower_array.shape
        or target_array.shape != upper_array.shape
    ):
        raise ValueError("interval arrays must share a shape")
    if np.any(lower_array > upper_array):
        raise ValueError("interval lower limits exceed upper limits")
    width = float(np.mean(upper_array - lower_array))
    covered = (target_array >= lower_array) & (target_array <= upper_array)
    return IntervalMetrics(mean_width=width, coverage=float(np.mean(covered)))


def empirical_prediction_interval(
    samples: np.ndarray,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("prediction samples require draws by observations")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in the unit interval")
    tail = (1.0 - confidence) / 2.0
    return np.quantile(matrix, tail, axis=0), np.quantile(matrix, 1.0 - tail, axis=0)


def bland_altman(target: np.ndarray, prediction: np.ndarray) -> BlandAltmanMetrics:
    target_array, prediction_array = _regression_arrays(target, prediction)
    differences = prediction_array - target_array
    averages = (prediction_array + target_array) / 2.0
    bias = float(np.mean(differences))
    standard_deviation = float(np.std(differences, ddof=1))
    correlation = pearsonr(averages, differences)
    return BlandAltmanMetrics(
        mean_bias=bias,
        lower_limit=bias - 1.96 * standard_deviation,
        upper_limit=bias + 1.96 * standard_deviation,
        proportional_bias_r=float(correlation.statistic),
        proportional_bias_p=float(correlation.pvalue),
    )


def coefficient_of_variation(target: np.ndarray, prediction: np.ndarray) -> float:
    target_array, prediction_array = _regression_arrays(target, prediction)
    mean = float(np.mean(target_array))
    if mean == 0.0:
        return float("nan")
    return root_mean_squared_error(target_array, prediction_array) / mean * 100.0


def rmse_standard_deviation_ratio(target: np.ndarray, prediction: np.ndarray) -> float:
    target_array, prediction_array = _regression_arrays(target, prediction)
    standard_deviation = float(np.std(target_array, ddof=1))
    if standard_deviation == 0.0:
        return float("nan")
    return root_mean_squared_error(target_array, prediction_array) / standard_deviation
