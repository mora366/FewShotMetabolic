from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import IterativeImputer

from .features import EXCLUDED_LABEL_FEATURES, canonical_feature_names

_ITERATIVE_IMPUTER_ENABLED = enable_iterative_imputer


@dataclass(frozen=True)
class NormalizationState:
    columns: tuple[str, ...]
    means: np.ndarray
    standard_deviations: np.ndarray


class MetabolicPreprocessor:
    def __init__(
        self, imputations: int = 5, seed: int = 42, epsilon: float = 1e-6
    ) -> None:
        if imputations != 5:
            raise ValueError("the analysis protocol uses five imputations")
        self.imputations = imputations
        self.seed = seed
        self.epsilon = epsilon
        self.states: list[NormalizationState] = []
        self.imputers: list[IterativeImputer] = []

    def _validate_columns(self, frame: pd.DataFrame) -> tuple[str, ...]:
        forbidden = set(frame.columns).intersection(EXCLUDED_LABEL_FEATURES)
        if forbidden:
            raise ValueError(f"label-defining inputs detected: {sorted(forbidden)}")
        expected = canonical_feature_names()
        missing = set(expected).difference(frame.columns)
        if missing:
            raise ValueError(f"missing required features: {sorted(missing)}")
        return expected

    def fit_transform(self, frame: pd.DataFrame) -> list[np.ndarray]:
        columns = self._validate_columns(frame)
        matrix = frame.loc[:, columns].to_numpy(dtype=np.float64)
        outputs: list[np.ndarray] = []
        self.states.clear()
        self.imputers.clear()
        for index in range(self.imputations):
            estimator = ExtraTreesRegressor(
                n_estimators=20,
                min_samples_leaf=2,
                random_state=self.seed + index,
                n_jobs=1,
            )
            imputer = IterativeImputer(
                estimator=estimator,
                max_iter=10,
                sample_posterior=False,
                random_state=self.seed + index,
                skip_complete=True,
            )
            imputed = imputer.fit_transform(matrix)
            means = np.mean(imputed, axis=0)
            standard_deviations = np.std(imputed, axis=0)
            state = NormalizationState(columns, means, standard_deviations)
            normalized = (imputed - means) / (standard_deviations + self.epsilon)
            self.imputers.append(imputer)
            self.states.append(state)
            outputs.append(normalized.astype(np.float32))
        return outputs

    def transform(self, frame: pd.DataFrame) -> list[np.ndarray]:
        columns = self._validate_columns(frame)
        if (
            len(self.imputers) != self.imputations
            or len(self.states) != self.imputations
        ):
            raise RuntimeError("preprocessor has not been fitted")
        matrix = frame.loc[:, columns].to_numpy(dtype=np.float64)
        outputs: list[np.ndarray] = []
        for imputer, state in zip(self.imputers, self.states, strict=True):
            imputed = imputer.transform(matrix)
            normalized = (imputed - state.means) / (
                state.standard_deviations + self.epsilon
            )
            outputs.append(normalized.astype(np.float32))
        return outputs


def derive_homa_ir(
    glucose_mmol_l: np.ndarray, insulin_micro_u_ml: np.ndarray
) -> np.ndarray:
    if np.any(glucose_mmol_l < 0.0) or np.any(insulin_micro_u_ml < 0.0):
        raise ValueError("HOMA-IR inputs must be nonnegative")
    return glucose_mmol_l * insulin_micro_u_ml / 22.5


def derive_homa_beta(
    glucose_mmol_l: np.ndarray, insulin_micro_u_ml: np.ndarray
) -> np.ndarray:
    denominator = glucose_mmol_l - 3.5
    if np.any(np.abs(denominator) < 1e-6):
        raise ValueError("HOMA-beta is undefined when glucose equals 3.5 mmol/L")
    return 20.0 * insulin_micro_u_ml / denominator
