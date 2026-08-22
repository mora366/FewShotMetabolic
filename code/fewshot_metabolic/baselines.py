from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ProbabilisticClassifier(Protocol):
    def fit(self, features: np.ndarray, target: np.ndarray) -> object: ...
    def predict_proba(self, features: np.ndarray) -> np.ndarray: ...


class NumericRegressor(Protocol):
    def fit(self, features: np.ndarray, target: np.ndarray) -> object: ...
    def predict(self, features: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class BaselinePrediction:
    name: str
    prediction: np.ndarray
    probability: np.ndarray | None


def logistic_regression(seed: int = 42) -> Pipeline:
    return Pipeline(
        (
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=seed)),
        )
    )


def support_vector_machine(seed: int = 42) -> Pipeline:
    return Pipeline(
        (
            ("scaler", StandardScaler()),
            ("classifier", SVC(probability=True, random_state=seed)),
        )
    )


def random_forest_classifier(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def ridge_regression(alpha: float = 1.0) -> Pipeline:
    return Pipeline((("scaler", StandardScaler()), ("regressor", Ridge(alpha=alpha))))


def random_forest_regressor(seed: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=1
    )


def standard_ensemble(seed: int = 42) -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("logistic", logistic_regression(seed)),
            ("forest", random_forest_classifier(seed)),
            ("svm", support_vector_machine(seed)),
        ],
        voting="soft",
    )


def fit_classifier(
    model: ProbabilisticClassifier,
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
    name: str,
) -> BaselinePrediction:
    fitted = model.fit(train_features, train_target)
    probability = fitted.predict_proba(test_features)[:, 1]
    prediction = (probability >= 0.5).astype(np.int64)
    return BaselinePrediction(name, prediction, probability)


def fit_regressor(
    model: NumericRegressor,
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
    name: str,
) -> BaselinePrediction:
    fitted = model.fit(train_features, train_target)
    prediction = np.asarray(fitted.predict(test_features), dtype=np.float64)
    return BaselinePrediction(name, prediction, None)


def pooled_pretrain_then_adapt(
    model: ProbabilisticClassifier,
    pooled_features: np.ndarray,
    pooled_target: np.ndarray,
    support_features: np.ndarray,
    support_target: np.ndarray,
    query_features: np.ndarray,
    name: str,
) -> BaselinePrediction:
    pooled = clone(model)
    pooled.fit(pooled_features, pooled_target)
    combined_features = np.concatenate((pooled_features, support_features), axis=0)
    combined_target = np.concatenate((pooled_target, support_target), axis=0)
    weights = np.concatenate(
        (
            np.ones(pooled_target.size),
            np.full(
                support_target.size, max(1, pooled_target.size // support_target.size)
            ),
        )
    )
    try:
        pooled.fit(
            combined_features, combined_target, classifier__sample_weight=weights
        )
    except TypeError:
        pooled.fit(combined_features, combined_target)
    probability = pooled.predict_proba(query_features)[:, 1]
    return BaselinePrediction(name, (probability >= 0.5).astype(np.int64), probability)


def bayesian_model_average(
    probabilities: list[np.ndarray], validation_targets: np.ndarray
) -> np.ndarray:
    if not probabilities:
        raise ValueError("Bayesian averaging requires component predictions")
    matrix = np.stack(probabilities, axis=0)
    if matrix.shape[1] != validation_targets.size:
        raise ValueError("component probability shapes differ from targets")
    losses = np.asarray(
        [log_loss(validation_targets, values, labels=[0, 1]) for values in matrix]
    )
    evidence = np.exp(-(losses - np.min(losses)))
    weights = evidence / np.sum(evidence)
    return np.sum(matrix * weights[:, None], axis=0)


def bootstrap_ensemble_probabilities(
    model: ProbabilisticClassifier,
    features: np.ndarray,
    target: np.ndarray,
    query: np.ndarray,
    members: int = 20,
    seed: int = 42,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    predictions = []
    for _ in range(members):
        indices = generator.integers(0, target.size, size=target.size)
        fitted = clone(model)
        fitted.fit(features[indices], target[indices])
        predictions.append(fitted.predict_proba(query)[:, 1])
    return np.stack(predictions, axis=0)


def classifier_registry(seed: int = 42) -> dict[str, ProbabilisticClassifier]:
    return {
        "logistic_regression": logistic_regression(seed),
        "random_forest": random_forest_classifier(seed),
        "support_vector_machine": support_vector_machine(seed),
        "ensemble_standard": standard_ensemble(seed),
    }


def regressor_registry(seed: int = 42) -> dict[str, NumericRegressor]:
    return {
        "ridge_regression": ridge_regression(),
        "random_forest": random_forest_regressor(seed),
    }
