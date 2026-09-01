"""Operações bayesianas elementares usadas nos primeiros exercícios."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalize(weights: ArrayLike) -> NDArray[np.float64]:
    """Normaliza pesos não negativos para que a soma seja igual a um."""

    values = np.asarray(weights, dtype=float)
    if values.ndim != 1:
        raise ValueError("Os pesos devem formar um vetor unidimensional.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Os pesos devem ser finitos e não negativos.")

    total = values.sum()
    if total <= 0:
        raise ValueError("A soma dos pesos deve ser positiva.")
    return values / total


def posterior_discrete(
    prior: ArrayLike,
    likelihood: ArrayLike,
) -> tuple[NDArray[np.float64], float]:
    """Calcula posterior e evidência para hipóteses discretas.

    Para cada hipótese ``H_i``, o peso não normalizado é
    ``p(H_i | I) p(D | H_i, I)``. A soma desses pesos é a evidência.
    """

    prior_values = np.asarray(prior, dtype=float)
    likelihood_values = np.asarray(likelihood, dtype=float)
    if prior_values.shape != likelihood_values.shape:
        raise ValueError("Prior e likelihood devem ter o mesmo formato.")

    prior_values = normalize(prior_values)
    if not np.all(np.isfinite(likelihood_values)) or np.any(likelihood_values < 0):
        raise ValueError("A likelihood deve ser finita e não negativa.")

    joint_weights = prior_values * likelihood_values
    evidence = float(joint_weights.sum())
    if evidence <= 0:
        raise ValueError("Os dados têm probabilidade nula sob todas as hipóteses.")
    return joint_weights / evidence, evidence


def gaussian_pdf(
    x: ArrayLike,
    mean: ArrayLike,
    sigma: float,
) -> NDArray[np.float64]:
    """Avalia a densidade Gaussiana com desvio-padrão positivo ``sigma``."""

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma deve ser finito e positivo.")

    x_values = np.asarray(x, dtype=float)
    mean_values = np.asarray(mean, dtype=float)
    z = (x_values - mean_values) / sigma
    return np.exp(-0.5 * z**2) / (np.sqrt(2.0 * np.pi) * sigma)


def posterior_disease_given_positive(
    prevalence: ArrayLike,
    sensitivity: float,
    false_positive_rate: float,
) -> NDArray[np.float64]:
    """Calcula ``P(doença | teste positivo)`` para um teste binário."""

    prevalence_values = np.asarray(prevalence, dtype=float)
    probabilities = np.concatenate(
        (
            prevalence_values.reshape(-1),
            np.asarray([sensitivity, false_positive_rate]),
        )
    )
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Todas as probabilidades devem pertencer ao intervalo [0, 1].")

    numerator = prevalence_values * sensitivity
    evidence = numerator + (1.0 - prevalence_values) * false_positive_rate
    if np.any(evidence <= 0):
        raise ValueError("A probabilidade de um teste positivo deve ser maior que zero.")
    return numerator / evidence
