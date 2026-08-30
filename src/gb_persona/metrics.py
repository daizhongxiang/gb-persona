from __future__ import annotations

import numpy as np

from .data import ProbabilityBundle
from .selection import SelectionResult


def ordinal_wasserstein_distance_matrix(bundle: ProbabilityBundle) -> np.ndarray:
    distance = np.zeros((len(bundle.persona_ids), len(bundle.persona_ids)), dtype=np.float64)
    for question_id in bundle.question_ids:
        probabilities = bundle.probabilities[question_id]
        option_count = probabilities.shape[1]
        cdf = np.cumsum(probabilities, axis=1)[:, :-1]
        distance += np.sum(
            np.abs(cdf[:, None, :] - cdf[None, :, :]), axis=2
        ) / (option_count - 1)
    distance /= len(bundle.question_ids)
    np.fill_diagonal(distance, 0.0)
    return distance


def classical_mds_features(distance: np.ndarray) -> np.ndarray:
    n = len(distance)
    centering = np.eye(n) - np.full((n, n), 1.0 / n)
    gram = -0.5 * centering @ (distance**2) @ centering
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    maximum = max(float(eigenvalues[0]), np.finfo(float).eps)
    positive = eigenvalues > maximum * 1e-12
    if not np.any(positive):
        raise ValueError("Classical MDS found no positive eigenvalues")
    return eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])


def normalized_ordinal_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.abs(np.cumsum(left)[:-1] - np.cumsum(right)[:-1]).sum()
        / (len(left) - 1)
    )


def evaluate_selection(
    bundle: ProbabilityBundle,
    weights: np.ndarray,
    result: SelectionResult,
) -> float:
    representatives = np.asarray(result.representative_indices, dtype=int)
    representative_weights = np.asarray(result.representative_weights, dtype=float)
    errors = []
    for question_id in bundle.question_ids:
        probabilities = bundle.probabilities[question_id]
        full_distribution = weights @ probabilities
        selected_distribution = representative_weights @ probabilities[representatives]
        errors.append(
            normalized_ordinal_wasserstein(selected_distribution, full_distribution)
        )
    return float(np.mean(errors))


def normalized_auec(budgets: list[int], errors: list[float]) -> float:
    order = np.argsort(budgets)
    x = np.asarray(budgets, dtype=float)[order]
    y = np.asarray(errors, dtype=float)[order]
    return float(np.trapz(y, x) / (x[-1] - x[0]))

