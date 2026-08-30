from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering


@dataclass(frozen=True)
class SelectionResult:
    method: str
    requested_budget: int
    representative_indices: tuple[int, ...]
    representative_weights: tuple[float, ...]
    clusters: tuple[tuple[int, ...], ...]
    stop_reason: str
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def actual_k(self) -> int:
        return len(self.representative_indices)


def _validate_inputs(distance: np.ndarray, weights: np.ndarray, budget: int) -> None:
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be square")
    if len(weights) != len(distance) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must align with distance and sum to one")
    if budget < 1 or budget >= len(weights):
        raise ValueError("budget must be positive and smaller than pool size")
    if not np.allclose(distance, distance.T) or not np.allclose(np.diag(distance), 0.0):
        raise ValueError("distance must be symmetric with a zero diagonal")


def _weighted_medoid(indices: np.ndarray, distance: np.ndarray, weights: np.ndarray) -> int:
    local = distance[np.ix_(indices, indices)]
    costs = local.T @ weights[indices]
    return int(indices[int(np.argmin(costs))])


def _assign_to_representatives(
    representative_indices: np.ndarray, distance: np.ndarray
) -> list[np.ndarray]:
    assignment = np.argmin(distance[:, representative_indices], axis=1)
    assignment[representative_indices] = np.arange(len(representative_indices))
    return [
        np.flatnonzero(assignment == cluster)
        for cluster in range(len(representative_indices))
    ]


def _result_from_clusters(
    method: str,
    budget: int,
    clusters: list[np.ndarray],
    distance: np.ndarray,
    weights: np.ndarray,
    stop_reason: str,
    diagnostics: dict[str, object] | None = None,
) -> SelectionResult:
    cleaned = [np.asarray(cluster, dtype=int) for cluster in clusters if len(cluster)]
    representatives = [_weighted_medoid(cluster, distance, weights) for cluster in cleaned]
    representative_weights = [float(weights[cluster].sum()) for cluster in cleaned]
    return SelectionResult(
        method,
        budget,
        tuple(representatives),
        tuple(representative_weights),
        tuple(tuple(int(value) for value in cluster) for cluster in cleaned),
        stop_reason,
        diagnostics or {},
    )


def _result_from_representatives(
    method: str,
    budget: int,
    representatives: np.ndarray,
    distance: np.ndarray,
    weights: np.ndarray,
    stop_reason: str,
    diagnostics: dict[str, object] | None = None,
) -> SelectionResult:
    representatives = np.asarray(representatives, dtype=int)
    clusters = _assign_to_representatives(representatives, distance)
    representative_weights = [float(weights[cluster].sum()) for cluster in clusters]
    return SelectionResult(
        method,
        budget,
        tuple(int(value) for value in representatives),
        tuple(representative_weights),
        tuple(tuple(int(value) for value in cluster) for cluster in clusters),
        stop_reason,
        diagnostics or {},
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    threshold = quantile * weights[order].sum()
    position = min(int(np.searchsorted(cumulative, threshold, side="left")), len(values) - 1)
    return float(values[order][position])


def _ball_cost(
    indices: np.ndarray,
    distance: np.ndarray,
    weights: np.ndarray,
    lambda_value: float,
    tail_quantile: float,
) -> float:
    medoid = _weighted_medoid(indices, distance, weights)
    radii = distance[indices, medoid]
    mass = float(weights[indices].sum())
    distortion = float(np.dot(weights[indices], radii))
    tail_radius = _weighted_quantile(radii, weights[indices], tail_quantile)
    return float((1.0 - lambda_value) * distortion + lambda_value * mass * tail_radius)


def _iterate_two_medoids(
    indices: np.ndarray,
    initial: tuple[int, int],
    distance: np.ndarray,
    weights: np.ndarray,
    minimum_child_size: int,
) -> tuple[list[np.ndarray], float] | None:
    medoids = np.asarray(initial, dtype=int)
    for _ in range(100):
        membership = np.argmin(distance[np.ix_(indices, medoids)], axis=1)
        clusters = [indices[membership == label] for label in range(2)]
        if min(len(cluster) for cluster in clusters) < minimum_child_size:
            return None
        updated = np.asarray(
            [_weighted_medoid(cluster, distance, weights) for cluster in clusters],
            dtype=int,
        )
        if np.array_equal(updated, medoids):
            break
        medoids = updated
    objective = sum(
        float(np.dot(weights[cluster], distance[cluster, medoid]))
        for cluster, medoid in zip(clusters, medoids)
    )
    return clusters, objective


def _weighted_two_medoids(
    indices: np.ndarray,
    distance: np.ndarray,
    weights: np.ndarray,
    minimum_child_size: int,
) -> list[np.ndarray] | None:
    if len(indices) < 2 * minimum_child_size:
        return None
    local = distance[np.ix_(indices, indices)]
    farthest = np.unravel_index(int(np.argmax(local)), local.shape)
    global_medoid = _weighted_medoid(indices, distance, weights)
    farthest_from_medoid = int(indices[int(np.argmax(distance[global_medoid, indices]))])
    initializations = [
        (int(indices[farthest[0]]), int(indices[farthest[1]])),
        (global_medoid, farthest_from_medoid),
    ]
    candidates = []
    for initial in initializations:
        if initial[0] == initial[1]:
            continue
        candidate = _iterate_two_medoids(
            indices, initial, distance, weights, minimum_child_size
        )
        if candidate is not None:
            candidates.append(candidate)
    return min(candidates, key=lambda item: item[1])[0] if candidates else None


class GBPersonaSelector:
    def __init__(self, lambda_value: float, tail_quantile: float, minimum_child_size: int):
        self.lambda_value = lambda_value
        self.tail_quantile = tail_quantile
        self.minimum_child_size = minimum_child_size

    def select(self, distance: np.ndarray, weights: np.ndarray, budget: int) -> SelectionResult:
        _validate_inputs(distance, weights, budget)
        clusters = [np.arange(len(weights), dtype=int)]
        gains: list[float] = []
        stop_reason = "budget_reached"
        while len(clusters) < budget:
            candidates = []
            for position, cluster in enumerate(clusters):
                children = _weighted_two_medoids(
                    cluster, distance, weights, self.minimum_child_size
                )
                if children is None:
                    continue
                gain = _ball_cost(
                    cluster, distance, weights, self.lambda_value, self.tail_quantile
                ) - sum(
                    _ball_cost(
                        child, distance, weights, self.lambda_value, self.tail_quantile
                    )
                    for child in children
                )
                candidates.append((gain, position, children))
            if not candidates:
                stop_reason = "no_legal_split"
                break
            gain, position, children = max(candidates, key=lambda item: (item[0], -item[1]))
            if gain <= 0:
                stop_reason = "no_positive_gain"
                break
            clusters[position : position + 1] = children
            gains.append(float(gain))
        return _result_from_clusters(
            "gb_persona",
            budget,
            clusters,
            distance,
            weights,
            stop_reason,
            {"split_gains": gains},
        )


class BehaviorKMedoidsSelector:
    def select(self, distance: np.ndarray, weights: np.ndarray, budget: int) -> SelectionResult:
        _validate_inputs(distance, weights, budget)
        medoids = [_weighted_medoid(np.arange(len(weights)), distance, weights)]
        while len(medoids) < budget:
            nearest = np.min(distance[:, medoids], axis=1)
            nearest[medoids] = -np.inf
            medoids.append(int(np.argmax(nearest)))
        medoids_array = np.asarray(medoids, dtype=int)
        for _ in range(100):
            clusters = _assign_to_representatives(medoids_array, distance)
            updated = np.asarray(
                [_weighted_medoid(cluster, distance, weights) for cluster in clusters],
                dtype=int,
            )
            if np.array_equal(updated, medoids_array):
                break
            medoids_array = updated
        clusters = _assign_to_representatives(medoids_array, distance)
        return _result_from_clusters(
            "behavior_kmedoids", budget, clusters, distance, weights, "converged"
        )


class WeightedRandomSelector:
    def __init__(self, seed: int):
        self.seed = seed

    def select(self, distance: np.ndarray, weights: np.ndarray, budget: int) -> SelectionResult:
        _validate_inputs(distance, weights, budget)
        representatives = np.random.default_rng(self.seed).choice(
            len(weights), size=budget, replace=False, p=weights / weights.sum()
        )
        return _result_from_representatives(
            "weighted_random",
            budget,
            representatives,
            distance,
            weights,
            "sampled",
            {"seed": self.seed},
        )


def _clusters_from_labels(labels: np.ndarray, budget: int) -> list[np.ndarray]:
    clusters = [np.flatnonzero(labels == name) for name in sorted(set(labels.tolist()))]
    if len(clusters) != budget or any(len(cluster) == 0 for cluster in clusters):
        raise ValueError(f"Clustering returned {len(clusters)} clusters for K={budget}")
    return clusters


class KMeansSelector:
    def __init__(self, seed: int, n_init: int = 10):
        self.seed = seed
        self.n_init = n_init

    def select(
        self,
        features: np.ndarray,
        distance: np.ndarray,
        weights: np.ndarray,
        budget: int,
    ) -> SelectionResult:
        _validate_inputs(distance, weights, budget)
        estimator = KMeans(
            n_clusters=budget,
            random_state=self.seed,
            n_init=self.n_init,
            algorithm="lloyd",
        )
        labels = estimator.fit_predict(features, sample_weight=weights)
        clusters = _clusters_from_labels(labels, budget)
        return _result_from_clusters(
            "kmeans",
            budget,
            clusters,
            distance,
            weights,
            "budget_reached",
            {"seed": self.seed, "n_init": self.n_init},
        )


def _rbf_kernel(distance: np.ndarray) -> tuple[np.ndarray, float]:
    triangle = distance[np.triu_indices(len(distance), 1)]
    positive = triangle[triangle > np.finfo(float).eps]
    sigma = float(np.median(positive)) if len(positive) else 1.0
    kernel = np.exp(-(distance**2) / (2.0 * sigma**2))
    kernel = (kernel + kernel.T) / 2.0
    np.fill_diagonal(kernel, 1.0)
    return kernel, sigma


class SpectralSelector:
    def __init__(self, seed: int, n_init: int = 10):
        self.seed = seed
        self.n_init = n_init

    def select(self, distance: np.ndarray, weights: np.ndarray, budget: int) -> SelectionResult:
        _validate_inputs(distance, weights, budget)
        kernel, sigma = _rbf_kernel(distance)
        labels = SpectralClustering(
            n_clusters=budget,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=self.seed,
            n_init=self.n_init,
        ).fit_predict(kernel)
        clusters = _clusters_from_labels(labels, budget)
        return _result_from_clusters(
            "spectral_clustering",
            budget,
            clusters,
            distance,
            weights,
            "budget_reached",
            {"seed": self.seed, "n_init": self.n_init, "rbf_sigma": sigma},
        )


class DensityPeaksSelector:
    def __init__(self, dc_quantile: float = 0.02):
        self.dc_quantile = dc_quantile

    def select(self, distance: np.ndarray, weights: np.ndarray, budget: int) -> SelectionResult:
        _validate_inputs(distance, weights, budget)
        positive = distance[np.triu_indices(len(distance), 1)]
        positive = positive[positive > np.finfo(float).eps]
        dc = float(np.quantile(positive, self.dc_quantile)) if len(positive) else 1.0
        kernel = np.exp(-((distance / dc) ** 2))
        np.fill_diagonal(kernel, 0.0)
        density = kernel @ weights
        order = np.lexsort((np.arange(len(weights)), -density))
        delta = np.zeros(len(weights), dtype=float)
        parent = np.full(len(weights), -1, dtype=int)
        delta[order[0]] = float(np.max(distance[order[0]]))
        for position, index in enumerate(order[1:], start=1):
            higher = order[:position]
            parent[index] = int(higher[int(np.argmin(distance[index, higher]))])
            delta[index] = float(distance[index, parent[index]])
        gamma = density * delta
        centers = np.lexsort((np.arange(len(weights)), -gamma))[:budget]
        labels = np.full(len(weights), -1, dtype=int)
        for label, center in enumerate(centers):
            labels[center] = label
        for index in order:
            if labels[index] < 0:
                labels[index] = labels[parent[index]]
        clusters = _clusters_from_labels(labels, budget)
        return _result_from_clusters(
            "density_peaks",
            budget,
            clusters,
            distance,
            weights,
            "budget_reached",
            {"dc_quantile": self.dc_quantile, "dc": dc},
        )


def _dbscan_clusters(labels: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    clusters = [
        np.flatnonzero(labels == label)
        for label in sorted(set(labels.tolist()) - {-1})
    ]
    noise = np.flatnonzero(labels == -1)
    clusters.extend(np.asarray([int(index)], dtype=int) for index in noise)
    return clusters, len(clusters) - len(noise), len(noise)


def _merge_closest_clusters(
    clusters: list[np.ndarray], distance: np.ndarray, target: int
) -> list[np.ndarray]:
    output = [np.asarray(cluster, dtype=int) for cluster in clusters]
    while len(output) > target:
        best: tuple[float, int, int] | None = None
        for left in range(len(output)):
            for right in range(left + 1, len(output)):
                candidate = (
                    float(distance[np.ix_(output[left], output[right])].mean()),
                    left,
                    right,
                )
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise ValueError("Could not find DBSCAN clusters to merge")
        _, left, right = best
        output[left] = np.concatenate([output[left], output[right]])
        del output[right]
    return output


def _split_farthest_clusters(
    clusters: list[np.ndarray],
    distance: np.ndarray,
    weights: np.ndarray,
    target: int,
) -> list[np.ndarray]:
    output = [np.asarray(cluster, dtype=int) for cluster in clusters]
    while len(output) < target:
        eligible = [position for position, cluster in enumerate(output) if len(cluster) > 1]
        position = max(
            eligible,
            key=lambda value: (
                float(
                    np.dot(
                        weights[output[value]],
                        distance[
                            np.ix_(
                                output[value],
                                [_weighted_medoid(output[value], distance, weights)],
                            )
                        ][:, 0],
                    )
                ),
                len(output[value]),
                -value,
            ),
        )
        cluster = output[position]
        local = distance[np.ix_(cluster, cluster)].copy()
        np.fill_diagonal(local, -np.inf)
        left_local, right_local = np.unravel_index(int(np.argmax(local)), local.shape)
        centers = [int(cluster[left_local]), int(cluster[right_local])]
        assignment = np.argmin(distance[np.ix_(cluster, centers)], axis=1)
        assignment[left_local] = 0
        assignment[right_local] = 1
        output[position : position + 1] = [
            cluster[assignment == label] for label in (0, 1)
        ]
    return output


class DBSCANSelector:
    def __init__(self, min_samples: int = 2):
        self.min_samples = min_samples

    def select(self, distance: np.ndarray, weights: np.ndarray, budget: int) -> SelectionResult:
        _validate_inputs(distance, weights, budget)
        triangle = distance[np.triu_indices(len(distance), 1)]
        positive = np.unique(triangle[triangle > np.finfo(float).eps])
        candidates = (
            np.concatenate(([max(float(positive[0]) / 2.0, 1e-12)], positive))
            if len(positive)
            else np.asarray([1e-12])
        )
        best = None
        for eps in candidates:
            labels = DBSCAN(
                eps=float(eps), min_samples=self.min_samples, metric="precomputed"
            ).fit_predict(distance)
            clusters, non_noise_count, noise_count = _dbscan_clusters(labels)
            score = (abs(len(clusters) - budget), float(weights[labels == -1].sum()), float(eps))
            candidate = (score, float(eps), clusters, non_noise_count, noise_count)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            raise ValueError("DBSCAN epsilon search produced no candidate")
        _, eps, clusters, non_noise_count, noise_count = best
        initial_k = len(clusters)
        if len(clusters) > budget:
            clusters = _merge_closest_clusters(clusters, distance, budget)
        elif len(clusters) < budget:
            clusters = _split_farthest_clusters(clusters, distance, weights, budget)
        return _result_from_clusters(
            "dbscan",
            budget,
            clusters,
            distance,
            weights,
            "budget_reconciled" if initial_k != budget else "budget_reached",
            {
                "eps": eps,
                "min_samples": self.min_samples,
                "raw_non_noise_cluster_count": non_noise_count,
                "raw_noise_count": noise_count,
            },
        )


class DemographicStratifiedSelector:
    def __init__(self, seed: int):
        self.seed = seed

    def select(
        self,
        distance: np.ndarray,
        weights: np.ndarray,
        budget: int,
        strata: np.ndarray,
    ) -> SelectionResult:
        _validate_inputs(distance, weights, budget)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(strata):
            groups[str(value)].append(index)
        names = sorted(groups)
        masses = {name: float(weights[groups[name]].sum()) for name in names}
        targets = {name: budget * masses[name] for name in names}
        allocation = {
            name: min(int(np.floor(targets[name])), len(groups[name])) for name in names
        }
        while sum(allocation.values()) < budget:
            eligible = [name for name in names if allocation[name] < len(groups[name])]
            chosen = max(
                eligible,
                key=lambda name: (
                    targets[name] - allocation[name],
                    masses[name],
                    -names.index(name),
                ),
            )
            allocation[chosen] += 1
        rng = np.random.default_rng(self.seed)
        selected: list[int] = []
        for name in names:
            count = allocation[name]
            if count:
                indices = np.asarray(groups[name], dtype=int)
                probabilities = weights[indices] / weights[indices].sum()
                selected.extend(
                    int(value)
                    for value in rng.choice(
                        indices, size=count, replace=False, p=probabilities
                    )
                )
        return _result_from_representatives(
            "demographic_stratified",
            budget,
            np.asarray(selected, dtype=int),
            distance,
            weights,
            "stratified_budget_reached",
            {"seed": self.seed, "allocation": allocation},
        )

