from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_experiment_data
from .metrics import (
    classical_mds_features,
    evaluate_selection,
    normalized_auec,
    ordinal_wasserstein_distance_matrix,
)
from .selection import (
    BehaviorKMedoidsSelector,
    DBSCANSelector,
    DemographicStratifiedSelector,
    DensityPeaksSelector,
    GBPersonaSelector,
    KMeansSelector,
    SelectionResult,
    SpectralSelector,
    WeightedRandomSelector,
)


METHOD_ORDER = (
    "GB-Persona",
    "K-means",
    "Behavior K-medoids",
    "Spectral Clustering",
    "Density Peaks",
    "DBSCAN",
    "Demographic Stratified",
    "Weighted Random",
)

INTERNAL_TO_DISPLAY = {
    "gb_persona": "GB-Persona",
    "kmeans": "K-means",
    "behavior_kmedoids": "Behavior K-medoids",
    "spectral_clustering": "Spectral Clustering",
    "density_peaks": "Density Peaks",
    "dbscan": "DBSCAN",
    "demographic_stratified": "Demographic Stratified",
    "weighted_random": "Weighted Random",
}


def _selection_record(
    pool_size: int,
    requested_budget: int,
    random_seed: int | None,
    result: SelectionResult,
    persona_ids: list[str],
) -> dict[str, Any]:
    return {
        "pool_size": pool_size,
        "method": INTERNAL_TO_DISPLAY[result.method],
        "requested_budget": requested_budget,
        "actual_k": result.actual_k,
        "random_seed": random_seed,
        "stop_reason": result.stop_reason,
        "representative_ids": [persona_ids[index] for index in result.representative_indices],
        "representative_weights": [float(value) for value in result.representative_weights],
    }


def compute_pool_results(
    repository_root: Path, pool_size: int
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    config = json.loads((repository_root / "config.json").read_text(encoding="utf-8"))
    alpha = float(config["gb_persona"]["jeffreys_alpha"])
    data = load_experiment_data(repository_root / "data", pool_size, alpha)
    budgets = [
        int(value)
        for value in config["evaluation"][str(pool_size)]["requested_budgets"]
    ]
    seeds = [int(value) for value in config["evaluation"]["random_seeds"]]

    distance = ordinal_wasserstein_distance_matrix(data.calibration)
    mds_features = classical_mds_features(distance)
    country_strata = (
        data.personas["country_region"].fillna("<MISSING>").astype(str).to_numpy()
    )
    persona_ids = data.personas["persona_id"].astype(str).tolist()

    gb_selector = GBPersonaSelector(
        lambda_value=float(config["gb_persona"]["lambda"]),
        tail_quantile=float(config["gb_persona"]["tail_quantile"]),
        minimum_child_size=int(config["gb_persona"]["minimum_child_size"]),
    )
    kmedoids_selector = BehaviorKMedoidsSelector()
    density_selector = DensityPeaksSelector(
        float(config["evaluation"]["density_peaks_dc_quantile"])
    )
    dbscan_selector = DBSCANSelector(
        int(config["evaluation"]["dbscan_min_samples"])
    )

    error_replicates: dict[tuple[str, int], list[float]] = {}
    selections: list[dict[str, Any]] = []

    def record(result: SelectionResult, budget: int, seed: int | None) -> None:
        display_name = INTERNAL_TO_DISPLAY[result.method]
        error_replicates.setdefault((display_name, budget), []).append(
            evaluate_selection(data.test, data.weights, result)
        )
        selections.append(
            _selection_record(pool_size, budget, seed, result, persona_ids)
        )

    for requested_budget in budgets:
        gb_result = gb_selector.select(distance, data.weights, requested_budget)
        comparison_k = gb_result.actual_k
        record(gb_result, requested_budget, None)
        record(
            kmedoids_selector.select(distance, data.weights, comparison_k),
            requested_budget,
            None,
        )
        record(
            density_selector.select(distance, data.weights, comparison_k),
            requested_budget,
            None,
        )
        record(
            dbscan_selector.select(distance, data.weights, comparison_k),
            requested_budget,
            None,
        )

        for seed in seeds:
            record(
                KMeansSelector(
                    seed, int(config["evaluation"]["kmeans_n_init"])
                ).select(mds_features, distance, data.weights, comparison_k),
                requested_budget,
                seed,
            )
            record(
                SpectralSelector(
                    seed, int(config["evaluation"]["kmeans_n_init"])
                ).select(distance, data.weights, comparison_k),
                requested_budget,
                seed,
            )
            record(
                DemographicStratifiedSelector(seed).select(
                    distance, data.weights, comparison_k, country_strata
                ),
                requested_budget,
                seed,
            )
            record(
                WeightedRandomSelector(seed).select(
                    distance, data.weights, comparison_k
                ),
                requested_budget,
                seed,
            )

    gb_actual_k = {
        int(item["requested_budget"]): int(item["actual_k"])
        for item in selections
        if item["method"] == "GB-Persona"
    }
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        errors = [
            float(np.mean(error_replicates[(method, budget)])) for budget in budgets
        ]
        auec = normalized_auec(budgets, errors)
        for budget, error in zip(budgets, errors):
            rows.append(
                {
                    "pool_size": pool_size,
                    "method": method,
                    "requested_budget": budget,
                    "actual_k": gb_actual_k[budget],
                    "mean_test_normalized_w1": error,
                    "normalized_auec": auec,
                    "algorithm_repetitions": len(
                        error_replicates[(method, budget)]
                    ),
                }
            )
    return pd.DataFrame(rows), selections


def compute_all_results(
    repository_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    selections: list[dict[str, Any]] = []
    for pool_size in (128, 64):
        frame, pool_selections = compute_pool_results(repository_root, pool_size)
        frames.append(frame)
        selections.extend(pool_selections)
    return pd.concat(frames, ignore_index=True), selections


def verify_against_paper(
    observed: pd.DataFrame, expected_path: Path, tolerance: float = 5e-12
) -> None:
    expected = pd.read_csv(expected_path)
    keys = ["pool_size", "method", "requested_budget"]
    observed = observed.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    if observed[keys].to_dict("records") != expected[keys].to_dict("records"):
        raise AssertionError("Observed methods, pools, or budgets do not match the paper")
    for column in ("actual_k", "algorithm_repetitions"):
        if not np.array_equal(observed[column].to_numpy(), expected[column].to_numpy()):
            raise AssertionError(f"Mismatch in {column}")
    for column in ("mean_test_normalized_w1", "normalized_auec"):
        difference = np.abs(
            observed[column].to_numpy(dtype=float)
            - expected[column].to_numpy(dtype=float)
        )
        if np.max(difference) > tolerance:
            position = int(np.argmax(difference))
            key = expected.loc[position, keys].to_dict()
            raise AssertionError(
                f"Mismatch in {column} at {key}: maximum absolute difference "
                f"{difference[position]:.3e} exceeds {tolerance:.1e}"
            )

