from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProbabilityBundle:
    persona_ids: tuple[str, ...]
    question_ids: tuple[str, ...]
    probabilities: dict[str, np.ndarray]
    counts: dict[str, np.ndarray]
    alpha: float


@dataclass(frozen=True)
class ExperimentData:
    personas: pd.DataFrame
    weights: np.ndarray
    calibration: ProbabilityBundle
    test: ProbabilityBundle


def _build_probability_bundle(
    responses: pd.DataFrame,
    persona_ids: list[str],
    question_ids: list[str],
    option_labels: dict[str, list[str]],
    repetitions: int,
    alpha: float,
) -> ProbabilityBundle:
    persona_index = {value: index for index, value in enumerate(persona_ids)}
    question_index = {value: index for index, value in enumerate(question_ids)}
    counts = {
        qid: np.zeros((len(persona_ids), len(option_labels[qid])), dtype=np.int64)
        for qid in question_ids
    }
    observed = np.zeros((len(persona_ids), len(question_ids)), dtype=np.int64)
    label_indices = {
        qid: {label: index for index, label in enumerate(option_labels[qid])}
        for qid in question_ids
    }

    for row in responses.itertuples(index=False):
        persona_id = str(row.persona_id)
        question_id = str(row.question_id)
        if persona_id not in persona_index or question_id not in question_index:
            continue
        persona_position = persona_index[persona_id]
        question_position = question_index[question_id]
        label = str(row.parsed_option)
        if label not in label_indices[question_id]:
            raise ValueError(f"Invalid response option {label!r} for {question_id}")
        observed[persona_position, question_position] += 1
        counts[question_id][persona_position, label_indices[question_id][label]] += 1

    bad = np.argwhere(observed != repetitions)
    if len(bad):
        examples = [
            (persona_ids[i], question_ids[q], int(observed[i, q]))
            for i, q in bad[:10]
        ]
        raise ValueError(
            f"Expected {repetitions} responses per persona-question; examples: {examples}"
        )

    probabilities = {
        qid: (matrix + alpha) / (repetitions + alpha * matrix.shape[1])
        for qid, matrix in counts.items()
    }
    return ProbabilityBundle(
        persona_ids=tuple(persona_ids),
        question_ids=tuple(question_ids),
        probabilities=probabilities,
        counts=counts,
        alpha=alpha,
    )


def load_experiment_data(data_dir: Path, pool_size: int, alpha: float) -> ExperimentData:
    if pool_size not in {64, 128}:
        raise ValueError("The paper reports only the 64- and 128-persona pools")

    personas = pd.read_csv(data_dir / "personas.csv", dtype={"persona_id": str})
    if len(personas) != 128 or personas["persona_id"].duplicated().any():
        raise ValueError("personas.csv must contain the frozen ordered 128-persona pool")
    personas = personas.iloc[:pool_size].copy().reset_index(drop=True)
    personas["population_weight"] = 1.0 / pool_size
    persona_ids = personas["persona_id"].tolist()
    weights = personas["population_weight"].to_numpy(dtype=float)

    splits = json.loads((data_dir / "question_splits.json").read_text(encoding="utf-8"))
    if set(splits) != {"calibration", "test"}:
        raise ValueError("question_splits.json must contain only calibration and test")
    if len(splits["calibration"]) != 20 or len(splits["test"]) != 20:
        raise ValueError("The paper protocol requires 20 calibration and 20 test questions")
    if set(splits["calibration"]) & set(splits["test"]):
        raise ValueError("Calibration and test question sets must be disjoint")

    questions = json.loads((data_dir / "questions.json").read_text(encoding="utf-8"))
    option_labels = {
        str(question["question_id"]): [str(option["label"]) for option in question["options"]]
        for question in questions
    }
    expected_question_ids = set(splits["calibration"]) | set(splits["test"])
    if set(option_labels) != expected_question_ids:
        raise ValueError("questions.json must contain exactly the 40 reported questions")

    responses = pd.read_csv(
        data_dir / "responses.csv.gz",
        dtype={"persona_id": str, "question_id": str, "parsed_option": str},
    )
    if responses.duplicated(["role", "persona_id", "question_id", "slot_id"]).any():
        raise ValueError("Duplicate response slots found")
    if set(responses["role"]) != {"calibration", "test"}:
        raise ValueError("Response data must contain only calibration and test roles")

    calibration = _build_probability_bundle(
        responses[responses["role"] == "calibration"],
        persona_ids,
        list(splits["calibration"]),
        option_labels,
        repetitions=8,
        alpha=alpha,
    )
    test = _build_probability_bundle(
        responses[responses["role"] == "test"],
        persona_ids,
        list(splits["test"]),
        option_labels,
        repetitions=8,
        alpha=alpha,
    )
    return ExperimentData(personas, weights, calibration, test)

