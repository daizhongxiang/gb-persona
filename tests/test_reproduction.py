from __future__ import annotations

import json
from pathlib import Path
import unittest

import pandas as pd

from gb_persona.data import load_experiment_data
from gb_persona.experiment import compute_all_results, verify_against_paper


ROOT = Path(__file__).resolve().parents[1]


class PaperReproductionTest(unittest.TestCase):
    def test_release_data_scope_and_counts(self) -> None:
        personas = pd.read_csv(ROOT / "data" / "personas.csv", dtype=str)
        responses = pd.read_csv(ROOT / "data" / "responses.csv.gz", dtype=str)
        self.assertEqual(len(personas), 128)
        self.assertEqual(len(responses), 40_960)
        self.assertEqual(set(responses["role"]), {"calibration", "test"})
        self.assertTrue(personas["persona_id"].str.fullmatch(r"P\d{3}").all())
        self.assertTrue(responses["persona_id"].str.fullmatch(r"P\d{3}").all())
        splits = json.loads((ROOT / "data" / "question_splits.json").read_text())
        families = json.loads((ROOT / "data" / "question_families.json").read_text())
        family_counts = {
            role: {
                family: len(set(question_ids) & set(splits[role]))
                for family, question_ids in families.items()
            }
            for role in ("calibration", "test")
        }
        self.assertEqual(family_counts["calibration"], family_counts["test"])
        self.assertEqual(
            family_counts["test"],
            {
                "child_qualities": 5,
                "neighbor_tolerance": 4,
                "gender_and_family_roles": 5,
                "work_and_social_change": 2,
                "wellbeing": 4,
            },
        )
        for pool_size in (64, 128):
            data = load_experiment_data(ROOT / "data", pool_size, alpha=0.5)
            self.assertEqual(len(data.calibration.question_ids), 20)
            self.assertEqual(len(data.test.question_ids), 20)

    def test_exact_paper_results(self) -> None:
        observed, _ = compute_all_results(ROOT)
        verify_against_paper(observed, ROOT / "results" / "paper_results.csv")


if __name__ == "__main__":
    unittest.main()
