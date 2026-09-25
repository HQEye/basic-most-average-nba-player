"""Checks that the saved results are right and that simulation.py reproduces them.

Run from the repo folder:  python -m unittest discover tests
"""
import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class SavedResults(unittest.TestCase):
    def test_final_rankings(self):
        rows = read_csv(DATA / "final_rankings.csv")
        self.assertEqual(len(rows), 272)
        self.assertEqual(rows[0]["Player"], "Jaylon Tyson")
        self.assertEqual(rows[0]["wins"], "2719")

    def test_da_silva_has_more_wins_but_worse_average(self):
        rows = {r["Player"]: r for r in read_csv(DATA / "final_rankings.csv")}
        tyson, da_silva = rows["Jaylon Tyson"], rows["Tristan Da Silva"]
        self.assertGreater(int(da_silva["wins"]), int(tyson["wins"]))
        self.assertGreater(float(da_silva["average_rank"]), float(tyson["average_rank"]))

    def test_phase_winners(self):
        winners = {
            "01_phase1_caveman_average.csv": "Gui Santos",
            "02_phase2_per100_min1000.csv": "Kyle Kuzma",
            "04_phase4_full_equal_features.csv": "Jaden Ivey",
            "05_phase5_add_play_style.csv": "Kyle Kuzma",
        }
        for name, player in winners.items():
            self.assertEqual(read_csv(DATA / "phases" / name)[0]["Player"], player)


class FullRerun(unittest.TestCase):
    """Reruns all 10,000 models and compares every value to data/final_rankings.csv."""

    def test_simulation_matches_saved_results(self):
        try:
            import numpy as np
            import pandas as pd
        except ImportError:
            self.skipTest("needs numpy and pandas (pip install -r requirements.txt)")
        import sys
        sys.path.insert(0, str(ROOT))
        from simulation import prepare_data, simulate

        players = prepare_data(DATA / "model_input.csv", DATA / "bios.csv")
        actual, _ = simulate(players)
        expected = pd.read_csv(DATA / "final_rankings.csv")
        self.assertEqual(actual.columns.tolist(), expected.columns.tolist())
        self.assertEqual(actual["Player"].tolist(), expected["Player"].tolist())
        for col in actual.columns:
            if pd.api.types.is_numeric_dtype(actual[col]):
                np.testing.assert_allclose(actual[col], expected[col], rtol=1e-10,
                                           atol=1e-10, equal_nan=True, err_msg=col)
            else:
                self.assertEqual(actual[col].fillna("").tolist(),
                                 expected[col].fillna("").tolist(), msg=col)


if __name__ == "__main__":
    unittest.main()
