"""Tests of evaluation semantics; does not load course labels or a model."""
import csv
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from verify import recompute_external_metrics, verify_predictions


class EvaluationTests(unittest.TestCase):
    def assert_reference_metrics(self, truth, predicted):
        metrics = recompute_external_metrics(truth, predicted)
        self.assertAlmostEqual(metrics["ari"], adjusted_rand_score(truth, predicted), places=12)
        self.assertAlmostEqual(metrics["nmi"], normalized_mutual_info_score(truth, predicted, average_method="arithmetic"), places=12)
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).sum(), len(truth))
        self.assertAlmostEqual(np.trace(np.asarray(metrics["confusion_matrix"])) / len(truth), metrics["acc"], places=12)
        return metrics

    def test_perfect_partition_with_permuted_ids(self):
        truth = ["bottle", "bottle", "cable", "cable", "tile", "tile"]
        result = self.assert_reference_metrics(truth, [17, 17, -9, -9, 402, 402])
        for key in ("acc", "ari", "nmi"):
            self.assertAlmostEqual(result[key], 1.0, places=12)
        self.assertEqual(result["confusion_matrix"], np.diag([2, 2, 2]).tolist())

    def test_cluster_and_class_renumbering_invariance(self):
        truth = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        predicted = [0, 0, 1, 1, 1, 2, 2, 2, 0]
        first = self.assert_reference_metrics(truth, predicted)
        second = self.assert_reference_metrics(
            np.asarray([9, -100, 4])[truth], np.asarray([-3, 72, 10])[predicted]
        )
        for key in ("acc", "ari", "nmi"):
            self.assertAlmostEqual(first[key], second[key], places=12)

    def test_uninformative_balanced_partition(self):
        result = self.assert_reference_metrics([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2])
        self.assertAlmostEqual(result["acc"], 1 / 3)
        self.assertAlmostEqual(result["nmi"], 0)
        self.assertLess(result["ari"], 0)
        self.assertEqual(result["confusion_columns"][-1], None)

    def test_single_cluster_is_not_perfect_for_multiclass_truth(self):
        result = self.assert_reference_metrics([0, 0, 1, 1, 2, 2], [7] * 6)
        self.assertAlmostEqual(result["acc"], 1 / 3)
        self.assertEqual(result["ari"], 0)
        self.assertEqual(result["nmi"], 0)

    def test_degenerate_reference_conventions(self):
        for truth, predicted in [([0], [8]), ([0, 0, 0], [4, 4, 4]), ([0, 1, 2], [8, 6, 4])]:
            result = self.assert_reference_metrics(truth, predicted)
            self.assertEqual(result["acc"], 1)
            self.assertEqual(result["ari"], 1)
            self.assertAlmostEqual(result["nmi"], 1, places=12)

    def test_seeded_random_partitions_match_reference(self):
        generator = np.random.default_rng(2026)
        for count in (2, 7, 31, 600):
            for true_k, predicted_k in ((2, 2), (6, 6), (3, 7), (7, 3)):
                truth = generator.integers(true_k, size=count)
                predicted = generator.integers(predicted_k, size=count)
                self.assert_reference_metrics(truth, predicted)

    def test_assignment_matches_exhaustive_search(self):
        truth = [0, 0, 0, 1, 1, 1, 2, 2]
        predicted = [2, 2, 1, 0, 1, 1, 0, 2]
        result = recompute_external_metrics(truth, predicted)
        table = np.asarray(result["contingency_matrix"])
        brute_force = max(sum(table[row, column] for row, column in enumerate(order)) for order in itertools.permutations(range(3)))
        self.assertEqual(result["acc"], brute_force / len(truth))

    def test_invalid_metric_inputs(self):
        for truth, predicted in [([], []), ([0, 1], [1]), ([[0, 1]], [[1, 0]]), ([0, np.nan], [0, 1])]:
            with self.assertRaises(ValueError):
                recompute_external_metrics(truth, predicted)


class SavedPredictionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.prediction_path = self.root / "predictions.csv"
        self.metric_path = self.root / "metrics.json"
        self.labels = {"001.png": "bottle", "002.png": "bottle", "003.png": "tile", "004.png": "tile"}
        self.rows = [["001.png", 9, "bottle"], ["002.png", 9, "bottle"], ["003.png", 3, "tile"], ["004.png", 3, "tile"]]

    def tearDown(self):
        self.folder.cleanup()

    def write_csv(self, rows=None, header=("filename", "cluster", "true_label")):
        with self.prediction_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(self.rows if rows is None else rows)

    def test_saved_result_and_reordered_rows(self):
        self.write_csv()
        result = verify_predictions(self.prediction_path, self.labels)
        self.metric_path.write_text(json.dumps({"external": result}), encoding="utf-8")
        self.write_csv(list(reversed(self.rows)))
        self.assertEqual(verify_predictions(self.prediction_path, self.labels, self.metric_path), result)

    def test_csv_without_truth_can_use_authoritative_labels(self):
        self.write_csv([row[:2] for row in self.rows], ("filename", "cluster"))
        self.assertEqual(verify_predictions(self.prediction_path, self.labels)["acc"], 1)
        with self.assertRaises(ValueError):
            verify_predictions(self.prediction_path)

    def test_tampered_truth_duplicate_and_incomplete_coverage_fail(self):
        cases = [self.rows[:-1], self.rows + [self.rows[0]], [["001.png", 9, "tile"]] + self.rows[1:], [["999.png", 9, "bottle"]] + self.rows[1:]]
        for rows in cases:
            self.write_csv(rows)
            with self.assertRaises(ValueError):
                verify_predictions(self.prediction_path, self.labels)

    def test_tampered_metric_fails(self):
        self.write_csv()
        metrics = verify_predictions(self.prediction_path, self.labels)
        metrics["ari"] = 0.123
        self.metric_path.write_text(json.dumps({"external": metrics}), encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_predictions(self.prediction_path, self.labels, self.metric_path)


if __name__ == "__main__":
    unittest.main()
