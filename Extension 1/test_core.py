"""Small boundary checks for conformal decisions and normal-only splitting."""
import unittest

import joblib
import numpy as np

from course_data import split_normal
from normal_bags import NormalBagDetector
from power_reference import PowerPrototypeReference
from unified import UnifiedDetector, conformal_threshold, normal_p_values


class CoreTests(unittest.TestCase):
    def test_ties_and_strict_threshold(self):
        cal = np.array([1.] * 18 + [2., 2.])
        scores = np.array([1., 2., 2.0001])
        threshold = conformal_threshold(cal, alpha=.05)
        self.assertEqual(threshold, 2.)
        np.testing.assert_allclose(normal_p_values(scores, cal), [1., 3./21, 1./21])
        np.testing.assert_array_equal(scores > threshold, [False, False, True])
        np.testing.assert_array_equal(scores > threshold, normal_p_values(scores, cal) <= .05)

    def test_small_calibration_has_infinite_threshold(self):
        cal = np.arange(18.)
        self.assertTrue(np.isposinf(conformal_threshold(cal, alpha=.05)))
        self.assertGreater(normal_p_values(np.array([1e12]), cal)[0], .05)
        self.assertEqual(conformal_threshold(np.arange(19.), alpha=.05), 18.)
        self.assertTrue(np.isposinf(conformal_threshold(np.arange(40.), alpha=.01)))
        self.assertGreater(normal_p_values(np.array([1e12]), np.arange(40.))[0], .01)

    def test_single_descriptor_bag_reduces_to_row_knn(self):
        fit = np.array([[-2., 0.], [-1., 1.], [0., 0.], [1., -1.], [2., 0.]])
        query = np.array([[.3, .2], [3., 1.]])
        model = NormalBagDetector(k=2, max_bank=20).fit(fit)
        transformed = model.transform(query)
        expected = [np.sort(np.linalg.norm(model.reference - row, axis=1))[:2].mean()/np.sqrt(2)
                    for row in transformed]
        np.testing.assert_allclose(model.scores(query), expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(model.scores(query[:, None, :]), expected, rtol=0, atol=1e-12)

    def test_top_pooling_uses_ceil_and_largest_distances(self):
        model = NormalBagDetector(k=1, max_bank=10, top_fraction=.02).fit(np.array([[0.], [1.], [2.]]))
        query = np.zeros((1, 51, 1)); query[0, -2:, 0] = [5., 10.]
        distances = model.descriptor_scores(query)[0]
        expected = (distances[-1]+distances[-2])/2
        self.assertAlmostEqual(model.scores(query)[0], expected, places=12)
        self.assertGreater(distances[-2], distances[:-2].max())

    def test_bag_scores_do_not_depend_on_batch(self):
        rng = np.random.default_rng(42)
        fit, query = rng.normal(size=(8, 7, 4)), rng.normal(size=(4, 7, 4))
        model = NormalBagDetector(k=3, max_bank=20, top_fraction=.3).fit(fit)
        together = model.scores(query)
        separate = np.array([model.scores(bag[None, :, :])[0] for bag in query])
        np.testing.assert_allclose(together, separate, rtol=0, atol=1e-12)

    def test_bank_owner_identity_and_sample_level_calibration(self):
        rng = np.random.default_rng(11)
        fit, cal = rng.normal(size=(6, 4, 3)), rng.normal(size=(40, 4, 3))
        model = NormalBagDetector(k=2, max_bank=9, alpha=.01).fit(fit).calibrate(cal)
        self.assertEqual(model.calibration.shape, (40,))
        self.assertTrue(np.isposinf(model.threshold))
        self.assertEqual(model.fit_shape, fit.shape)
        np.testing.assert_array_equal(model.reference_owners, model.memory_indices//4)
        self.assertTrue(set(model.reference_owners).issubset(set(range(6))))
        expected = model.transform(fit.reshape(-1, 3))[model.memory_indices]
        np.testing.assert_allclose(model.reference, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(model.predict(cal)["prediction"], np.zeros(40, dtype=int))

    def test_scores_do_not_depend_on_other_batch_rows(self):
        rng = np.random.default_rng(12)
        model = UnifiedDetector(k=3, max_components=3).fit(rng.normal(size=(40, 5)))
        query = rng.normal(size=(7, 5))
        together = model.scores(query)
        for name, scores in together.items():
            individual = [model.scores(row[None, :])[name][0] for row in query]
            np.testing.assert_allclose(scores, individual, rtol=0, atol=1e-12)

    def test_residual_detects_discarded_direction(self):
        x = np.column_stack((np.linspace(-2, 2, 40), np.zeros(40)))
        model = UnifiedDetector(k=3).fit(x)
        scores = model.scores(np.array([[0., 0.], [0., 10.]]))
        self.assertEqual(model.q, 1)
        self.assertEqual(scores["knn_only"][0], scores["knn_only"][1])
        self.assertGreater(scores["knn_residual"][1], scores["knn_residual"][0] + 5.)

    def test_deduplication_prevents_cross_split_copies(self):
        unique = np.arange(60.).reshape(10, 6)
        x = np.concatenate((unique, unique[[1, 1, 4, 7, 7, 7]]))
        fit, cal, retained = split_normal(x, seed=42, fraction=.2, deduplicate=True)
        np.testing.assert_array_equal(retained, np.arange(10))
        self.assertEqual((len(fit), len(cal)), (8, 2))
        self.assertFalse({tuple(row) for row in x[fit]} & {tuple(row) for row in x[cal]})
        other = split_normal(x, seed=42, fraction=.2, deduplicate=True)
        for a, b in zip((fit, cal, retained), other):
            np.testing.assert_array_equal(a, b)

    def test_power_reference_requires_all_six_finite_attributes(self):
        rng = np.random.default_rng(28)
        fit = rng.normal(size=(60, 6))
        model = PowerPrototypeReference().fit(fit)
        invalid = [np.empty((0, 6)), np.ones((4, 5)), np.ones((4, 2, 6))]
        nonfinite = fit[:4].copy()
        nonfinite[0, 0] = np.nan  # An unselected original feature is still validated.
        invalid.append(nonfinite)
        for values in invalid:
            with self.subTest(shape=values.shape):
                with self.assertRaises(ValueError):
                    model.scores(values)
        with self.assertRaises(ValueError):
            PowerPrototypeReference().fit(fit[:2])
        constant = fit.copy()
        constant[:, 1] = 3.
        with self.assertRaises(ValueError):
            PowerPrototypeReference().fit(constant)

    def test_power_reference_batches_and_singleton_bags_agree(self):
        rng = np.random.default_rng(29)
        fit, query = rng.normal(size=(90, 6)), rng.normal(size=(8, 6))
        model = PowerPrototypeReference().fit(fit)
        self.assertIs(type(model.core), NormalBagDetector)
        self.assertEqual((model.core.metric, model.core.distance,
                          model.core.reference_mode, model.core.k),
                         ("identity", "manhattan", "median", 1))
        self.assertEqual(model.core.dimension, 1)
        expected_reference = np.median(model.transform(fit), axis=0, keepdims=True)
        np.testing.assert_allclose(model.core.reference, expected_reference, atol=1e-12, rtol=0)
        expected = np.abs(model.transform(query) - expected_reference).sum(axis=1)
        together = model.scores(query)
        separate = [model.scores(row[None, :])[0] for row in query]
        np.testing.assert_allclose(together, expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(together, separate, atol=1e-12, rtol=0)
        np.testing.assert_allclose(together, model.scores(query[:, None, :]), atol=1e-12, rtol=0)
        # The joint tail requires BOTH high feature 2 and low feature 6.
        coordinates = np.array([[1., -2.], [-1., -2.], [1., 2.], [0., -2.]])
        directional = np.zeros((4, 6))
        directional[:, [1, 5]] = model.power.inverse_transform(model.scaler.inverse_transform(coordinates))
        np.testing.assert_allclose(model.scores(directional), [2., 0., 0., 0.], atol=1e-12, rtol=0)

    def test_power_calibration_cannot_refit_the_representation(self):
        rng = np.random.default_rng(30)
        fit = rng.normal(size=(100, 6))
        calibration = rng.normal(loc=4., scale=2., size=(120, 6))
        model = PowerPrototypeReference(alpha=.05).fit(fit)
        fitted_state = joblib.hash((model.power, model.scaler, model.core, model.fit_count))
        expected_scores = model.scores(calibration)
        model.calibrate(calibration)
        self.assertEqual(joblib.hash((model.power, model.scaler, model.core, model.fit_count)), fitted_state)
        np.testing.assert_array_equal(model.calibration, expected_scores)
        self.assertEqual(model.threshold, conformal_threshold(expected_scores, .05))

    def test_power_reference_keeps_equal_threshold_scores_normal(self):
        rng = np.random.default_rng(31)
        model = PowerPrototypeReference(alpha=.2).fit(rng.normal(size=(80, 6)))
        # Repeated calibration rows create an exact score tie at the threshold.
        calibration = np.repeat(rng.normal(size=(1, 6)), 40, axis=0)
        model.calibrate(calibration)
        result = model.predict(calibration)
        np.testing.assert_array_equal(result["score"], np.full(40, model.threshold))
        np.testing.assert_array_equal(result["prediction"], np.zeros(40, dtype=int))
        np.testing.assert_array_equal(result["p_normal"], np.ones(40))


if __name__ == "__main__":
    unittest.main(verbosity=2)
