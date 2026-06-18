"""
阶段46测试：自动特征工程与选择
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.auto_feature import (
    FeatureTransformer, MutualInfoSelector, GeneticFeatureEngineer,
    FeaturePipeline, AutoFeatureController, FeatureDef
)


class TestFeatureTransformer(unittest.TestCase):
    def test_init(self):
        t = FeatureTransformer()
        self.assertGreater(len(t.BINARY_OPS), 0)
        self.assertGreater(len(t.UNARY_OPS), 0)
        logger.info("init passed")

    def test_apply_unary(self):
        t = FeatureTransformer()
        X = np.random.randn(10, 3)
        names = ["a", "b", "c"]
        result, defn = t.apply_unary(X, names, 0, "square")
        self.assertEqual(len(result), 10)
        self.assertEqual(defn.operation, "square")
        logger.info("apply_unary passed")

    def test_apply_binary(self):
        t = FeatureTransformer()
        X = np.random.randn(10, 3)
        names = ["a", "b", "c"]
        result, defn = t.apply_binary(X, names, 0, 1, "mul")
        self.assertEqual(len(result), 10)
        self.assertEqual(defn.operation, "mul")
        logger.info("apply_binary passed")

    def test_generate_all(self):
        t = FeatureTransformer()
        X = np.random.randn(20, 3)
        names = ["a", "b", "c"]
        new_X, new_names, defs = t.generate_all(X, names, max_features=20)
        self.assertGreater(len(defs), 0)
        self.assertEqual(new_X.shape[0], 20)
        logger.info("generate_all passed: %d features", len(defs))


class TestMutualInfoSelector(unittest.TestCase):
    def test_compute_mutual_info(self):
        sel = MutualInfoSelector(n_bins=5)
        x = np.random.randn(100)
        y = (x > 0).astype(int)
        mi = sel.compute_mutual_info(x, y)
        self.assertGreater(mi, 0)
        logger.info("compute_mutual_info passed: mi=%.4f", mi)

    def test_fit(self):
        sel = MutualInfoSelector(n_bins=5)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        names = ["f0", "f1", "f2", "f3", "f4"]
        selected = sel.fit(X, y, names, n_select=3)
        self.assertEqual(len(selected), 3)
        logger.info("fit passed: %s", selected)

    def test_transform(self):
        sel = MutualInfoSelector(n_bins=5)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        names = ["f0", "f1", "f2", "f3", "f4"]
        sel.fit(X, y, names, n_select=3)
        result = sel.transform(X, names)
        self.assertEqual(result.shape[1], 3)
        logger.info("transform passed")

    def test_get_scores(self):
        sel = MutualInfoSelector(n_bins=5)
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        names = ["a", "b", "c"]
        sel.fit(X, y, names, n_select=2)
        scores = sel.get_scores()
        self.assertEqual(len(scores), 3)
        logger.info("get_scores passed")


class TestGeneticFeatureEngineer(unittest.TestCase):
    def test_evolve(self):
        eng = GeneticFeatureEngineer(population_size=10, n_generations=3)
        X = np.random.randn(30, 3)
        y = (X[:, 0] > 0).astype(float)
        names = ["a", "b", "c"]
        features = eng.evolve(X, y, names)
        self.assertGreater(len(features), 0)
        logger.info("evolve passed: %d features", len(features))

    def test_fitness_history(self):
        eng = GeneticFeatureEngineer(population_size=5, n_generations=2)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        names = ["a", "b", "c"]
        eng.evolve(X, y, names)
        self.assertGreater(len(eng.fitness_history), 0)
        logger.info("fitness_history passed")


class TestFeaturePipeline(unittest.TestCase):
    def test_fit_transform(self):
        pipe = FeaturePipeline(max_generated=20, n_select=5)
        X = np.random.randn(30, 3)
        y = np.random.randint(0, 2, 30)
        names = ["a", "b", "c"]
        result_X, result_names = pipe.fit_transform(X, y, names)
        self.assertEqual(result_X.shape[1], 5)
        self.assertEqual(len(result_names), 5)
        logger.info("fit_transform passed: %d -> %d", X.shape[1], result_X.shape[1])

    def test_transform(self):
        pipe = FeaturePipeline(max_generated=20, n_select=3)
        X = np.random.randn(30, 3)
        y = np.random.randint(0, 2, 30)
        names = ["a", "b", "c"]
        pipe.fit_transform(X, y, names)
        X_new = np.random.randn(10, 3)
        result = pipe.transform(X_new, names)
        self.assertEqual(result.shape[0], 10)
        self.assertLessEqual(result.shape[1], 3)
        logger.info("transform passed: shape=%s", result.shape)

    def test_get_feature_importance(self):
        pipe = FeaturePipeline(max_generated=10, n_select=3)
        X = np.random.randn(30, 3)
        y = np.random.randint(0, 2, 30)
        names = ["a", "b", "c"]
        pipe.fit_transform(X, y, names)
        importance = pipe.get_feature_importance()
        self.assertGreater(len(importance), 0)
        logger.info("get_feature_importance passed")


class TestAutoFeatureController(unittest.TestCase):
    def test_init(self):
        ctrl = AutoFeatureController(max_generated=20, n_select=5)
        self.assertFalse(ctrl.is_fitted)
        logger.info("init passed")

    def test_init_with_genetic(self):
        ctrl = AutoFeatureController(use_genetic=True, genetic_generations=2)
        self.assertTrue(ctrl.use_genetic)
        self.assertIsNotNone(ctrl.genetic_engineer)
        logger.info("init_with_genetic passed")

    def test_fit(self):
        ctrl = AutoFeatureController(max_generated=20, n_select=5)
        X = np.random.randn(30, 3)
        y = np.random.randint(0, 2, 30)
        result = ctrl.fit(X, y, ["a", "b", "c"])
        self.assertTrue(ctrl.is_fitted)
        self.assertIn("selected_features", result)
        self.assertIn("feature_importance", result)
        logger.info("fit passed: %d selected", result["selected_features"])

    def test_fit_with_genetic(self):
        ctrl = AutoFeatureController(max_generated=10, n_select=3, use_genetic=True, genetic_generations=2)
        X = np.random.randn(20, 3)
        y = np.random.randint(0, 2, 20)
        result = ctrl.fit(X, y, ["a", "b", "c"])
        self.assertIn("genetic_features", result)
        logger.info("fit_with_genetic passed")

    def test_transform(self):
        ctrl = AutoFeatureController(max_generated=20, n_select=3)
        X = np.random.randn(30, 3)
        y = np.random.randint(0, 2, 30)
        ctrl.fit(X, y, ["a", "b", "c"])
        X_new = np.random.randn(10, 3)
        result = ctrl.transform(X_new)
        self.assertEqual(result.shape[0], 10)
        logger.info("transform passed")

    def test_transform_not_fitted(self):
        ctrl = AutoFeatureController()
        with self.assertRaises(ValueError):
            ctrl.transform(np.random.randn(5, 3))
        logger.info("transform_not_fitted exception passed")

    def test_get_stats(self):
        ctrl = AutoFeatureController(max_generated=20, n_select=5)
        X = np.random.randn(30, 3)
        y = np.random.randint(0, 2, 30)
        ctrl.fit(X, y, ["a", "b", "c"])
        stats = ctrl.get_stats()
        self.assertTrue(stats["is_fitted"])
        logger.info("get_stats passed: %s", stats)

    def test_save_load(self):
        ctrl = AutoFeatureController(max_generated=20, n_select=5)
        X = np.random.randn(30, 3)
        y = np.random.randint(0, 2, 30)
        ctrl.fit(X, y, ["a", "b", "c"])
        saved = ctrl.save()
        self.assertIn("selected_feature_names", saved)

        ctrl2 = AutoFeatureController()
        ctrl2.load(saved)
        self.assertTrue(ctrl2.is_fitted)
        self.assertEqual(ctrl2.selected_feature_names, ctrl.selected_feature_names)
        logger.info("save_load passed")


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
    unittest.main(verbosity=2)
