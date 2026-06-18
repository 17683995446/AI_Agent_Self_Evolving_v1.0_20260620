"""
阶段45测试：不确定性量化与贝叶斯神经网络
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.uncertainty_quantification import (
    MCDropoutNetwork, DeepEnsemble, BayesianLinearRegression,
    UncertaintyEstimator, UncertaintyController, UncertaintyResult
)


class TestMCDropoutNetwork(unittest.TestCase):
    def test_init(self):
        net = MCDropoutNetwork(4, 16, 1, dropout_rate=0.2, n_mc_samples=10)
        self.assertEqual(net.input_dim, 4)
        logger.info("init passed")

    def test_predict(self):
        net = MCDropoutNetwork(4, 16, 1)
        x = np.random.randn(5, 4)
        out = net.predict(x)
        self.assertEqual(out.shape, (5, 1))
        logger.info("predict passed")

    def test_predict_with_uncertainty(self):
        net = MCDropoutNetwork(4, 16, 1, n_mc_samples=10)
        x = np.random.randn(5, 4)
        result = net.predict_with_uncertainty(x)
        self.assertEqual(result.mean.shape, (5, 1))
        self.assertGreaterEqual(result.epistemic_uncertainty, 0)
        self.assertGreaterEqual(result.confidence, 0)
        self.assertEqual(result.n_samples, 10)
        logger.info("predict_with_uncertainty passed: conf=%.4f", result.confidence)

    def test_train_step(self):
        net = MCDropoutNetwork(4, 8, 1)
        x = np.random.randn(10, 4)
        y = np.random.randn(10, 1)
        loss = net.train_step(x, y, lr=0.01)
        self.assertIsInstance(loss, float)
        logger.info("train_step passed: loss=%.4f", loss)


class TestDeepEnsemble(unittest.TestCase):
    def test_init(self):
        ens = DeepEnsemble(4, 16, 1, n_models=3)
        self.assertEqual(ens.n_models, 3)
        logger.info("init passed")

    def test_predict_with_uncertainty(self):
        ens = DeepEnsemble(4, 16, 1, n_models=3)
        x = np.random.randn(5, 4)
        result = ens.predict_with_uncertainty(x)
        self.assertEqual(result.mean.shape, (5, 1))
        self.assertEqual(result.n_samples, 3)
        logger.info("predict_with_uncertainty passed")

    def test_train_model(self):
        ens = DeepEnsemble(4, 8, 1, n_models=3)
        x = np.random.randn(10, 4)
        y = np.random.randn(10, 1)
        loss = ens.train_model(0, x, y, lr=0.01, n_steps=2)
        self.assertIsInstance(loss, float)
        logger.info("train_model passed: loss=%.4f", loss)

    def test_train_all(self):
        ens = DeepEnsemble(4, 8, 1, n_models=3)
        x = np.random.randn(10, 4)
        y = np.random.randn(10, 1)
        losses = ens.train_all(x, y, lr=0.01, n_steps=2)
        self.assertEqual(len(losses), 3)
        logger.info("train_all passed")


class TestBayesianLinearRegression(unittest.TestCase):
    def test_init(self):
        blr = BayesianLinearRegression(4)
        self.assertEqual(blr.input_dim, 4)
        logger.info("init passed")

    def test_fit(self):
        blr = BayesianLinearRegression(4)
        X = np.random.randn(20, 4)
        y = np.random.randn(20)
        blr.fit(X, y)
        self.assertEqual(blr.post_mean.shape, (4,))
        logger.info("fit passed")

    def test_predict_with_uncertainty(self):
        blr = BayesianLinearRegression(4)
        X = np.random.randn(20, 4)
        y = np.random.randn(20)
        blr.fit(X, y)
        X_test = np.random.randn(5, 4)
        result = blr.predict_with_uncertainty(X_test)
        self.assertEqual(result.mean.shape, (5, 1))
        self.assertGreaterEqual(result.epistemic_uncertainty, 0)
        logger.info("predict_with_uncertainty passed")


class TestUncertaintyEstimator(unittest.TestCase):
    def test_mc_dropout(self):
        est = UncertaintyEstimator(method="mc_dropout", input_dim=4, hidden_dim=8, n_mc_samples=5)
        x = np.random.randn(5, 4)
        result = est.predict(x)
        self.assertIsInstance(result, UncertaintyResult)
        logger.info("mc_dropout passed")

    def test_ensemble(self):
        est = UncertaintyEstimator(method="ensemble", input_dim=4, hidden_dim=8, n_models=3)
        x = np.random.randn(5, 4)
        result = est.predict(x)
        self.assertIsInstance(result, UncertaintyResult)
        logger.info("ensemble passed")

    def test_bayesian_lr(self):
        est = UncertaintyEstimator(method="bayesian_lr", input_dim=4)
        X = np.random.randn(20, 4)
        y = np.random.randn(20)
        est.train(X, y)
        result = est.predict(X[:5])
        self.assertIsInstance(result, UncertaintyResult)
        logger.info("bayesian_lr passed")

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            UncertaintyEstimator(method="unknown", input_dim=4)
        logger.info("unknown_method exception passed")


class TestUncertaintyController(unittest.TestCase):
    def test_init(self):
        ctrl = UncertaintyController(4, 1, method="mc_dropout", n_mc_samples=5)
        self.assertEqual(ctrl.method, "mc_dropout")
        logger.info("init passed")

    def test_train_predict(self):
        ctrl = UncertaintyController(4, 1, method="mc_dropout", hidden_dim=8, n_mc_samples=5)
        x = np.random.randn(10, 4)
        y = np.random.randn(10, 1)
        ctrl.train(x, y, lr=0.01)
        result = ctrl.predict(x[:5])
        self.assertIn("mean", result)
        self.assertIn("confidence", result)
        self.assertIn("is_reliable", result)
        logger.info("train_predict passed: conf=%.4f", result["confidence"])

    def test_get_confidence_stats(self):
        ctrl = UncertaintyController(4, 1, method="mc_dropout", hidden_dim=8, n_mc_samples=5)
        x = np.random.randn(5, 4)
        ctrl.predict(x)
        stats = ctrl.get_confidence_stats()
        self.assertIn("mean_confidence", stats)
        logger.info("get_confidence_stats passed: %s", stats)

    def test_get_stats(self):
        ctrl = UncertaintyController(4, 1, method="ensemble", n_models=3)
        stats = ctrl.get_stats()
        self.assertEqual(stats["method"], "ensemble")
        logger.info("get_stats passed")

    def test_save_load(self):
        ctrl = UncertaintyController(4, 1, method="mc_dropout", n_mc_samples=5)
        ctrl.step_count = 10
        saved = ctrl.save()
        ctrl2 = UncertaintyController(4, 1, method="mc_dropout", n_mc_samples=5)
        ctrl2.load(saved)
        self.assertEqual(ctrl2.step_count, 10)
        logger.info("save_load passed")


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
    unittest.main(verbosity=2)
