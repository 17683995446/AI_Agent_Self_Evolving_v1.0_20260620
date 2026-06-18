"""
阶段41测试：对抗鲁棒性训练
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.adversarial_robustness import (
    PerturbationBudget, AdversarialExampleGenerator,
    RobustnessEvaluator, AdversarialTrainer,
    RobustnessCertifier, AdversarialRobustnessController
)


def dummy_forward(x):
    """简单线性模型前向"""
    w = np.array([[1.0, -1.0], [-1.0, 1.0], [0.5, -0.5]])
    return x @ w


def dummy_predict(x):
    logits = dummy_forward(x)
    return np.argmax(logits, axis=-1)


class TestPerturbationBudget(unittest.TestCase):
    def test_is_valid_linf(self):
        budget = PerturbationBudget(epsilon=0.1, norm="linf")
        orig = np.zeros(5)
        pert = np.ones(5) * 0.05
        self.assertTrue(budget.is_valid(orig, pert))
        logger.info("is_valid_linf passed")

    def test_is_valid_linf_exceed(self):
        budget = PerturbationBudget(epsilon=0.1, norm="linf")
        orig = np.zeros(5)
        pert = np.ones(5) * 0.2
        self.assertFalse(budget.is_valid(orig, pert))
        logger.info("is_valid_linf_exceed passed")

    def test_clip_linf(self):
        budget = PerturbationBudget(epsilon=0.1, norm="linf")
        orig = np.zeros(5)
        pert = np.ones(5) * 0.5
        clipped = budget.clip(orig, pert)
        self.assertTrue(budget.is_valid(orig, clipped))
        logger.info("clip_linf passed")

    def test_clip_l2(self):
        budget = PerturbationBudget(epsilon=1.0, norm="l2")
        orig = np.zeros(5)
        pert = np.ones(5) * 10.0
        clipped = budget.clip(orig, pert)
        self.assertTrue(budget.is_valid(orig, clipped))
        logger.info("clip_l2 passed")


class TestAdversarialExampleGenerator(unittest.TestCase):
    def test_fgsm(self):
        gen = AdversarialExampleGenerator(dummy_forward)
        x = np.array([1.0, 0.0, 0.0])
        y = 0
        budget = PerturbationBudget(epsilon=0.1)
        adv = gen.fgsm(x, y, budget)
        self.assertTrue(budget.is_valid(x, adv))
        logger.info("fgsm passed")

    def test_pgd(self):
        gen = AdversarialExampleGenerator(dummy_forward)
        x = np.array([1.0, 0.0, 0.0])
        y = 0
        budget = PerturbationBudget(epsilon=0.1)
        adv = gen.pgd(x, y, budget, steps=5)
        self.assertTrue(budget.is_valid(x, adv))
        logger.info("pgd passed")

    def test_generate_fgsm(self):
        gen = AdversarialExampleGenerator(dummy_forward)
        x = np.array([1.0, 0.0, 0.0])
        y = 0
        adv = gen.generate(x, y, method="fgsm", budget=PerturbationBudget(epsilon=0.1))
        self.assertTrue(PerturbationBudget(epsilon=0.1).is_valid(x, adv))
        logger.info("generate_fgsm passed")

    def test_generate_pgd(self):
        gen = AdversarialExampleGenerator(dummy_forward)
        x = np.array([1.0, 0.0, 0.0])
        y = 0
        adv = gen.generate(x, y, method="pgd", budget=PerturbationBudget(epsilon=0.1), steps=3)
        self.assertTrue(PerturbationBudget(epsilon=0.1).is_valid(x, adv))
        logger.info("generate_pgd passed")

    def test_unknown_method(self):
        gen = AdversarialExampleGenerator(dummy_forward)
        with self.assertRaises(ValueError):
            gen.generate(np.zeros(3), 0, method="unknown")
        logger.info("unknown_method exception passed")


class TestRobustnessEvaluator(unittest.TestCase):
    def test_evaluate(self):
        evalr = RobustnessEvaluator(dummy_forward, dummy_predict)
        x = np.random.randn(10, 3)
        y = dummy_predict(x)
        result = evalr.evaluate(x, y, budget=PerturbationBudget(epsilon=0.05))
        self.assertIn("clean_accuracy", result)
        self.assertIn("adversarial_accuracy", result)
        self.assertIn("accuracy_drop", result)
        logger.info("evaluate passed: %s", result)

    def test_evaluate_robustness_curve(self):
        evalr = RobustnessEvaluator(dummy_forward, dummy_predict)
        x = np.random.randn(5, 3)
        y = dummy_predict(x)
        results = evalr.evaluate_robustness_curve(x, y, epsilons=[0.0, 0.05, 0.1])
        self.assertEqual(len(results), 3)
        logger.info("evaluate_robustness_curve passed")


class TestAdversarialTrainer(unittest.TestCase):
    def test_train_step(self):
        model = {"W": np.random.randn(3, 2)}

        def forward(x):
            return x @ model["W"]

        def update(x, y):
            # 简化更新：随机扰动权重
            model["W"] += np.random.randn(3, 2) * 0.001
            return 0.5

        trainer = AdversarialTrainer(model, forward, update,
                                     budget=PerturbationBudget(epsilon=0.05),
                                     adversarial_ratio=0.3)
        x = np.random.randn(10, 3)
        y = np.random.randint(0, 2, 10)
        metrics = trainer.train_step(x, y)
        self.assertIn("loss", metrics)
        self.assertIn("clean_acc", metrics)
        logger.info("train_step passed: %s", metrics)

    def test_train(self):
        model = {"W": np.random.randn(3, 2)}

        def forward(x):
            return x @ model["W"]

        def update(x, y):
            model["W"] += np.random.randn(3, 2) * 0.001
            return 0.5

        trainer = AdversarialTrainer(model, forward, update,
                                     budget=PerturbationBudget(epsilon=0.05),
                                     adversarial_ratio=0.3)

        def data_gen():
            return np.random.randn(8, 3), np.random.randint(0, 2, 8)

        result = trainer.train(data_gen, n_steps=3)
        self.assertIn("final_clean_acc", result)
        logger.info("train passed: %s", result)


class TestRobustnessCertifier(unittest.TestCase):
    def test_estimate_local_lipschitz(self):
        cert = RobustnessCertifier(dummy_forward)
        x = np.array([1.0, 0.0, 0.0])
        lip = cert.estimate_local_lipschitz(x, n_samples=20)
        self.assertGreaterEqual(lip, 0.0)
        logger.info("estimate_local_lipschitz passed: %.4f", lip)

    def test_certify_radius(self):
        cert = RobustnessCertifier(dummy_forward)
        x = np.array([1.0, 0.0, 0.0])
        radius = cert.certify_radius(x, y_true=0)
        self.assertGreaterEqual(radius, 0.0)
        logger.info("certify_radius passed: %.4f", radius)


class TestAdversarialRobustnessController(unittest.TestCase):
    def test_init(self):
        ctrl = AdversarialRobustnessController(None, dummy_forward, dummy_predict)
        self.assertFalse(ctrl.get_stats()["has_trainer"])
        logger.info("init passed")

    def test_init_with_trainer(self):
        def update(x, y):
            return 0.5
        ctrl = AdversarialRobustnessController(None, dummy_forward, dummy_predict, update)
        self.assertTrue(ctrl.get_stats()["has_trainer"])
        logger.info("init_with_trainer passed")

    def test_generate_attack(self):
        ctrl = AdversarialRobustnessController(None, dummy_forward, dummy_predict)
        x = np.array([1.0, 0.0, 0.0])
        y = 0
        adv = ctrl.generate_attack(x, y, method="fgsm", epsilon=0.1)
        self.assertTrue(PerturbationBudget(epsilon=0.1).is_valid(x, adv))
        logger.info("generate_attack passed")

    def test_evaluate_robustness(self):
        ctrl = AdversarialRobustnessController(None, dummy_forward, dummy_predict)
        x = np.random.randn(5, 3)
        y = dummy_predict(x)
        result = ctrl.evaluate_robustness(x, y, epsilon=0.05)
        self.assertIn("clean_accuracy", result)
        logger.info("evaluate_robustness passed: %s", result)

    def test_certify(self):
        ctrl = AdversarialRobustnessController(None, dummy_forward, dummy_predict)
        x = np.array([1.0, 0.0, 0.0])
        result = ctrl.certify(x, y_true=0)
        self.assertIn("certified_radius", result)
        logger.info("certify passed: %s", result)

    def test_adversarial_train(self):
        model = {"W": np.random.randn(3, 2)}

        def forward(x):
            return x @ model["W"]

        def update(x, y):
            model["W"] += np.random.randn(3, 2) * 0.001
            return 0.5

        ctrl = AdversarialRobustnessController(model, forward, dummy_predict, update)

        def data_gen():
            return np.random.randn(8, 3), np.random.randint(0, 2, 8)

        result = ctrl.adversarial_train(data_gen, n_steps=3)
        self.assertIn("final_clean_acc", result)
        logger.info("adversarial_train passed: %s", result)

    def test_adversarial_train_no_update(self):
        ctrl = AdversarialRobustnessController(None, dummy_forward, dummy_predict)
        with self.assertRaises(ValueError):
            ctrl.adversarial_train(lambda: (np.zeros((2, 3)), np.zeros(2)))
        logger.info("adversarial_train_no_update exception passed")

    def test_save_load(self):
        ctrl = AdversarialRobustnessController(None, dummy_forward, dummy_predict)
        data = ctrl.save()
        self.assertIn("has_trainer", data)
        ctrl.load(data)
        logger.info("save_load passed")


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
    unittest.main(verbosity=2)
