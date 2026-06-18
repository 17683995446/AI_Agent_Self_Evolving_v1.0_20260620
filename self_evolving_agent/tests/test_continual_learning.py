"""
阶段44测试：持续学习与灾难性遗忘防护
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.continual_learning import (
    EWCRegularizer, ExperienceReplay, ProgressiveNetwork,
    ForgettingMonitor, ContinualLearningController, TaskExperience
)


class TestEWCRegularizer(unittest.TestCase):
    def test_init(self):
        ewc = EWCRegularizer(lambda_ewc=1.0)
        self.assertEqual(ewc.lambda_ewc, 1.0)
        logger.info("init passed")

    def test_compute_fisher(self):
        ewc = EWCRegularizer()
        params = {"W": np.random.randn(4, 3)}
        def forward(x):
            return np.dot(x, params["W"])
        data = np.random.randn(10, 4)
        labels = np.random.randint(0, 3, 10)
        fisher = ewc.compute_fisher_information(forward, params, data, labels, n_samples=5)
        self.assertIn("W", fisher)
        self.assertEqual(fisher["W"].shape, (4, 3))
        logger.info("compute_fisher passed")

    def test_register_task(self):
        ewc = EWCRegularizer()
        params = {"W": np.random.randn(4, 3)}
        def forward(x):
            return np.dot(x, params["W"])
        data = np.random.randn(10, 4)
        labels = np.random.randint(0, 3, 10)
        ewc.register_task(forward, params, data, labels)
        self.assertEqual(ewc.task_count, 1)
        logger.info("register_task passed")

    def test_compute_penalty(self):
        ewc = EWCRegularizer(lambda_ewc=0.5)
        params = {"W": np.random.randn(4, 3)}
        def forward(x):
            return np.dot(x, params["W"])
        data = np.random.randn(10, 4)
        labels = np.random.randint(0, 3, 10)
        ewc.register_task(forward, params, data, labels)
        penalty = ewc.compute_penalty(params)
        self.assertEqual(penalty, 0.0)  # 当前参数=最优参数，惩罚为0
        params["W"] += 0.1
        penalty2 = ewc.compute_penalty(params)
        self.assertGreater(penalty2, 0.0)
        logger.info("compute_penalty passed: %.4f", penalty2)

    def test_get_param_importance(self):
        ewc = EWCRegularizer()
        params = {"W": np.random.randn(4, 3)}
        def forward(x):
            return np.dot(x, params["W"])
        data = np.random.randn(10, 4)
        labels = np.random.randint(0, 3, 10)
        ewc.register_task(forward, params, data, labels)
        importance = ewc.get_param_importance()
        self.assertIn("W", importance)
        logger.info("get_param_importance passed")


class TestExperienceReplay(unittest.TestCase):
    def test_add_task(self):
        replay = ExperienceReplay(capacity_per_task=50)
        data = np.random.randn(100, 4)
        labels = np.random.randint(0, 3, 100)
        replay.add_task("task1", data, labels)
        self.assertIn("task1", replay.experiences)
        self.assertEqual(len(replay.experiences["task1"].data), 50)
        logger.info("add_task passed")

    def test_sample_replay_batch(self):
        replay = ExperienceReplay(capacity_per_task=50)
        replay.add_task("task1", np.random.randn(100, 4), np.random.randint(0, 3, 100))
        replay.add_task("task2", np.random.randn(100, 4), np.random.randint(0, 3, 100))
        data, labels = replay.sample_replay_batch(batch_size=16, exclude_task="task2")
        self.assertEqual(len(data), 16)
        logger.info("sample_replay_batch passed")

    def test_sample_empty(self):
        replay = ExperienceReplay()
        data, labels = replay.sample_replay_batch()
        self.assertEqual(len(data), 0)
        logger.info("sample_empty passed")

    def test_len(self):
        replay = ExperienceReplay(capacity_per_task=30)
        replay.add_task("t1", np.random.randn(50, 4), np.random.randint(0, 2, 50))
        self.assertEqual(len(replay), 30)
        logger.info("len passed")


class TestProgressiveNetwork(unittest.TestCase):
    def test_init(self):
        net = ProgressiveNetwork(4, 16, 2)
        self.assertEqual(net.get_active_column_count(), 1)
        logger.info("init passed")

    def test_forward(self):
        net = ProgressiveNetwork(4, 16, 2)
        x = np.random.randn(5, 4)
        out = net.forward(x)
        self.assertEqual(out.shape, (5, 2))
        logger.info("forward passed")

    def test_add_task(self):
        net = ProgressiveNetwork(4, 16, 2)
        net.add_task()
        self.assertEqual(net.get_active_column_count(), 2)
        x = np.random.randn(5, 4)
        out = net.forward(x)
        self.assertEqual(out.shape, (5, 2))
        logger.info("add_task passed")

    def test_freeze(self):
        net = ProgressiveNetwork(4, 16, 2)
        net.freeze_column(0)
        logger.info("freeze passed")


class TestForgettingMonitor(unittest.TestCase):
    def test_register_baseline(self):
        mon = ForgettingMonitor()
        mon.register_baseline("task1", 0.9)
        self.assertIn("task1", mon.baselines)
        logger.info("register_baseline passed")

    def test_record_performance(self):
        mon = ForgettingMonitor()
        mon.register_baseline("task1", 0.9)
        mon.record_performance("task1", 0.7)
        self.assertEqual(len(mon.history["task1"]), 2)
        logger.info("record_performance passed")

    def test_compute_forgetting(self):
        mon = ForgettingMonitor()
        mon.register_baseline("task1", 0.9)
        mon.record_performance("task1", 0.6)
        result = mon.compute_forgetting("task1")
        self.assertAlmostEqual(result["forgetting"], 0.3)
        self.assertGreater(result["forgetting_ratio"], 0)
        logger.info("compute_forgetting passed: %s", result)

    def test_get_all_forgetting(self):
        mon = ForgettingMonitor()
        mon.register_baseline("t1", 0.9)
        mon.register_baseline("t2", 0.8)
        mon.record_performance("t1", 0.7)
        all_f = mon.get_all_forgetting()
        self.assertEqual(len(all_f), 2)
        logger.info("get_all_forgetting passed")

    def test_get_summary(self):
        mon = ForgettingMonitor()
        mon.register_baseline("t1", 0.9)
        mon.register_baseline("t2", 0.8)
        mon.record_performance("t1", 0.6)
        summary = mon.get_summary()
        self.assertIn("mean_forgetting", summary)
        logger.info("get_summary passed: %s", summary)


class TestContinualLearningController(unittest.TestCase):
    def test_init_ewc(self):
        ctrl = ContinualLearningController(4, 2, strategy="ewc")
        self.assertEqual(ctrl.strategy, "ewc")
        logger.info("init ewc passed")

    def test_init_replay(self):
        ctrl = ContinualLearningController(4, 2, strategy="replay")
        self.assertEqual(ctrl.strategy, "replay")
        logger.info("init replay passed")

    def test_init_ewc_replay(self):
        ctrl = ContinualLearningController(4, 2, strategy="ewc_replay")
        self.assertIsNotNone(ctrl.ewc)
        self.assertIsNotNone(ctrl.replay)
        logger.info("init ewc_replay passed")

    def test_init_progressive(self):
        ctrl = ContinualLearningController(4, 2, strategy="progressive")
        self.assertIsNotNone(ctrl.progressive)
        logger.info("init progressive passed")

    def test_learn_task(self):
        ctrl = ContinualLearningController(4, 2, strategy="ewc_replay")
        data = np.random.randn(30, 4)
        labels = np.random.randint(0, 2, 30)
        result = ctrl.learn_task("task1", data, labels, n_steps=3, lr=0.01)
        self.assertIn("accuracy", result)
        self.assertEqual(ctrl.task_count, 1)
        logger.info("learn_task passed: %s", result)

    def test_learn_multiple_tasks(self):
        ctrl = ContinualLearningController(4, 2, strategy="ewc_replay")
        for i in range(3):
            data = np.random.randn(20, 4)
            labels = np.random.randint(0, 2, 20)
            ctrl.learn_task(f"task{i}", data, labels, n_steps=3, lr=0.01)
        self.assertEqual(ctrl.task_count, 3)
        logger.info("learn_multiple_tasks passed")

    def test_evaluate_task(self):
        ctrl = ContinualLearningController(4, 2, strategy="ewc")
        data = np.random.randn(20, 4)
        labels = np.random.randint(0, 2, 20)
        ctrl.learn_task("t1", data, labels, n_steps=3, lr=0.01)
        result = ctrl.evaluate_task("t1", data, labels)
        self.assertIn("accuracy", result)
        self.assertIn("forgetting", result)
        logger.info("evaluate_task passed: %s", result)

    def test_get_stats(self):
        ctrl = ContinualLearningController(4, 2, strategy="ewc_replay")
        data = np.random.randn(20, 4)
        labels = np.random.randint(0, 2, 20)
        ctrl.learn_task("t1", data, labels, n_steps=3, lr=0.01)
        stats = ctrl.get_stats()
        self.assertEqual(stats["task_count"], 1)
        logger.info("get_stats passed: %s", stats)

    def test_save_load(self):
        ctrl = ContinualLearningController(4, 2, strategy="ewc")
        data = np.random.randn(20, 4)
        labels = np.random.randint(0, 2, 20)
        ctrl.learn_task("t1", data, labels, n_steps=3, lr=0.01)
        saved = ctrl.save()
        self.assertIn("model_params", saved)

        ctrl2 = ContinualLearningController(4, 2, strategy="ewc")
        ctrl2.load(saved)
        self.assertEqual(ctrl2.task_count, 1)
        logger.info("save_load passed")


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
    unittest.main(verbosity=2)
