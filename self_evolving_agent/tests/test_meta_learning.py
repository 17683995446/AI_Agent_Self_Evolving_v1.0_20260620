"""
阶段39测试：元学习与快速适应
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.meta_learning import (
    Task, TaskDistribution, SimpleModel, MAMLModel,
    MetaLearner, FastAdapter, MetaReplayBuffer, MetaLearningController
)


class TestTaskDistribution(unittest.TestCase):
    def test_sample_task(self):
        dist = TaskDistribution(8, n_classes=3, k_shot=5, q_query=10)
        task = dist.sample_task()
        self.assertEqual(task.support_x.shape[0], 15)  # 3 classes * 5 shots
        self.assertEqual(task.query_x.shape[0], 30)    # 3 classes * 10 queries
        logger.info("sample_task passed: support=%s, query=%s", task.support_x.shape, task.query_x.shape)

    def test_sample_batch(self):
        dist = TaskDistribution(8, n_classes=3, k_shot=5, q_query=10)
        batch = dist.sample_batch(4)
        self.assertEqual(len(batch), 4)
        logger.info("sample_batch passed")


class TestSimpleModel(unittest.TestCase):
    def test_forward(self):
        model = SimpleModel(8, 5)
        x = np.random.randn(10, 8)
        logits = model.forward(x)
        self.assertEqual(logits.shape, (10, 5))
        logger.info("forward passed")

    def test_predict(self):
        model = SimpleModel(8, 5)
        x = np.random.randn(10, 8)
        preds = model.predict(x)
        self.assertEqual(len(preds), 10)
        self.assertTrue(all(0 <= p < 5 for p in preds))
        logger.info("predict passed")

    def test_loss(self):
        model = SimpleModel(8, 5)
        x = np.random.randn(10, 8)
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        loss = model.loss(x, y)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
        logger.info("loss passed: %.4f", loss)

    def test_accuracy(self):
        model = SimpleModel(8, 5)
        x = np.random.randn(10, 8)
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        acc = model.accuracy(x, y)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
        logger.info("accuracy passed: %.4f", acc)

    def test_copy(self):
        model = SimpleModel(8, 5)
        model2 = model.copy()
        np.testing.assert_array_equal(model.W1, model2.W1)
        logger.info("copy passed")

    def test_get_set_params(self):
        model = SimpleModel(8, 5)
        params = model.get_params()
        model.set_params(params)
        for k in params:
            np.testing.assert_array_equal(params[k], model.get_params()[k])
        logger.info("get_set_params passed")


class TestMAMLModel(unittest.TestCase):
    def test_adapt(self):
        model = SimpleModel(8, 5)
        maml = MAMLModel(model, inner_lr=0.01, inner_steps=3)
        dist = TaskDistribution(8, n_classes=3, k_shot=5, q_query=10)
        task = dist.sample_task()
        adapted = maml.adapt(task)
        self.assertIsNotNone(adapted)
        logger.info("adapt passed")

    def test_meta_loss(self):
        model = SimpleModel(8, 5)
        maml = MAMLModel(model, inner_lr=0.01, inner_steps=2)
        dist = TaskDistribution(8, n_classes=3, k_shot=5, q_query=10)
        task = dist.sample_task()
        loss, acc = maml.meta_loss(task)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        logger.info("meta_loss passed: loss=%.4f, acc=%.4f", loss, acc)


class TestMetaLearner(unittest.TestCase):
    def test_meta_train_step(self):
        model = SimpleModel(8, 5)
        maml = MAMLModel(model, inner_lr=0.01, inner_steps=2)
        dist = TaskDistribution(8, n_classes=3, k_shot=5, q_query=10)
        learner = MetaLearner(maml, meta_lr=0.001, task_distribution=dist)
        metrics = learner.meta_train_step(batch_size=2)
        self.assertIn("meta_loss", metrics)
        self.assertIn("meta_accuracy", metrics)
        logger.info("meta_train_step passed: %s", metrics)

    def test_train(self):
        model = SimpleModel(8, 5)
        maml = MAMLModel(model, inner_lr=0.01, inner_steps=2)
        dist = TaskDistribution(8, n_classes=3, k_shot=5, q_query=10)
        learner = MetaLearner(maml, meta_lr=0.001, task_distribution=dist)
        result = learner.train(n_iterations=5, batch_size=2)
        self.assertIn("final_loss", result)
        self.assertIn("steps", result)
        logger.info("train passed: %s", result)

    def test_evaluate_on_new_task(self):
        model = SimpleModel(8, 5)
        maml = MAMLModel(model, inner_lr=0.01, inner_steps=2)
        dist = TaskDistribution(8, n_classes=3, k_shot=5, q_query=10)
        learner = MetaLearner(maml, meta_lr=0.001, task_distribution=dist)
        learner.train(n_iterations=3, batch_size=2)
        eval_result = learner.evaluate_on_new_task(n_eval_tasks=3)
        self.assertIn("acc_before_adapt", eval_result)
        self.assertIn("acc_after_adapt", eval_result)
        self.assertIn("improvement", eval_result)
        logger.info("evaluate_on_new_task passed: %s", eval_result)


class TestFastAdapter(unittest.TestCase):
    def test_adapt(self):
        model = SimpleModel(8, 5)
        adapter = FastAdapter(model, lr=0.01, max_steps=3)
        support_x = np.random.randn(15, 8)
        support_y = np.array([0, 1, 2] * 5)
        adapted = adapter.adapt(support_x, support_y)
        self.assertIsNotNone(adapted)
        logger.info("FastAdapter adapt passed")

    def test_predict(self):
        model = SimpleModel(8, 5)
        adapter = FastAdapter(model)
        x = np.random.randn(5, 8)
        preds = adapter.predict(x)
        self.assertEqual(len(preds), 5)
        logger.info("FastAdapter predict passed")

    def test_get_stats(self):
        model = SimpleModel(8, 5)
        adapter = FastAdapter(model)
        stats = adapter.get_stats()
        self.assertEqual(stats["adapted"], False)
        logger.info("FastAdapter stats passed")


class TestMetaReplayBuffer(unittest.TestCase):
    def test_add_and_sample(self):
        buf = MetaReplayBuffer(capacity=10)
        dist = TaskDistribution(8, n_classes=3, k_shot=5, q_query=10)
        for _ in range(5):
            buf.add(dist.sample_task())
        self.assertEqual(len(buf), 5)
        sampled = buf.sample(3)
        self.assertEqual(len(sampled), 3)
        logger.info("MetaReplayBuffer add_and_sample passed")

    def test_capacity(self):
        buf = MetaReplayBuffer(capacity=3)
        dist = TaskDistribution(8, n_classes=3, k_shot=5, q_query=10)
        for _ in range(5):
            buf.add(dist.sample_task())
        self.assertEqual(len(buf), 3)
        logger.info("MetaReplayBuffer capacity passed")


class TestMetaLearningController(unittest.TestCase):
    def test_init(self):
        ctrl = MetaLearningController(8, 5, n_classes=3, k_shot=5)
        self.assertEqual(ctrl.input_dim, 8)
        self.assertEqual(ctrl.output_dim, 5)
        logger.info("MetaLearningController init passed")

    def test_meta_train(self):
        ctrl = MetaLearningController(8, 5, n_classes=3, k_shot=5)
        result = ctrl.meta_train(n_iterations=3, batch_size=2)
        self.assertIn("final_loss", result)
        logger.info("meta_train passed: %s", result)

    def test_adapt_to_task(self):
        ctrl = MetaLearningController(8, 5, n_classes=3, k_shot=5)
        support_x = np.random.randn(15, 8)
        support_y = np.array([0, 1, 2] * 5)
        adapted = ctrl.adapt_to_task(support_x, support_y)
        self.assertIsNotNone(adapted)
        logger.info("adapt_to_task passed")

    def test_evaluate_adaptation(self):
        ctrl = MetaLearningController(8, 5, n_classes=3, k_shot=5)
        ctrl.meta_train(n_iterations=3, batch_size=2)
        result = ctrl.evaluate_adaptation(n_tasks=3)
        self.assertIn("improvement", result)
        logger.info("evaluate_adaptation passed: %s", result)

    def test_get_stats(self):
        ctrl = MetaLearningController(8, 5, n_classes=3, k_shot=5)
        stats = ctrl.get_stats()
        self.assertIn("meta_steps", stats)
        logger.info("get_stats passed: %s", stats)

    def test_save_load(self):
        ctrl = MetaLearningController(8, 5, n_classes=3, k_shot=5)
        ctrl.meta_train(n_iterations=2, batch_size=2)
        data = ctrl.save()
        self.assertIn("model_params", data)

        ctrl2 = MetaLearningController(8, 5, n_classes=3, k_shot=5)
        ctrl2.load(data)
        self.assertEqual(ctrl2.meta_learner.step_count, ctrl.meta_learner.step_count)
        logger.info("save_load passed")


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
    unittest.main(verbosity=2)
