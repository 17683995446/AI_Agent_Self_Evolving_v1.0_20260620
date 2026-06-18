"""
阶段42测试：自监督学习与表征学习
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.self_supervised import (
    RepresentationEncoder, DataAugmentor, ContrastiveLearner,
    MaskedPredictor, RepresentationEvaluator, SelfSupervisedController
)


class TestRepresentationEncoder(unittest.TestCase):
    def test_init(self):
        enc = RepresentationEncoder(10, latent_dim=32)
        self.assertEqual(enc.input_dim, 10)
        self.assertEqual(enc.latent_dim, 32)
        logger.info("Encoder init passed")

    def test_encode(self):
        enc = RepresentationEncoder(10, latent_dim=32)
        x = np.random.randn(5, 10)
        z = enc.encode(x)
        self.assertEqual(z.shape, (5, 32))
        logger.info("encode passed: shape=%s", z.shape)

    def test_copy(self):
        enc = RepresentationEncoder(10, latent_dim=32)
        enc2 = enc.copy()
        for k in enc.params:
            np.testing.assert_array_equal(enc.params[k], enc2.params[k])
        logger.info("copy passed")


class TestDataAugmentor(unittest.TestCase):
    def test_augment(self):
        aug = DataAugmentor(noise_std=0.1, mask_ratio=0.15)
        x = np.random.randn(10)
        x_aug = aug.augment(x)
        self.assertEqual(x_aug.shape, x.shape)
        self.assertFalse(np.array_equal(x, x_aug))
        logger.info("augment passed")

    def test_augment_batch(self):
        aug = DataAugmentor()
        x = np.random.randn(8, 10)
        x_aug = aug.augment_batch(x)
        self.assertEqual(x_aug.shape, x.shape)
        logger.info("augment_batch passed")


class TestContrastiveLearner(unittest.TestCase):
    def test_compute_loss(self):
        enc = RepresentationEncoder(10, latent_dim=32)
        learner = ContrastiveLearner(enc, temperature=0.5)
        x = np.random.randn(16, 10)
        loss = learner.compute_loss(x)
        self.assertIsInstance(loss, float)
        logger.info("compute_loss passed: loss=%.4f", loss)

    def test_train_step(self):
        enc = RepresentationEncoder(10, latent_dim=16)
        learner = ContrastiveLearner(enc, temperature=0.5)
        x = np.random.randn(8, 10)
        metrics = learner.train_step(x, lr=0.001)
        self.assertIn("contrastive_loss", metrics)
        logger.info("train_step passed: %s", metrics)

    def test_get_embeddings(self):
        enc = RepresentationEncoder(10, latent_dim=32)
        learner = ContrastiveLearner(enc)
        x = np.random.randn(5, 10)
        emb = learner.get_embeddings(x)
        self.assertEqual(emb.shape, (5, 32))
        logger.info("get_embeddings passed")


class TestMaskedPredictor(unittest.TestCase):
    def test_compute_loss(self):
        pred = MaskedPredictor(10, mask_ratio=0.2)
        x = np.random.randn(8, 10)
        loss = pred.compute_loss(x)
        self.assertIsInstance(loss, float)
        logger.info("masked compute_loss passed: loss=%.4f", loss)

    def test_train_step(self):
        pred = MaskedPredictor(10, mask_ratio=0.2)
        x = np.random.randn(8, 10)
        metrics = pred.train_step(x, lr=0.001)
        self.assertIn("masked_loss", metrics)
        logger.info("masked train_step passed: %s", metrics)

    def test_reconstruct(self):
        pred = MaskedPredictor(10, mask_ratio=0.2)
        x = np.random.randn(5, 10)
        recon = pred.reconstruct(x)
        self.assertEqual(recon.shape, x.shape)
        logger.info("reconstruct passed")


class TestRepresentationEvaluator(unittest.TestCase):
    def test_linear_probe(self):
        ev = RepresentationEvaluator()
        embeddings = np.random.randn(50, 16)
        labels = np.random.randint(0, 3, 50)
        result = ev.evaluate_linear_probe(embeddings, labels)
        self.assertIn("linear_probe_accuracy", result)
        logger.info("linear_probe passed: acc=%.4f", result["linear_probe_accuracy"])

    def test_clustering(self):
        ev = RepresentationEvaluator()
        embeddings = np.random.randn(50, 16)
        labels = np.random.randint(0, 3, 50)
        result = ev.evaluate_clustering(embeddings, labels)
        self.assertIn("clustering_purity", result)
        logger.info("clustering passed: purity=%.4f", result["clustering_purity"])

    def test_uniformity(self):
        ev = RepresentationEvaluator()
        embeddings = np.random.randn(50, 16)
        u = ev.evaluate_uniformity(embeddings)
        self.assertIsInstance(u, float)
        logger.info("uniformity passed: %.4f", u)


class TestSelfSupervisedController(unittest.TestCase):
    def test_init_contrastive(self):
        ctrl = SelfSupervisedController(10, latent_dim=16, method="contrastive")
        self.assertEqual(ctrl.method, "contrastive")
        logger.info("init contrastive passed")

    def test_init_masked(self):
        ctrl = SelfSupervisedController(10, method="masked")
        self.assertEqual(ctrl.method, "masked")
        logger.info("init masked passed")

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            SelfSupervisedController(10, method="unknown")
        logger.info("unknown method exception passed")

    def test_train_contrastive(self):
        ctrl = SelfSupervisedController(10, latent_dim=16, method="contrastive")
        data = np.random.randn(50, 10)
        result = ctrl.train(data, n_steps=3, lr=0.001, batch_size=8)
        self.assertIn("final_loss", result)
        logger.info("train contrastive passed: %s", result)

    def test_train_masked(self):
        ctrl = SelfSupervisedController(10, method="masked")
        data = np.random.randn(50, 10)
        result = ctrl.train(data, n_steps=3, lr=0.001, batch_size=8)
        self.assertIn("final_loss", result)
        logger.info("train masked passed: %s", result)

    def test_get_embeddings(self):
        ctrl = SelfSupervisedController(10, latent_dim=16, method="contrastive")
        data = np.random.randn(5, 10)
        emb = ctrl.get_embeddings(data)
        self.assertEqual(emb.shape, (5, 16))
        logger.info("get_embeddings passed")

    def test_evaluate(self):
        ctrl = SelfSupervisedController(10, latent_dim=16, method="contrastive")
        data = np.random.randn(50, 10)
        labels = np.random.randint(0, 3, 50)
        result = ctrl.evaluate(data, labels)
        self.assertIn("linear_probe", result)
        self.assertIn("clustering", result)
        logger.info("evaluate passed")

    def test_get_stats(self):
        ctrl = SelfSupervisedController(10, method="contrastive")
        stats = ctrl.get_stats()
        self.assertEqual(stats["method"], "contrastive")
        logger.info("get_stats passed: %s", stats)

    def test_save_load(self):
        ctrl = SelfSupervisedController(10, latent_dim=16, method="contrastive")
        data = np.random.randn(20, 10)
        ctrl.train(data, n_steps=2, batch_size=8)
        saved = ctrl.save()
        self.assertIn("encoder_params", saved)

        ctrl2 = SelfSupervisedController(10, latent_dim=16, method="contrastive")
        ctrl2.load(saved)
        self.assertEqual(ctrl2.step_count, ctrl.step_count)
        logger.info("save_load passed")


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
    unittest.main(verbosity=2)
