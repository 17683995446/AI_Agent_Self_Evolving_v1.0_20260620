"""
阶段43测试：多模态融合引擎
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.multimodal_fusion import (
    ModalityConfig, ModalityEncoder, CrossModalAligner,
    AttentionFusion, MultimodalPredictor, MultimodalController
)


class TestModalityEncoder(unittest.TestCase):
    def test_init(self):
        cfg = ModalityConfig("numeric", 10, 32)
        enc = ModalityEncoder(cfg)
        self.assertEqual(enc.config.name, "numeric")
        logger.info("init passed")

    def test_encode(self):
        cfg = ModalityConfig("numeric", 10, 32)
        enc = ModalityEncoder(cfg)
        x = np.random.randn(5, 10)
        z = enc.encode(x)
        self.assertEqual(z.shape, (5, 32))
        logger.info("encode passed")

    def test_get_set_params(self):
        cfg = ModalityConfig("numeric", 10, 32)
        enc = ModalityEncoder(cfg)
        params = enc.get_params()
        enc.set_params(params)
        for k in params:
            np.testing.assert_array_equal(params[k], enc.params[k])
        logger.info("get_set_params passed")


class TestCrossModalAligner(unittest.TestCase):
    def test_compute_alignment_loss(self):
        aligner = CrossModalAligner(latent_dim=16)
        embs = {
            "mod1": np.random.randn(8, 16),
            "mod2": np.random.randn(8, 16),
        }
        loss = aligner.compute_alignment_loss(embs)
        self.assertIsInstance(loss, float)
        logger.info("alignment_loss passed: %.4f", loss)

    def test_align_same_dim(self):
        aligner = CrossModalAligner(latent_dim=16)
        emb = np.random.randn(5, 16)
        aligned = aligner.align(emb, 16)
        np.testing.assert_array_equal(emb, aligned)
        logger.info("align_same_dim passed")

    def test_align_different_dim(self):
        aligner = CrossModalAligner(latent_dim=16)
        emb = np.random.randn(5, 32)
        aligned = aligner.align(emb, 16)
        self.assertEqual(aligned.shape, (5, 16))
        logger.info("align_different_dim passed")


class TestAttentionFusion(unittest.TestCase):
    def test_compute_attention(self):
        fusion = AttentionFusion(n_modalities=3, latent_dim=16)
        embs = [np.random.randn(8, 16) for _ in range(3)]
        fused, weights = fusion.compute_attention(embs)
        self.assertEqual(fused.shape, (8, 16))
        self.assertEqual(len(weights), 3)
        logger.info("compute_attention passed")

    def test_fuse(self):
        fusion = AttentionFusion(n_modalities=2, latent_dim=16)
        embs = [np.random.randn(8, 16) for _ in range(2)]
        fused = fusion.fuse(embs)
        self.assertEqual(fused.shape, (8, 16))
        logger.info("fuse passed")

    def test_update_weights(self):
        fusion = AttentionFusion(n_modalities=3, latent_dim=16)
        losses = [0.1, 0.5, 0.3]
        fusion.update_modality_weights(losses)
        self.assertEqual(len(fusion.modality_weights), 3)
        logger.info("update_weights passed: %s", fusion.modality_weights)


class TestMultimodalPredictor(unittest.TestCase):
    def test_predict(self):
        pred = MultimodalPredictor(latent_dim=16, output_dim=2)
        x = np.random.randn(5, 16)
        out = pred.predict(x)
        self.assertEqual(out.shape, (5, 2))
        logger.info("predict passed")

    def test_compute_loss(self):
        pred = MultimodalPredictor(latent_dim=16, output_dim=1)
        x = np.random.randn(5, 16)
        y = np.random.randn(5, 1)
        loss = pred.compute_loss(x, y)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
        logger.info("compute_loss passed: %.4f", loss)


class TestMultimodalController(unittest.TestCase):
    def test_init(self):
        configs = [
            ModalityConfig("mod1", 10, 16),
            ModalityConfig("mod2", 8, 16),
        ]
        ctrl = MultimodalController(configs, output_dim=1, latent_dim=16)
        self.assertEqual(len(ctrl.encoders), 2)
        logger.info("init passed")

    def test_encode_modalities(self):
        configs = [ModalityConfig("mod1", 10, 16), ModalityConfig("mod2", 8, 16)]
        ctrl = MultimodalController(configs, latent_dim=16)
        data = {"mod1": np.random.randn(5, 10), "mod2": np.random.randn(5, 8)}
        embs = ctrl.encode_modalities(data)
        self.assertEqual(len(embs), 2)
        self.assertEqual(embs["mod1"].shape, (5, 16))
        logger.info("encode_modalities passed")

    def test_fuse(self):
        configs = [ModalityConfig("mod1", 10, 16), ModalityConfig("mod2", 8, 16)]
        ctrl = MultimodalController(configs, latent_dim=16)
        data = {"mod1": np.random.randn(5, 10), "mod2": np.random.randn(5, 8)}
        fused = ctrl.fuse(data)
        self.assertEqual(fused.shape[1], 16)
        logger.info("fuse passed: shape=%s", fused.shape)

    def test_predict(self):
        configs = [ModalityConfig("mod1", 10, 16), ModalityConfig("mod2", 8, 16)]
        ctrl = MultimodalController(configs, output_dim=2, latent_dim=16)
        data = {"mod1": np.random.randn(5, 10), "mod2": np.random.randn(5, 8)}
        out = ctrl.predict(data)
        self.assertEqual(out.shape, (5, 2))
        logger.info("predict passed")

    def test_train_step(self):
        configs = [ModalityConfig("mod1", 10, 16), ModalityConfig("mod2", 8, 16)]
        ctrl = MultimodalController(configs, output_dim=1, latent_dim=16)
        data = {"mod1": np.random.randn(5, 10), "mod2": np.random.randn(5, 8)}
        targets = np.random.randn(5, 1)
        metrics = ctrl.train_step(data, targets, lr=0.001)
        self.assertIn("fusion_loss", metrics)
        logger.info("train_step passed: %s", metrics)

    def test_get_stats(self):
        configs = [ModalityConfig("mod1", 10, 16)]
        ctrl = MultimodalController(configs, latent_dim=16)
        stats = ctrl.get_stats()
        self.assertEqual(stats["n_modalities"], 1)
        logger.info("get_stats passed")

    def test_save_load(self):
        configs = [ModalityConfig("mod1", 10, 16), ModalityConfig("mod2", 8, 16)]
        ctrl = MultimodalController(configs, latent_dim=16)
        ctrl.step_count = 5
        data = ctrl.save()
        self.assertIn("modality_configs", data)

        ctrl2 = MultimodalController(configs, latent_dim=16)
        ctrl2.load(data)
        self.assertEqual(ctrl2.step_count, 5)
        logger.info("save_load passed")


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
    unittest.main(verbosity=2)
