"""
阶段38测试：神经网络架构搜索(NAS)
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.nas_engine import (
    LayerSpec, ArchitectureSpec, SearchSpace,
    WeightSharingNAS, ArchitectureEvaluator,
    EvolutionarySearcher, NASController
)


class TestArchitectureSpec(unittest.TestCase):
    def test_add_layer(self):
        arch = ArchitectureSpec(input_dim=4, output_dim=2)
        lid = arch.add_layer("fc", params={"dim": 32})
        self.assertEqual(len(arch.layers), 1)
        self.assertEqual(arch.layers[0].layer_type, "fc")
        logger.info("add_layer passed")

    def test_add_connection(self):
        arch = ArchitectureSpec(input_dim=4, output_dim=2)
        arch.add_layer("input", "input", dim=4)
        arch.add_layer("fc", "fc1", dim=32)
        arch.add_connection("input", "fc1")
        self.assertEqual(len(arch.connections), 1)
        logger.info("add_connection passed")

    def test_hash(self):
        arch = ArchitectureSpec(input_dim=4, output_dim=2)
        arch.add_layer("fc", "fc1", dim=32)
        h = arch.hash()
        self.assertEqual(len(h), 12)
        logger.info("hash passed: %s", h)

    def test_copy(self):
        arch = ArchitectureSpec(input_dim=4, output_dim=2)
        arch.add_layer("fc", "fc1", dim=32)
        arch2 = arch.copy()
        self.assertEqual(len(arch2.layers), 1)
        arch2.layers[0].params["dim"] = 64
        self.assertEqual(arch.layers[0].params["dim"], 32)
        logger.info("copy passed")

    def test_to_dict(self):
        arch = ArchitectureSpec(input_dim=4, output_dim=2)
        arch.add_layer("fc", "fc1", dim=32)
        d = arch.to_dict()
        self.assertIn("layers", d)
        self.assertIn("connections", d)
        logger.info("to_dict passed")


class TestSearchSpace(unittest.TestCase):
    def test_sample_random(self):
        ss = SearchSpace(4, 2, max_layers=3)
        arch = ss.sample_random()
        self.assertGreater(len(arch.layers), 0)
        self.assertEqual(arch.input_dim, 4)
        self.assertEqual(arch.output_dim, 2)
        logger.info("sample_random passed: layers=%d", len(arch.layers))

    def test_mutate(self):
        ss = SearchSpace(4, 2, max_layers=5)
        arch = ss.sample_random()
        mutated = ss.mutate(arch)
        self.assertIsNotNone(mutated)
        logger.info("mutate passed")

    def test_crossover(self):
        ss = SearchSpace(4, 2)
        a1 = ss.sample_random()
        a2 = ss.sample_random()
        child = ss.crossover(a1, a2)
        self.assertGreater(len(child.layers), 0)
        logger.info("crossover passed")

    def test_allow_skip_false(self):
        ss = SearchSpace(4, 2, allow_skip=False, max_layers=3)
        arch = ss.sample_random()
        self.assertEqual(arch.input_dim, 4)
        logger.info("allow_skip_false passed")


class TestWeightSharingNAS(unittest.TestCase):
    def test_init(self):
        ss = SearchSpace(4, 2)
        nas = WeightSharingNAS(ss, 4, 2)
        self.assertEqual(nas.state_dim, 4)
        logger.info("WeightSharingNAS init passed")

    def test_get_weight(self):
        ss = SearchSpace(4, 2)
        nas = WeightSharingNAS(ss, 4, 2)
        w = nas.get_weight("fc", 4, 32)
        self.assertEqual(w.shape, (4, 32))
        logger.info("get_weight passed")

    def test_evaluate_and_cache(self):
        ss = SearchSpace(4, 2)
        nas = WeightSharingNAS(ss, 4, 2)
        arch = ArchitectureSpec(input_dim=4, output_dim=2)
        arch.add_layer("fc", "fc1", dim=32)

        def eval_fn(a):
            return 1.0

        score1 = nas.evaluate_architecture(arch, eval_fn)
        score2 = nas.evaluate_architecture(arch, eval_fn)
        self.assertEqual(score1, score2)
        self.assertEqual(len(nas.architecture_cache), 1)
        logger.info("evaluate_and_cache passed")

    def test_top_architectures(self):
        ss = SearchSpace(4, 2)
        nas = WeightSharingNAS(ss, 4, 2)
        for i in range(5):
            arch = ArchitectureSpec(input_dim=4, output_dim=2)
            arch.add_layer("fc", f"fc{i}", dim=32)
            nas.evaluate_architecture(arch, lambda a: float(i))
        top = nas.get_top_architectures(k=3)
        self.assertEqual(len(top), 3)
        self.assertEqual(top[0][1], 4.0)
        logger.info("top_architectures passed")


class TestArchitectureEvaluator(unittest.TestCase):
    def test_default_evaluate(self):
        ev = ArchitectureEvaluator()
        arch = ArchitectureSpec(input_dim=4, output_dim=2)
        arch.add_layer("fc", "fc1", dim=48)
        score = ev.default_evaluate(arch)
        self.assertIsInstance(score, float)
        logger.info("default_evaluate passed: score=%.4f", score)

    def test_evaluate_with_proxy(self):
        ev = ArchitectureEvaluator(proxy_task_fn=lambda a: 5.0)
        arch = ArchitectureSpec()
        score = ev.evaluate(arch)
        self.assertEqual(score, 5.0)
        logger.info("evaluate_with_proxy passed")

    def test_evaluate_history(self):
        ev = ArchitectureEvaluator()
        arch = ArchitectureSpec()
        ev.evaluate(arch)
        self.assertEqual(len(ev.eval_history), 1)
        logger.info("evaluate_history passed")


class TestEvolutionarySearcher(unittest.TestCase):
    def test_initialize_population(self):
        ss = SearchSpace(4, 2)
        searcher = EvolutionarySearcher(ss, population_size=5)
        ev = ArchitectureEvaluator()
        searcher.initialize_population(ev)
        self.assertEqual(len(searcher.population), 5)
        logger.info("initialize_population passed")

    def test_evolve(self):
        ss = SearchSpace(4, 2)
        searcher = EvolutionarySearcher(ss, population_size=5)
        ev = ArchitectureEvaluator()
        searcher.initialize_population(ev)
        best = searcher.evolve(ev)
        self.assertIsNotNone(best)
        self.assertEqual(searcher.generation, 1)
        logger.info("evolve passed")

    def test_get_best(self):
        ss = SearchSpace(4, 2)
        searcher = EvolutionarySearcher(ss, population_size=5)
        ev = ArchitectureEvaluator()
        searcher.initialize_population(ev)
        best_arch, best_score = searcher.get_best()
        self.assertIsNotNone(best_arch)
        logger.info("get_best passed: score=%.4f", best_score)


class TestNASController(unittest.TestCase):
    def test_init(self):
        ctrl = NASController(4, 2, generations=2, population_size=4)
        self.assertEqual(ctrl.state_dim, 4)
        self.assertEqual(ctrl.action_dim, 2)
        logger.info("NASController init passed")

    def test_search(self):
        ctrl = NASController(4, 2, generations=2, population_size=4, max_layers=3)
        result = ctrl.search()
        self.assertIn("best_architecture", result)
        self.assertIn("best_score", result)
        self.assertIn("history", result)
        self.assertEqual(len(result["history"]), 2)
        logger.info("search passed: best_score=%.4f", result["best_score"])

    def test_evaluate_architecture(self):
        ctrl = NASController(4, 2, generations=1, population_size=3, max_layers=3)
        arch = ArchitectureSpec(input_dim=4, output_dim=2)
        arch.add_layer("fc", "fc1", dim=32)
        score = ctrl.evaluate_architecture(arch)
        self.assertIsInstance(score, float)
        logger.info("evaluate_architecture passed")

    def test_get_stats(self):
        ctrl = NASController(4, 2, generations=1, population_size=3, max_layers=3)
        ctrl.search()
        stats = ctrl.get_stats()
        self.assertIn("best_score", stats)
        self.assertIn("evaluated_architectures", stats)
        logger.info("get_stats passed: %s", stats)

    def test_save_load(self):
        ctrl = NASController(4, 2, generations=1, population_size=3, max_layers=3)
        ctrl.search()
        data = ctrl.save()
        self.assertIn("best_architecture", data)

        ctrl2 = NASController(4, 2, generations=1, population_size=3, max_layers=3)
        ctrl2.load(data)
        self.assertEqual(ctrl2.best_score, ctrl.best_score)
        if ctrl.best_architecture:
            self.assertIsNotNone(ctrl2.best_architecture)
        logger.info("save_load passed")


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
    unittest.main(verbosity=2)
