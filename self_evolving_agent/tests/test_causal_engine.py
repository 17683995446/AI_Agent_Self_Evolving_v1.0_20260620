"""
阶段40测试：因果推断引擎
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.causal_engine import (
    CausalVariable, CausalGraph, CausalDiscovery,
    BackdoorCriterion, DoCalculus, CausalInferenceController
)


class TestCausalGraph(unittest.TestCase):
    def test_add_variable(self):
        g = CausalGraph()
        g.add_variable("X")
        self.assertIn("X", g.nodes)
        logger.info("add_variable passed")

    def test_add_edge(self):
        g = CausalGraph()
        g.add_edge("X", "Y")
        self.assertIn("Y", g.edges["X"])
        logger.info("add_edge passed")

    def test_get_parents(self):
        g = CausalGraph()
        g.add_edge("X", "Y")
        parents = g.get_parents("Y")
        self.assertEqual(parents, {"X"})
        logger.info("get_parents passed")

    def test_get_children(self):
        g = CausalGraph()
        g.add_edge("X", "Y")
        children = g.get_children("X")
        self.assertEqual(children, ["Y"])
        logger.info("get_children passed")

    def test_get_ancestors(self):
        g = CausalGraph()
        g.add_edge("X", "Y")
        g.add_edge("Y", "Z")
        ancestors = g.get_ancestors("Z")
        self.assertEqual(ancestors, {"X", "Y"})
        logger.info("get_ancestors passed")

    def test_get_descendants(self):
        g = CausalGraph()
        g.add_edge("X", "Y")
        g.add_edge("Y", "Z")
        descendants = g.get_descendants("X")
        self.assertEqual(descendants, {"Y", "Z"})
        logger.info("get_descendants passed")

    def test_is_dag_true(self):
        g = CausalGraph()
        g.add_edge("X", "Y")
        g.add_edge("Y", "Z")
        self.assertTrue(g.is_dag())
        logger.info("is_dag_true passed")

    def test_is_dag_false(self):
        g = CausalGraph()
        g.add_edge("X", "Y")
        g.add_edge("Y", "Z")
        g.add_edge("Z", "X")
        self.assertFalse(g.is_dag())
        logger.info("is_dag_false passed")

    def test_get_backdoor_paths(self):
        g = CausalGraph()
        g.add_edge("Z", "X")
        g.add_edge("Z", "Y")
        g.add_edge("X", "Y")
        paths = g.get_backdoor_paths("X", "Y")
        self.assertGreater(len(paths), 0)
        logger.info("get_backdoor_paths passed")

    def test_to_dict(self):
        g = CausalGraph()
        g.add_edge("X", "Y")
        d = g.to_dict()
        self.assertIn("nodes", d)
        self.assertIn("edges", d)
        logger.info("to_dict passed")


class TestCausalDiscovery(unittest.TestCase):
    def test_partial_correlation(self):
        cd = CausalDiscovery()
        x = np.random.randn(100)
        y = x * 0.5 + np.random.randn(100) * 0.1
        pc = cd.partial_correlation(x, y)
        self.assertGreater(abs(pc), 0.5)
        logger.info("partial_correlation passed: %.4f", pc)

    def test_partial_correlation_with_cond(self):
        cd = CausalDiscovery()
        z = np.random.randn(100)
        x = z * 0.8 + np.random.randn(100) * 0.1
        y = z * 0.8 + np.random.randn(100) * 0.1
        pc = cd.partial_correlation(x, y, z)
        self.assertLess(abs(pc), 0.3)
        logger.info("partial_correlation_with_cond passed: %.4f", pc)

    def test_discover(self):
        cd = CausalDiscovery()
        n = 200
        z = np.random.randn(n)
        x = z * 0.8 + np.random.randn(n) * 0.1
        y = x * 0.5 + z * 0.3 + np.random.randn(n) * 0.1
        data = {"Z": z, "X": x, "Y": y}
        graph = cd.discover(data, ["Z", "X", "Y"])
        self.assertGreater(len(graph.nodes), 0)
        self.assertTrue(graph.is_dag())
        logger.info("discover passed: nodes=%d", len(graph.nodes))


class TestBackdoorCriterion(unittest.TestCase):
    def test_find_valid_adjustment_sets(self):
        g = CausalGraph()
        g.add_edge("Z", "X")
        g.add_edge("Z", "Y")
        g.add_edge("X", "Y")
        bd = BackdoorCriterion(g)
        sets = bd.find_valid_adjustment_sets("X", "Y")
        self.assertGreater(len(sets), 0)
        logger.info("find_valid_adjustment_sets passed: %s", sets)

    def test_estimate_ate_no_confounder(self):
        g = CausalGraph()
        g.add_edge("X", "Y")
        bd = BackdoorCriterion(g)
        n = 200
        x = np.random.binomial(1, 0.5, n)
        y = x * 2.0 + np.random.randn(n) * 0.5
        data = {"X": x, "Y": y}
        ate = bd.estimate_ate(data, "X", "Y")
        self.assertGreater(ate, 1.0)
        self.assertLess(ate, 3.0)
        logger.info("estimate_ate_no_confounder passed: %.4f", ate)

    def test_estimate_ate_with_confounder(self):
        g = CausalGraph()
        g.add_edge("Z", "X")
        g.add_edge("Z", "Y")
        g.add_edge("X", "Y")
        bd = BackdoorCriterion(g)
        n = 500
        z = np.random.randn(n)
        x = (z + np.random.randn(n) * 0.5) > 0
        y = x.astype(float) * 1.5 + z * 2.0 + np.random.randn(n) * 0.5
        data = {"Z": z, "X": x.astype(float), "Y": y}
        ate = bd.estimate_ate(data, "X", "Y", adjustment_set={"Z"})
        self.assertGreater(ate, 0.5)
        self.assertLess(ate, 2.5)
        logger.info("estimate_ate_with_confounder passed: %.4f", ate)


class TestDoCalculus(unittest.TestCase):
    def test_identify(self):
        g = CausalGraph()
        g.add_edge("Z", "X")
        g.add_edge("Z", "Y")
        g.add_edge("X", "Y")
        dc = DoCalculus(g)
        method = dc.identify("X", "Y")
        self.assertEqual(method, "backdoor")
        logger.info("identify passed: %s", method)

    def test_estimate_intervention(self):
        g = CausalGraph()
        g.add_edge("X", "Y")
        dc = DoCalculus(g)
        n = 200
        x = np.random.binomial(1, 0.5, n)
        y = x * 2.0 + np.random.randn(n) * 0.5
        data = {"X": x, "Y": y}
        effect = dc.estimate_intervention(data, "X", "Y", treatment_value=1, control_value=0)
        self.assertIsInstance(effect, float)
        logger.info("estimate_intervention passed: %.4f", effect)


class TestCausalInferenceController(unittest.TestCase):
    def test_discover_graph(self):
        ctrl = CausalInferenceController()
        n = 200
        z = np.random.randn(n)
        x = z * 0.8 + np.random.randn(n) * 0.1
        y = x * 0.5 + z * 0.3 + np.random.randn(n) * 0.1
        data = {"Z": z, "X": x, "Y": y}
        graph = ctrl.discover_graph(data, ["Z", "X", "Y"])
        self.assertIsNotNone(graph)
        logger.info("discover_graph passed")

    def test_set_graph(self):
        ctrl = CausalInferenceController()
        g = CausalGraph()
        g.add_edge("X", "Y")
        ctrl.set_graph(g)
        self.assertEqual(ctrl.graph, g)
        logger.info("set_graph passed")

    def test_estimate_causal_effect(self):
        ctrl = CausalInferenceController()
        g = CausalGraph()
        g.add_edge("X", "Y")
        ctrl.set_graph(g)
        n = 200
        x = np.random.binomial(1, 0.5, n)
        y = x * 2.0 + np.random.randn(n) * 0.5
        data = {"X": x, "Y": y}
        result = ctrl.estimate_causal_effect(data, "X", "Y")
        self.assertTrue(result["identifiable"])
        self.assertIn("ate", result)
        logger.info("estimate_causal_effect passed: %s", result)

    def test_get_mediators(self):
        ctrl = CausalInferenceController()
        g = CausalGraph()
        g.add_edge("X", "M")
        g.add_edge("M", "Y")
        g.add_edge("X", "Y")
        ctrl.set_graph(g)
        mediators = ctrl.get_mediators("X", "Y")
        self.assertIn("M", mediators)
        logger.info("get_mediators passed: %s", mediators)

    def test_get_confounders(self):
        ctrl = CausalInferenceController()
        g = CausalGraph()
        g.add_edge("Z", "X")
        g.add_edge("Z", "Y")
        g.add_edge("X", "Y")
        ctrl.set_graph(g)
        confounders = ctrl.get_confounders("X", "Y")
        self.assertIn("Z", confounders)
        logger.info("get_confounders passed: %s", confounders)

    def test_get_stats(self):
        ctrl = CausalInferenceController()
        g = CausalGraph()
        g.add_edge("X", "Y")
        ctrl.set_graph(g)
        stats = ctrl.get_stats()
        self.assertTrue(stats["is_dag"])
        logger.info("get_stats passed: %s", stats)

    def test_save_load(self):
        ctrl = CausalInferenceController()
        g = CausalGraph()
        g.add_edge("X", "Y")
        ctrl.set_graph(g)
        data = ctrl.save()
        self.assertIn("graph", data)

        ctrl2 = CausalInferenceController()
        ctrl2.load(data)
        self.assertIsNotNone(ctrl2.graph)
        logger.info("save_load passed")


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
    unittest.main(verbosity=2)
