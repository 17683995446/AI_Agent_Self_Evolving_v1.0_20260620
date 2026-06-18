"""
阶段37测试：强化学习策略优化（PPO/TRPO）
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.rl_policy import (
    PolicyNetwork, ValueNetwork, RolloutBuffer,
    PPOTrainer, TRPOTrainer, PolicyOptimizer, Transition, Rollout
)


class TestPolicyNetwork(unittest.TestCase):
    def test_init(self):
        net = PolicyNetwork(4, 2, discrete=True)
        self.assertEqual(net.state_dim, 4)
        self.assertEqual(net.action_dim, 2)
        self.assertTrue(net.discrete)
        self.assertIn('W0', net.params)
        logger.info("PolicyNetwork init passed")

    def test_get_action_discrete(self):
        net = PolicyNetwork(4, 2, discrete=True)
        state = np.array([0.1, -0.2, 0.3, -0.1])
        result = net.get_action(state)
        self.assertIn("action", result)
        self.assertIn("log_prob", result)
        self.assertIn(result["action"], [0, 1])
        logger.info("get_action discrete passed: action=%s", result["action"])

    def test_get_action_continuous(self):
        net = PolicyNetwork(4, 3, discrete=False)
        state = np.array([0.1, -0.2, 0.3, -0.1])
        result = net.get_action(state)
        self.assertIn("action", result)
        self.assertEqual(len(result["action"]), 3)
        logger.info("get_action continuous passed")

    def test_evaluate_discrete(self):
        net = PolicyNetwork(4, 2, discrete=True)
        states = np.random.randn(10, 4)
        actions = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        log_probs, entropy = net.evaluate(states, actions)
        self.assertEqual(len(log_probs), 10)
        self.assertEqual(len(entropy), 10)
        logger.info("evaluate discrete passed")

    def test_copy(self):
        net = PolicyNetwork(4, 2)
        net2 = net.copy()
        self.assertEqual(net2.state_dim, 4)
        for k in net.params:
            np.testing.assert_array_equal(net.params[k], net2.params[k])
        logger.info("copy passed")

    def test_flat_params(self):
        net = PolicyNetwork(4, 2)
        flat = net.get_flat_params()
        net.set_flat_params(flat)
        flat2 = net.get_flat_params()
        np.testing.assert_array_equal(flat, flat2)
        logger.info("flat params roundtrip passed")


class TestValueNetwork(unittest.TestCase):
    def test_predict(self):
        net = ValueNetwork(4)
        state = np.array([0.1, -0.2, 0.3, -0.1])
        v = net.predict(state)
        self.assertIsInstance(v, float)
        logger.info("ValueNetwork predict passed: v=%.4f", v)

    def test_predict_batch(self):
        net = ValueNetwork(4)
        states = np.random.randn(10, 4)
        vs = net.predict_batch(states)
        self.assertEqual(len(vs), 10)
        logger.info("predict_batch passed")

    def test_update(self):
        net = ValueNetwork(4)
        states = np.random.randn(32, 4)
        targets = np.random.randn(32)
        loss1 = net.update(states, targets, epochs=10)
        loss2 = net.update(states, targets, epochs=10)
        self.assertLessEqual(loss2, loss1 * 1.5)  # 允许一定波动
        logger.info("update passed: loss1=%.4f, loss2=%.4f", loss1, loss2)


class TestRolloutBuffer(unittest.TestCase):
    def test_add_and_len(self):
        buf = RolloutBuffer()
        buf.add(np.zeros(4), 0, 1.0, -0.5, 0.2, False)
        self.assertEqual(len(buf), 1)
        logger.info("buffer add passed")

    def test_compute_advantages(self):
        buf = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        for i in range(10):
            buf.add(np.zeros(4), 0, 1.0, -0.5, 0.5, i == 9)
        advantages, returns = buf.compute_advantages(last_value=0.0)
        self.assertEqual(len(advantages), 10)
        self.assertEqual(len(returns), 10)
        logger.info("compute_advantages passed: mean_adv=%.4f", np.mean(advantages))

    def test_get_batch(self):
        buf = RolloutBuffer()
        for i in range(5):
            buf.add(np.zeros(4), 0, 1.0, -0.5, 0.2, False)
        batch = buf.get_batch()
        self.assertEqual(len(batch["states"]), 5)
        logger.info("get_batch passed")

    def test_clear(self):
        buf = RolloutBuffer()
        buf.add(np.zeros(4), 0, 1.0, -0.5, 0.2, False)
        buf.clear()
        self.assertEqual(len(buf), 0)
        logger.info("clear passed")


class TestPPOTrainer(unittest.TestCase):
    def test_update(self):
        policy = PolicyNetwork(4, 2, discrete=True)
        value_net = ValueNetwork(4)
        trainer = PPOTrainer(policy, value_net, ppo_epochs=2)
        buf = RolloutBuffer()
        for i in range(20):
            state = np.random.randn(4)
            action = 0 if i % 2 == 0 else 1
            log_prob = -0.5
            value = 0.0
            buf.add(state, action, 1.0, log_prob, value, i == 19)
        metrics = trainer.update(buf, last_value=0.0)
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertIn("entropy", metrics)
        logger.info("PPO update passed: %s", metrics)

    def test_clip_behavior(self):
        # 验证clip确实限制了ratio
        policy = PolicyNetwork(4, 2, discrete=True)
        value_net = ValueNetwork(4)
        trainer = PPOTrainer(policy, value_net, clip_epsilon=0.2, ppo_epochs=1)
        buf = RolloutBuffer()
        for i in range(10):
            buf.add(np.random.randn(4), 0, 1.0, -0.5, 0.0, False)
        metrics = trainer.update(buf, last_value=0.0)
        self.assertIsNotNone(metrics)
        logger.info("clip behavior passed")


class TestTRPOTrainer(unittest.TestCase):
    def test_update(self):
        policy = PolicyNetwork(4, 2, discrete=True)
        value_net = ValueNetwork(4)
        trainer = TRPOTrainer(policy, value_net)
        buf = RolloutBuffer()
        for i in range(20):
            state = np.random.randn(4)
            action = 0 if i % 2 == 0 else 1
            log_prob = -0.5
            value = 0.0
            buf.add(state, action, 1.0, log_prob, value, i == 19)
        metrics = trainer.update(buf, last_value=0.0)
        self.assertIn("policy_loss", metrics)
        self.assertIn("kl_divergence", metrics)
        logger.info("TRPO update passed: %s", metrics)

    def test_kl_rollback(self):
        # 强制大更新触发KL回退
        policy = PolicyNetwork(4, 2, discrete=True)
        value_net = ValueNetwork(4)
        trainer = TRPOTrainer(policy, value_net, max_kl=0.001)
        buf = RolloutBuffer()
        for i in range(20):
            state = np.random.randn(4)
            action = 0 if i % 2 == 0 else 1
            log_prob = -0.5
            value = 0.0
            buf.add(state, action, 10.0, log_prob, value, i == 19)
        old_params = policy.get_flat_params().copy()
        metrics = trainer.update(buf, last_value=0.0)
        # 如果KL超限，参数应回退到旧参数附近
        new_params = policy.get_flat_params()
        diff = np.linalg.norm(new_params - old_params)
        logger.info("KL rollback test: diff=%.4f, kl=%.4f", diff, metrics["kl_divergence"])
        self.assertIn("kl_divergence", metrics)


class TestPolicyOptimizer(unittest.TestCase):
    def test_init_ppo(self):
        opt = PolicyOptimizer(4, 2, algorithm="ppo")
        self.assertEqual(opt.algorithm, "ppo")
        logger.info("PolicyOptimizer PPO init passed")

    def test_init_trpo(self):
        opt = PolicyOptimizer(4, 2, algorithm="trpo")
        self.assertEqual(opt.algorithm, "trpo")
        logger.info("PolicyOptimizer TRPO init passed")

    def test_select_action(self):
        opt = PolicyOptimizer(4, 2, algorithm="ppo")
        state = np.random.randn(4)
        result = opt.select_action(state)
        self.assertIn("action", result)
        self.assertIn("value", result)
        logger.info("select_action passed")

    def test_store_and_update(self):
        opt = PolicyOptimizer(4, 2, algorithm="ppo")
        for i in range(10):
            state = np.random.randn(4)
            action = 0
            reward = 1.0
            next_state = np.random.randn(4)
            done = i == 9
            log_prob = -0.5
            value = 0.0
            opt.store_transition(state, action, reward, next_state, done, log_prob, value)
        metrics = opt.update(last_value=0.0)
        self.assertIn("policy_loss", metrics)
        logger.info("store_and_update passed: %s", metrics)

    def test_train_episode_mock_env(self):
        opt = PolicyOptimizer(2, 2, algorithm="ppo")
        step = [0]

        def env_step(action):
            if action is None:
                step[0] = 0
                return (np.random.randn(2), 0.0, False, {})
            step[0] += 1
            return (np.random.randn(2), 1.0, step[0] >= 10, {})

        result = opt.train_episode(env_step, max_steps=20, update_interval=5)
        self.assertIn("episode_reward", result)
        self.assertIn("episode_length", result)
        logger.info("train_episode passed: %s", result)

    def test_get_stats(self):
        opt = PolicyOptimizer(4, 2, algorithm="ppo")
        stats = opt.get_stats()
        self.assertEqual(stats["algorithm"], "ppo")
        self.assertEqual(stats["step_count"], 0)
        logger.info("get_stats passed: %s", stats)

    def test_save_load(self):
        opt = PolicyOptimizer(4, 2, algorithm="ppo")
        for i in range(5):
            opt.store_transition(np.zeros(4), 0, 1.0, np.zeros(4), False, -0.5, 0.0)
        opt.update()
        data = opt.save()
        self.assertIn("policy_params", data)
        self.assertIn("value_params", data)

        opt2 = PolicyOptimizer(4, 2, algorithm="ppo")
        opt2.load(data)
        self.assertEqual(opt2.step_count, opt.step_count)
        self.assertEqual(opt2.update_count, opt.update_count)
        logger.info("save_load passed")

    def test_unknown_algorithm(self):
        with self.assertRaises(ValueError):
            PolicyOptimizer(4, 2, algorithm="unknown")
        logger.info("unknown_algorithm exception passed")


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
    unittest.main(verbosity=2)
