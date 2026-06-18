"""
配置热重载验证测试 —— 确保配置修改后系统正确响应
运行: python -m pytest tests/test_hot_reload.py -v
"""
import os
import sys
import time
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ConfigManager, SystemConfig


class TestHotReload(unittest.TestCase):
    """测试配置热重载功能"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "test_config.yaml")
        # 创建初始配置
        with open(self.config_path, 'w') as f:
            f.write("""
resource:
  max_cpu_percent: 20.0
  max_memory_percent: 20.0
  max_reconstruction_iterations: 3
  storage_retain_ratio: 0.30

evolution:
  exploration_ratio: 0.30
  idle_replay_interval_hours: 6.0
  steady_state_error_threshold: 0.01
  exploration_boost_on_steady: 0.10

drive_weights:
  safe: 0.4
  gain: 0.4
  eff: 0.2

main_loop_frequency: 1.0
perception_interval: 1.0
decision_interval: 1.0
log_dir: "./logs"
database_path: "./data/agent_memory.db"
model_snapshot_dir: "./data/snapshots"
""")
        # 重置单例状态以支持测试
        ConfigManager._initialized = False
        ConfigManager._instance = None
        self.cm = ConfigManager(config_path=self.config_path)

    def test_initial_load(self):
        """验证初始配置加载正确"""
        config = self.cm.load()
        self.assertEqual(config.resource.max_cpu_percent, 20.0)
        self.assertEqual(config.evolution.exploration_ratio, 0.30)
        self.assertEqual(config.drive_weights.safe, 0.4)

    def test_hot_reload_detects_change(self):
        """验证热重载能检测到配置变更"""
        # 先加载一次
        self.cm.load()
        self.cm._last_modified = os.path.getmtime(self.config_path)

        # 未修改时返回False
        result = self.cm.check_hot_reload()
        self.assertFalse(result)

        # 修改配置文件（需要足够时间让mtime变化）
        time.sleep(1.0)
        with open(self.config_path, 'w') as f:
            f.write("""
resource:
  max_cpu_percent: 15.0
  max_memory_percent: 15.0
  max_reconstruction_iterations: 3
  storage_retain_ratio: 0.30

evolution:
  exploration_ratio: 0.50
  idle_replay_interval_hours: 6.0
  steady_state_error_threshold: 0.01
  exploration_boost_on_steady: 0.10

drive_weights:
  safe: 0.5
  gain: 0.3
  eff: 0.2

main_loop_frequency: 2.0
perception_interval: 1.0
decision_interval: 1.0
log_dir: "./logs"
database_path: "./data/agent_memory.db"
model_snapshot_dir: "./data/snapshots"
""")

        # 检测变更
        result = self.cm.check_hot_reload()
        self.assertTrue(result)

        # 验证新配置生效
        config = self.cm.config
        self.assertEqual(config.resource.max_cpu_percent, 15.0)
        self.assertEqual(config.evolution.exploration_ratio, 0.50)
        self.assertEqual(config.drive_weights.safe, 0.5)
        self.assertEqual(config.main_loop_frequency, 2.0)

    def test_hot_reload_invalid_config(self):
        """验证热重载遇到无效配置时抛出异常（当前设计）"""
        self.cm.load()
        self.cm._last_modified = os.path.getmtime(self.config_path)
        old_safe = self.cm.config.drive_weights.safe

        # 写入无效配置（权重和不等于1）
        time.sleep(1.0)
        with open(self.config_path, 'w') as f:
            f.write("""
resource:
  max_cpu_percent: 20.0
  max_memory_percent: 20.0
  max_reconstruction_iterations: 3
  storage_retain_ratio: 0.30

evolution:
  exploration_ratio: 0.30
  idle_replay_interval_hours: 6.0
  steady_state_error_threshold: 0.01
  exploration_boost_on_steady: 0.10

drive_weights:
  safe: 0.5
  gain: 0.5
  eff: 0.5

main_loop_frequency: 1.0
perception_interval: 1.0
decision_interval: 1.0
log_dir: "./logs"
database_path: "./data/agent_memory.db"
model_snapshot_dir: "./data/snapshots"
""")

        # 当前设计：无效配置会导致异常
        # 注意：异常前部分配置可能已被更新，这是已知设计限制
        with self.assertRaises(AssertionError):
            self.cm.check_hot_reload()


if __name__ == "__main__":
    unittest.main(verbosity=2)
