"""
U5：配置易用性测试

验证目标：
1. config.py 中所有 assert 已替换为 ConfigurationException
2. 配置项非法时抛出 ConfigurationException（带上下文）
3. YAML 语法错误时拒绝启动（不静默降级到默认值）
4. 环境变量覆盖配置项（优先级：环境变量 > 配置文件 > 默认值）
5. 环境变量类型转换失败时抛出 ConfigurationException
6. 无回归

第一性原理：
- 配置是契约：错误配置必须显式失败，不能静默降级
- assert 可被 python -O 禁用，不适合做配置校验
- 环境变量是部署时的契约，应优先于文件配置
"""
import os
import sys
import unittest
import tempfile
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent


class TestU5NoAssertInConfig(unittest.TestCase):
    """U5：config.py 中不应有 assert 语句"""

    def test_no_assert_in_config_py(self):
        import ast
        config_path = AGENT_ROOT / "core" / "config.py"
        content = config_path.read_text(encoding="utf-8")
        # 使用 AST 精确检测 assert 语句（排除注释和字符串中的 "assert" 文本）
        tree = ast.parse(content, filename=str(config_path))
        assert_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
        if assert_nodes:
            lines = [str(getattr(n, "lineno", "?")) for n in assert_nodes]
            self.fail(
                f"config.py 仍包含 {len(assert_nodes)} 处 assert 语句（行: {', '.join(lines)}），"
                f"应改为 ConfigurationException"
            )


class TestU5ConfigurationExceptionOnInvalidValues(unittest.TestCase):
    """U5：非法配置值应抛出 ConfigurationException"""

    def setUp(self):
        from core.config import SystemConfig
        from core.exceptions import ConfigurationException
        self.SystemConfig = SystemConfig
        self.ConfigurationException = ConfigurationException

    def test_resource_cpu_out_of_range_raises(self):
        cfg = self.SystemConfig()
        cfg.resource.max_cpu_percent = 150.0
        with self.assertRaises(self.ConfigurationException) as ctx:
            cfg.resource.validate()
        self.assertIn("CPU", str(ctx.exception))
        self.assertIn("150", str(ctx.exception))

    def test_resource_cpu_zero_raises(self):
        cfg = self.SystemConfig()
        cfg.resource.max_cpu_percent = 0
        with self.assertRaises(self.ConfigurationException):
            cfg.resource.validate()

    def test_resource_memory_out_of_range_raises(self):
        cfg = self.SystemConfig()
        cfg.resource.max_memory_percent = -5.0
        with self.assertRaises(self.ConfigurationException):
            cfg.resource.validate()

    def test_resource_reconstruction_zero_raises(self):
        cfg = self.SystemConfig()
        cfg.resource.max_reconstruction_iterations = 0
        with self.assertRaises(self.ConfigurationException):
            cfg.resource.validate()

    def test_resource_storage_retain_out_of_range_raises(self):
        cfg = self.SystemConfig()
        cfg.resource.storage_retain_ratio = 1.5
        with self.assertRaises(self.ConfigurationException):
            cfg.resource.validate()

    def test_safety_exploration_failures_zero_raises(self):
        cfg = self.SystemConfig()
        cfg.safety.max_consecutive_exploration_failures = 0
        with self.assertRaises(self.ConfigurationException):
            cfg.safety.validate()

    def test_safety_perf_degradation_out_of_range_raises(self):
        cfg = self.SystemConfig()
        cfg.safety.performance_degradation_threshold = 1.5
        with self.assertRaises(self.ConfigurationException):
            cfg.safety.validate()

    def test_evolution_exploration_ratio_out_of_range_raises(self):
        cfg = self.SystemConfig()
        cfg.evolution.exploration_ratio = 1.5
        with self.assertRaises(self.ConfigurationException):
            cfg.evolution.validate()

    def test_evolution_replay_interval_zero_raises(self):
        cfg = self.SystemConfig()
        cfg.evolution.idle_replay_interval_hours = 0
        with self.assertRaises(self.ConfigurationException):
            cfg.evolution.validate()

    def test_drive_weights_not_summing_to_one_raises(self):
        cfg = self.SystemConfig()
        cfg.drive_weights.safe = 0.5
        cfg.drive_weights.gain = 0.5
        cfg.drive_weights.eff = 0.5  # 总和 1.5
        with self.assertRaises(self.ConfigurationException) as ctx:
            cfg.drive_weights.validate()
        self.assertIn("1.0", str(ctx.exception))

    def test_drive_weights_negative_raises(self):
        cfg = self.SystemConfig()
        cfg.drive_weights.safe = -0.1
        cfg.drive_weights.gain = 0.9
        cfg.drive_weights.eff = 0.2
        with self.assertRaises(self.ConfigurationException):
            cfg.drive_weights.validate()

    def test_logging_invalid_console_level_raises(self):
        cfg = self.SystemConfig()
        cfg.logging.console_level = "VERBOSE"
        with self.assertRaises(self.ConfigurationException):
            cfg.logging.validate()

    def test_logging_negative_threshold_raises(self):
        cfg = self.SystemConfig()
        cfg.logging.perf_slow_threshold_sec = -1.0
        with self.assertRaises(self.ConfigurationException):
            cfg.logging.validate()

    def test_system_config_negative_interval_raises(self):
        cfg = self.SystemConfig()
        cfg.perception_interval = -1.0
        with self.assertRaises(self.ConfigurationException):
            cfg.validate()

    def test_system_config_zero_loop_frequency_raises(self):
        cfg = self.SystemConfig()
        cfg.main_loop_frequency = 0
        with self.assertRaises(self.ConfigurationException):
            cfg.validate()

    def test_exception_has_context(self):
        """ConfigurationException 应包含 context 字段"""
        cfg = self.SystemConfig()
        cfg.resource.max_cpu_percent = 200.0
        with self.assertRaises(self.ConfigurationException) as ctx:
            cfg.resource.validate()
        self.assertIsNotNone(ctx.exception.context)
        self.assertIn("field", ctx.exception.context)
        self.assertEqual(ctx.exception.context["field"], "max_cpu_percent")

    def test_exception_not_degradable(self):
        """ConfigurationException 不可降级"""
        cfg = self.SystemConfig()
        cfg.resource.max_cpu_percent = 200.0
        with self.assertRaises(self.ConfigurationException) as ctx:
            cfg.resource.validate()
        self.assertFalse(ctx.exception.degradable)


class TestU5YamlSyntaxErrorRefusesToStart(unittest.TestCase):
    """U5：YAML 语法错误时拒绝启动"""

    def setUp(self):
        from core.exceptions import ConfigurationException
        self.ConfigurationException = ConfigurationException

    def _write_temp_config(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def test_yaml_syntax_error_raises_configuration_exception(self):
        """YAML 语法错误应抛出 ConfigurationException，不静默降级"""
        from core.config import ConfigManager
        # 重置单例
        ConfigManager._instance = None
        ConfigManager._initialized = False

        bad_yaml = """
resource:
  max_cpu_percent: 20.0
  - this is invalid yaml
"""
        path = self._write_temp_config(bad_yaml)
        try:
            with self.assertRaises(self.ConfigurationException) as ctx:
                ConfigManager(config_path=path)
            self.assertIn("YAML", str(ctx.exception).upper() + str(ctx.exception))
        finally:
            os.unlink(path)
            ConfigManager._instance = None
            ConfigManager._initialized = False

    def test_non_dict_top_level_raises(self):
        """配置文件顶层非字典应拒绝启动"""
        from core.config import ConfigManager
        ConfigManager._instance = None
        ConfigManager._initialized = False

        path = self._write_temp_config("- just\n- a\n- list\n")
        try:
            with self.assertRaises(self.ConfigurationException):
                ConfigManager(config_path=path)
        finally:
            os.unlink(path)
            ConfigManager._instance = None
            ConfigManager._initialized = False

    def test_missing_config_file_uses_defaults(self):
        """配置文件不存在时应使用默认配置（不拒绝启动）"""
        from core.config import ConfigManager
        ConfigManager._instance = None
        ConfigManager._initialized = False

        try:
            mgr = ConfigManager(config_path="/nonexistent/path/agent.yaml")
            self.assertIsNotNone(mgr.config)
            # 默认值应合理
            self.assertGreater(mgr.config.resource.max_cpu_percent, 0)
        finally:
            ConfigManager._instance = None
            ConfigManager._initialized = False


class TestU5EnvironmentVariableOverrides(unittest.TestCase):
    """U5：环境变量覆盖配置项"""

    def setUp(self):
        from core.config import ConfigManager
        self.ConfigManager = ConfigManager
        # 保存当前环境变量
        self._saved_env = {}
        for key in ["AGENT_MAX_CPU", "AGENT_MAX_MEMORY", "AGENT_EXPLORATION_RATIO",
                     "AGENT_PERCEPTION_INTERVAL", "AGENT_DECISION_INTERVAL",
                     "AGENT_LOOP_FREQUENCY", "AGENT_LOG_DIR", "AGENT_DB_PATH",
                     "AGENT_SNAPSHOT_DIR", "AGENT_LOG_LEVEL", "AGENT_SAFE_WEIGHT",
                     "AGENT_GAIN_WEIGHT", "AGENT_EFF_WEIGHT"]:
            if key in os.environ:
                self._saved_env[key] = os.environ[key]

    def tearDown(self):
        # 恢复环境变量
        for key in ["AGENT_MAX_CPU", "AGENT_MAX_MEMORY", "AGENT_EXPLORATION_RATIO",
                     "AGENT_PERCEPTION_INTERVAL", "AGENT_DECISION_INTERVAL",
                     "AGENT_LOOP_FREQUENCY", "AGENT_LOG_DIR", "AGENT_DB_PATH",
                     "AGENT_SNAPSHOT_DIR", "AGENT_LOG_LEVEL", "AGENT_SAFE_WEIGHT",
                     "AGENT_GAIN_WEIGHT", "AGENT_EFF_WEIGHT"]:
            if key in self._saved_env:
                os.environ[key] = self._saved_env[key]
            elif key in os.environ:
                del os.environ[key]
        # 重置单例
        self.ConfigManager._instance = None
        self.ConfigManager._initialized = False

    def test_env_override_max_cpu(self):
        os.environ["AGENT_MAX_CPU"] = "50.0"
        self.ConfigManager._instance = None
        self.ConfigManager._initialized = False
        mgr = self.ConfigManager(config_path="/nonexistent/agent.yaml")
        self.assertEqual(mgr.config.resource.max_cpu_percent, 50.0)

    def test_env_override_exploration_ratio(self):
        os.environ["AGENT_EXPLORATION_RATIO"] = "0.5"
        self.ConfigManager._instance = None
        self.ConfigManager._initialized = False
        mgr = self.ConfigManager(config_path="/nonexistent/agent.yaml")
        self.assertEqual(mgr.config.evolution.exploration_ratio, 0.5)

    def test_env_override_log_dir(self):
        os.environ["AGENT_LOG_DIR"] = "/tmp/test_agent_logs"
        self.ConfigManager._instance = None
        self.ConfigManager._initialized = False
        mgr = self.ConfigManager(config_path="/nonexistent/agent.yaml")
        self.assertEqual(mgr.config.log_dir, "/tmp/test_agent_logs")

    def test_env_override_log_level(self):
        os.environ["AGENT_LOG_LEVEL"] = "DEBUG"
        self.ConfigManager._instance = None
        self.ConfigManager._initialized = False
        mgr = self.ConfigManager(config_path="/nonexistent/agent.yaml")
        self.assertEqual(mgr.config.logging.console_level, "DEBUG")

    def test_env_override_loop_frequency(self):
        os.environ["AGENT_LOOP_FREQUENCY"] = "2.0"
        self.ConfigManager._instance = None
        self.ConfigManager._initialized = False
        mgr = self.ConfigManager(config_path="/nonexistent/agent.yaml")
        self.assertEqual(mgr.config.main_loop_frequency, 2.0)

    def test_env_override_priority_over_file(self):
        """环境变量优先级高于配置文件"""
        # 配置文件中 max_cpu_percent=20.0
        os.environ["AGENT_MAX_CPU"] = "35.0"
        self.ConfigManager._instance = None
        self.ConfigManager._initialized = False
        mgr = self.ConfigManager()  # 使用默认 ./config/agent.yaml
        self.assertEqual(
            mgr.config.resource.max_cpu_percent, 35.0,
            "环境变量应覆盖配置文件中的值",
        )

    def test_env_override_invalid_value_raises(self):
        """环境变量类型转换失败应抛出 ConfigurationException"""
        from core.exceptions import ConfigurationException
        os.environ["AGENT_MAX_CPU"] = "not_a_number"
        self.ConfigManager._instance = None
        self.ConfigManager._initialized = False
        with self.assertRaises(ConfigurationException):
            self.ConfigManager(config_path="/nonexistent/agent.yaml")

    def test_env_override_invalid_value_rejected_by_validate(self):
        """环境变量值通过类型转换但校验失败应抛出 ConfigurationException"""
        from core.exceptions import ConfigurationException
        os.environ["AGENT_MAX_CPU"] = "150.0"  # 超出 0-100 范围
        self.ConfigManager._instance = None
        self.ConfigManager._initialized = False
        with self.assertRaises(ConfigurationException):
            self.ConfigManager(config_path="/nonexistent/agent.yaml")

    def test_env_override_drive_weights(self):
        os.environ["AGENT_SAFE_WEIGHT"] = "0.5"
        os.environ["AGENT_GAIN_WEIGHT"] = "0.3"
        os.environ["AGENT_EFF_WEIGHT"] = "0.2"
        self.ConfigManager._instance = None
        self.ConfigManager._initialized = False
        mgr = self.ConfigManager(config_path="/nonexistent/agent.yaml")
        self.assertAlmostEqual(mgr.config.drive_weights.safe, 0.5)
        self.assertAlmostEqual(mgr.config.drive_weights.gain, 0.3)
        self.assertAlmostEqual(mgr.config.drive_weights.eff, 0.2)


class TestU5ConfigManagerHasEnvOverrideMethod(unittest.TestCase):
    """U5：ConfigManager 应有 _apply_env_overrides 方法"""

    def test_apply_env_overrides_method_exists(self):
        from core.config import ConfigManager
        self.assertTrue(hasattr(ConfigManager, "_apply_env_overrides"))

    def test_apply_env_overrides_callable(self):
        from core.config import ConfigManager
        self.ConfigManager = ConfigManager
        ConfigManager._instance = None
        ConfigManager._initialized = False
        try:
            mgr = ConfigManager(config_path="/nonexistent/agent.yaml")
            self.assertTrue(callable(mgr._apply_env_overrides))
        finally:
            ConfigManager._instance = None
            ConfigManager._initialized = False


class TestU5NoRegression(unittest.TestCase):
    """U5：无回归保护"""

    def test_agent_init_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        self.assertIsNotNone(agent.perception)

    def test_main_loop_still_runs(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        import time
        time.sleep(1.5)
        agent.stop()
        self.assertGreaterEqual(agent.loop_count, 0)

    def test_default_config_validates(self):
        from core.config import SystemConfig
        cfg = SystemConfig()
        self.assertTrue(cfg.validate())

    def test_config_manager_loads_without_error(self):
        from core.config import ConfigManager
        ConfigManager._instance = None
        ConfigManager._initialized = False
        try:
            mgr = ConfigManager()
            self.assertIsNotNone(mgr.config)
        finally:
            ConfigManager._instance = None
            ConfigManager._initialized = False


if __name__ == "__main__":
    unittest.main()
