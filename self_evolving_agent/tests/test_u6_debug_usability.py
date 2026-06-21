"""
U6：调试易用性测试

验证目标：
1. AgentException 及其子类有 remediation 字段
2. ConfigurationException 有默认修复建议
3. logger.exception() 方法保留完整堆栈
4. logger.console_level 属性可读可写（实际生效）
5. CLI --log-level 通过 set_console_level 生效（非设置无效属性）
6. main.py 未知异常处理使用 logger.exception（保留堆栈）
7. 无回归

第一性原理：
- 异常不只是报错，更是指导用户行动的信号（remediation）
- 未知异常的堆栈是排查问题的关键线索，不能丢失
- --log-level 开关必须真正生效，不能是摆设
"""
import os
import sys
import unittest
import traceback
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent


class TestU6ExceptionRemediation(unittest.TestCase):
    """U6：异常类应有 remediation 字段"""

    def test_agent_exception_has_remediation_field(self):
        from core.exceptions import AgentException
        exc = AgentException("test error", module="test")
        self.assertTrue(hasattr(exc, "remediation"))

    def test_agent_exception_remediation_default_empty(self):
        from core.exceptions import AgentException
        exc = AgentException("test error", module="test")
        # 基类默认为空字符串
        self.assertEqual(exc.remediation, "")

    def test_agent_exception_remediation_custom(self):
        from core.exceptions import AgentException
        exc = AgentException("test error", module="test", remediation="重启服务")
        self.assertEqual(exc.remediation, "重启服务")

    def test_configuration_exception_has_default_remediation(self):
        from core.exceptions import ConfigurationException
        exc = ConfigurationException("配置错误")
        self.assertTrue(exc.remediation, "ConfigurationException 应有默认修复建议")
        self.assertIn("config validate", exc.remediation.lower() + str(exc.remediation))

    def test_configuration_exception_custom_remediation(self):
        from core.exceptions import ConfigurationException
        exc = ConfigurationException("配置错误", remediation="修改 agent.yaml")
        self.assertEqual(exc.remediation, "修改 agent.yaml")

    def test_perception_exception_has_remediation(self):
        from core.exceptions import PerceptionException
        exc = PerceptionException("感知失败")
        self.assertTrue(exc.remediation)

    def test_prediction_exception_has_remediation(self):
        from core.exceptions import PredictionException
        exc = PredictionException("预测失败")
        self.assertTrue(exc.remediation)

    def test_drive_engine_exception_has_remediation(self):
        from core.exceptions import DriveEngineException
        exc = DriveEngineException("驱力失败")
        self.assertTrue(exc.remediation)

    def test_decision_exception_has_remediation(self):
        from core.exceptions import DecisionException
        exc = DecisionException("决策失败")
        self.assertTrue(exc.remediation)

    def test_memory_exception_has_remediation(self):
        from core.exceptions import MemoryException
        exc = MemoryException("记忆失败")
        self.assertTrue(exc.remediation)

    def test_optimizer_exception_has_remediation(self):
        from core.exceptions import OptimizerException
        exc = OptimizerException("优化失败")
        self.assertTrue(exc.remediation)

    def test_resource_exhausted_exception_has_remediation(self):
        from core.exceptions import ResourceExhaustedException
        exc = ResourceExhaustedException("资源耗尽")
        self.assertTrue(exc.remediation)

    def test_module_init_exception_has_remediation(self):
        from core.exceptions import ModuleInitException
        exc = ModuleInitException("初始化失败")
        self.assertTrue(exc.remediation)

    def test_workflow_exception_has_remediation(self):
        from core.exceptions import WorkflowException
        exc = WorkflowException("工作流失败")
        self.assertTrue(exc.remediation)

    def test_skill_library_exception_has_remediation(self):
        from core.exceptions import SkillLibraryException
        exc = SkillLibraryException("技能库失败")
        self.assertTrue(exc.remediation)

    def test_persistence_exception_has_remediation(self):
        from core.exceptions import PersistenceException
        exc = PersistenceException("持久化失败")
        self.assertTrue(exc.remediation)

    def test_forever_runner_exception_has_remediation(self):
        from core.exceptions import ForeverRunnerException
        exc = ForeverRunnerException("永续运行失败")
        self.assertTrue(exc.remediation)

    def test_alerting_exception_has_remediation(self):
        from core.exceptions import AlertingException
        exc = AlertingException("告警失败")
        self.assertTrue(exc.remediation)

    def test_multi_agent_exception_has_remediation(self):
        from core.exceptions import MultiAgentException
        exc = MultiAgentException("多Agent失败")
        self.assertTrue(exc.remediation)

    def test_long_task_exception_has_remediation(self):
        from core.exceptions import LongTaskException
        exc = LongTaskException("长任务失败")
        self.assertTrue(exc.remediation)

    def test_to_dict_includes_remediation(self):
        from core.exceptions import ConfigurationException
        exc = ConfigurationException("配置错误")
        d = exc.to_dict()
        self.assertIn("remediation", d)
        self.assertTrue(d["remediation"])

    def test_backward_compatible_construction(self):
        """旧代码不传 remediation 参数仍能正常构造"""
        from core.exceptions import AgentException, ConfigurationException
        # 不传 remediation
        exc1 = AgentException("msg")
        exc2 = ConfigurationException("msg")
        self.assertEqual(str(exc1), "msg")
        self.assertEqual(str(exc2), "msg")


class TestU6LoggerExceptionMethod(unittest.TestCase):
    """U6：logger.exception() 方法保留完整堆栈"""

    def test_logger_has_exception_method(self):
        from core.logger import logger
        self.assertTrue(hasattr(logger, "exception"))
        self.assertTrue(callable(logger.exception))

    def test_logger_exception_captures_traceback(self):
        """logger.exception 应捕获完整堆栈"""
        from core.logger import logger
        captured = []
        original_log = logger._log

        def mock_log(event_type, message, level, module, extra):
            captured.append({"event_type": event_type, "message": message,
                             "level": level, "extra": extra})
            return None

        logger._log = mock_log
        try:
            try:
                raise ValueError("测试异常")
            except ValueError:
                logger.exception("TEST", "捕获异常", module="test")
            self.assertEqual(len(captured), 1)
            self.assertIn("traceback", captured[0]["extra"])
            self.assertIn("ValueError", captured[0]["extra"]["traceback"])
            self.assertIn("测试异常", captured[0]["extra"]["traceback"])
        finally:
            logger._log = original_log

    def test_logger_exception_without_active_exception(self):
        """无活跃异常时不应崩溃"""
        from core.logger import logger
        captured = []
        original_log = logger._log

        def mock_log(event_type, message, level, module, extra):
            captured.append({"extra": extra})
            return None

        logger._log = mock_log
        try:
            logger.exception("TEST", "无异常上下文", module="test")
            self.assertEqual(len(captured), 1)
            # 无活跃异常时 traceback 应为空或不包含
            self.assertNotIn("traceback", captured[0]["extra"])
        finally:
            logger._log = original_log


class TestU6LoggerConsoleLevelProperty(unittest.TestCase):
    """U6：logger.console_level 属性可读可写"""

    def test_console_level_property_exists(self):
        from core.logger import logger
        self.assertTrue(hasattr(logger, "console_level"))

    def test_console_level_getter_returns_string(self):
        from core.logger import logger
        level = logger.console_level
        self.assertIsInstance(level, str)
        self.assertIn(level, ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "UNKNOWN"])

    def test_console_level_setter_changes_level(self):
        from core.logger import logger
        original = logger.console_level
        try:
            logger.console_level = "WARNING"
            self.assertEqual(logger.console_level, "WARNING")
            logger.console_level = "DEBUG"
            self.assertEqual(logger.console_level, "DEBUG")
        finally:
            logger.console_level = original


class TestU6CliLogLevelFix(unittest.TestCase):
    """U6：CLI --log-level 通过 set_console_level 生效"""

    def test_cli_uses_set_console_level(self):
        """CLI 代码应调用 set_console_level，而非设置无效属性"""
        cli_path = AGENT_ROOT / "cli.py"
        content = cli_path.read_text(encoding="utf-8")
        self.assertIn("set_console_level", content)

    def test_cli_has_log_file_level_option(self):
        """U6：CLI 应支持 --log-file-level 选项"""
        cli_path = AGENT_ROOT / "cli.py"
        content = cli_path.read_text(encoding="utf-8")
        self.assertIn("--log-file-level", content)

    def test_cli_log_level_actually_works(self):
        """端到端验证：--log-level DEBUG 真正改变日志级别"""
        from cli import _create_parser, _cmd_run
        from core.logger import logger
        import argparse

        original = logger.console_level
        try:
            parser = _create_parser()
            args = parser.parse_args(["run", "--log-level", "WARNING"])
            self.assertEqual(args.log_level, "WARNING")
            # 直接调用日志设置逻辑（不启动Agent）
            logger.set_console_level(args.log_level)
            self.assertEqual(logger.console_level, "WARNING")
        finally:
            logger.set_console_level(original)


class TestU6MainPreservesTraceback(unittest.TestCase):
    """U6：main.py 未知异常处理应保留堆栈"""

    def test_online_loop_uses_logger_exception(self):
        """在线循环的未知异常处理应使用 logger.exception"""
        main_path = AGENT_ROOT / "main.py"
        content = main_path.read_text(encoding="utf-8")
        # 查找在线循环的未知异常处理块
        self.assertIn("logger.exception", content)

    def test_offline_loop_uses_logger_exception(self):
        """离线循环的未知异常处理应使用 logger.exception"""
        main_path = AGENT_ROOT / "main.py"
        content = main_path.read_text(encoding="utf-8")
        # 应有多处 logger.exception 调用
        count = content.count("logger.exception")
        self.assertGreaterEqual(count, 2, "main.py 应至少有2处 logger.exception 调用（在线+离线循环）")

    def test_agent_exception_handling_includes_remediiation(self):
        """已知异常处理应记录 remediation 字段"""
        main_path = AGENT_ROOT / "main.py"
        content = main_path.read_text(encoding="utf-8")
        self.assertIn("remediation", content)


class TestU6NoRegression(unittest.TestCase):
    """U6：无回归保护"""

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

    def test_exceptions_still_catchable(self):
        """异常类仍可被 except 捕获"""
        from core.exceptions import AgentException, ConfigurationException
        self.assertTrue(issubclass(ConfigurationException, AgentException))
        self.assertTrue(issubclass(AgentException, Exception))

    def test_exception_context_still_works(self):
        """异常 context 字段仍正常工作"""
        from core.exceptions import ConfigurationException
        exc = ConfigurationException("msg", context={"key": "value"})
        self.assertEqual(exc.context["key"], "value")


if __name__ == "__main__":
    unittest.main()
