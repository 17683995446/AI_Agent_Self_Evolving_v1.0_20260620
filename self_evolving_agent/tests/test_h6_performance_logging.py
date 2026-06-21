"""
H6阶段验证测试 —— 性能优化与日志分级

验证点：
1. PerformanceMonitor单例存在且可创建
2. track()上下文管理器正确计时
3. record()直接记录
4. 统计聚合正确（count/total/avg/max/min/p99）
5. 慢操作检测
6. 启用/禁用控制
7. StructuredLogger运行时级别切换
8. 采样控制
9. 函数追踪开关
10. LoggingConfig数据类
11. SystemConfig集成logging字段
12. Agent初始化应用日志配置
13. Agent性能查询接口
14. 场景配置覆盖logging
"""

import os
import sys
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.performance import (
    PerformanceMonitor, perf_monitor, OperationStats, track_performance,
)
from core.logger import StructuredLogger, logger, log_execution
from core.config import LoggingConfig, SystemConfig, config_manager


class TestH6PerformanceMonitor:
    """验证性能监控器"""

    def test_perf_monitor_exists(self):
        assert PerformanceMonitor is not None

    def test_perf_monitor_singleton(self):
        """全局单例"""
        m1 = PerformanceMonitor()
        m2 = PerformanceMonitor()
        assert m1 is m2
        assert m1 is perf_monitor

    def test_track_context_manager(self):
        """track上下文管理器计时"""
        m = PerformanceMonitor.__new__(PerformanceMonitor)  # 临时实例避免污染单例
        m._max_samples = 1000
        m._stats = {}
        m._lock = __import__('threading').RLock()
        m._enabled = True
        m._slow_threshold = 0.1

        with m.track("unit_test_op"):
            time.sleep(0.001)
        stats = m.get_stats("unit_test_op")
        assert stats["count"] == 1
        assert stats["total_time"] > 0

    def test_record_directly(self):
        """直接record记录"""
        m = PerformanceMonitor.__new__(PerformanceMonitor)
        m._max_samples = 1000
        m._stats = {}
        m._lock = __import__('threading').RLock()
        m._enabled = True
        m._slow_threshold = 0.1

        m.record("direct_op", 0.05)
        m.record("direct_op", 0.03)
        stats = m.get_stats("direct_op")
        assert stats["count"] == 2
        assert stats["total_time"] == pytest.approx(0.08, abs=1e-6)
        assert stats["avg_time"] == pytest.approx(0.04, abs=1e-6)
        assert stats["max_time"] == pytest.approx(0.05, abs=1e-6)
        assert stats["min_time"] == pytest.approx(0.03, abs=1e-6)

    def test_disabled_monitor_no_record(self):
        """禁用后不记录"""
        m = PerformanceMonitor.__new__(PerformanceMonitor)
        m._max_samples = 1000
        m._stats = {}
        m._lock = __import__('threading').RLock()
        m._enabled = False
        m._slow_threshold = 0.1

        m.record("disabled_op", 0.01)
        assert m.get_stats("disabled_op") == {}

    def test_slow_operation_detection(self):
        """慢操作检测"""
        m = PerformanceMonitor.__new__(PerformanceMonitor)
        m._max_samples = 1000
        m._stats = {}
        m._lock = __import__('threading').RLock()
        m._enabled = True
        m._slow_threshold = 0.05

        m.record("fast_op", 0.01)
        m.record("slow_op", 0.1)
        slow = m.get_slow_operations()
        assert len(slow) == 1
        assert slow[0]["name"] == "slow_op"

    def test_get_summary(self):
        """全局摘要"""
        m = PerformanceMonitor.__new__(PerformanceMonitor)
        m._max_samples = 1000
        m._stats = {}
        m._lock = __import__('threading').RLock()
        m._enabled = True
        m._slow_threshold = 0.1

        m.record("a", 0.01)
        m.record("b", 0.02)
        summary = m.get_summary()
        assert summary["operation_count"] == 2
        assert summary["total_calls"] == 2
        assert summary["enabled"] is True

    def test_reset(self):
        """清空统计"""
        m = PerformanceMonitor.__new__(PerformanceMonitor)
        m._max_samples = 1000
        m._stats = {}
        m._lock = __import__('threading').RLock()
        m._enabled = True
        m._slow_threshold = 0.1

        m.record("x", 0.01)
        assert len(m.get_stats()) == 1
        m.reset()
        assert len(m.get_stats()) == 0

    def test_report_string(self):
        """报告生成字符串"""
        m = PerformanceMonitor.__new__(PerformanceMonitor)
        m._max_samples = 1000
        m._stats = {}
        m._lock = __import__('threading').RLock()
        m._enabled = True
        m._slow_threshold = 0.1

        # 空报告
        assert "暂无数据" in m.report()
        # 有数据
        m.record("op1", 0.01)
        report = m.report()
        assert "性能报告" in report
        assert "op1" in report

    def test_percentile(self):
        """分位数计算"""
        m = PerformanceMonitor.__new__(PerformanceMonitor)
        m._max_samples = 1000
        m._stats = {}
        m._lock = __import__('threading').RLock()
        m._enabled = True
        m._slow_threshold = 0.1

        for i in range(100):
            m.record("pct_op", 0.001 * (i + 1))
        stats = m.get_stats("pct_op")
        assert stats["p50"] > 0
        assert stats["p99"] >= stats["p50"]

    def test_track_performance_decorator(self):
        """装饰器工作"""
        m = PerformanceMonitor.__new__(PerformanceMonitor)
        m._max_samples = 1000
        m._stats = {}
        m._lock = __import__('threading').RLock()
        m._enabled = True
        m._slow_threshold = 0.1

        # 临时替换全局单例
        import core.performance as perf_mod
        original = perf_mod.perf_monitor
        perf_mod.perf_monitor = m
        try:
            @track_performance("decorated_op")
            def my_func():
                time.sleep(0.001)
                return 42

            result = my_func()
            assert result == 42
            assert m.get_stats("decorated_op")["count"] == 1
        finally:
            perf_mod.perf_monitor = original


class TestH6LoggerLevelControl:
    """验证日志运行时级别控制"""

    def test_set_console_level(self):
        """控制台级别切换"""
        import logging
        logger.set_console_level("WARNING")
        assert logger._console_handler.level == logging.WARNING
        # 恢复
        logger.set_console_level("INFO")
        assert logger._console_handler.level == logging.INFO

    def test_set_file_level(self):
        """文件级别切换"""
        import logging
        logger.set_file_level("ERROR")
        assert logger._file_handler.level == logging.ERROR
        # 恢复
        logger.set_file_level("DEBUG")

    def test_invalid_level_ignored(self):
        """非法级别不崩溃"""
        import logging
        original = logger._console_handler.level
        logger.set_console_level("INVALID_LEVEL")
        assert logger._console_handler.level == original  # 未改变

    def test_func_trace_toggle(self):
        """函数追踪开关"""
        logger.set_func_trace_enabled(False)
        assert logger.func_trace_enabled is False
        logger.set_func_trace_enabled(True)
        assert logger.func_trace_enabled is True
        # 恢复默认
        logger.set_func_trace_enabled(False)


class TestH6LoggerSampling:
    """验证采样控制"""

    def test_sample_rate_basic(self):
        """每N次记录1次"""
        logger.set_sample_rate("H6_TEST_SAMPLE", 3)
        results = []
        for _ in range(7):
            r = logger.info("H6_TEST_SAMPLE", "test")
            results.append(r is not None)
        # 每3次记1次：第2、5次记录
        assert results.count(True) == 2
        logger.clear_sample_rate("H6_TEST_SAMPLE")

    def test_sample_rate_disable(self):
        """every_n=0禁用该类型"""
        logger.set_sample_rate("H6_TEST_DISABLE", 0)
        r = logger.info("H6_TEST_DISABLE", "should skip")
        assert r is None
        logger.clear_sample_rate("H6_TEST_DISABLE")

    def test_sample_rate_1_means_all(self):
        """every_n=1表示全记录"""
        logger.set_sample_rate("H6_TEST_ALL", 1)
        count = 0
        for _ in range(5):
            r = logger.info("H6_TEST_ALL", "test")
            if r is not None:
                count += 1
        assert count == 5
        logger.clear_sample_rate("H6_TEST_ALL")

    def test_clear_sample_rate(self):
        """清除采样后全记录"""
        logger.set_sample_rate("H6_TEST_CLEAR", 100)
        r1 = logger.info("H6_TEST_CLEAR", "skip")
        assert r1 is None
        logger.clear_sample_rate("H6_TEST_CLEAR")
        r2 = logger.info("H6_TEST_CLEAR", "record")
        assert r2 is not None

    def test_unsampled_always_recorded(self):
        """未配置采样的event_type全记录"""
        r = logger.info("H6_TEST_UNSAMPLED", "always")
        assert r is not None


class TestH6LogExecutionDecorator:
    """验证log_execution装饰器与H6开关集成"""

    def test_decorator_works_with_trace_enabled(self):
        logger.set_func_trace_enabled(True)
        @log_execution
        def foo():
            return 100
        assert foo() == 100
        logger.set_func_trace_enabled(False)

    def test_decorator_works_with_trace_disabled(self):
        logger.set_func_trace_enabled(False)
        @log_execution
        def bar():
            return 200
        assert bar() == 200

    def test_decorator_propagates_exception(self):
        logger.set_func_trace_enabled(False)
        @log_execution
        def raise_func():
            raise ValueError("test error")
        with pytest.raises(ValueError, match="test error"):
            raise_func()


class TestH6LoggingConfig:
    """验证LoggingConfig数据类"""

    def test_logging_config_exists(self):
        assert LoggingConfig is not None

    def test_default_values(self):
        cfg = LoggingConfig()
        assert cfg.console_level == "INFO"
        assert cfg.file_level == "DEBUG"
        assert cfg.func_trace_enabled is False
        assert cfg.perf_monitor_enabled is True
        assert cfg.perf_slow_threshold_sec == 0.1
        assert cfg.perf_report_interval_loops == 100
        assert "FUNC_ENTER" in cfg.sample_rates

    def test_validate_valid(self):
        cfg = LoggingConfig()
        assert cfg.validate() is True

    def test_validate_invalid_console_level(self):
        cfg = LoggingConfig(console_level="BOGUS")
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_validate_negative_threshold(self):
        cfg = LoggingConfig(perf_slow_threshold_sec=-0.1)
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_to_dict(self):
        cfg = LoggingConfig()
        d = cfg.to_dict()
        assert "console_level" in d
        assert "file_level" in d
        assert "sample_rates" in d
        assert "perf_monitor_enabled" in d


class TestH6SystemConfigIntegration:
    """验证SystemConfig集成logging字段"""

    def test_system_config_has_logging(self):
        cfg = SystemConfig()
        assert hasattr(cfg, "logging")
        assert isinstance(cfg.logging, LoggingConfig)

    def test_system_config_validate_includes_logging(self):
        cfg = SystemConfig()
        # 不应抛异常
        cfg.validate()

    def test_system_config_to_dict_includes_logging(self):
        cfg = SystemConfig()
        d = cfg.to_dict()
        assert "logging" in d
        assert "console_level" in d["logging"]


class TestH6AgentIntegration:
    """验证Agent集成H6"""

    def test_agent_has_performance_methods(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert hasattr(agent, "get_performance_stats")
        assert hasattr(agent, "get_performance_report")

    def test_agent_applies_logging_config_on_init(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        # 默认配置应已应用：console_level=INFO
        import logging
        assert logger._console_handler.level == logging.INFO
        # func_trace默认False
        assert logger.func_trace_enabled is False

    def test_agent_performance_stats_structure(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        stats = agent.get_performance_stats()
        assert "summary" in stats
        assert "operations" in stats
        assert "slow_operations" in stats
        assert "operation_count" in stats["summary"]

    def test_agent_performance_report_string(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        report = agent.get_performance_report()
        assert isinstance(report, str)

    def test_agent_health_report_includes_performance(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start_time = time.time()
        report = agent._generate_health_report()
        assert "performance_summary" in report


class TestH6ScenarioLoggingOverride:
    """验证场景配置可覆盖logging"""

    def test_scenario_loader_applies_logging_override(self, tmp_path):
        from core.scenario_loader import ScenarioLoader
        p = tmp_path / "s.yaml"
        p.write_text(
            "scenarios:\n"
            "  test_logging:\n"
            "    description: 'test'\n"
            "    config_overrides:\n"
            "      logging:\n"
            "        console_level: 'DEBUG'\n"
            "        func_trace_enabled: true\n"
        )
        loader = ScenarioLoader(scenarios_path=str(p))
        s = loader.load("test_logging")
        assert config_manager.config.logging.console_level == "DEBUG"
        assert config_manager.config.logging.func_trace_enabled is True
        # 恢复
        config_manager.config.logging.console_level = "INFO"
        config_manager.config.logging.func_trace_enabled = False
