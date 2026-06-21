"""
U24：性能分析 —— 测试

测试覆盖：
1. PerfCliHelper 的所有方法
2. format_stats / format_slow / format_summary / format_report 格式化输出
3. reset / set_slow_threshold / set_enabled 控制
4. CLI集成（_cmd_perf 分发）
5. 无回归（U18-U23 仍可用）
"""

import os
import sys
import time
import pytest
from unittest.mock import patch, MagicMock

# 路径设置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def fresh_perf_monitor():
    """每个测试前重置性能监控器单例"""
    from core.performance import perf_monitor
    perf_monitor.reset()
    perf_monitor.set_enabled(True)
    perf_monitor.set_slow_threshold(0.1)
    yield perf_monitor
    perf_monitor.reset()


@pytest.fixture
def helper(fresh_perf_monitor):
    """创建PerfCliHelper，使用已重置的监控器"""
    from core.perf_cli import PerfCliHelper
    return PerfCliHelper()


# ==================== TestU24PerfCliHelper ====================

class TestU24PerfCliHelper:
    """测试PerfCliHelper基础功能"""

    def test_helper_initialization(self, helper):
        """测试初始化"""
        assert helper is not None
        assert helper.monitor is not None

    def test_helper_uses_singleton_monitor(self, helper):
        """测试使用单例监控器"""
        from core.performance import perf_monitor
        assert helper.monitor is perf_monitor

    def test_get_stats_empty(self, helper):
        """测试空统计"""
        stats = helper.get_stats()
        assert stats == {}

    def test_get_stats_single_op(self, helper, fresh_perf_monitor):
        """测试单操作统计"""
        fresh_perf_monitor.record("test_op", 0.05)
        stats = helper.get_stats("test_op")
        assert stats["name"] == "test_op"
        assert stats["count"] == 1
        assert stats["total_time"] == pytest.approx(0.05, abs=1e-6)

    def test_get_stats_nonexistent_op(self, helper):
        """测试获取不存在的操作统计"""
        stats = helper.get_stats("nonexistent")
        assert stats == {}

    def test_get_stats_all(self, helper, fresh_perf_monitor):
        """测试获取全部统计"""
        fresh_perf_monitor.record("op1", 0.01)
        fresh_perf_monitor.record("op2", 0.02)
        stats = helper.get_stats()
        assert "op1" in stats
        assert "op2" in stats
        assert stats["op1"]["count"] == 1
        assert stats["op2"]["count"] == 1

    def test_get_summary(self, helper, fresh_perf_monitor):
        """测试获取摘要"""
        fresh_perf_monitor.record("op1", 0.01)
        fresh_perf_monitor.record("op1", 0.02)
        summary = helper.get_summary()
        assert summary["operation_count"] == 1
        assert summary["total_calls"] == 2
        assert summary["enabled"] is True
        assert "slow_threshold_sec" in summary

    def test_get_slow_operations_empty(self, helper):
        """测试空慢操作列表"""
        slow = helper.get_slow_operations()
        assert slow == []

    def test_get_slow_operations_with_data(self, helper, fresh_perf_monitor):
        """测试有数据的慢操作列表"""
        fresh_perf_monitor.record("fast_op", 0.01)
        fresh_perf_monitor.record("slow_op", 0.5)  # 超过默认阈值0.1
        slow = helper.get_slow_operations()
        assert len(slow) == 1
        assert slow[0]["name"] == "slow_op"

    def test_get_slow_operations_custom_threshold(self, helper, fresh_perf_monitor):
        """测试自定义阈值的慢操作列表"""
        fresh_perf_monitor.record("op1", 0.05)
        fresh_perf_monitor.record("op2", 0.15)
        # 阈值0.1：op2是慢操作
        slow = helper.get_slow_operations(0.1)
        assert len(slow) == 1
        assert slow[0]["name"] == "op2"
        # 阈值0.2：无慢操作
        slow = helper.get_slow_operations(0.2)
        assert slow == []

    def test_reset(self, helper, fresh_perf_monitor):
        """测试重置"""
        fresh_perf_monitor.record("op1", 0.01)
        assert helper.get_summary()["operation_count"] == 1
        helper.reset()
        assert helper.get_summary()["operation_count"] == 0

    def test_set_slow_threshold(self, helper, fresh_perf_monitor):
        """测试设置慢操作阈值"""
        helper.set_slow_threshold(0.5)
        fresh_perf_monitor.record("op1", 0.3)  # 超过0.1但不超过0.5
        slow = helper.get_slow_operations()  # 使用默认阈值
        assert slow == []  # 阈值已改为0.5，0.3不算慢

    def test_set_enabled(self, helper, fresh_perf_monitor):
        """测试启用/禁用"""
        helper.set_enabled(False)
        fresh_perf_monitor.record("op1", 0.01)
        # 禁用后不记录
        assert helper.get_stats() == {}
        helper.set_enabled(True)
        fresh_perf_monitor.record("op1", 0.01)
        assert "op1" in helper.get_stats()


# ==================== TestU24FormatStats ====================

class TestU24FormatStats:
    """测试format_stats格式化"""

    def test_format_stats_empty(self, helper):
        """测试空统计格式化"""
        result = helper.format_stats()
        assert "暂无性能数据" in result

    def test_format_stats_all(self, helper, fresh_perf_monitor):
        """测试全部统计格式化"""
        fresh_perf_monitor.record("op1", 0.01)
        fresh_perf_monitor.record("op2", 0.02)
        result = helper.format_stats()
        assert "性能统计" in result
        assert "op1" in result
        assert "op2" in result

    def test_format_stats_single_op(self, helper, fresh_perf_monitor):
        """测试单操作统计格式化"""
        fresh_perf_monitor.record("test_op", 0.05)
        result = helper.format_stats("test_op")
        assert "操作统计" in result
        assert "test_op" in result
        assert "调用次数" in result
        assert "P99" in result

    def test_format_stats_nonexistent_op(self, helper):
        """测试不存在的操作格式化"""
        result = helper.format_stats("nonexistent")
        assert "不存在" in result

    def test_format_stats_sorted_by_total_time(self, helper, fresh_perf_monitor):
        """测试按总耗时降序排列"""
        fresh_perf_monitor.record("slow_op", 0.05)
        fresh_perf_monitor.record("slow_op", 0.05)
        fresh_perf_monitor.record("fast_op", 0.001)
        result = helper.format_stats()
        # slow_op总耗时更大，应排在前面
        slow_pos = result.index("slow_op")
        fast_pos = result.index("fast_op")
        assert slow_pos < fast_pos


# ==================== TestU24FormatSlow ====================

class TestU24FormatSlow:
    """测试format_slow格式化"""

    def test_format_slow_empty(self, helper):
        """测试空慢操作格式化"""
        result = helper.format_slow()
        assert "无慢操作" in result

    def test_format_slow_with_data(self, helper, fresh_perf_monitor):
        """测试有数据的慢操作格式化"""
        fresh_perf_monitor.record("slow_op", 0.5)
        result = helper.format_slow()
        assert "慢操作" in result
        assert "slow_op" in result

    def test_format_slow_custom_threshold(self, helper, fresh_perf_monitor):
        """测试自定义阈值慢操作格式化"""
        fresh_perf_monitor.record("op1", 0.05)
        fresh_perf_monitor.record("op2", 0.15)
        result = helper.format_slow(0.1)
        assert "op2" in result
        assert "op1" not in result

    def test_format_slow_no_slow_ops_with_custom_threshold(self, helper, fresh_perf_monitor):
        """测试自定义阈值无慢操作"""
        fresh_perf_monitor.record("op1", 0.01)
        result = helper.format_slow(0.5)
        assert "无慢操作" in result


# ==================== TestU24FormatSummary ====================

class TestU24FormatSummary:
    """测试format_summary格式化"""

    def test_format_summary_empty(self, helper):
        """测试空摘要格式化"""
        result = helper.format_summary()
        assert "性能摘要" in result
        assert "操作种类" in result
        assert "总调用次数" in result

    def test_format_summary_with_data(self, helper, fresh_perf_monitor):
        """测试有数据的摘要格式化"""
        fresh_perf_monitor.record("op1", 0.01)
        fresh_perf_monitor.record("op1", 0.02)
        result = helper.format_summary()
        assert "性能摘要" in result
        assert "1" in result  # 操作种类=1
        assert "2" in result  # 总调用=2

    def test_format_summary_disabled(self, helper, fresh_perf_monitor):
        """测试禁用状态的摘要"""
        helper.set_enabled(False)
        result = helper.format_summary()
        assert "禁用" in result


# ==================== TestU24FormatReport ====================

class TestU24FormatReport:
    """测试format_report格式化"""

    def test_format_report_empty(self, helper):
        """测试空报告格式化"""
        result = helper.format_report()
        assert "暂无数据" in result

    def test_format_report_with_data(self, helper, fresh_perf_monitor):
        """测试有数据的报告格式化"""
        fresh_perf_monitor.record("op1", 0.01)
        fresh_perf_monitor.record("op2", 0.5)  # 慢操作
        result = helper.format_report()
        assert "性能报告" in result
        assert "op1" in result
        assert "op2" in result


# ==================== TestU24CliIntegration ====================

class TestU24CliIntegration:
    """测试CLI集成"""

    def test_perf_parser_exists(self):
        """测试perf子命令parser存在"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["perf"])
        assert args.command == "perf"

    def test_perf_stats_parser(self):
        """测试perf stats parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["perf", "stats"])
        assert args.command == "perf"
        assert args.perf_action == "stats"
        assert args.name is None

    def test_perf_stats_with_name_parser(self):
        """测试perf stats带操作名"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["perf", "stats", "perceive"])
        assert args.perf_action == "stats"
        assert args.name == "perceive"

    def test_perf_slow_parser(self):
        """测试perf slow parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["perf", "slow"])
        assert args.perf_action == "slow"
        assert args.threshold is None

    def test_perf_slow_with_threshold_parser(self):
        """测试perf slow带阈值"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["perf", "slow", "--threshold", "0.2"])
        assert args.perf_action == "slow"
        assert args.threshold == 0.2

    def test_perf_summary_parser(self):
        """测试perf summary parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["perf", "summary"])
        assert args.perf_action == "summary"

    def test_perf_report_parser(self):
        """测试perf report parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["perf", "report"])
        assert args.perf_action == "report"

    def test_perf_reset_parser(self):
        """测试perf reset parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["perf", "reset"])
        assert args.perf_action == "reset"

    def test_cmd_perf_stats_all(self, capsys, fresh_perf_monitor):
        """测试_cmd_perf stats全部"""
        from cli import _cmd_perf, _create_parser
        fresh_perf_monitor.record("test_op", 0.01)
        parser = _create_parser()
        args = parser.parse_args(["perf", "stats"])
        ret = _cmd_perf(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "性能统计" in out
        assert "test_op" in out

    def test_cmd_perf_stats_single(self, capsys, fresh_perf_monitor):
        """测试_cmd_perf stats单操作"""
        from cli import _cmd_perf, _create_parser
        fresh_perf_monitor.record("test_op", 0.01)
        parser = _create_parser()
        args = parser.parse_args(["perf", "stats", "test_op"])
        ret = _cmd_perf(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "操作统计" in out
        assert "test_op" in out

    def test_cmd_perf_stats_nonexistent(self, capsys, helper):
        """测试_cmd_perf stats不存在的操作"""
        from cli import _cmd_perf, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["perf", "stats", "nonexistent"])
        ret = _cmd_perf(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "不存在" in out

    def test_cmd_perf_slow_empty(self, capsys, helper):
        """测试_cmd_perf slow空"""
        from cli import _cmd_perf, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["perf", "slow"])
        ret = _cmd_perf(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "无慢操作" in out

    def test_cmd_perf_slow_with_data(self, capsys, fresh_perf_monitor):
        """测试_cmd_perf slow有数据"""
        from cli import _cmd_perf, _create_parser
        fresh_perf_monitor.record("slow_op", 0.5)
        parser = _create_parser()
        args = parser.parse_args(["perf", "slow"])
        ret = _cmd_perf(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "slow_op" in out

    def test_cmd_perf_slow_custom_threshold(self, capsys, fresh_perf_monitor):
        """测试_cmd_perf slow自定义阈值"""
        from cli import _cmd_perf, _create_parser
        fresh_perf_monitor.record("op1", 0.05)
        fresh_perf_monitor.record("op2", 0.15)
        parser = _create_parser()
        args = parser.parse_args(["perf", "slow", "--threshold", "0.1"])
        ret = _cmd_perf(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "op2" in out
        assert "op1" not in out

    def test_cmd_perf_summary(self, capsys, fresh_perf_monitor):
        """测试_cmd_perf summary"""
        from cli import _cmd_perf, _create_parser
        fresh_perf_monitor.record("op1", 0.01)
        parser = _create_parser()
        args = parser.parse_args(["perf", "summary"])
        ret = _cmd_perf(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "性能摘要" in out

    def test_cmd_perf_report(self, capsys, fresh_perf_monitor):
        """测试_cmd_perf report"""
        from cli import _cmd_perf, _create_parser
        fresh_perf_monitor.record("op1", 0.01)
        parser = _create_parser()
        args = parser.parse_args(["perf", "report"])
        ret = _cmd_perf(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "性能报告" in out or "暂无数据" in out

    def test_cmd_perf_reset(self, capsys, fresh_perf_monitor):
        """测试_cmd_perf reset"""
        from cli import _cmd_perf, _create_parser
        fresh_perf_monitor.record("op1", 0.01)
        parser = _create_parser()
        args = parser.parse_args(["perf", "reset"])
        ret = _cmd_perf(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "重置" in out
        assert fresh_perf_monitor.get_summary()["operation_count"] == 0

    def test_cmd_perf_no_action(self, capsys, helper):
        """测试_cmd_perf无操作"""
        from cli import _cmd_perf, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["perf"])
        ret = _cmd_perf(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "请指定操作" in out

    def test_cli_main_perf_dispatch(self, capsys, fresh_perf_monitor):
        """测试cli_main分发perf命令"""
        from cli import cli_main
        fresh_perf_monitor.record("test_op", 0.01)
        ret = cli_main(["perf", "stats"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "性能统计" in out

    def test_cli_main_perf_reset_dispatch(self, capsys, fresh_perf_monitor):
        """测试cli_main分发perf reset"""
        from cli import cli_main
        fresh_perf_monitor.record("op1", 0.01)
        ret = cli_main(["perf", "reset"])
        assert ret == 0
        assert fresh_perf_monitor.get_summary()["operation_count"] == 0


# ==================== TestU24NoRegression ====================

class TestU24NoRegression:
    """验证U24不破坏既有功能"""

    def test_u18_themes_still_works(self):
        """U18 themes模块仍可导入"""
        from core.themes import ThemeManager
        mgr = ThemeManager()
        assert mgr is not None

    def test_u19_config_editor_still_works(self):
        """U19 config_editor模块仍可导入"""
        from core.config_editor import ConfigEditor
        assert ConfigEditor is not None

    def test_u20_history_still_works(self):
        """U20 history模块仍可导入"""
        from core.history import CommandHistory
        assert CommandHistory is not None

    def test_u21_context_help_still_works(self):
        """U21 context_help模块仍可导入"""
        from core.context_help import ContextualHelpProvider
        assert ContextualHelpProvider is not None

    def test_u22_log_viewer_still_works(self):
        """U22 log_viewer模块仍可导入"""
        from core.log_viewer import LogManager, LogReader
        assert LogManager is not None
        assert LogReader is not None

    def test_u23_alerts_still_works(self):
        """U23 alert_cli模块仍可导入"""
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        assert helper is not None

    def test_existing_commands_still_dispatch(self):
        """既有命令仍可正常分发"""
        from cli import _create_parser
        parser = _create_parser()
        # 验证所有既有命令仍可解析
        for cmd in ["run", "status", "snapshot", "logs", "alerts", "perf"]:
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_performance_monitor_unchanged(self, fresh_perf_monitor):
        """PerformanceMonitor核心API未变"""
        from core.performance import perf_monitor
        perf_monitor.record("test", 0.01)
        stats = perf_monitor.get_stats("test")
        assert stats["count"] == 1
        summary = perf_monitor.get_summary()
        assert "operation_count" in summary
        slow = perf_monitor.get_slow_operations()
        assert isinstance(slow, list)
