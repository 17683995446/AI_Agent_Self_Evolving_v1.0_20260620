"""
U32：永续运行与自愈控制 —— 测试

测试覆盖：
1. ForeverCliHelper 的所有方法
2. enable / disable / is_enabled / get_stats / get_health_report / get_health_history
3. trigger_self_heal / trigger_cleanup / set_auto_recovery
4. format_stats / format_health_report / format_history / format_enable_result / format_disable_result
   / format_self_heal_result / format_cleanup_result / format_auto_recovery / format_auto_recovery_set
5. CLI集成（_cmd_forever 分发）
6. 无回归（U27-U31 仍可用）
"""

import os
import sys
import time
import pytest

# 路径设置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def helper():
    """创建ForeverCliHelper（确保停止状态）"""
    from core.forever_cli import ForeverCliHelper
    h = ForeverCliHelper()
    yield h
    # 清理：确保runner停止
    if h.runner._running:
        h.runner.stop()


# ==================== TestU32ForeverCliHelper ====================

class TestU32ForeverCliHelper:
    """测试ForeverCliHelper基础功能"""

    def test_helper_initialization(self, helper):
        """测试初始化"""
        assert helper is not None
        assert helper.runner is not None
        assert helper._auto_recovery_enabled is True

    def test_enable(self, helper):
        """测试启用永续运行"""
        success = helper.enable()
        assert success is True
        assert helper.is_enabled() is True

    def test_enable_with_intervals(self, helper):
        """测试自定义间隔启用"""
        success = helper.enable(health_check_interval=60.0, resource_cleanup_interval=600.0)
        assert success is True
        assert helper.runner._health_interval == 60.0
        assert helper.runner._cleanup_interval == 600.0

    def test_enable_already_running(self, helper):
        """测试已启用时再启用"""
        helper.enable()
        success = helper.enable()
        assert success is False

    def test_disable(self, helper):
        """测试禁用永续运行"""
        helper.enable()
        success = helper.disable()
        assert success is True
        assert helper.is_enabled() is False

    def test_disable_not_running(self, helper):
        """测试未启用时禁用"""
        success = helper.disable()
        assert success is False

    def test_is_enabled_default_false(self, helper):
        """测试默认未启用"""
        assert helper.is_enabled() is False

    def test_get_health_report(self, helper):
        """测试获取健康报告"""
        report = helper.get_health_report()
        assert "level" in report
        assert "score" in report
        assert "issues" in report
        assert "timestamp" in report

    def test_get_health_report_no_agent(self, helper):
        """测试无Agent时的健康报告"""
        report = helper.get_health_report()
        # 无Agent时应该是warning级别
        assert report["level"] == "warning"
        assert "未附加Agent" in report["issues"]

    def test_get_health_history_empty(self, helper):
        """测试空健康历史"""
        history = helper.get_health_history()
        assert history == []

    def test_get_health_history_with_data(self, helper):
        """测试有数据的健康历史"""
        # 手动触发健康检查并记录到历史
        report = helper.runner.check_health()
        helper.runner._health_history.append(report)
        history = helper.get_health_history()
        assert len(history) >= 1

    def test_get_health_history_limit(self, helper):
        """测试健康历史限制"""
        for _ in range(15):
            report = helper.runner.check_health()
            helper.runner._health_history.append(report)
        history = helper.get_health_history(limit=5)
        assert len(history) <= 5

    def test_trigger_self_heal(self, helper):
        """测试触发自愈"""
        report = helper.trigger_self_heal()
        assert "level" in report
        assert helper.runner._self_heal_count >= 1

    def test_trigger_cleanup(self, helper):
        """测试触发清理"""
        initial_count = helper.runner._cleanup_count
        cleanup_count = helper.trigger_cleanup()
        assert cleanup_count == initial_count + 1

    def test_get_stats(self, helper):
        """测试获取统计"""
        stats = helper.get_stats()
        assert "running" in stats
        assert "uptime_seconds" in stats
        assert "restart_count" in stats
        assert "self_heal_count" in stats
        assert "cleanup_count" in stats
        assert "auto_recovery_enabled" in stats

    def test_get_stats_auto_recovery(self, helper):
        """测试统计中包含自动恢复状态"""
        stats = helper.get_stats()
        assert stats["auto_recovery_enabled"] is True

    def test_set_auto_recovery(self, helper):
        """测试设置自动恢复"""
        helper.set_auto_recovery(False)
        assert helper.is_auto_recovery_enabled() is False
        helper.set_auto_recovery(True)
        assert helper.is_auto_recovery_enabled() is True

    def test_is_auto_recovery_enabled_default(self, helper):
        """测试默认自动恢复启用"""
        assert helper.is_auto_recovery_enabled() is True


# ==================== TestU32FormatStats ====================

class TestU32FormatStats:
    """测试format_stats格式化"""

    def test_format_stats(self, helper):
        """测试统计格式化"""
        result = helper.format_stats()
        assert "永续运行统计" in result
        assert "运行状态" in result
        assert "运行时长" in result
        assert "重启次数" in result
        assert "自愈次数" in result
        assert "清理次数" in result
        assert "自动恢复" in result

    def test_format_stats_running(self, helper):
        """测试运行中统计"""
        helper.enable()
        result = helper.format_stats()
        assert "运行中" in result

    def test_format_stats_stopped(self, helper):
        """测试停止状态统计"""
        result = helper.format_stats()
        assert "已停止" in result

    def test_format_stats_auto_recovery_enabled(self, helper):
        """测试自动恢复启用"""
        result = helper.format_stats()
        assert "启用" in result

    def test_format_stats_auto_recovery_disabled(self, helper):
        """测试自动恢复禁用"""
        helper.set_auto_recovery(False)
        result = helper.format_stats()
        assert "禁用" in result


# ==================== TestU32FormatHealthReport ====================

class TestU32FormatHealthReport:
    """测试format_health_report格式化"""

    def test_format_health_report(self, helper):
        """测试健康报告格式化"""
        result = helper.format_health_report()
        assert "健康报告" in result
        assert "健康等级" in result
        assert "健康评分" in result

    def test_format_health_report_no_agent(self, helper):
        """测试无Agent健康报告"""
        result = helper.format_health_report()
        assert "警告" in result
        assert "未附加Agent" in result

    def test_format_health_report_has_issues(self, helper):
        """测试健康报告包含问题"""
        result = helper.format_health_report()
        assert "问题" in result


# ==================== TestU32FormatHistory ====================

class TestU32FormatHistory:
    """测试format_history格式化"""

    def test_format_history_empty(self, helper):
        """测试空历史格式化"""
        result = helper.format_history()
        assert "无健康历史记录" in result

    def test_format_history_with_data(self, helper):
        """测试有数据的历史格式化"""
        report = helper.runner.check_health()
        helper.runner._health_history.append(report)
        result = helper.format_history()
        assert "健康历史" in result
        assert "警告" in result  # 无Agent时为warning

    def test_format_history_with_limit(self, helper):
        """测试限制条数"""
        for _ in range(5):
            report = helper.runner.check_health()
            helper.runner._health_history.append(report)
        result = helper.format_history(limit=3)
        assert "健康历史" in result


# ==================== TestU32FormatEnableDisable ====================

class TestU32FormatEnableDisable:
    """测试启用/禁用结果格式化"""

    def test_format_enable_result_success(self, helper):
        """测试启用成功格式化"""
        result = helper.format_enable_result(True, 30.0, 300.0)
        assert "永续运行已启用" in result
        assert "30.0" in result
        assert "300.0" in result

    def test_format_enable_result_already_running(self, helper):
        """测试已启用格式化"""
        result = helper.format_enable_result(False, 30.0, 300.0)
        assert "已在运行中" in result

    def test_format_disable_result_success(self, helper):
        """测试禁用成功格式化"""
        result = helper.format_disable_result(True)
        assert "永续运行已禁用" in result

    def test_format_disable_result_not_running(self, helper):
        """测试未启用禁用格式化"""
        result = helper.format_disable_result(False)
        assert "未启用" in result


# ==================== TestU32FormatSelfHealAndCleanup ====================

class TestU32FormatSelfHealAndCleanup:
    """测试自愈/清理结果格式化"""

    def test_format_self_heal_result(self, helper):
        """测试自愈结果格式化"""
        report = helper.get_health_report()
        result = helper.format_self_heal_result(report)
        assert "自愈已触发" in result
        assert "自愈前健康等级" in result

    def test_format_cleanup_result(self, helper):
        """测试清理结果格式化"""
        result = helper.format_cleanup_result(5)
        assert "资源回收已触发" in result
        assert "5" in result

    def test_format_self_heal_result_with_issues(self, helper):
        """测试带问题的自愈结果"""
        report = helper.get_health_report()
        result = helper.format_self_heal_result(report)
        assert "未附加Agent" in result


# ==================== TestU32FormatAutoRecovery ====================

class TestU32FormatAutoRecovery:
    """测试自动恢复格式化"""

    def test_format_auto_recovery_enabled(self, helper):
        """测试自动恢复启用状态"""
        result = helper.format_auto_recovery()
        assert "自动恢复状态" in result
        assert "启用" in result

    def test_format_auto_recovery_disabled(self, helper):
        """测试自动恢复禁用状态"""
        helper.set_auto_recovery(False)
        result = helper.format_auto_recovery()
        assert "禁用" in result

    def test_format_auto_recovery_set_true(self, helper):
        """测试设置自动恢复启用"""
        result = helper.format_auto_recovery_set(True)
        assert "自动恢复已设置" in result
        assert "启用" in result

    def test_format_auto_recovery_set_false(self, helper):
        """测试设置自动恢复禁用"""
        result = helper.format_auto_recovery_set(False)
        assert "自动恢复已设置" in result
        assert "禁用" in result


# ==================== TestU32FormatUptime ====================

class TestU32FormatUptime:
    """测试运行时长格式化"""

    def test_format_uptime_zero(self):
        """测试0秒"""
        from core.forever_cli import ForeverCliHelper
        assert ForeverCliHelper._format_uptime(0) == "0秒"

    def test_format_uptime_seconds(self):
        """测试秒"""
        from core.forever_cli import ForeverCliHelper
        assert "45秒" in ForeverCliHelper._format_uptime(45)

    def test_format_uptime_minutes(self):
        """测试分钟"""
        from core.forever_cli import ForeverCliHelper
        result = ForeverCliHelper._format_uptime(125)  # 2分5秒
        assert "2分钟" in result
        assert "5秒" in result

    def test_format_uptime_hours(self):
        """测试小时"""
        from core.forever_cli import ForeverCliHelper
        result = ForeverCliHelper._format_uptime(3725)  # 1小时2分5秒
        assert "1小时" in result

    def test_format_uptime_days(self):
        """测试天"""
        from core.forever_cli import ForeverCliHelper
        result = ForeverCliHelper._format_uptime(90061)  # 1天1小时1分1秒
        assert "1天" in result


# ==================== TestU32CliIntegration ====================

class TestU32CliIntegration:
    """测试CLI集成"""

    def test_forever_parser_exists(self):
        """测试forever子命令解析器存在"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever"])
        assert args.command == "forever"

    def test_forever_enable_parser(self):
        """测试forever enable parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "enable"])
        assert args.forever_action == "enable"
        assert args.health_interval == 30.0
        assert args.cleanup_interval == 300.0

    def test_forever_enable_parser_with_intervals(self):
        """测试forever enable带间隔"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "enable", "--health-interval", "60", "--cleanup-interval", "600"])
        assert args.health_interval == 60.0
        assert args.cleanup_interval == 600.0

    def test_forever_disable_parser(self):
        """测试forever disable parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "disable"])
        assert args.forever_action == "disable"

    def test_forever_stats_parser(self):
        """测试forever stats parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "stats"])
        assert args.forever_action == "stats"

    def test_forever_health_parser(self):
        """测试forever health parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "health"])
        assert args.forever_action == "health"

    def test_forever_history_parser(self):
        """测试forever history parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "history"])
        assert args.forever_action == "history"
        assert args.limit == 10

    def test_forever_history_parser_with_limit(self):
        """测试forever history带limit"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "history", "--limit", "20"])
        assert args.limit == 20

    def test_forever_self_heal_parser(self):
        """测试forever self-heal parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "self-heal"])
        assert args.forever_action == "self-heal"

    def test_forever_cleanup_parser(self):
        """测试forever cleanup parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "cleanup"])
        assert args.forever_action == "cleanup"

    def test_forever_auto_recovery_parser(self):
        """测试forever auto-recovery parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "auto-recovery"])
        assert args.forever_action == "auto-recovery"

    def test_forever_auto_recovery_enable_parser(self):
        """测试forever auto-recovery --enable"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "auto-recovery", "--enable"])
        assert args.enable is True

    def test_forever_auto_recovery_disable_parser(self):
        """测试forever auto-recovery --disable"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "auto-recovery", "--disable"])
        assert args.disable is True

    def test_cmd_forever_stats(self, capsys):
        """测试_cmd_forever stats"""
        from cli import _cmd_forever, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "stats"])
        ret = _cmd_forever(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "永续运行统计" in out

    def test_cmd_forever_health(self, capsys):
        """测试_cmd_forever health"""
        from cli import _cmd_forever, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "health"])
        ret = _cmd_forever(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "健康报告" in out

    def test_cmd_forever_history_empty(self, capsys):
        """测试_cmd_forever history空"""
        from cli import _cmd_forever, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "history"])
        ret = _cmd_forever(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "无健康历史记录" in out

    def test_cmd_forever_self_heal(self, capsys):
        """测试_cmd_forever self-heal"""
        from cli import _cmd_forever, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "self-heal"])
        ret = _cmd_forever(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "自愈已触发" in out

    def test_cmd_forever_cleanup(self, capsys):
        """测试_cmd_forever cleanup"""
        from cli import _cmd_forever, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "cleanup"])
        ret = _cmd_forever(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "资源回收已触发" in out

    def test_cmd_forever_auto_recovery_status(self, capsys):
        """测试_cmd_forever auto-recovery状态"""
        from cli import _cmd_forever, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "auto-recovery"])
        ret = _cmd_forever(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "自动恢复状态" in out

    def test_cmd_forever_auto_recovery_enable(self, capsys):
        """测试_cmd_forever auto-recovery --enable"""
        from cli import _cmd_forever, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "auto-recovery", "--enable"])
        ret = _cmd_forever(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "自动恢复已设置" in out

    def test_cmd_forever_auto_recovery_disable(self, capsys):
        """测试_cmd_forever auto-recovery --disable"""
        from cli import _cmd_forever, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever", "auto-recovery", "--disable"])
        ret = _cmd_forever(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "自动恢复已设置" in out

    def test_cmd_forever_enable_disable(self, capsys):
        """测试_cmd_forever enable + disable（同一helper实例）"""
        from cli import _cmd_forever, _create_parser
        from core.forever_cli import ForeverCliHelper
        parser = _create_parser()
        helper = ForeverCliHelper()
        try:
            # enable
            args = parser.parse_args(["forever", "enable"])
            success = helper.enable(args.health_interval, args.cleanup_interval)
            out = helper.format_enable_result(success, args.health_interval, args.cleanup_interval)
            assert "永续运行已启用" in out
            # disable
            success = helper.disable()
            out = helper.format_disable_result(success)
            assert "永续运行已禁用" in out
        finally:
            if helper.runner._running:
                helper.runner.stop()

    def test_cmd_forever_no_action(self, capsys):
        """测试_cmd_forever无操作"""
        from cli import _cmd_forever, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["forever"])
        ret = _cmd_forever(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "请指定操作" in out

    def test_cli_main_forever_dispatch(self, capsys):
        """测试cli_main分发forever命令"""
        from cli import cli_main
        ret = cli_main(["forever", "stats"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "永续运行统计" in out


# ==================== TestU32NoRegression ====================

class TestU32NoRegression:
    """验证U32不破坏既有功能"""

    def test_u27_export_still_works(self):
        """U27 export_cli模块仍可导入"""
        from core.export_cli import ExportCliHelper
        assert ExportCliHelper is not None

    def test_u28_plugins_still_works(self):
        """U28 plugins_cli模块仍可导入"""
        from core.plugins_cli import PluginsCliHelper
        assert PluginsCliHelper is not None

    def test_u29_skill_still_works(self):
        """U29 skill_cli模块仍可导入"""
        from core.skill_cli import SkillCliHelper
        assert SkillCliHelper is not None

    def test_u30_long_task_still_works(self):
        """U30 long_task_cli模块仍可导入"""
        from core.long_task_cli import LongTaskCliHelper
        assert LongTaskCliHelper is not None

    def test_u31_workflow_still_works(self):
        """U31 workflow_cli模块仍可导入"""
        from core.workflow_cli import WorkflowCliHelper
        assert WorkflowCliHelper is not None

    def test_forever_cli_module_importable(self):
        """U32 forever_cli模块可导入"""
        from core.forever_cli import ForeverCliHelper
        assert ForeverCliHelper is not None

    def test_forever_subcommand_in_help(self, capsys):
        """forever子命令出现在帮助中"""
        from cli import _create_parser
        parser = _create_parser()
        parser.print_help()
        out = capsys.readouterr().out
        assert "forever" in out
