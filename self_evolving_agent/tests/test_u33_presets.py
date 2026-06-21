"""
U33：预设与场景发现 —— 测试

测试覆盖：
1. PresetsCliHelper 的所有方法
2. ScenariosCliHelper 的所有方法
3. CLI集成（_cmd_presets / _cmd_scenarios 分发）
4. 无回归（U28-U32 仍可用）
"""

import os
import sys
import pytest

# 路径设置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def presets_helper():
    """创建PresetsCliHelper"""
    from core.presets_cli import PresetsCliHelper
    return PresetsCliHelper()


@pytest.fixture
def scenarios_helper():
    """创建ScenariosCliHelper"""
    from core.presets_cli import ScenariosCliHelper
    return ScenariosCliHelper()


# ==================== TestU33PresetsCliHelper ====================

class TestU33PresetsCliHelper:
    """测试PresetsCliHelper基础功能"""

    def test_helper_initialization(self, presets_helper):
        """测试初始化"""
        assert presets_helper is not None
        assert len(presets_helper._presets) >= 4

    def test_list_presets(self, presets_helper):
        """测试列出预设"""
        presets = presets_helper.list_presets()
        assert len(presets) >= 4
        names = [p["name"] for p in presets]
        assert "balanced" in names
        assert "aggressive" in names
        assert "conservative" in names
        assert "eco" in names

    def test_list_presets_fields(self, presets_helper):
        """测试预设字段完整"""
        presets = presets_helper.list_presets()
        for p in presets:
            assert "name" in p
            assert "label" in p
            assert "description" in p

    def test_get_preset_info(self, presets_helper):
        """测试获取预设详情"""
        info = presets_helper.get_preset_info("balanced")
        assert info is not None
        assert info["name"] == "balanced"
        assert "config" in info

    def test_get_preset_info_nonexistent(self, presets_helper):
        """测试获取不存在的预设"""
        info = presets_helper.get_preset_info("nonexistent")
        assert info is None

    def test_get_preset_info_config(self, presets_helper):
        """测试预设配置内容"""
        info = presets_helper.get_preset_info("aggressive")
        assert "resource" in info["config"]
        assert "drive_weights" in info["config"]
        assert "evolution" in info["config"]
        assert "main_loop_frequency" in info["config"]

    def test_apply_preset(self, presets_helper):
        """测试应用预设"""
        success = presets_helper.apply_preset("balanced")
        assert success is True

    def test_apply_preset_nonexistent(self, presets_helper):
        """测试应用不存在的预设"""
        success = presets_helper.apply_preset("nonexistent")
        assert success is False

    def test_get_available_names(self, presets_helper):
        """测试获取可用名称"""
        names = presets_helper.get_available_names()
        assert "balanced" in names
        assert "aggressive" in names
        assert len(names) >= 4


# ==================== TestU33PresetsFormat ====================

class TestU33PresetsFormat:
    """测试PresetsCliHelper格式化"""

    def test_format_list(self, presets_helper):
        """测试列表格式化"""
        result = presets_helper.format_list()
        assert "配置预设" in result
        assert "balanced" in result
        assert "aggressive" in result

    def test_format_info(self, presets_helper):
        """测试详情格式化"""
        result = presets_helper.format_info("balanced")
        assert "预设详情" in result
        assert "balanced" in result
        assert "资源约束" in result
        assert "驱力权重" in result

    def test_format_info_nonexistent(self, presets_helper):
        """测试不存在预设详情"""
        result = presets_helper.format_info("nonexistent")
        assert "不存在" in result

    def test_format_apply_result_success(self, presets_helper):
        """测试应用成功格式化"""
        result = presets_helper.format_apply_result("balanced", True)
        assert "已应用" in result
        assert "balanced" in result

    def test_format_apply_result_failure(self, presets_helper):
        """测试应用失败格式化"""
        result = presets_helper.format_apply_result("nonexistent", False)
        assert "失败" in result


# ==================== TestU33ScenariosCliHelper ====================

class TestU33ScenariosCliHelper:
    """测试ScenariosCliHelper基础功能"""

    def test_helper_initialization(self, scenarios_helper):
        """测试初始化"""
        assert scenarios_helper is not None
        assert scenarios_helper.loader is not None

    def test_list_scenarios(self, scenarios_helper):
        """测试列出场景"""
        scenarios = scenarios_helper.list_scenarios()
        assert len(scenarios) >= 1
        # 至少应有 system_monitor
        assert "system_monitor" in scenarios

    def test_get_scenario_info(self, scenarios_helper):
        """测试获取场景信息"""
        info = scenarios_helper.get_scenario_info("system_monitor")
        assert info is not None
        assert "description" in info

    def test_get_scenario_info_nonexistent(self, scenarios_helper):
        """测试获取不存在的场景"""
        info = scenarios_helper.get_scenario_info("nonexistent")
        assert info is None

    def test_get_scenario_info_has_modules(self, scenarios_helper):
        """测试场景信息包含模块配置"""
        info = scenarios_helper.get_scenario_info("system_monitor")
        if info:
            assert "modules" in info or "config_overrides" in info


# ==================== TestU33ScenariosFormat ====================

class TestU33ScenariosFormat:
    """测试ScenariosCliHelper格式化"""

    def test_format_list(self, scenarios_helper):
        """测试列表格式化"""
        result = scenarios_helper.format_list()
        assert "运行场景" in result
        assert "system_monitor" in result

    def test_format_info(self, scenarios_helper):
        """测试详情格式化"""
        result = scenarios_helper.format_info("system_monitor")
        assert "场景详情" in result
        assert "system_monitor" in result

    def test_format_info_nonexistent(self, scenarios_helper):
        """测试不存在场景详情"""
        result = scenarios_helper.format_info("nonexistent")
        assert "不存在" in result


# ==================== TestU33CliIntegration ====================

class TestU33CliIntegration:
    """测试CLI集成"""

    def test_presets_parser_exists(self):
        """测试presets子命令解析器存在"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["presets"])
        assert args.command == "presets"

    def test_presets_list_parser(self):
        """测试presets list parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["presets", "list"])
        assert args.presets_action == "list"

    def test_presets_show_parser(self):
        """测试presets show parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["presets", "show", "balanced"])
        assert args.presets_action == "show"
        assert args.name == "balanced"

    def test_presets_apply_parser(self):
        """测试presets apply parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["presets", "apply", "aggressive"])
        assert args.presets_action == "apply"
        assert args.name == "aggressive"

    def test_scenarios_parser_exists(self):
        """测试scenarios子命令解析器存在"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["scenarios"])
        assert args.command == "scenarios"

    def test_scenarios_list_parser(self):
        """测试scenarios list parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["scenarios", "list"])
        assert args.scenarios_action == "list"

    def test_scenarios_info_parser(self):
        """测试scenarios info parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["scenarios", "info", "system_monitor"])
        assert args.scenarios_action == "info"
        assert args.name == "system_monitor"

    def test_cmd_presets_list(self, capsys):
        """测试_cmd_presets list"""
        from cli import _cmd_presets, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["presets", "list"])
        ret = _cmd_presets(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "配置预设" in out

    def test_cmd_presets_show(self, capsys):
        """测试_cmd_presets show"""
        from cli import _cmd_presets, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["presets", "show", "balanced"])
        ret = _cmd_presets(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "预设详情" in out
        assert "balanced" in out

    def test_cmd_presets_show_nonexistent(self, capsys):
        """测试_cmd_presets show不存在"""
        from cli import _cmd_presets, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["presets", "show", "nonexistent"])
        ret = _cmd_presets(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "不存在" in out

    def test_cmd_presets_apply(self, capsys):
        """测试_cmd_presets apply"""
        from cli import _cmd_presets, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["presets", "apply", "balanced"])
        ret = _cmd_presets(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "已应用" in out

    def test_cmd_presets_apply_nonexistent(self, capsys):
        """测试_cmd_presets apply不存在"""
        from cli import _cmd_presets, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["presets", "apply", "nonexistent"])
        ret = _cmd_presets(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "失败" in out

    def test_cmd_presets_no_action(self, capsys):
        """测试_cmd_presets无操作"""
        from cli import _cmd_presets, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["presets"])
        ret = _cmd_presets(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "请指定操作" in out

    def test_cmd_scenarios_list(self, capsys):
        """测试_cmd_scenarios list"""
        from cli import _cmd_scenarios, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["scenarios", "list"])
        ret = _cmd_scenarios(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "运行场景" in out

    def test_cmd_scenarios_info(self, capsys):
        """测试_cmd_scenarios info"""
        from cli import _cmd_scenarios, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["scenarios", "info", "system_monitor"])
        ret = _cmd_scenarios(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "场景详情" in out

    def test_cmd_scenarios_info_nonexistent(self, capsys):
        """测试_cmd_scenarios info不存在"""
        from cli import _cmd_scenarios, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["scenarios", "info", "nonexistent"])
        ret = _cmd_scenarios(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "不存在" in out

    def test_cmd_scenarios_no_action(self, capsys):
        """测试_cmd_scenarios无操作"""
        from cli import _cmd_scenarios, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["scenarios"])
        ret = _cmd_scenarios(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "请指定操作" in out

    def test_cli_main_presets_dispatch(self, capsys):
        """测试cli_main分发presets命令"""
        from cli import cli_main
        ret = cli_main(["presets", "list"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "配置预设" in out

    def test_cli_main_scenarios_dispatch(self, capsys):
        """测试cli_main分发scenarios命令"""
        from cli import cli_main
        ret = cli_main(["scenarios", "list"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "运行场景" in out


# ==================== TestU33NoRegression ====================

class TestU33NoRegression:
    """验证U33不破坏既有功能"""

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

    def test_u32_forever_still_works(self):
        """U32 forever_cli模块仍可导入"""
        from core.forever_cli import ForeverCliHelper
        assert ForeverCliHelper is not None

    def test_presets_cli_module_importable(self):
        """U33 presets_cli模块可导入"""
        from core.presets_cli import PresetsCliHelper, ScenariosCliHelper
        assert PresetsCliHelper is not None
        assert ScenariosCliHelper is not None

    def test_presets_subcommand_in_help(self, capsys):
        """presets子命令出现在帮助中"""
        from cli import _create_parser
        parser = _create_parser()
        parser.print_help()
        out = capsys.readouterr().out
        assert "presets" in out

    def test_scenarios_subcommand_in_help(self, capsys):
        """scenarios子命令出现在帮助中"""
        from cli import _create_parser
        parser = _create_parser()
        parser.print_help()
        out = capsys.readouterr().out
        assert "scenarios" in out
