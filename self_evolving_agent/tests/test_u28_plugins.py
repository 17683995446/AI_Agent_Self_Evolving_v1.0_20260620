"""
U28：插件管理 —— 测试

测试覆盖：
1. PluginsCliHelper 的所有方法
2. list_plugins / get_plugin_info / get_plugin_usage 数据查询
3. format_list / format_info / format_usage / format_types 格式化
4. CLI集成（_cmd_plugins 分发）
5. 无回归（U23-U27 仍可用）
"""

import os
import sys
import pytest

# 路径设置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def helper():
    """创建PluginsCliHelper"""
    from core.plugins_cli import PluginsCliHelper
    return PluginsCliHelper()


# ==================== TestU28PluginsCliHelper ====================

class TestU28PluginsCliHelper:
    """测试PluginsCliHelper基础功能"""

    def test_helper_initialization(self, helper):
        """测试初始化"""
        assert helper is not None

    def test_registry_lazy_loading(self, helper):
        """测试注册表延迟加载"""
        assert helper._registry is None
        reg = helper.registry
        assert reg is not None
        assert helper._registry is not None

    def test_list_module_types(self, helper):
        """测试列出模块类型"""
        types = helper.list_module_types()
        assert "perception" in types
        assert "prediction" in types
        assert "drive_engine" in types
        assert "decision" in types
        assert "error_eval" in types
        assert "optimizer" in types
        assert "memory" in types

    def test_list_plugins_all(self, helper):
        """测试列出所有插件"""
        plugins = helper.list_plugins()
        assert isinstance(plugins, dict)
        # 至少应该有一些模块类型
        assert len(plugins) > 0

    def test_list_plugins_by_type(self, helper):
        """测试按类型列出插件"""
        plugins = helper.list_plugins("perception")
        assert "perception" in plugins
        # perception可能加载成功也可能失败（取决于依赖）
        assert isinstance(plugins["perception"], list)

    def test_list_plugins_invalid_type(self, helper):
        """测试无效模块类型"""
        plugins = helper.list_plugins("invalid_type")
        assert "invalid_type" in plugins
        assert plugins["invalid_type"] == []

    def test_get_load_errors(self, helper):
        """测试获取加载错误"""
        # 触发registry加载
        _ = helper.registry
        errors = helper.get_load_errors()
        assert isinstance(errors, list)


# ==================== TestU28GetPluginInfo ====================

class TestU28GetPluginInfo:
    """测试get_plugin_info"""

    def test_get_plugin_info_existing(self, helper):
        """测试获取存在的插件信息"""
        # 先找出一个存在的插件
        plugins = helper.list_plugins()
        found = False
        for mt, names in plugins.items():
            if names:
                info = helper.get_plugin_info(mt, names[0])
                assert info is not None
                assert info["module_type"] == mt
                assert info["name"] == names[0]
                assert "class_name" in info
                assert "module_path" in info
                assert "docstring" in info
                found = True
                break
        if not found:
            pytest.skip("无可用插件测试")

    def test_get_plugin_info_nonexistent(self, helper):
        """测试获取不存在的插件信息"""
        info = helper.get_plugin_info("perception", "nonexistent")
        assert info is None

    def test_get_plugin_info_invalid_type(self, helper):
        """测试无效模块类型"""
        info = helper.get_plugin_info("invalid_type", "default")
        assert info is None

    def test_get_plugin_info_default_mark(self, helper):
        """测试默认插件标记"""
        plugins = helper.list_plugins("perception")
        if "default" in plugins.get("perception", []):
            info = helper.get_plugin_info("perception", "default")
            assert info is not None
            assert info["is_default"] is True


# ==================== TestU28GetPluginUsage ====================

class TestU28GetPluginUsage:
    """测试get_plugin_usage"""

    def test_get_plugin_usage_existing(self, helper):
        """测试获取存在的插件使用方式"""
        plugins = helper.list_plugins()
        found = False
        for mt, names in plugins.items():
            if names:
                usage = helper.get_plugin_usage(mt, names[0])
                assert usage is not None
                assert "config_key" in usage
                assert "config_example" in usage
                assert "python_example" in usage
                assert "env_var" in usage
                found = True
                break
        if not found:
            pytest.skip("无可用插件测试")

    def test_get_plugin_usage_nonexistent(self, helper):
        """测试获取不存在的插件使用方式"""
        usage = helper.get_plugin_usage("perception", "nonexistent")
        assert usage is None

    def test_python_example_contains_module_config(self, helper):
        """测试Python示例包含module_config"""
        plugins = helper.list_plugins()
        found = False
        for mt, names in plugins.items():
            if names:
                usage = helper.get_plugin_usage(mt, names[0])
                assert "module_config" in usage["python_example"]
                found = True
                break
        if not found:
            pytest.skip("无可用插件测试")


# ==================== TestU28FormatList ====================

class TestU28FormatList:
    """测试format_list格式化"""

    def test_format_list_all(self, helper):
        """测试格式化所有插件列表"""
        result = helper.format_list()
        assert "插件列表" in result
        # 应该包含所有模块类型
        assert "感知" in result or "perception" in result

    def test_format_list_by_type(self, helper):
        """测试按类型格式化插件列表"""
        result = helper.format_list("perception")
        assert "插件列表" in result
        assert "perception" in result

    def test_format_list_empty_type(self, helper):
        """测试空类型的格式化"""
        result = helper.format_list("invalid_type")
        assert "无可用插件" in result


# ==================== TestU28FormatInfo ====================

class TestU28FormatInfo:
    """测试format_info格式化"""

    def test_format_info_existing(self, helper):
        """测试格式化存在的插件详情"""
        plugins = helper.list_plugins()
        found = False
        for mt, names in plugins.items():
            if names:
                result = helper.format_info(mt, names[0])
                assert "插件详情" in result
                assert "类名" in result
                assert "模块路径" in result
                found = True
                break
        if not found:
            pytest.skip("无可用插件测试")

    def test_format_info_nonexistent(self, helper):
        """测试格式化不存在的插件详情"""
        result = helper.format_info("perception", "nonexistent")
        assert "不存在" in result


# ==================== TestU28FormatUsage ====================

class TestU28FormatUsage:
    """测试format_usage格式化"""

    def test_format_usage_existing(self, helper):
        """测试格式化存在的插件使用方式"""
        plugins = helper.list_plugins()
        found = False
        for mt, names in plugins.items():
            if names:
                result = helper.format_usage(mt, names[0])
                assert "插件使用" in result
                assert "配置键" in result
                assert "配置示例" in result
                assert "Python示例" in result
                found = True
                break
        if not found:
            pytest.skip("无可用插件测试")

    def test_format_usage_nonexistent(self, helper):
        """测试格式化不存在的插件使用方式"""
        result = helper.format_usage("perception", "nonexistent")
        assert "不存在" in result


# ==================== TestU28FormatTypes ====================

class TestU28FormatTypes:
    """测试format_types格式化"""

    def test_format_types(self, helper):
        """测试格式化模块类型列表"""
        result = helper.format_types()
        assert "模块类型" in result
        assert "perception" in result
        assert "prediction" in result
        assert "感知" in result
        assert "预测" in result


# ==================== TestU28CliIntegration ====================

class TestU28CliIntegration:
    """测试CLI集成"""

    def test_plugins_parser_exists(self):
        """测试plugins子命令parser存在"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["plugins"])
        assert args.command == "plugins"

    def test_plugins_list_parser(self):
        """测试plugins list parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["plugins", "list"])
        assert args.plugins_action == "list"
        assert args.module_type is None

    def test_plugins_list_with_type_parser(self):
        """测试plugins list带模块类型"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["plugins", "list", "prediction"])
        assert args.plugins_action == "list"
        assert args.module_type == "prediction"

    def test_plugins_info_parser(self):
        """测试plugins info parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["plugins", "info", "prediction", "enhanced"])
        assert args.plugins_action == "info"
        assert args.module_type == "prediction"
        assert args.name == "enhanced"

    def test_plugins_use_parser(self):
        """测试plugins use parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["plugins", "use", "prediction", "enhanced"])
        assert args.plugins_action == "use"
        assert args.module_type == "prediction"
        assert args.name == "enhanced"

    def test_plugins_types_parser(self):
        """测试plugins types parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["plugins", "types"])
        assert args.plugins_action == "types"

    def test_cmd_plugins_list(self, capsys):
        """测试_cmd_plugins list"""
        from cli import _cmd_plugins, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["plugins", "list"])
        ret = _cmd_plugins(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "插件列表" in out

    def test_cmd_plugins_list_with_type(self, capsys):
        """测试_cmd_plugins list带类型"""
        from cli import _cmd_plugins, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["plugins", "list", "perception"])
        ret = _cmd_plugins(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "插件列表" in out

    def test_cmd_plugins_types(self, capsys):
        """测试_cmd_plugins types"""
        from cli import _cmd_plugins, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["plugins", "types"])
        ret = _cmd_plugins(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "模块类型" in out

    def test_cmd_plugins_info_nonexistent(self, capsys):
        """测试_cmd_plugins info不存在的插件"""
        from cli import _cmd_plugins, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["plugins", "info", "perception", "nonexistent"])
        ret = _cmd_plugins(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "不存在" in out

    def test_cmd_plugins_use_nonexistent(self, capsys):
        """测试_cmd_plugins use不存在的插件"""
        from cli import _cmd_plugins, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["plugins", "use", "perception", "nonexistent"])
        ret = _cmd_plugins(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "不存在" in out

    def test_cmd_plugins_no_action(self, capsys):
        """测试_cmd_plugins无操作"""
        from cli import _cmd_plugins, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["plugins"])
        ret = _cmd_plugins(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "请指定操作" in out

    def test_cli_main_plugins_dispatch(self, capsys):
        """测试cli_main分发plugins命令"""
        from cli import cli_main
        ret = cli_main(["plugins", "types"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "模块类型" in out


# ==================== TestU28NoRegression ====================

class TestU28NoRegression:
    """验证U28不破坏既有功能"""

    def test_u23_alerts_still_works(self):
        """U23 alert_cli模块仍可导入"""
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        assert helper is not None

    def test_u24_perf_still_works(self):
        """U24 perf_cli模块仍可导入"""
        from core.perf_cli import PerfCliHelper
        helper = PerfCliHelper()
        assert helper is not None

    def test_u25_agents_still_works(self):
        """U25 agents_cli模块仍可导入"""
        from core.agents_cli import AgentsCliHelper
        assert AgentsCliHelper is not None

    def test_u26_snapshot_still_works(self):
        """U26 snapshot_cli模块仍可导入"""
        from core.snapshot_cli import SnapshotCliHelper
        assert SnapshotCliHelper is not None

    def test_u27_export_still_works(self):
        """U27 export_cli模块仍可导入"""
        from core.export_cli import ExportCliHelper
        helper = ExportCliHelper()
        assert helper is not None

    def test_existing_commands_still_dispatch(self):
        """既有命令仍可正常分发"""
        from cli import _create_parser
        parser = _create_parser()
        for cmd in ["run", "status", "snapshot", "logs", "alerts", "perf", "agents", "export", "plugins"]:
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_registry_unchanged(self):
        """ModuleRegistry核心API未变"""
        from core.registry import ModuleRegistry
        reg = ModuleRegistry()
        assert hasattr(reg, "register_perception")
        assert hasattr(reg, "register_prediction")
        assert hasattr(reg, "create_perception")
        assert hasattr(reg, "list_predictions")
        assert hasattr(reg, "is_registered")
