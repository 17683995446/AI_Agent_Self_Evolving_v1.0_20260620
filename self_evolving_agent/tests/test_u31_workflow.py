"""
U31：工作流管理 —— 测试

测试覆盖：
1. WorkflowCliHelper 的所有方法
2. submit_goal / get_workflow_status / list_workflows / get_stats
3. format_list / format_status / format_stats / format_submit_result / format_goal_types
4. CLI集成（_cmd_workflow 分发）
5. 无回归（U26-U30 仍可用）
"""

import os
import sys
import pytest

# 路径设置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def helper():
    """创建WorkflowCliHelper"""
    from core.workflow_cli import WorkflowCliHelper
    return WorkflowCliHelper()


@pytest.fixture
def helper_with_workflow(helper):
    """创建带有工作流的helper"""
    workflow_id = helper.submit_goal("monitor_and_alert", "测试工作流")
    return helper, workflow_id


# ==================== TestU31WorkflowCliHelper ====================

class TestU31WorkflowCliHelper:
    """测试WorkflowCliHelper基础功能"""

    def test_helper_initialization(self, helper):
        """测试初始化"""
        assert helper is not None
        assert helper.engine is not None

    def test_submit_goal(self, helper):
        """测试提交目标"""
        workflow_id = helper.submit_goal("monitor_and_alert", "测试")
        assert workflow_id is not None
        assert len(workflow_id) > 0

    def test_submit_goal_with_params(self, helper):
        """测试带参数提交"""
        workflow_id = helper.submit_goal("custom", "测试", {"key": "value"})
        assert workflow_id is not None

    def test_get_workflow_status(self, helper_with_workflow):
        """测试获取工作流状态"""
        helper, workflow_id = helper_with_workflow
        status = helper.get_workflow_status(workflow_id)
        assert status is not None
        assert status["workflow_id"] == workflow_id
        assert "tasks" in status

    def test_get_workflow_status_nonexistent(self, helper):
        """测试获取不存在的工作流"""
        status = helper.get_workflow_status("nonexistent")
        assert status is None

    def test_list_workflows_empty(self, helper):
        """测试空工作流列表"""
        workflows = helper.list_workflows()
        assert workflows == []

    def test_list_workflows_with_data(self, helper_with_workflow):
        """测试有数据的工作流列表"""
        helper, workflow_id = helper_with_workflow
        workflows = helper.list_workflows()
        assert len(workflows) == 1
        assert workflows[0]["workflow_id"] == workflow_id

    def test_get_stats(self, helper_with_workflow):
        """测试获取统计"""
        helper, workflow_id = helper_with_workflow
        stats = helper.get_stats()
        assert "total_workflows" in stats or "total" in stats

    def test_get_available_goal_types(self, helper):
        """测试获取可用目标类型"""
        types = helper.get_available_goal_types()
        assert "monitor_and_alert" in types
        assert "optimize_resource" in types
        assert "custom" in types


# ==================== TestU31FormatList ====================

class TestU31FormatList:
    """测试format_list格式化"""

    def test_format_list_empty(self, helper):
        """测试空列表格式化"""
        result = helper.format_list()
        assert "无工作流" in result

    def test_format_list_with_data(self, helper_with_workflow):
        """测试有数据的列表格式化"""
        helper, workflow_id = helper_with_workflow
        result = helper.format_list()
        assert "工作流列表" in result
        assert workflow_id in result


# ==================== TestU31FormatStatus ====================

class TestU31FormatStatus:
    """测试format_status格式化"""

    def test_format_status(self, helper_with_workflow):
        """测试工作流状态格式化"""
        helper, workflow_id = helper_with_workflow
        result = helper.format_status(workflow_id)
        assert "工作流详情" in result
        assert "任务列表" in result

    def test_format_status_nonexistent(self, helper):
        """测试不存在的工作流状态格式化"""
        result = helper.format_status("nonexistent")
        assert "不存在" in result


# ==================== TestU31FormatStats ====================

class TestU31FormatStats:
    """测试format_stats格式化"""

    def test_format_stats(self, helper_with_workflow):
        """测试统计格式化"""
        helper, workflow_id = helper_with_workflow
        result = helper.format_stats()
        assert "工作流引擎统计" in result


# ==================== TestU31FormatSubmitResult ====================

class TestU31FormatSubmitResult:
    """测试format_submit_result格式化"""

    def test_format_submit_result(self, helper):
        """测试提交结果格式化"""
        result = helper.format_submit_result("abc123", "monitor_and_alert")
        assert "提交成功" in result
        assert "abc123" in result
        assert "监控并告警" in result


# ==================== TestU31FormatGoalTypes ====================

class TestU31FormatGoalTypes:
    """测试format_goal_types格式化"""

    def test_format_goal_types(self, helper):
        """测试目标类型列表格式化"""
        result = helper.format_goal_types()
        assert "可用目标类型" in result
        assert "monitor_and_alert" in result


# ==================== TestU31CliIntegration ====================

class TestU31CliIntegration:
    """测试CLI集成"""

    def test_workflow_parser_exists(self):
        """测试workflow子命令parser存在"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["workflow"])
        assert args.command == "workflow"

    def test_workflow_submit_parser(self):
        """测试workflow submit parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["workflow", "submit", "monitor_and_alert"])
        assert args.workflow_action == "submit"
        assert args.goal_type == "monitor_and_alert"

    def test_workflow_submit_with_desc_parser(self):
        """测试workflow submit带描述"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["workflow", "submit", "custom", "--desc", "my_workflow"])
        assert args.desc == "my_workflow"

    def test_workflow_submit_with_params_parser(self):
        """测试workflow submit带参数"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["workflow", "submit", "custom", "--params", '{"k":"v"}'])
        assert args.params == '{"k":"v"}'

    def test_workflow_list_parser(self):
        """测试workflow list parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["workflow", "list"])
        assert args.workflow_action == "list"

    def test_workflow_status_parser(self):
        """测试workflow status parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["workflow", "status", "abc123"])
        assert args.workflow_action == "status"
        assert args.id == "abc123"

    def test_workflow_stats_parser(self):
        """测试workflow stats parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["workflow", "stats"])
        assert args.workflow_action == "stats"

    def test_workflow_goals_parser(self):
        """测试workflow goals parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["workflow", "goals"])
        assert args.workflow_action == "goals"

    def test_cmd_workflow_list_empty(self, capsys):
        """测试_cmd_workflow list空"""
        from cli import _cmd_workflow, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["workflow", "list"])
        ret = _cmd_workflow(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "无工作流" in out

    def test_cmd_workflow_submit(self, capsys):
        """测试_cmd_workflow submit"""
        from cli import _cmd_workflow, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["workflow", "submit", "monitor_and_alert"])
        ret = _cmd_workflow(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "提交成功" in out

    def test_cmd_workflow_submit_invalid_params(self, capsys):
        """测试_cmd_workflow submit无效参数"""
        from cli import _cmd_workflow, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["workflow", "submit", "custom", "--params", "invalid_json"])
        ret = _cmd_workflow(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "JSON解析失败" in out

    def test_cmd_workflow_stats(self, capsys):
        """测试_cmd_workflow stats"""
        from cli import _cmd_workflow, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["workflow", "stats"])
        ret = _cmd_workflow(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "工作流引擎统计" in out

    def test_cmd_workflow_goals(self, capsys):
        """测试_cmd_workflow goals"""
        from cli import _cmd_workflow, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["workflow", "goals"])
        ret = _cmd_workflow(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "可用目标类型" in out

    def test_cmd_workflow_status_nonexistent(self, capsys):
        """测试_cmd_workflow status不存在"""
        from cli import _cmd_workflow, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["workflow", "status", "nonexistent"])
        ret = _cmd_workflow(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "不存在" in out

    def test_cmd_workflow_no_action(self, capsys):
        """测试_cmd_workflow无操作"""
        from cli import _cmd_workflow, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["workflow"])
        ret = _cmd_workflow(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "请指定操作" in out

    def test_cli_main_workflow_dispatch(self, capsys):
        """测试cli_main分发workflow命令"""
        from cli import cli_main
        ret = cli_main(["workflow", "goals"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "可用目标类型" in out


# ==================== TestU31NoRegression ====================

class TestU31NoRegression:
    """验证U31不破坏既有功能"""

    def test_u26_snapshot_still_works(self):
        """U26 snapshot_cli模块仍可导入"""
        from core.snapshot_cli import SnapshotCliHelper
        assert SnapshotCliHelper is not None

    def test_u27_export_still_works(self):
        """U27 export_cli模块仍可导入"""
        from core.export_cli import ExportCliHelper
        helper = ExportCliHelper()
        assert helper is not None

    def test_u28_plugins_still_works(self):
        """U28 plugins_cli模块仍可导入"""
        from core.plugins_cli import PluginsCliHelper
        helper = PluginsCliHelper()
        assert helper is not None

    def test_u29_skill_still_works(self):
        """U29 skill_cli模块仍可导入"""
        from core.skill_cli import SkillCliHelper
        helper = SkillCliHelper()
        assert helper is not None

    def test_u30_long_task_still_works(self):
        """U30 long_task_cli模块仍可导入"""
        from core.long_task_cli import LongTaskCliHelper
        helper = LongTaskCliHelper()
        assert helper is not None

    def test_existing_commands_still_dispatch(self):
        """既有命令仍可正常分发"""
        from cli import _create_parser
        parser = _create_parser()
        for cmd in ["run", "status", "snapshot", "logs", "alerts", "perf",
                     "agents", "export", "plugins", "skill", "long-task", "workflow"]:
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_workflow_engine_unchanged(self):
        """WorkflowEngine核心API未变"""
        from core.workflow_engine import WorkflowEngine, Workflow, WorkflowTask
        engine = WorkflowEngine()
        assert hasattr(engine, "submit_workflow")
        assert hasattr(engine, "get_workflow_status")
        assert hasattr(engine, "get_stats")
        assert hasattr(engine, "start")
        assert hasattr(engine, "stop")
