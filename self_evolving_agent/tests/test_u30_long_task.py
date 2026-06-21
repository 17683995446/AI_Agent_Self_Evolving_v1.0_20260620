"""
U30：长任务管理 —— 测试

测试覆盖：
1. LongTaskCliHelper 的所有方法
2. plan_task / list_tasks / get_task_status / advance_task / pause/resume / get_stats
3. format_list / format_status / format_stats / format_plan_result / format_advance_result / format_templates
4. CLI集成（_cmd_long_task 分发）
5. 无回归（U25-U29 仍可用）
"""

import os
import sys
import pytest

# 路径设置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def helper():
    """创建LongTaskCliHelper"""
    from core.long_task_cli import LongTaskCliHelper
    return LongTaskCliHelper()


@pytest.fixture
def helper_with_task(helper):
    """创建带有任务的helper"""
    result = helper.plan_task("system_optimization", "test_task", "测试任务")
    return helper, result["task_id"]


# ==================== TestU30LongTaskCliHelper ====================

class TestU30LongTaskCliHelper:
    """测试LongTaskCliHelper基础功能"""

    def test_helper_initialization(self, helper):
        """测试初始化"""
        assert helper is not None
        assert helper.planner is not None

    def test_plan_task(self, helper):
        """测试规划任务"""
        result = helper.plan_task("system_optimization", "test", "desc")
        assert "task_id" in result
        assert result["name"] == "test"
        assert len(result["subgoals"]) > 0

    def test_plan_task_custom_template(self, helper):
        """测试自定义模板名"""
        result = helper.plan_task("custom_template", "test", "desc")
        assert "task_id" in result
        # 自定义模板会创建单子目标任务
        assert len(result["subgoals"]) == 1

    def test_list_tasks_empty(self, helper):
        """测试空任务列表"""
        tasks = helper.list_tasks()
        assert tasks == []

    def test_list_tasks_with_data(self, helper_with_task):
        """测试有数据的任务列表"""
        helper, task_id = helper_with_task
        tasks = helper.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == task_id

    def test_get_task_status(self, helper_with_task):
        """测试获取任务状态"""
        helper, task_id = helper_with_task
        status = helper.get_task_status(task_id)
        assert status is not None
        assert status["task_id"] == task_id
        assert "subgoals" in status

    def test_get_task_status_nonexistent(self, helper):
        """测试获取不存在的任务"""
        status = helper.get_task_status("nonexistent")
        assert status is None

    def test_advance_task(self, helper_with_task):
        """测试推进任务"""
        helper, task_id = helper_with_task
        result = helper.advance_task(task_id, 0.8)
        assert "completed" in result or "progress" in result or "error" in result

    def test_advance_task_nonexistent(self, helper):
        """测试推进不存在的任务"""
        result = helper.advance_task("nonexistent", 0.8)
        assert "error" in result

    def test_advance_task_low_drive(self, helper_with_task):
        """测试低驱力推进"""
        helper, task_id = helper_with_task
        result = helper.advance_task(task_id, 0.1)  # 低于默认阈值0.2
        assert result.get("skipped") is True

    def test_pause_task_not_running(self, helper_with_task):
        """测试暂停非运行状态的任务"""
        helper, task_id = helper_with_task
        # 任务处于planned状态，不能暂停
        success = helper.pause_task(task_id)
        assert success is False

    def test_pause_resume_task(self, helper_with_task):
        """测试暂停恢复任务"""
        helper, task_id = helper_with_task
        # 先推进使任务进入running状态
        helper.advance_task(task_id, 0.8)
        # 暂停
        success = helper.pause_task(task_id)
        assert success is True
        # 恢复
        success = helper.resume_task(task_id)
        assert success is True

    def test_resume_task_not_paused(self, helper_with_task):
        """测试恢复非暂停状态的任务"""
        helper, task_id = helper_with_task
        success = helper.resume_task(task_id)
        assert success is False

    def test_get_stats(self, helper_with_task):
        """测试获取统计"""
        helper, task_id = helper_with_task
        stats = helper.get_stats()
        assert stats["total_tasks"] == 1
        assert "templates" in stats

    def test_get_available_templates(self, helper):
        """测试获取可用模板"""
        templates = helper.get_available_templates()
        assert "system_optimization" in templates
        assert "continuous_monitoring" in templates
        assert "fault_recovery" in templates
        assert "capability_enhancement" in templates


# ==================== TestU30FormatList ====================

class TestU30FormatList:
    """测试format_list格式化"""

    def test_format_list_empty(self, helper):
        """测试空列表格式化"""
        result = helper.format_list()
        assert "无长任务" in result

    def test_format_list_with_data(self, helper_with_task):
        """测试有数据的列表格式化"""
        helper, task_id = helper_with_task
        result = helper.format_list()
        assert "长任务列表" in result
        assert task_id in result


# ==================== TestU30FormatStatus ====================

class TestU30FormatStatus:
    """测试format_status格式化"""

    def test_format_status(self, helper_with_task):
        """测试任务状态格式化"""
        helper, task_id = helper_with_task
        result = helper.format_status(task_id)
        assert "长任务详情" in result
        assert "子目标列表" in result

    def test_format_status_nonexistent(self, helper):
        """测试不存在的任务状态格式化"""
        result = helper.format_status("nonexistent")
        assert "不存在" in result


# ==================== TestU30FormatStats ====================

class TestU30FormatStats:
    """测试format_stats格式化"""

    def test_format_stats(self, helper_with_task):
        """测试统计格式化"""
        helper, task_id = helper_with_task
        result = helper.format_stats()
        assert "长任务统计" in result
        assert "总任务数" in result


# ==================== TestU30FormatPlanResult ====================

class TestU30FormatPlanResult:
    """测试format_plan_result格式化"""

    def test_format_plan_result(self, helper):
        """测试规划结果格式化"""
        result = helper.plan_task("system_optimization", "test", "desc")
        formatted = helper.format_plan_result(result)
        assert "规划成功" in formatted
        assert "任务ID" in formatted


# ==================== TestU30FormatAdvanceResult ====================

class TestU30FormatAdvanceResult:
    """测试format_advance_result格式化"""

    def test_format_advance_result_progress(self, helper_with_task):
        """测试推进结果格式化（进度）"""
        helper, task_id = helper_with_task
        result = helper.advance_task(task_id, 0.8)
        formatted = helper.format_advance_result(result)
        assert "任务推进结果" in formatted

    def test_format_advance_result_error(self, helper):
        """测试推进结果格式化（错误）"""
        result = {"error": "任务不存在"}
        formatted = helper.format_advance_result(result)
        assert "错误" in formatted

    def test_format_advance_result_skipped(self, helper):
        """测试推进结果格式化（跳过）"""
        result = {"skipped": True, "reason": "驱力低"}
        formatted = helper.format_advance_result(result)
        assert "跳过" in formatted


# ==================== TestU30FormatTemplates ====================

class TestU30FormatTemplates:
    """测试format_templates格式化"""

    def test_format_templates(self, helper):
        """测试模板列表格式化"""
        result = helper.format_templates()
        assert "可用长任务模板" in result
        assert "system_optimization" in result


# ==================== TestU30CliIntegration ====================

class TestU30CliIntegration:
    """测试CLI集成"""

    def test_longtask_parser_exists(self):
        """测试long-task子命令parser存在"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task"])
        assert args.command == "long-task"

    def test_longtask_plan_parser(self):
        """测试long-task plan parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "plan", "system_optimization"])
        assert args.longtask_action == "plan"
        assert args.template == "system_optimization"

    def test_longtask_plan_with_name_parser(self):
        """测试long-task plan带名称"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "plan", "system_optimization", "--name", "my_task"])
        assert args.name == "my_task"

    def test_longtask_list_parser(self):
        """测试long-task list parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "list"])
        assert args.longtask_action == "list"

    def test_longtask_status_parser(self):
        """测试long-task status parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "status", "abc123"])
        assert args.longtask_action == "status"
        assert args.id == "abc123"

    def test_longtask_advance_parser(self):
        """测试long-task advance parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "advance", "abc123"])
        assert args.longtask_action == "advance"
        assert args.id == "abc123"
        assert args.drive == 0.8

    def test_longtask_advance_with_drive_parser(self):
        """测试long-task advance带驱力"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "advance", "abc123", "--drive", "0.9"])
        assert args.drive == 0.9

    def test_longtask_pause_parser(self):
        """测试long-task pause parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "pause", "abc123"])
        assert args.longtask_action == "pause"

    def test_longtask_resume_parser(self):
        """测试long-task resume parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "resume", "abc123"])
        assert args.longtask_action == "resume"

    def test_longtask_stats_parser(self):
        """测试long-task stats parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "stats"])
        assert args.longtask_action == "stats"

    def test_longtask_templates_parser(self):
        """测试long-task templates parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "templates"])
        assert args.longtask_action == "templates"

    def test_cmd_longtask_list_empty(self, capsys):
        """测试_cmd_longtask list空"""
        from cli import _cmd_long_task, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "list"])
        ret = _cmd_long_task(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "无长任务" in out

    def test_cmd_longtask_plan(self, capsys):
        """测试_cmd_longtask plan"""
        from cli import _cmd_long_task, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "plan", "system_optimization"])
        ret = _cmd_long_task(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "规划成功" in out

    def test_cmd_longtask_stats(self, capsys):
        """测试_cmd_longtask stats"""
        from cli import _cmd_long_task, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "stats"])
        ret = _cmd_long_task(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "长任务统计" in out

    def test_cmd_longtask_templates(self, capsys):
        """测试_cmd_longtask templates"""
        from cli import _cmd_long_task, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "templates"])
        ret = _cmd_long_task(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "可用长任务模板" in out

    def test_cmd_longtask_status_nonexistent(self, capsys):
        """测试_cmd_longtask status不存在"""
        from cli import _cmd_long_task, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "status", "nonexistent"])
        ret = _cmd_long_task(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "不存在" in out

    def test_cmd_longtask_advance_nonexistent(self, capsys):
        """测试_cmd_longtask advance不存在"""
        from cli import _cmd_long_task, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "advance", "nonexistent"])
        ret = _cmd_long_task(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "错误" in out or "任务推进结果" in out

    def test_cmd_longtask_pause_nonexistent(self, capsys):
        """测试_cmd_longtask pause不存在"""
        from cli import _cmd_long_task, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task", "pause", "nonexistent"])
        ret = _cmd_long_task(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "失败" in out

    def test_cmd_longtask_no_action(self, capsys):
        """测试_cmd_longtask无操作"""
        from cli import _cmd_long_task, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["long-task"])
        ret = _cmd_long_task(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "请指定操作" in out

    def test_cli_main_longtask_dispatch(self, capsys):
        """测试cli_main分发long-task命令"""
        from cli import cli_main
        ret = cli_main(["long-task", "templates"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "可用长任务模板" in out


# ==================== TestU30NoRegression ====================

class TestU30NoRegression:
    """验证U30不破坏既有功能"""

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

    def test_existing_commands_still_dispatch(self):
        """既有命令仍可正常分发"""
        from cli import _create_parser
        parser = _create_parser()
        for cmd in ["run", "status", "snapshot", "logs", "alerts", "perf",
                     "agents", "export", "plugins", "skill", "long-task"]:
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_long_task_planner_unchanged(self):
        """LongTaskPlanner核心API未变"""
        from core.long_task_planner import LongTaskPlanner, LongTask, SubGoal
        planner = LongTaskPlanner()
        assert hasattr(planner, "plan_task")
        assert hasattr(planner, "get_task")
        assert hasattr(planner, "list_tasks")
        assert hasattr(planner, "advance_task")
        assert hasattr(planner, "pause_task")
        assert hasattr(planner, "resume_task")
        assert hasattr(planner, "get_stats")
