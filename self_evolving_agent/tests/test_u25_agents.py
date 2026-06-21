"""
U25：多Agent控制台 —— 测试

测试覆盖：
1. AgentsCliHelper 的所有方法
2. format_status / format_list / format_health / format_delegation_result 格式化
3. spawn / stop / restart / delegate / recover 生命周期管理
4. CLI集成（_cmd_agents 分发）
5. 无回归（U20-U24 仍可用）
"""

import os
import sys
import time
import json
import pytest

# 路径设置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def helper():
    """创建AgentsCliHelper，测试后清理"""
    from core.agents_cli import AgentsCliHelper
    h = AgentsCliHelper()
    yield h
    try:
        h.stop_all()
    except Exception:
        pass


# ==================== TestU25AgentsCliHelper ====================

class TestU25AgentsCliHelper:
    """测试AgentsCliHelper基础功能"""

    def test_helper_initialization(self, helper):
        """测试初始化"""
        assert helper is not None
        assert helper.orchestrator is not None

    def test_get_cluster_status_empty(self, helper):
        """测试空集群状态"""
        status = helper.get_cluster_status()
        assert status["agent_count"] == 0
        assert status["delegation_count"] == 0

    def test_list_agents_empty(self, helper):
        """测试空Agent列表"""
        agents = helper.list_agents()
        assert agents == []

    def test_spawn_agent(self, helper):
        """测试创建子Agent"""
        agent_id = helper.spawn_agent("perception")
        assert agent_id == "perception-1"
        agents = helper.list_agents()
        assert len(agents) == 1
        assert agents[0]["agent_id"] == "perception-1"

    def test_spawn_agent_with_custom_id(self, helper):
        """测试自定义ID创建子Agent"""
        agent_id = helper.spawn_agent("perception", "my-perception")
        assert agent_id == "my-perception"

    def test_spawn_multiple_agents_same_role(self, helper):
        """测试同角色创建多个Agent"""
        id1 = helper.spawn_agent("perception")
        id2 = helper.spawn_agent("perception")
        assert id1 == "perception-1"
        assert id2 == "perception-2"

    def test_spawn_different_roles(self, helper):
        """测试创建不同角色Agent"""
        helper.spawn_agent("perception")
        helper.spawn_agent("prediction")
        helper.spawn_agent("decision")
        helper.spawn_agent("execution")
        agents = helper.list_agents()
        assert len(agents) == 4

    def test_spawn_invalid_role(self, helper):
        """测试无效角色"""
        with pytest.raises(ValueError):
            helper.spawn_agent("invalid_role")

    def test_spawn_default_agents(self, helper):
        """测试创建默认子Agent集群"""
        ids = helper.spawn_default_agents()
        assert len(ids) == 4
        assert "perception-1" in ids
        assert "prediction-1" in ids
        assert "decision-1" in ids
        assert "execution-1" in ids

    def test_stop_agent(self, helper):
        """测试停止子Agent"""
        agent_id = helper.spawn_agent("perception")
        success = helper.stop_agent(agent_id)
        assert success is True
        assert helper.list_agents() == []

    def test_stop_nonexistent_agent(self, helper):
        """测试停止不存在的Agent"""
        success = helper.stop_agent("nonexistent")
        assert success is False

    def test_restart_agent(self, helper):
        """测试重启子Agent"""
        agent_id = helper.spawn_agent("perception")
        success = helper.restart_agent(agent_id)
        assert success is True
        agents = helper.list_agents()
        assert len(agents) == 1

    def test_restart_nonexistent_agent(self, helper):
        """测试重启不存在的Agent"""
        success = helper.restart_agent("nonexistent")
        assert success is False

    def test_health_check_empty(self, helper):
        """测试空集群健康检查"""
        health = helper.health_check()
        assert health["total"] == 0
        assert health["healthy"] == 0
        assert health["unhealthy"] == []

    def test_health_check_with_agents(self, helper):
        """测试有Agent的健康检查"""
        helper.spawn_agent("perception")
        health = helper.health_check()
        assert health["total"] == 1
        assert health["healthy"] == 1

    def test_auto_recover_no_issues(self, helper):
        """测试无故障时自动恢复"""
        helper.spawn_agent("perception")
        recovered = helper.auto_recover()
        assert recovered == 0

    def test_get_delegation_history_empty(self, helper):
        """测试空委派历史"""
        history = helper.get_delegation_history()
        assert history == []

    def test_delegate_invalid_task_type(self, helper):
        """测试无效任务类型委派"""
        result = helper.delegate_task("invalid_task", {})
        assert result["success"] is False
        assert "未知任务类型" in result["error"]


# ==================== TestU25FormatStatus ====================

class TestU25FormatStatus:
    """测试format_status格式化"""

    def test_format_status_empty(self, helper):
        """测试空集群状态格式化"""
        result = helper.format_status()
        assert "多Agent集群状态" in result
        assert "Agent数量: 0" in result

    def test_format_status_with_agents(self, helper):
        """测试有Agent的状态格式化"""
        helper.spawn_agent("perception")
        result = helper.format_status()
        assert "多Agent集群状态" in result
        assert "Agent数量: 1" in result


# ==================== TestU25FormatList ====================

class TestU25FormatList:
    """测试format_list格式化"""

    def test_format_list_empty(self, helper):
        """测试空Agent列表格式化"""
        result = helper.format_list()
        assert "无子Agent" in result

    def test_format_list_with_agents(self, helper):
        """测试有Agent的列表格式化"""
        helper.spawn_agent("perception")
        helper.spawn_agent("prediction")
        result = helper.format_list()
        assert "子Agent列表" in result
        assert "perception-1" in result
        assert "prediction-1" in result
        assert "感知" in result
        assert "预测" in result


# ==================== TestU25FormatHealth ====================

class TestU25FormatHealth:
    """测试format_health格式化"""

    def test_format_health_empty(self, helper):
        """测试空集群健康检查格式化"""
        result = helper.format_health()
        assert "健康检查" in result
        assert "总数: 0" in result

    def test_format_health_with_agents(self, helper):
        """测试有Agent的健康检查格式化"""
        helper.spawn_agent("perception")
        result = helper.format_health()
        assert "健康检查" in result
        assert "总数: 1" in result
        assert "健康: 1" in result


# ==================== TestU25FormatDelegationResult ====================

class TestU25FormatDelegationResult:
    """测试format_delegation_result格式化"""

    def test_format_delegation_result_success(self, helper):
        """测试成功委派结果格式化"""
        result = {
            "task_type": "predict",
            "success": True,
            "agent_id": "prediction-1",
            "elapsed": 0.05,
            "result": {"prediction": "high_load"},
        }
        formatted = helper.format_delegation_result(result)
        assert "委派结果" in formatted
        assert "predict" in formatted
        assert "是" in formatted
        assert "prediction-1" in formatted

    def test_format_delegation_result_failure(self, helper):
        """测试失败委派结果格式化"""
        result = {
            "task_type": "predict",
            "success": False,
            "error": "Agent不存在",
            "elapsed": 0.0,
        }
        formatted = helper.format_delegation_result(result)
        assert "委派结果" in formatted
        assert "否" in formatted
        assert "Agent不存在" in formatted

    def test_format_delegation_result_long_result(self, helper):
        """测试长结果截断"""
        long_result = {"data": "x" * 300}
        result = {
            "task_type": "perceive",
            "success": True,
            "agent_id": "perception-1",
            "elapsed": 0.01,
            "result": long_result,
        }
        formatted = helper.format_delegation_result(result)
        assert "..." in formatted


# ==================== TestU25FormatSpawnResult ====================

class TestU25FormatSpawnResult:
    """测试format_spawn_result格式化"""

    def test_format_spawn_result(self, helper):
        """测试创建结果格式化"""
        result = helper.format_spawn_result("perception-1", "perception")
        assert "perception-1" in result
        assert "感知" in result


# ==================== TestU25FormatHistory ====================

class TestU25FormatHistory:
    """测试format_history格式化"""

    def test_format_history_empty(self, helper):
        """测试空委派历史格式化"""
        result = helper.format_history()
        assert "无委派历史" in result


# ==================== TestU25CliIntegration ====================

class TestU25CliIntegration:
    """测试CLI集成"""

    def test_agents_parser_exists(self):
        """测试agents子命令parser存在"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents"])
        assert args.command == "agents"

    def test_agents_status_parser(self):
        """测试agents status parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "status"])
        assert args.agents_action == "status"

    def test_agents_list_parser(self):
        """测试agents list parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "list"])
        assert args.agents_action == "list"

    def test_agents_spawn_parser(self):
        """测试agents spawn parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "spawn", "perception"])
        assert args.agents_action == "spawn"
        assert args.role == "perception"

    def test_agents_spawn_with_id_parser(self):
        """测试agents spawn带ID"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "spawn", "perception", "--id", "my-agent"])
        assert args.role == "perception"
        assert args.id == "my-agent"

    def test_agents_spawn_default_parser(self):
        """测试agents spawn-default parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "spawn-default"])
        assert args.agents_action == "spawn-default"

    def test_agents_delegate_parser(self):
        """测试agents delegate parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "delegate", "predict", '{"cpu": 0.8}'])
        assert args.agents_action == "delegate"
        assert args.task_type == "predict"
        assert args.payload == '{"cpu": 0.8}'

    def test_agents_restart_parser(self):
        """测试agents restart parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "restart", "perception-1"])
        assert args.agents_action == "restart"
        assert args.agent_id == "perception-1"

    def test_agents_health_parser(self):
        """测试agents health parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "health"])
        assert args.agents_action == "health"

    def test_agents_recover_parser(self):
        """测试agents recover parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "recover"])
        assert args.agents_action == "recover"

    def test_agents_stop_parser(self):
        """测试agents stop parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "stop", "perception-1"])
        assert args.agents_action == "stop"
        assert args.agent_id == "perception-1"

    def test_agents_history_parser(self):
        """测试agents history parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "history"])
        assert args.agents_action == "history"

    def test_cmd_agents_status(self, capsys):
        """测试_cmd_agents status"""
        from cli import _cmd_agents, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "status"])
        ret = _cmd_agents(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "多Agent集群状态" in out

    def test_cmd_agents_list_empty(self, capsys):
        """测试_cmd_agents list空"""
        from cli import _cmd_agents, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "list"])
        ret = _cmd_agents(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "无子Agent" in out

    def test_cmd_agents_spawn(self, capsys):
        """测试_cmd_agents spawn"""
        from cli import _cmd_agents, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "spawn", "perception"])
        ret = _cmd_agents(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "perception-1" in out
        assert "感知" in out

    def test_cmd_agents_spawn_default(self, capsys):
        """测试_cmd_agents spawn-default"""
        from cli import _cmd_agents, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "spawn-default"])
        ret = _cmd_agents(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "perception-1" in out

    def test_cmd_agents_delegate_invalid_json(self, capsys):
        """测试_cmd_agents delegate无效JSON"""
        from cli import _cmd_agents, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "delegate", "predict", "invalid_json"])
        ret = _cmd_agents(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "JSON解析失败" in out

    def test_cmd_agents_restart_nonexistent(self, capsys):
        """测试_cmd_agents restart不存在的Agent"""
        from cli import _cmd_agents, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "restart", "nonexistent"])
        ret = _cmd_agents(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "不存在" in out

    def test_cmd_agents_health(self, capsys):
        """测试_cmd_agents health"""
        from cli import _cmd_agents, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "health"])
        ret = _cmd_agents(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "健康检查" in out

    def test_cmd_agents_recover(self, capsys):
        """测试_cmd_agents recover"""
        from cli import _cmd_agents, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "recover"])
        ret = _cmd_agents(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "已恢复" in out

    def test_cmd_agents_stop_nonexistent(self, capsys):
        """测试_cmd_agents stop不存在的Agent"""
        from cli import _cmd_agents, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "stop", "nonexistent"])
        ret = _cmd_agents(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "不存在" in out

    def test_cmd_agents_history(self, capsys):
        """测试_cmd_agents history"""
        from cli import _cmd_agents, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents", "history"])
        ret = _cmd_agents(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "委派历史" in out or "无委派历史" in out

    def test_cmd_agents_no_action(self, capsys):
        """测试_cmd_agents无操作"""
        from cli import _cmd_agents, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["agents"])
        ret = _cmd_agents(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "请指定操作" in out

    def test_cli_main_agents_dispatch(self, capsys):
        """测试cli_main分发agents命令"""
        from cli import cli_main
        ret = cli_main(["agents", "status"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "多Agent集群状态" in out


# ==================== TestU25NoRegression ====================

class TestU25NoRegression:
    """验证U25不破坏既有功能"""

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
        from core.log_viewer import LogManager
        assert LogManager is not None

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

    def test_existing_commands_still_dispatch(self):
        """既有命令仍可正常分发"""
        from cli import _create_parser
        parser = _create_parser()
        for cmd in ["run", "status", "snapshot", "logs", "alerts", "perf", "agents"]:
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_orchestrator_unchanged(self):
        """AgentOrchestrator核心API未变"""
        from core.agent_orchestrator import AgentOrchestrator
        orch = AgentOrchestrator()
        assert hasattr(orch, "spawn_agent")
        assert hasattr(orch, "delegate_task")
        assert hasattr(orch, "get_cluster_status")
        assert hasattr(orch, "health_check")
        try:
            orch.stop_all()
        except Exception:
            pass
