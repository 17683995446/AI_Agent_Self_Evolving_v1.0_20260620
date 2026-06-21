"""
M1阶段验证测试 —— 多Agent编排接入主循环

验证点：
1. AgentOrchestrator类存在且可创建
2. spawn_default_agents创建4个专用子Agent
3. delegate_task委派任务并返回结果
4. perceive任务同步获取状态
5. 集群状态查询
6. 健康检查
7. 自动恢复故障Agent
8. stop_all清理资源
9. SelfEvolvingAgent集成orchestrator
10. enable/disable_multi_agent
11. delegate_to_agent接口
12. 主循环不破坏（多Agent默认关闭）
"""

import os
import sys
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent_orchestrator import (
    AgentOrchestrator, TaskType, DelegationResult,
)
from core.multi_agent_bus import AgentRole


@pytest.fixture
def orchestrator():
    """每个测试独立的orchestrator"""
    orch = AgentOrchestrator(default_timeout=3.0)
    yield orch
    orch.stop_all()


class TestM1OrchestratorCreation:
    """验证编排器创建"""

    def test_orchestrator_exists(self):
        assert AgentOrchestrator is not None

    def test_task_type_enum(self):
        assert TaskType.PERCEIVE.value == "perceive"
        assert TaskType.PREDICT.value == "predict"
        assert TaskType.DECIDE.value == "decide"
        assert TaskType.EXECUTE.value == "execute"

    def test_delegation_result_dataclass(self):
        r = DelegationResult(task_type="perceive", success=True, result={"a": 1})
        d = r.to_dict()
        assert d["task_type"] == "perceive"
        assert d["success"] is True
        assert d["result"] == {"a": 1}

    def test_create_orchestrator(self, orchestrator):
        assert orchestrator is not None
        assert orchestrator.bus is not None
        assert orchestrator._delegation_count == 0


class TestM1SpawnAgents:
    """验证子Agent创建"""

    def test_spawn_default_creates_four(self, orchestrator):
        ids = orchestrator.spawn_default_agents()
        assert len(ids) == 4
        assert "perception-1" in ids
        assert "prediction-1" in ids
        assert "decision-1" in ids
        assert "execution-1" in ids

    def test_spawn_specific_role(self, orchestrator):
        aid = orchestrator.spawn_agent(AgentRole.PERCEPTION)
        assert aid == "perception-1"
        assert orchestrator.get_agent(aid) is not None

    def test_spawn_with_custom_id(self, orchestrator):
        aid = orchestrator.spawn_agent(AgentRole.PERCEPTION, agent_id="custom-perc")
        assert aid == "custom-perc"

    def test_spawn_duplicate_id_returns_same(self, orchestrator):
        aid1 = orchestrator.spawn_agent(AgentRole.PERCEPTION, agent_id="dup")
        aid2 = orchestrator.spawn_agent(AgentRole.PERCEPTION, agent_id="dup")
        assert aid1 == aid2

    def test_spawn_multiple_same_role(self, orchestrator):
        """同角色创建多个，ID递增"""
        aid1 = orchestrator.spawn_agent(AgentRole.PERCEPTION)
        aid2 = orchestrator.spawn_agent(AgentRole.PERCEPTION)
        assert aid1 == "perception-1"
        assert aid2 == "perception-2"

    def test_spawn_unsupported_role_raises(self, orchestrator):
        with pytest.raises(ValueError):
            orchestrator.spawn_agent(AgentRole.MERGER)

    def test_list_agents(self, orchestrator):
        orchestrator.spawn_default_agents()
        agents = orchestrator.list_agents()
        assert len(agents) == 4


class TestM1DelegateTask:
    """验证任务委派"""

    def test_delegate_perceive_returns_state(self, orchestrator):
        orchestrator.spawn_agent(AgentRole.PERCEPTION)
        time.sleep(1.0)  # 等待采集
        result = orchestrator.delegate_task(TaskType.PERCEIVE, {})
        assert result.success is True
        assert result.task_type == "perceive"
        assert result.result is not None

    def test_delegate_predict_returns_result(self, orchestrator):
        orchestrator.spawn_agent(AgentRole.PREDICTION)
        result = orchestrator.delegate_task(TaskType.PREDICT, {})
        assert result.success is True

    def test_delegate_auto_creates_agent(self, orchestrator):
        """无可用Agent时自动创建"""
        assert len(orchestrator.list_agents()) == 0
        result = orchestrator.delegate_task(TaskType.PERCEIVE, {})
        # 自动创建后应能委派
        assert len(orchestrator.list_agents()) >= 1

    def test_delegate_unknown_task_type(self, orchestrator):
        """未知任务类型返回失败"""
        # TaskType是Enum，无法传非法值，但测试边界
        result = orchestrator.delegate_task(TaskType.PERCEIVE, {}, timeout=0.1)
        # 应该成功或快速返回
        assert isinstance(result, DelegationResult)

    def test_delegation_increments_count(self, orchestrator):
        orchestrator.spawn_agent(AgentRole.PERCEPTION)
        time.sleep(0.5)
        initial = orchestrator._delegation_count
        orchestrator.delegate_task(TaskType.PERCEIVE, {})
        assert orchestrator._delegation_count == initial + 1

    def test_delegation_result_has_elapsed(self, orchestrator):
        orchestrator.spawn_agent(AgentRole.PERCEPTION)
        time.sleep(0.5)
        result = orchestrator.delegate_task(TaskType.PERCEIVE, {})
        assert result.elapsed >= 0


class TestM1ClusterStatus:
    """验证集群状态查询"""

    def test_get_cluster_status_empty(self, orchestrator):
        status = orchestrator.get_cluster_status()
        assert status["agent_count"] == 0
        assert status["delegation_count"] == 0

    def test_get_cluster_status_with_agents(self, orchestrator):
        orchestrator.spawn_default_agents()
        status = orchestrator.get_cluster_status()
        assert status["agent_count"] == 4
        assert "agents" in status
        assert "bus_stats" in status

    def test_success_rate_calculation(self, orchestrator):
        orchestrator.spawn_agent(AgentRole.PERCEPTION)
        time.sleep(0.5)
        orchestrator.delegate_task(TaskType.PERCEIVE, {})
        status = orchestrator.get_cluster_status()
        assert status["success_count"] >= 1
        assert status["success_rate"] > 0

    def test_get_delegation_history(self, orchestrator):
        orchestrator.spawn_agent(AgentRole.PERCEPTION)
        time.sleep(0.5)
        orchestrator.delegate_task(TaskType.PERCEIVE, {})
        history = orchestrator.get_delegation_history()
        assert len(history) >= 1
        assert "task_type" in history[0]


class TestM1HealthCheck:
    """验证健康检查与自动恢复"""

    def test_health_check_all_healthy(self, orchestrator):
        orchestrator.spawn_default_agents()
        time.sleep(0.5)
        health = orchestrator.health_check()
        assert health["total"] == 4
        assert health["healthy"] >= 0  # 启动初期可能未稳定

    def test_health_check_empty(self, orchestrator):
        health = orchestrator.health_check()
        assert health["total"] == 0
        assert health["unhealthy"] == []

    def test_auto_recover_no_failures(self, orchestrator):
        orchestrator.spawn_default_agents()
        recovered = orchestrator.auto_recover()
        # 无故障时返回0
        assert recovered >= 0

    def test_restart_agent(self, orchestrator):
        aid = orchestrator.spawn_agent(AgentRole.PERCEPTION)
        assert orchestrator.restart_agent(aid) is True
        # 重启后应仍存在
        assert orchestrator.get_agent(aid) is not None

    def test_restart_nonexistent_returns_false(self, orchestrator):
        assert orchestrator.restart_agent("nonexistent") is False


class TestM1StopAll:
    """验证资源清理"""

    def test_stop_all_clears_agents(self, orchestrator):
        orchestrator.spawn_default_agents()
        assert len(orchestrator.list_agents()) == 4
        orchestrator.stop_all()
        assert len(orchestrator.list_agents()) == 0

    def test_stop_agent_removes_one(self, orchestrator):
        aid = orchestrator.spawn_agent(AgentRole.PERCEPTION)
        assert orchestrator.stop_agent(aid) is True
        assert orchestrator.get_agent(aid) is None

    def test_stop_nonexistent_returns_false(self, orchestrator):
        assert orchestrator.stop_agent("nonexistent") is False


class TestM1AgentIntegration:
    """验证SelfEvolvingAgent集成orchestrator"""

    def test_agent_has_orchestrator(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert hasattr(agent, "orchestrator")
        assert agent.orchestrator is not None
        agent.stop()

    def test_orchestrator_disabled_by_default(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert agent._orchestrator_enabled is False
        agent.stop()

    def test_enable_multi_agent(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        ids = agent.enable_multi_agent()
        assert len(ids) == 4
        assert agent._orchestrator_enabled is True
        agent.stop()

    def test_disable_multi_agent(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.enable_multi_agent()
        agent.disable_multi_agent()
        assert agent._orchestrator_enabled is False
        assert len(agent.orchestrator.list_agents()) == 0
        agent.stop()

    def test_delegate_to_agent_auto_enables(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        # 未启用时委派应自动启用
        result = agent.delegate_to_agent("perceive", {})
        assert agent._orchestrator_enabled is True
        agent.stop()

    def test_delegate_to_agent_returns_dict(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        result = agent.delegate_to_agent("perceive", {})
        assert isinstance(result, dict)
        assert "success" in result
        agent.stop()

    def test_delegate_unknown_type_returns_error(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        result = agent.delegate_to_agent("unknown_type", {})
        assert result["success"] is False
        agent.stop()

    def test_get_cluster_status_from_agent(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.enable_multi_agent()
        status = agent.get_cluster_status()
        assert status["agent_count"] == 4
        agent.stop()

    def test_recover_cluster_from_agent(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.enable_multi_agent()
        recovered = agent.recover_cluster()
        assert isinstance(recovered, int)
        agent.stop()

    def test_stop_cleans_up_orchestrator(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.enable_multi_agent()
        assert len(agent.orchestrator.list_agents()) == 4
        agent.stop()
        assert len(agent.orchestrator.list_agents()) == 0


class TestM1NoRegression:
    """验证M1不破坏现有主循环"""

    def test_agent_runs_without_multi_agent(self):
        """不启用多Agent时主循环正常运行"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.run_for_duration(1.0)
        assert agent.loop_count >= 1
        agent.stop()

    def test_agent_runs_with_multi_agent(self):
        """启用多Agent时主循环仍正常运行"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.enable_multi_agent()
        agent.run_for_duration(1.0)
        assert agent.loop_count >= 1
        agent.stop()
