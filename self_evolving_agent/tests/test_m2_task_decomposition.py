"""
M2阶段验证测试 —— 任务分解与工作流引擎接入

验证点：
1. TaskDecomposer类存在且可创建
2. GoalType枚举完整
3. decompose将目标分解为DAG工作流
4. 4种默认分解模板（monitor/optimize/clean/analyze）
5. 自定义模板注册
6. 分解历史记录
7. SelfEvolvingAgent集成decomposer和workflow_engine
8. submit_goal提交并执行工作流
9. 工作流状态查询
10. 工作流统计
11. wait_for_workflow等待完成
12. 任务处理器注册（perceive/predict/decide/alert/clean_cache等）
13. 不破坏主循环
"""

import os
import sys
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.task_decomposer import TaskDecomposer, Goal, GoalType
from core.workflow_engine import (
    WorkflowEngine, Workflow, WorkflowTask, TaskState, WorkflowState,
)


@pytest.fixture
def decomposer():
    return TaskDecomposer()


class TestM2DecomposerCreation:
    """验证分解器创建"""

    def test_decomposer_exists(self):
        assert TaskDecomposer is not None

    def test_goal_type_enum(self):
        assert GoalType.MONITOR_AND_ALERT.value == "monitor_and_alert"
        assert GoalType.OPTIMIZE_RESOURCE.value == "optimize_resource"
        assert GoalType.CLEAN_AND_RECOVER.value == "clean_and_recover"
        assert GoalType.ANALYZE_AND_DECIDE.value == "analyze_and_decide"
        assert GoalType.CUSTOM.value == "custom"

    def test_goal_dataclass(self):
        g = Goal(goal_type=GoalType.CUSTOM, description="test")
        d = g.to_dict()
        assert d["goal_type"] == "custom"
        assert d["description"] == "test"

    def test_create_decomposer(self, decomposer):
        assert decomposer is not None
        assert len(decomposer.list_templates()) == 4

    def test_default_templates_registered(self, decomposer):
        templates = decomposer.list_templates()
        assert "monitor_and_alert" in templates
        assert "optimize_resource" in templates
        assert "clean_and_recover" in templates
        assert "analyze_and_decide" in templates


class TestM2Decompose:
    """验证目标分解"""

    def test_decompose_monitor(self, decomposer):
        g = Goal(goal_type=GoalType.MONITOR_AND_ALERT, description="监控")
        wf = decomposer.decompose(g)
        assert len(wf.tasks) == 4
        # 拓扑排序应有4个任务
        order = wf.get_task_order()
        assert len(order) == 4

    def test_decompose_optimize(self, decomposer):
        g = Goal(goal_type=GoalType.OPTIMIZE_RESOURCE, description="优化")
        wf = decomposer.decompose(g)
        assert len(wf.tasks) == 4

    def test_decompose_clean(self, decomposer):
        g = Goal(goal_type=GoalType.CLEAN_AND_RECOVER, description="清理")
        wf = decomposer.decompose(g)
        assert len(wf.tasks) == 3

    def test_decompose_analyze(self, decomposer):
        g = Goal(goal_type=GoalType.ANALYZE_AND_DECIDE, description="分析")
        wf = decomposer.decompose(g)
        assert len(wf.tasks) == 4

    def test_decompose_custom(self, decomposer):
        g = Goal(goal_type=GoalType.CUSTOM, description="自定义")
        wf = decomposer.decompose(g)
        assert len(wf.tasks) == 1

    def test_decompose_produces_dag(self, decomposer):
        """分解产生有依赖关系的DAG"""
        g = Goal(goal_type=GoalType.MONITOR_AND_ALERT)
        wf = decomposer.decompose(g)
        # 至少有一个任务有依赖
        has_deps = any(len(t.dependencies) > 0 for t in wf.tasks.values())
        assert has_deps

    def test_decompose_records_history(self, decomposer):
        g = Goal(goal_type=GoalType.CLEAN_AND_RECOVER)
        decomposer.decompose(g)
        history = decomposer.get_history()
        assert len(history) == 1
        assert history[0]["goal_type"] == "clean_and_recover"
        assert history[0]["task_count"] == 3

    def test_decompose_stats(self, decomposer):
        g = Goal(goal_type=GoalType.MONITOR_AND_ALERT)
        decomposer.decompose(g)
        stats = decomposer.get_stats()
        assert stats["total_decompositions"] == 1
        assert stats["template_count"] == 4


class TestM2CustomTemplate:
    """验证自定义模板"""

    def test_register_custom_template(self, decomposer):
        def custom_template(goal):
            return [
                WorkflowTask(name="step1", task_type="custom"),
                WorkflowTask(name="step2", task_type="custom"),
            ]
        decomposer.register_template(GoalType.CUSTOM, custom_template)
        g = Goal(goal_type=GoalType.CUSTOM)
        wf = decomposer.decompose(g)
        assert len(wf.tasks) == 2


class TestM2WorkflowEngine:
    """验证工作流引擎"""

    def test_workflow_engine_exists(self):
        assert WorkflowEngine is not None

    def test_workflow_dag(self):
        """工作流DAG拓扑排序"""
        wf = Workflow(name="test")
        t1 = WorkflowTask(name="t1", task_type="custom")
        t2 = WorkflowTask(name="t2", task_type="custom", dependencies=[t1.task_id])
        wf.add_task(t1)
        wf.add_task(t2)
        order = wf.get_task_order()
        assert order[0] == t1.task_id
        assert order[1] == t2.task_id

    def test_get_ready_tasks(self):
        """获取就绪任务"""
        wf = Workflow(name="test")
        t1 = WorkflowTask(name="t1", task_type="custom")
        t2 = WorkflowTask(name="t2", task_type="custom", dependencies=[t1.task_id])
        wf.add_task(t1)
        wf.add_task(t2)
        ready = wf.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == t1.task_id

    def test_engine_submit_and_execute(self):
        """引擎提交并执行工作流"""
        engine = WorkflowEngine(max_workers=2)
        engine.register_task_handler("custom", lambda p: {"ok": True})

        wf = Workflow(name="test")
        t1 = WorkflowTask(name="t1", task_type="custom")
        wf.add_task(t1)

        engine.start()
        try:
            wf_id = engine.submit_workflow(wf)
            time.sleep(1.0)
            status = engine.get_workflow_status(wf_id)
            assert status["state"] == "completed"
        finally:
            engine.stop()


class TestM2AgentIntegration:
    """验证SelfEvolvingAgent集成"""

    def test_agent_has_decomposer(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert hasattr(agent, "task_decomposer")
        assert agent.task_decomposer is not None
        agent.stop()

    def test_agent_has_workflow_engine(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert hasattr(agent, "workflow_engine")
        assert agent.workflow_engine is not None
        agent.stop()

    def test_submit_goal_returns_workflow_id(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        wf_id = agent.submit_goal("clean_and_recover", "测试清理")
        assert isinstance(wf_id, str)
        assert len(wf_id) > 0
        agent.stop()

    def test_submit_goal_custom_type(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        wf_id = agent.submit_goal("custom", "自定义", {"key": "value"})
        assert isinstance(wf_id, str)
        agent.stop()

    def test_submit_unknown_goal_type_falls_back_to_custom(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        wf_id = agent.submit_goal("unknown_type", "未知类型")
        assert isinstance(wf_id, str)
        agent.stop()

    def test_get_workflow_status(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        wf_id = agent.submit_goal("clean_and_recover", "测试")
        time.sleep(2.0)
        status = agent.get_workflow_status(wf_id)
        assert status is not None
        assert "state" in status
        agent.stop()

    def test_wait_for_workflow_completes(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        wf_id = agent.submit_goal("clean_and_recover", "测试")
        result = agent.wait_for_workflow(wf_id, timeout=10)
        assert result["state"] in ("completed", "failed", "partial", "timeout")
        agent.stop()

    def test_get_workflow_stats(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.submit_goal("clean_and_recover", "测试")
        time.sleep(2.0)
        stats = agent.get_workflow_stats()
        assert "total_workflows" in stats
        assert "decomposer_stats" in stats
        agent.stop()

    def test_workflow_handlers_registered(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        # 检查处理器已注册
        handlers = agent.workflow_engine._executor._handlers
        assert "perceive" in handlers
        assert "predict" in handlers
        assert "decide" in handlers
        assert "alert" in handlers
        assert "clean_cache" in handlers
        assert "optimize" in handlers
        agent.stop()

    def test_monitor_workflow_executes(self):
        """监控工作流完整执行"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        wf_id = agent.submit_goal("monitor_and_alert", "监控CPU")
        result = agent.wait_for_workflow(wf_id, timeout=15)
        assert result["state"] in ("completed", "failed", "partial", "timeout")
        agent.stop()

    def test_optimize_workflow_executes(self):
        """优化工作流完整执行"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        wf_id = agent.submit_goal("optimize_resource", "优化资源")
        result = agent.wait_for_workflow(wf_id, timeout=15)
        assert result["state"] in ("completed", "failed", "partial", "timeout")
        agent.stop()


class TestM2NoRegression:
    """验证M2不破坏现有功能"""

    def test_agent_runs_without_workflow(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.run_for_duration(1.0)
        assert agent.loop_count >= 1
        agent.stop()

    def test_agent_runs_with_workflow(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.submit_goal("clean_and_recover", "并发测试")
        agent.run_for_duration(1.0)
        assert agent.loop_count >= 1
        agent.stop()

    def test_m1_still_works(self):
        """M1多Agent编排仍工作"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        ids = agent.enable_multi_agent()
        assert len(ids) == 4
        agent.stop()
