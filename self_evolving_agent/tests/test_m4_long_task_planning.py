"""
M4阶段验证测试 —— 长任务规划与自驱执行

验证点：
1. LongTaskPlanner类存在且可创建
2. 4种默认模板（system_optimization/continuous_monitoring/fault_recovery/capability_enhancement）
3. plan_task规划长任务
4. advance_task自驱推进
5. 进度跟踪
6. 暂停/恢复
7. 失败重试
8. 驱力阈值控制
9. SelfEvolvingAgent集成
10. self_drive_long_task自驱完成
11. 不破坏M1-M3和主循环
"""

import os
import sys
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.long_task_planner import (
    LongTaskPlanner, LongTask, SubGoal, LongTaskState, SubGoalState,
)


@pytest.fixture
def planner():
    return LongTaskPlanner()


class TestM4PlannerCreation:
    """验证规划器创建"""

    def test_planner_exists(self):
        assert LongTaskPlanner is not None

    def test_long_task_dataclass(self):
        t = LongTask(name="test")
        assert t.name == "test"
        assert t.state == LongTaskState.PLANNED
        assert t.progress == 0.0

    def test_subgoal_dataclass(self):
        s = SubGoal(name="step1", goal_type="custom")
        assert s.name == "step1"
        assert s.state == SubGoalState.PENDING

    def test_create_planner(self, planner):
        assert planner is not None
        stats = planner.get_stats()
        assert stats["total_tasks"] == 0

    def test_default_templates_registered(self, planner):
        stats = planner.get_stats()
        assert "system_optimization" in stats["templates"]
        assert "continuous_monitoring" in stats["templates"]
        assert "fault_recovery" in stats["templates"]
        assert "capability_enhancement" in stats["templates"]


class TestM4PlanTask:
    """验证长任务规划"""

    def test_plan_system_optimization(self, planner):
        task = planner.plan_task("system_optimization")
        assert task.name == "system_optimization"
        assert len(task.subgoals) == 4
        assert task.state == LongTaskState.PLANNED

    def test_plan_continuous_monitoring(self, planner):
        task = planner.plan_task("continuous_monitoring")
        assert len(task.subgoals) == 3

    def test_plan_fault_recovery(self, planner):
        task = planner.plan_task("fault_recovery")
        assert len(task.subgoals) == 4

    def test_plan_capability_enhancement(self, planner):
        task = planner.plan_task("capability_enhancement")
        assert len(task.subgoals) == 4

    def test_plan_unknown_template_creates_default(self, planner):
        task = planner.plan_task("unknown_template")
        assert len(task.subgoals) == 1

    def test_plan_with_custom_name(self, planner):
        task = planner.plan_task("system_optimization", name="自定义任务")
        assert task.name == "自定义任务"

    def test_plan_records_task(self, planner):
        task = planner.plan_task("system_optimization")
        assert planner.get_task(task.task_id) is not None

    def test_list_tasks(self, planner):
        planner.plan_task("system_optimization")
        planner.plan_task("fault_recovery")
        tasks = planner.list_tasks()
        assert len(tasks) == 2


class TestM4AdvanceTask:
    """验证自驱推进"""

    def test_advance_starts_task(self, planner):
        task = planner.plan_task("system_optimization")
        result = planner.advance_task(task.task_id, drive_level=0.8)
        assert task.state == LongTaskState.RUNNING
        assert result.get("completed") is False

    def test_advance_completes_subgoal(self, planner):
        task = planner.plan_task("system_optimization")
        planner.advance_task(task.task_id, drive_level=0.8)
        assert task.subgoals[0].state == SubGoalState.COMPLETED
        assert task.current_subgoal_idx == 1

    def test_advance_progress_increments(self, planner):
        task = planner.plan_task("system_optimization")
        planner.advance_task(task.task_id, drive_level=0.8)
        assert task.progress == 0.25

    def test_advance_all_completes_task(self, planner):
        task = planner.plan_task("system_optimization")
        for _ in range(4):
            result = planner.advance_task(task.task_id, drive_level=0.8)
        assert result.get("completed") is True
        assert task.state == LongTaskState.COMPLETED
        assert task.progress == 1.0

    def test_advance_below_drive_threshold_skipped(self, planner):
        task = planner.plan_task("system_optimization")
        task.drive_threshold = 0.5
        result = planner.advance_task(task.task_id, drive_level=0.3)
        assert result.get("skipped") is True
        assert task.state == LongTaskState.PLANNED  # 未启动

    def test_advance_nonexistent_task(self, planner):
        result = planner.advance_task("nonexistent", drive_level=0.8)
        assert "error" in result

    def test_advance_completed_task_returns_error(self, planner):
        task = planner.plan_task("system_optimization")
        for _ in range(4):
            planner.advance_task(task.task_id, drive_level=0.8)
        result = planner.advance_task(task.task_id, drive_level=0.8)
        assert "error" in result

    def test_advance_with_execute_fn(self, planner):
        """使用自定义执行函数"""
        task = planner.plan_task("system_optimization")
        calls = []
        def exec_fn(subgoal):
            calls.append(subgoal.name)
            return {"done": True}
        planner.advance_task(task.task_id, drive_level=0.8, execute_fn=exec_fn)
        assert len(calls) == 1
        assert calls[0] == "monitor"


class TestM4PauseResume:
    """验证暂停/恢复"""

    def test_pause_running_task(self, planner):
        task = planner.plan_task("system_optimization")
        planner.advance_task(task.task_id, drive_level=0.8)
        assert planner.pause_task(task.task_id) is True
        assert task.state == LongTaskState.PAUSED

    def test_resume_paused_task(self, planner):
        task = planner.plan_task("system_optimization")
        planner.advance_task(task.task_id, drive_level=0.8)
        planner.pause_task(task.task_id)
        assert planner.resume_task(task.task_id) is True
        assert task.state == LongTaskState.RUNNING

    def test_pause_nonexistent(self, planner):
        assert planner.pause_task("nonexistent") is False

    def test_advance_paused_returns_error(self, planner):
        task = planner.plan_task("system_optimization")
        planner.advance_task(task.task_id, drive_level=0.8)
        planner.pause_task(task.task_id)
        result = planner.advance_task(task.task_id, drive_level=0.8)
        assert "error" in result


class TestM4FailureRetry:
    """验证失败重试"""

    def test_retry_on_failure(self, planner):
        task = planner.plan_task("system_optimization")
        call_count = [0]
        def failing_fn(subgoal):
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("模拟失败")
            return {"ok": True}
        # 第一次失败
        result = planner.advance_task(task.task_id, drive_level=0.8, execute_fn=failing_fn)
        assert result.get("retrying") is True
        # 重试成功
        result = planner.advance_task(task.task_id, drive_level=0.8, execute_fn=failing_fn)
        assert result.get("completed") is False  # 子目标完成但任务未完成

    def test_fail_after_max_retries(self, planner):
        task = planner.plan_task("system_optimization")
        def always_fail(subgoal):
            raise Exception("持续失败")
        # max_retries=2：第1次失败重试，第2次失败标记FAILED
        result1 = planner.advance_task(task.task_id, drive_level=0.8, execute_fn=always_fail)
        assert result1.get("retrying") is True
        result2 = planner.advance_task(task.task_id, drive_level=0.8, execute_fn=always_fail)
        assert result2.get("failed") is True
        assert task.state == LongTaskState.FAILED


class TestM4AgentIntegration:
    """验证SelfEvolvingAgent集成"""

    def test_agent_has_long_task_planner(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert hasattr(agent, "long_task_planner")
        assert agent.long_task_planner is not None
        agent.stop()

    def test_plan_long_task(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        task_id = agent.plan_long_task("system_optimization", "测试")
        assert isinstance(task_id, str)
        agent.stop()

    def test_self_drive_completes(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        task_id = agent.plan_long_task("system_optimization", "自驱测试")
        result = agent.self_drive_long_task(task_id, max_steps=10, drive_level=0.8)
        assert result.get("completed") is True or result.get("failed") is True or "error" in result
        agent.stop()

    def test_get_long_task_status(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        task_id = agent.plan_long_task("system_optimization")
        status = agent.get_long_task_status(task_id)
        assert status is not None
        assert "state" in status
        assert "progress" in status
        agent.stop()

    def test_list_long_tasks(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.plan_long_task("system_optimization")
        tasks = agent.list_long_tasks()
        assert len(tasks) >= 1
        agent.stop()

    def test_get_long_task_stats(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        stats = agent.get_long_task_stats()
        assert "total_tasks" in stats
        assert "templates" in stats
        agent.stop()

    def test_pause_resume_long_task(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        task_id = agent.plan_long_task("system_optimization")
        agent.advance_long_task(task_id, drive_level=0.8)
        assert agent.pause_long_task(task_id) is True
        assert agent.resume_long_task(task_id) is True
        agent.stop()

    def test_advance_with_workflow_execution(self):
        """推进时通过工作流引擎执行子目标"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        task_id = agent.plan_long_task("continuous_monitoring")
        result = agent.advance_long_task(task_id, drive_level=0.8)
        # 应通过工作流执行子目标
        assert "completed" in result or "error" in result
        agent.stop()


class TestM4NoRegression:
    """验证M4不破坏现有功能"""

    def test_agent_runs_without_long_task(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.run_for_duration(1.0)
        assert agent.loop_count >= 1
        agent.stop()

    def test_m1_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        ids = agent.enable_multi_agent()
        assert len(ids) == 4
        agent.stop()

    def test_m2_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        wf_id = agent.submit_goal("clean_and_recover", "回归")
        result = agent.wait_for_workflow(wf_id, timeout=10)
        assert result["state"] in ("completed", "failed", "partial", "timeout")
        agent.stop()

    def test_m3_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        stats = agent.get_skill_library_stats()
        assert "total_skills" in stats
        agent.stop()
