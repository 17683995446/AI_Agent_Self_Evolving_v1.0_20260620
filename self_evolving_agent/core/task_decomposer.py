"""
M2：任务分解器 —— 将长任务/高层目标自动分解为可执行的DAG工作流

第一性原理：
1. 长任务=多步骤目标，必须分解才能执行
2. 分解=识别子任务+确定依赖关系，形成DAG
3. 子任务应可独立执行、可重试、可并行
4. 分解策略可学习：从成功的工作流中提炼模式

核心能力：
- decompose(goal): 将高层目标分解为Workflow DAG
- 预置分解模板（监控/优化/告警/清理等常见任务）
- 支持自定义分解策略
- 与WorkflowEngine协作执行分解后的工作流
"""

import os
import sys
import time
import uuid
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger
from core.workflow_engine import Workflow, WorkflowTask, TaskState


class GoalType(Enum):
    """目标类型 —— 决定分解策略"""
    MONITOR_AND_ALERT = "monitor_and_alert"      # 监控并告警
    OPTIMIZE_RESOURCE = "optimize_resource"      # 优化资源
    CLEAN_AND_RECOVER = "clean_and_recover"      # 清理并恢复
    ANALYZE_AND_DECIDE = "analyze_and_decide"    # 分析并决策
    CUSTOM = "custom"                            # 自定义


@dataclass
class Goal:
    """高层目标"""
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal_type: GoalType = GoalType.CUSTOM
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    priority: int = 2
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "goal_type": self.goal_type.value,
            "description": self.description,
            "params": self.params,
            "priority": self.priority,
        }


class TaskDecomposer:
    """任务分解器 —— 将目标分解为工作流DAG

    设计要点：
    - 每种GoalType有对应的分解模板
    - 分解产生WorkflowTask列表，含依赖关系
    - 模板可扩展（register_template）
    - 分解结果可追溯（记录分解历史）
    """

    def __init__(self):
        self._templates: Dict[GoalType, Callable[[Goal], List[WorkflowTask]]] = {}
        self._decomposition_history: List[Dict[str, Any]] = []
        self._register_default_templates()
        logger.info("DECOMPOSER", "任务分解器初始化", module="task_decomposer")

    def _register_default_templates(self) -> None:
        """注册默认分解模板"""
        self._templates[GoalType.MONITOR_AND_ALERT] = self._decompose_monitor
        self._templates[GoalType.OPTIMIZE_RESOURCE] = self._decompose_optimize
        self._templates[GoalType.CLEAN_AND_RECOVER] = self._decompose_clean
        self._templates[GoalType.ANALYZE_AND_DECIDE] = self._decompose_analyze

    def register_template(self, goal_type: GoalType,
                          template: Callable[[Goal], List[WorkflowTask]]) -> None:
        """注册自定义分解模板"""
        self._templates[goal_type] = template
        logger.debug("DECOMPOSER", f"注册模板: {goal_type.value}", module="task_decomposer")

    def decompose(self, goal: Goal) -> Workflow:
        """将目标分解为工作流

        Args:
            goal: 高层目标

        Returns:
            可执行的Workflow
        """
        template = self._templates.get(goal.goal_type)
        if template is None:
            # 自定义目标：生成单任务工作流
            tasks = [WorkflowTask(
                name="custom_task",
                task_type="custom",
                payload=goal.params,
            )]
        else:
            tasks = template(goal)

        workflow = Workflow(name=f"workflow-{goal.goal_type.value}")
        for task in tasks:
            workflow.add_task(task)

        # 记录分解历史
        self._decomposition_history.append({
            "goal_id": goal.goal_id,
            "goal_type": goal.goal_type.value,
            "workflow_id": workflow.workflow_id,
            "task_count": len(tasks),
            "task_order": workflow.get_task_order(),
            "timestamp": time.time(),
        })

        logger.info("DECOMPOSER",
                    f"目标分解: {goal.goal_type.value} -> {len(tasks)}个任务, 工作流={workflow.workflow_id}",
                    module="task_decomposer")
        return workflow

    # ---------- 默认分解模板 ----------

    def _decompose_monitor(self, goal: Goal) -> List[WorkflowTask]:
        """监控并告警：感知→预测→决策→(条件)告警"""
        t1 = WorkflowTask(name="perceive", task_type="perceive",
                          payload=goal.params, max_retries=2)
        t2 = WorkflowTask(name="predict", task_type="predict",
                          dependencies=[], max_retries=2)
        t2.dependencies = [t1.task_id]
        t3 = WorkflowTask(name="decide", task_type="decide",
                          dependencies=[t2.task_id], max_retries=2)
        t4 = WorkflowTask(name="alert_if_needed", task_type="alert",
                          dependencies=[t3.task_id], max_retries=1)
        return [t1, t2, t3, t4]

    def _decompose_optimize(self, goal: Goal) -> List[WorkflowTask]:
        """优化资源：感知→分析→优化→验证"""
        t1 = WorkflowTask(name="perceive", task_type="perceive",
                          payload=goal.params, max_retries=2)
        t2 = WorkflowTask(name="analyze", task_type="analyze",
                          dependencies=[t1.task_id], max_retries=2)
        t3 = WorkflowTask(name="optimize", task_type="optimize",
                          dependencies=[t2.task_id], max_retries=2)
        t4 = WorkflowTask(name="verify", task_type="perceive",
                          dependencies=[t3.task_id], max_retries=2)
        return [t1, t2, t3, t4]

    def _decompose_clean(self, goal: Goal) -> List[WorkflowTask]:
        """清理并恢复：感知→清理→验证"""
        t1 = WorkflowTask(name="perceive", task_type="perceive",
                          payload=goal.params, max_retries=2)
        t2 = WorkflowTask(name="clean_cache", task_type="clean_cache",
                          dependencies=[t1.task_id], max_retries=1)
        t3 = WorkflowTask(name="verify", task_type="perceive",
                          dependencies=[t2.task_id], max_retries=2)
        return [t1, t2, t3]

    def _decompose_analyze(self, goal: Goal) -> List[WorkflowTask]:
        """分析并决策：感知→预测→分析→决策"""
        t1 = WorkflowTask(name="perceive", task_type="perceive",
                          payload=goal.params, max_retries=2)
        t2 = WorkflowTask(name="predict", task_type="predict",
                          dependencies=[t1.task_id], max_retries=2)
        t3 = WorkflowTask(name="analyze", task_type="analyze",
                          dependencies=[t2.task_id], max_retries=2)
        t4 = WorkflowTask(name="decide", task_type="decide",
                          dependencies=[t3.task_id], max_retries=2)
        return [t1, t2, t3, t4]

    # ---------- 查询 ----------

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取分解历史"""
        return self._decomposition_history[-limit:]

    def list_templates(self) -> List[str]:
        """列出已注册的分解模板"""
        return [gt.value for gt in self._templates.keys()]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "template_count": len(self._templates),
            "total_decompositions": len(self._decomposition_history),
            "templates": self.list_templates(),
        }


if __name__ == "__main__":
    # 自测
    d = TaskDecomposer()
    g = Goal(goal_type=GoalType.MONITOR_AND_ALERT, description="监控CPU并告警")
    wf = d.decompose(g)
    print(f"工作流: {wf.name}, 任务数: {len(wf.tasks)}")
    print(f"执行顺序: {wf.get_task_order()}")
    print(f"统计: {d.get_stats()}")
