"""
M4：长任务规划与自驱执行 —— 让Agent自主驱动自己完成长任务

第一性原理：
1. 长任务=多步骤目标，需分解为可管理的子目标
2. 自驱=内在动机驱动持续行动，无需外部触发
3. 进度可跟踪、可恢复、可调整
4. 失败子目标可重试或重新规划

核心能力：
- LongTask：长任务（含多个子目标）
- LongTaskPlanner：规划长任务，分解子目标
- 自驱执行循环：基于驱力自动推进子目标
- 进度跟踪与动态调整
- 与M2工作流引擎协作执行子目标
"""

import os
import sys
import time
import uuid
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


class SubGoalState(Enum):
    """子目标状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class LongTaskState(Enum):
    """长任务状态"""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class SubGoal:
    """子目标 —— 长任务的一个步骤"""
    subgoal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    goal_type: str = "custom"  # 对应M2的GoalType
    params: Dict[str, Any] = field(default_factory=dict)
    state: SubGoalState = SubGoalState.PENDING
    workflow_id: Optional[str] = None  # 执行时关联的工作流ID
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subgoal_id": self.subgoal_id,
            "name": self.name,
            "description": self.description,
            "goal_type": self.goal_type,
            "params": self.params,
            "state": self.state.value,
            "workflow_id": self.workflow_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "retry_count": self.retry_count,
        }


@dataclass
class LongTask:
    """长任务 —— 由多个子目标组成的复合任务

    Attributes:
        task_id: 唯一ID
        name: 任务名称
        description: 任务描述
        subgoals: 子目标列表（按顺序执行）
        state: 任务状态
        created_at: 创建时间
        started_at: 开始时间
        completed_at: 完成时间
        current_subgoal_idx: 当前执行的子目标索引
        drive_threshold: 驱力阈值，低于此值不自动推进
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    subgoals: List[SubGoal] = field(default_factory=list)
    state: LongTaskState = LongTaskState.PLANNED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    current_subgoal_idx: int = 0
    drive_threshold: float = 0.2  # 自驱阈值

    def add_subgoal(self, subgoal: SubGoal) -> None:
        self.subgoals.append(subgoal)

    @property
    def progress(self) -> float:
        """完成进度 [0, 1]"""
        if not self.subgoals:
            return 0.0
        completed = sum(1 for s in self.subgoals if s.state == SubGoalState.COMPLETED)
        return completed / len(self.subgoals)

    @property
    def current_subgoal(self) -> Optional[SubGoal]:
        if self.current_subgoal_idx < len(self.subgoals):
            return self.subgoals[self.current_subgoal_idx]
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "state": self.state.value,
            "progress": round(self.progress, 4),
            "current_subgoal_idx": self.current_subgoal_idx,
            "subgoals": [s.to_dict() for s in self.subgoals],
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class LongTaskPlanner:
    """长任务规划器 —— 将高层目标分解为多步骤长任务

    设计要点：
    - 预置长任务模板（系统优化/持续监控/故障恢复/能力提升）
    - 支持自定义规划
    - 与M2工作流引擎协作执行子目标
    - 自驱推进：基于驱力自动执行下一个子目标
    """

    def __init__(self):
        self._tasks: Dict[str, LongTask] = {}
        self._templates: Dict[str, Callable[[], LongTask]] = {}
        self._register_default_templates()
        self._lock = threading.Lock()
        logger.info("LT_PLANNER", "长任务规划器初始化", module="long_task_planner")

    def _register_default_templates(self) -> None:
        """注册默认长任务模板"""
        self._templates["system_optimization"] = self._plan_system_optimization
        self._templates["continuous_monitoring"] = self._plan_continuous_monitoring
        self._templates["fault_recovery"] = self._plan_fault_recovery
        self._templates["capability_enhancement"] = self._plan_capability_enhancement

    def register_template(self, name: str, template: Callable[[], LongTask]) -> None:
        """注册自定义长任务模板"""
        self._templates[name] = template

    def plan_task(self, template_name: str, name: str = "",
                  description: str = "") -> LongTask:
        """规划长任务

        Args:
            template_name: 模板名称
            name: 任务名称
            description: 任务描述

        Returns:
            LongTask
        """
        template = self._templates.get(template_name)
        if template is None:
            # 默认：创建单子目标任务
            task = LongTask(name=name or template_name, description=description)
            task.add_subgoal(SubGoal(
                name="default_step", goal_type="custom",
                description=description,
            ))
        else:
            task = template()
            if name:
                task.name = name
            if description:
                task.description = description

        with self._lock:
            self._tasks[task.task_id] = task
        logger.info("LT_PLANNER", f"长任务规划: {task.name} [{template_name}], 子目标={len(task.subgoals)}",
                    module="long_task_planner")
        return task

    def get_task(self, task_id: str) -> Optional[LongTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> List[LongTask]:
        with self._lock:
            return list(self._tasks.values())

    def advance_task(self, task_id: str, drive_level: float,
                     execute_fn: Optional[Callable[[SubGoal], Dict[str, Any]]] = None) -> Dict[str, Any]:
        """自驱推进长任务 —— 执行下一个子目标

        第一性原理：驱力足够时自动推进，无需外部触发

        Args:
            task_id: 长任务ID
            drive_level: 当前驱力水平 [0, 1]
            execute_fn: 子目标执行函数（接收SubGoal，返回结果dict）

        Returns:
            推进结果
        """
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            return {"error": "任务不存在"}
        if task.state in (LongTaskState.COMPLETED, LongTaskState.FAILED, LongTaskState.PAUSED):
            return {"error": f"任务状态={task.state.value}, 无法推进"}
        if drive_level < task.drive_threshold:
            return {"skipped": True, "reason": f"驱力{drive_level:.2f}低于阈值{task.drive_threshold}"}

        # 启动任务
        if task.state == LongTaskState.PLANNED:
            task.state = LongTaskState.RUNNING
            task.started_at = time.time()

        # 获取当前子目标
        subgoal = task.current_subgoal
        if subgoal is None:
            # 所有子目标完成
            task.state = LongTaskState.COMPLETED
            task.completed_at = time.time()
            logger.info("LT_PLANNER", f"长任务完成: {task.name}, 进度=100%",
                        module="long_task_planner")
            return {"completed": True, "progress": 1.0}

        # 执行子目标
        subgoal.state = SubGoalState.IN_PROGRESS
        subgoal.started_at = time.time()

        try:
            if execute_fn is not None:
                result = execute_fn(subgoal)
            else:
                result = {"executed": True, "default": True}

            subgoal.result = result
            subgoal.state = SubGoalState.COMPLETED
            subgoal.completed_at = time.time()
            task.current_subgoal_idx += 1

            logger.info("LT_PLANNER",
                        f"子目标完成: {subgoal.name}, 进度={task.progress:.0%}",
                        module="long_task_planner")

            # 检查是否全部完成
            if task.current_subgoal_idx >= len(task.subgoals):
                task.state = LongTaskState.COMPLETED
                task.completed_at = time.time()
                return {"completed": True, "progress": 1.0, "result": result}

            return {"completed": False, "progress": task.progress,
                    "next_subgoal": task.current_subgoal.name if task.current_subgoal else None}

        except Exception as e:
            subgoal.retry_count += 1
            if subgoal.retry_count >= subgoal.max_retries:
                subgoal.state = SubGoalState.FAILED
                subgoal.completed_at = time.time()
                task.state = LongTaskState.FAILED
                logger.error("LT_PLANNER", f"子目标失败(超过重试): {subgoal.name}: {e}",
                             module="long_task_planner")
                return {"failed": True, "error": str(e)}
            else:
                subgoal.state = SubGoalState.PENDING  # 重置以重试
                logger.warning("LT_PLANNER",
                               f"子目标失败(将重试): {subgoal.name} 第{subgoal.retry_count}次: {e}",
                               module="long_task_planner")
                return {"retrying": True, "error": str(e),
                        "retry_count": subgoal.retry_count}

    def pause_task(self, task_id: str) -> bool:
        """暂停长任务"""
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None or task.state != LongTaskState.RUNNING:
            return False
        task.state = LongTaskState.PAUSED
        logger.info("LT_PLANNER", f"长任务暂停: {task.name}", module="long_task_planner")
        return True

    def resume_task(self, task_id: str) -> bool:
        """恢复长任务"""
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None or task.state != LongTaskState.PAUSED:
            return False
        task.state = LongTaskState.RUNNING
        logger.info("LT_PLANNER", f"长任务恢复: {task.name}", module="long_task_planner")
        return True

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._tasks)
            running = sum(1 for t in self._tasks.values() if t.state == LongTaskState.RUNNING)
            completed = sum(1 for t in self._tasks.values() if t.state == LongTaskState.COMPLETED)
            failed = sum(1 for t in self._tasks.values() if t.state == LongTaskState.FAILED)
        return {
            "total_tasks": total,
            "running": running,
            "completed": completed,
            "failed": failed,
            "templates": list(self._templates.keys()),
        }

    # ---------- 默认长任务模板 ----------

    def _plan_system_optimization(self) -> LongTask:
        """系统优化长任务：监控→分析→优化→验证→总结"""
        task = LongTask(name="system_optimization", description="系统优化长任务")
        task.add_subgoal(SubGoal(name="monitor", goal_type="monitor_and_alert",
                                  description="监控当前系统状态"))
        task.add_subgoal(SubGoal(name="analyze", goal_type="analyze_and_decide",
                                  description="分析系统瓶颈"))
        task.add_subgoal(SubGoal(name="optimize", goal_type="optimize_resource",
                                  description="执行资源优化"))
        task.add_subgoal(SubGoal(name="verify", goal_type="monitor_and_alert",
                                  description="验证优化效果"))
        return task

    def _plan_continuous_monitoring(self) -> LongTask:
        """持续监控长任务：循环监控→告警→恢复"""
        task = LongTask(name="continuous_monitoring", description="持续监控长任务")
        task.add_subgoal(SubGoal(name="monitor_1", goal_type="monitor_and_alert",
                                  description="第一轮监控"))
        task.add_subgoal(SubGoal(name="monitor_2", goal_type="monitor_and_alert",
                                  description="第二轮监控"))
        task.add_subgoal(SubGoal(name="monitor_3", goal_type="monitor_and_alert",
                                  description="第三轮监控"))
        return task

    def _plan_fault_recovery(self) -> LongTask:
        """故障恢复长任务：检测→清理→重启→验证"""
        task = LongTask(name="fault_recovery", description="故障恢复长任务")
        task.add_subgoal(SubGoal(name="detect", goal_type="analyze_and_decide",
                                  description="检测故障"))
        task.add_subgoal(SubGoal(name="clean", goal_type="clean_and_recover",
                                  description="清理资源"))
        task.add_subgoal(SubGoal(name="recover", goal_type="optimize_resource",
                                  description="恢复服务"))
        task.add_subgoal(SubGoal(name="verify", goal_type="monitor_and_alert",
                                  description="验证恢复"))
        return task

    def _plan_capability_enhancement(self) -> LongTask:
        """能力提升长任务：评估→学习→进化→验证"""
        task = LongTask(name="capability_enhancement", description="能力提升长任务")
        task.add_subgoal(SubGoal(name="assess", goal_type="analyze_and_decide",
                                  description="评估当前能力"))
        task.add_subgoal(SubGoal(name="learn", goal_type="custom",
                                  description="学习新技能"))
        task.add_subgoal(SubGoal(name="evolve", goal_type="custom",
                                  description="进化现有技能"))
        task.add_subgoal(SubGoal(name="verify", goal_type="monitor_and_alert",
                                  description="验证能力提升"))
        return task


if __name__ == "__main__":
    planner = LongTaskPlanner()
    task = planner.plan_task("system_optimization")
    print(f"长任务: {task.name}, 子目标: {len(task.subgoals)}")
    # 自驱推进
    for i in range(5):
        result = planner.advance_task(task.task_id, drive_level=0.8)
        print(f"  步骤{i}: {result}")
        if result.get("completed") or result.get("failed"):
            break
    print(f"最终进度: {task.progress:.0%}")
