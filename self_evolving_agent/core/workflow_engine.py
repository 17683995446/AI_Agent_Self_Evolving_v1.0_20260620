"""
任务编排与工作流引擎 —— DAG建模、依赖驱动、并行执行、失败重试

设计原则（第一性原理）：
1. 复杂即简单：任何复杂工作流都是原子任务的组合
2. 依赖驱动：前置任务完成才触发后续，像多米诺骨牌
3. 并行最大化：无依赖的任务同时执行，节省时间
4. 容错：失败自动重试，重试失败自动回滚
"""

import os
import sys
import time
import threading
import uuid
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


class TaskState(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


class WorkflowState(Enum):
    """工作流状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class WorkflowTask:
    """工作流中的任务节点"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    task_type: str = ""           # 任务类型标识
    payload: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # 依赖的任务ID
    max_retries: int = 2
    timeout_sec: float = 30.0
    retry_delay_sec: float = 1.0

    # 运行时状态
    state: TaskState = TaskState.PENDING
    attempt: int = 0
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "task_type": self.task_type,
            "state": self.state.value,
            "attempt": self.attempt,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class Workflow:
    """工作流定义"""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    tasks: Dict[str, WorkflowTask] = field(default_factory=dict)
    state: WorkflowState = WorkflowState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def add_task(self, task: WorkflowTask) -> None:
        self.tasks[task.task_id] = task

    def get_ready_tasks(self) -> List[WorkflowTask]:
        """获取所有依赖已满足的任务"""
        ready = []
        for task in self.tasks.values():
            if task.state != TaskState.PENDING:
                continue
            deps_satisfied = all(
                self.tasks.get(dep, WorkflowTask()).state == TaskState.SUCCESS
                for dep in task.dependencies
            )
            if deps_satisfied:
                ready.append(task)
        return ready

    def get_task_order(self) -> List[str]:
        """拓扑排序获取任务执行顺序"""
        in_degree = {tid: 0 for tid in self.tasks}
        adj = defaultdict(list)
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep in self.tasks:
                    adj[dep].append(task.task_id)
                    in_degree[task.task_id] += 1

        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        order = []
        while queue:
            tid = queue.popleft()
            order.append(tid)
            for next_tid in adj[tid]:
                in_degree[next_tid] -= 1
                if in_degree[next_tid] == 0:
                    queue.append(next_tid)

        return order

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "state": self.state.value,
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class TaskExecutor:
    """任务执行器 —— 实际执行单个任务"""

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}

    def register_handler(self, task_type: str, handler: Callable) -> None:
        """注册任务处理器"""
        self._handlers[task_type] = handler

    def execute(self, task: WorkflowTask) -> Any:
        """执行任务"""
        handler = self._handlers.get(task.task_type)
        if not handler:
            raise ValueError(f"无处理器: {task.task_type}")
        return handler(task.payload)


class WorkflowEngine:
    """工作流引擎 —— 调度、执行、监控工作流"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._executor = TaskExecutor()
        self._workflows: Dict[str, Workflow] = {}
        self._running = False
        self._lock = threading.Lock()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._active_threads: Set[threading.Thread] = set()

    def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """注册任务处理器"""
        self._executor.register_handler(task_type, handler)

    def submit_workflow(self, workflow: Workflow) -> str:
        """提交工作流"""
        with self._lock:
            self._workflows[workflow.workflow_id] = workflow
        logger.info("WORKFLOW", f"工作流提交: {workflow.workflow_id}", module="workflow_engine")
        return workflow.workflow_id

    def start(self) -> None:
        """启动调度器"""
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("WORKFLOW", "工作流引擎启动", module="workflow_engine")

    def stop(self) -> None:
        """停止调度器"""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        for t in list(self._active_threads):
            t.join(timeout=2.0)
        logger.info("WORKFLOW", "工作流引擎停止", module="workflow_engine")

    def _scheduler_loop(self) -> None:
        """调度循环"""
        while self._running:
            with self._lock:
                for workflow in self._workflows.values():
                    if workflow.state == WorkflowState.PENDING:
                        workflow.state = WorkflowState.RUNNING
                        workflow.started_at = time.time()

                    if workflow.state == WorkflowState.RUNNING:
                        self._schedule_workflow(workflow)
            time.sleep(0.5)

    def _schedule_workflow(self, workflow: Workflow) -> None:
        """调度单个工作流"""
        ready_tasks = workflow.get_ready_tasks()
        running_count = sum(1 for t in workflow.tasks.values() if t.state == TaskState.RUNNING)
        available_slots = self.max_workers - running_count

        for task in ready_tasks[:available_slots]:
            if len(self._active_threads) >= self.max_workers:
                break
            self._execute_task_async(workflow, task)

        # 检查工作流是否完成
        all_done = all(t.state in (TaskState.SUCCESS, TaskState.SKIPPED, TaskState.FAILED)
                       for t in workflow.tasks.values())
        if all_done:
            any_failed = any(t.state == TaskState.FAILED for t in workflow.tasks.values())
            workflow.state = WorkflowState.FAILED if any_failed else WorkflowState.COMPLETED
            workflow.completed_at = time.time()
            logger.info("WORKFLOW", f"工作流完成: {workflow.workflow_id} 状态={workflow.state.value}",
                        module="workflow_engine")

    def _execute_task_async(self, workflow: Workflow, task: WorkflowTask) -> None:
        """异步执行任务"""
        def run():
            task.state = TaskState.RUNNING
            task.started_at = time.time()
            task.attempt += 1

            try:
                # 收集依赖任务的输出，合并到payload
                merged_payload = dict(task.payload)
                for dep_id in task.dependencies:
                    dep_task = workflow.tasks.get(dep_id)
                    if dep_task and dep_task.result is not None:
                        # 使用任务名作为键注入结果
                        key = dep_task.name or dep_id
                        merged_payload[key] = dep_task.result

                # 创建临时任务对象用于执行
                exec_task = WorkflowTask(
                    task_id=task.task_id,
                    name=task.name,
                    task_type=task.task_type,
                    payload=merged_payload,
                )
                result = self._executor.execute(exec_task)
                task.result = result
                task.state = TaskState.SUCCESS
                task.completed_at = time.time()
                logger.info("WORKFLOW", f"任务成功: {task.name} ({task.task_id})", module="workflow_engine")
            except Exception as e:
                task.error = str(e)
                if task.attempt <= task.max_retries:
                    task.state = TaskState.PENDING  # 直接重新进入待执行队列
                    logger.warning("WORKFLOW", f"任务失败，将重试: {task.name} ({task.task_id}) 第{task.attempt}次",
                                   module="workflow_engine")
                else:
                    task.state = TaskState.FAILED
                    task.completed_at = time.time()
                    logger.error("WORKFLOW", f"任务失败: {task.name} ({task.task_id}): {e}",
                                 module="workflow_engine")
            finally:
                self._active_threads.discard(threading.current_thread())

        t = threading.Thread(target=run, name=f"task-{task.task_id}", daemon=True)
        self._active_threads.add(t)
        t.start()

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """获取工作流状态"""
        with self._lock:
            wf = self._workflows.get(workflow_id)
            return wf.to_dict() if wf else None

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            total = len(self._workflows)
            completed = sum(1 for w in self._workflows.values() if w.state == WorkflowState.COMPLETED)
            failed = sum(1 for w in self._workflows.values() if w.state == WorkflowState.FAILED)
            running = sum(1 for w in self._workflows.values() if w.state == WorkflowState.RUNNING)
            return {
                "total_workflows": total,
                "completed": completed,
                "failed": failed,
                "running": running,
                "active_threads": len(self._active_threads),
            }
