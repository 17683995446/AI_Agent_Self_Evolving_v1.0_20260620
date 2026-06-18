"""
通用Agent集群与任务分派算法 —— 全功能Agent组成集群，按负载分派任务

设计原则（第一性原理）：
1. 全功能：每个Agent都能独立完成完整任务
2. 负载均衡：任务分配给最空闲的Agent
3. 容错：任务超时自动重试，失败自动切换Agent
4. 弹性：Agent可随时加入或离开集群
"""

import os
import sys
import time
import threading
import uuid
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger
from core.multi_agent_bus import (
    MultiAgentBus, AgentMessage, AgentRole, MessageType, MessagePriority,
)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Task:
    """任务定义"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: str = ""           # predict/decide/execute/perceive
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = MessagePriority.NORMAL.value
    assigned_to: str = ""         # Agent ID
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "priority": self.priority,
            "assigned_to": self.assigned_to,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
        }


class GeneralAgent:
    """通用Agent —— 全功能，可执行任何任务"""

    def __init__(self, agent_id: str, bus: MultiAgentBus):
        self.agent_id = agent_id
        self.bus = bus
        self._running = False
        self._current_task: Optional[Task] = None
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self.bus.register_agent(self.agent_id, AgentRole.GENERAL,
                                capabilities=["predict", "decide", "execute", "perceive"])
        self.bus.subscribe(MessageType.TASK.value, self._on_task)
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        logger.info("GEN_AGENT", f"通用Agent启动: {self.agent_id}", module="general_agent_cluster")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.bus.unregister_agent(self.agent_id)
        logger.info("GEN_AGENT", f"通用Agent停止: {self.agent_id}", module="general_agent_cluster")

    def _heartbeat_loop(self) -> None:
        while self._running:
            load = 1.0 if self._current_task else 0.0
            self.bus.heartbeat(self.agent_id, load_score=load)
            time.sleep(5.0)

    def _on_task(self, msg: AgentMessage) -> None:
        """接收任务消息"""
        if msg.target != self.agent_id:
            return  # 不是发给我的

        task_data = msg.payload.get("task")
        if not task_data:
            return

        task = Task(
            task_id=task_data.get("task_id", ""),
            task_type=task_data.get("task_type", ""),
            payload=task_data.get("payload", {}),
            priority=task_data.get("priority", MessagePriority.NORMAL.value),
            assigned_to=self.agent_id,
        )
        self._execute_task(task)

    def _execute_task(self, task: Task) -> None:
        """执行任务"""
        self._current_task = task
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        try:
            # 模拟任务执行
            result = self._process(task.task_type, task.payload)
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            self._completed_tasks += 1

            # 发送结果
            self._send_result(task)
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            self._failed_tasks += 1
            self._send_result(task)
        finally:
            self._current_task = None

    def _process(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """实际处理逻辑"""
        if task_type == "predict":
            return {"prediction": payload.get("input", 0) * 1.05}
        elif task_type == "decide":
            val = payload.get("input", 0)
            return {"action": "throttle" if val > 80 else "boost" if val < 20 else "noop"}
        elif task_type == "execute":
            return {"executed": payload.get("action", "noop")}
        elif task_type == "perceive":
            return {"state": {"cpu": 50.0, "mem": 60.0}}
        else:
            return {"result": "unknown_task_type"}

    def _send_result(self, task: Task) -> None:
        msg = AgentMessage(
            msg_type=MessageType.TASK_RESULT.value,
            sender=self.agent_id,
            target="orchestrator",  # 发给调度中心
            priority=task.priority,
            payload={"task_id": task.task_id, "result": task.to_dict()},
        )
        self.bus.send(msg)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "completed_tasks": self._completed_tasks,
            "failed_tasks": self._failed_tasks,
            "busy": self._current_task is not None,
        }


class TaskDispatcher:
    """任务调度器 —— 负责任务分派、超时监控、重试"""

    def __init__(self, bus: MultiAgentBus):
        self.bus = bus
        self._tasks: Dict[str, Task] = {}
        self._results: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._timeout_sec = 10.0
        self._callback_registry: Dict[str, Callable] = {}

        # 订阅任务结果
        self.bus.subscribe(MessageType.TASK_RESULT.value, self._on_result)

    def start(self) -> None:
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("DISPATCHER", "任务调度器启动", module="general_agent_cluster")

    def stop(self) -> None:
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("DISPATCHER", "任务调度器停止", module="general_agent_cluster")

    def dispatch(self, task_type: str, payload: Dict[str, Any],
                 priority: int = MessagePriority.NORMAL.value,
                 callback: Optional[Callable] = None) -> str:
        """分派任务，返回task_id"""
        task = Task(task_type=task_type, payload=payload, priority=priority)

        with self._lock:
            self._tasks[task.task_id] = task
            if callback:
                self._callback_registry[task.task_id] = callback

        # 选择负载最低的通用Agent
        agent = self.bus.get_least_loaded_agent(AgentRole.GENERAL)
        if agent is None:
            task.status = TaskStatus.FAILED
            task.error = "无可用Agent"
            return task.task_id

        task.assigned_to = agent.agent_id
        task.status = TaskStatus.ASSIGNED

        # 发送任务
        msg = AgentMessage(
            msg_type=MessageType.TASK.value,
            sender="orchestrator",
            target=agent.agent_id,
            priority=priority,
            payload={"task": task.to_dict()},
        )
        self.bus.send(msg)

        logger.info("DISPATCHER", f"任务分派: {task.task_id} -> {agent.agent_id}",
                    module="general_agent_cluster")
        return task.task_id

    def _on_result(self, msg: AgentMessage) -> None:
        """接收任务结果"""
        result_data = msg.payload.get("result", {})
        task_id = result_data.get("task_id", "")

        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus(result_data.get("status", "failed"))
                task.result = result_data.get("result")
                task.error = result_data.get("error")
                task.completed_at = time.time()
                self._results[task_id] = result_data

                # 调用回调
                callback = self._callback_registry.pop(task_id, None)
                if callback:
                    try:
                        callback(task)
                    except Exception as e:
                        logger.warning("DISPATCHER", f"回调异常: {e}", module="general_agent_cluster")

    def _monitor_loop(self) -> None:
        """监控任务超时"""
        while self._running:
            now = time.time()
            with self._lock:
                for task in list(self._tasks.values()):
                    if task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.RUNNING):
                        if task.started_at and (now - task.started_at) > self._timeout_sec:
                            task.status = TaskStatus.TIMEOUT
                            if task.retry_count < task.max_retries:
                                task.retry_count += 1
                                task.status = TaskStatus.PENDING
                                task.started_at = None
                                # 重新分派
                                self._redispatch(task)
            time.sleep(2.0)

    def _redispatch(self, task: Task) -> None:
        """重新分派任务"""
        agent = self.bus.get_least_loaded_agent(AgentRole.GENERAL)
        if agent is None:
            return
        task.assigned_to = agent.agent_id
        msg = AgentMessage(
            msg_type=MessageType.TASK.value,
            sender="orchestrator",
            target=agent.agent_id,
            priority=task.priority,
            payload={"task": task.to_dict()},
        )
        self.bus.send(msg)
        logger.info("DISPATCHER", f"任务重试: {task.task_id} -> {agent.agent_id}",
                    module="general_agent_cluster")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.to_dict() if task else None

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._tasks)
            completed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED)
            pending = sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)
            return {
                "total_tasks": total,
                "completed": completed,
                "failed": failed,
                "pending": pending,
            }
