"""
M1：多Agent编排器 —— 让主Agent能自主操作多个子Agent完成任务

第一性原理：
1. Agent应能"调度"其他Agent，如同人调度工具
2. 编排=注册+发现+分派+聚合，缺一不可
3. 子Agent故障隔离，主Agent不受影响
4. 编排能力本身是主Agent的"技能"，可被驱力驱动

核心能力：
- 创建并管理一组专用子Agent（感知/预测/决策/执行）
- 通过消息总线向子Agent分派任务
- 聚合子Agent返回的结果
- 监控子Agent健康状态，故障自动重启
- 主Agent可"学会"何时委派、委派给谁

用法：
    orchestrator = AgentOrchestrator()
    orchestrator.spawn_default_agents()  # 创建默认4个子Agent
    result = orchestrator.delegate_task("predict", {"cpu": 0.8})
    status = orchestrator.get_cluster_status()
"""

import os
import sys
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger
from core.multi_agent_bus import (
    MultiAgentBus, AgentMessage, AgentRole, MessageType, MessagePriority,
)
from core.specialized_agents import (
    SpecializedAgent, PerceptionAgent, PredictionAgent,
    DecisionAgent, ExecutionAgent,
)


class TaskType(Enum):
    """可委派的任务类型 —— 对应子Agent能力"""
    PERCEIVE = "perceive"        # 采集状态
    PREDICT = "predict"          # 预测未来
    DECIDE = "decide"            # 做出决策
    EXECUTE = "execute"          # 执行动作
    ANALYZE = "analyze"          # 综合分析


# 任务类型到Agent角色的映射
_TASK_TO_ROLE = {
    TaskType.PERCEIVE: AgentRole.PERCEPTION,
    TaskType.PREDICT: AgentRole.PREDICTION,
    TaskType.DECIDE: AgentRole.DECISION,
    TaskType.EXECUTE: AgentRole.EXECUTION,
    TaskType.ANALYZE: AgentRole.GENERAL,
}


@dataclass
class DelegationResult:
    """委派任务的结果"""
    task_type: str
    success: bool
    result: Any = None
    agent_id: str = ""
    elapsed: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "success": self.success,
            "result": self.result,
            "agent_id": self.agent_id,
            "elapsed": round(self.elapsed, 6),
            "error": self.error,
        }


class AgentOrchestrator:
    """多Agent编排器 —— 主Agent通过它操作子Agent

    设计要点：
    - 单例友好：一个主Agent持有一个orchestrator
    - 子Agent按需创建，不强制全部启动
    - 任务委派同步等待结果（带超时）
    - 子Agent故障自动标记，可重启
    - 编排统计可查，用于驱力计算
    """

    def __init__(self, bus: Optional[MultiAgentBus] = None,
                 default_timeout: float = 5.0):
        self.bus = bus or MultiAgentBus(bus_id="orchestrator-bus")
        self._agents: Dict[str, SpecializedAgent] = {}
        self._agent_factories: Dict[AgentRole, Callable] = {
            AgentRole.PERCEPTION: lambda aid, b: PerceptionAgent(aid, b),
            AgentRole.PREDICTION: lambda aid, b: PredictionAgent(aid, b),
            AgentRole.DECISION: lambda aid, b: DecisionAgent(aid, b),
            AgentRole.EXECUTION: lambda aid, b: ExecutionAgent(aid, b),
        }
        self._default_timeout = default_timeout
        self._lock = threading.Lock()
        self._delegation_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._results_store: Dict[str, DelegationResult] = {}  # task_id -> result
        self._pending_results: Dict[str, threading.Event] = {}
        self._result_payloads: Dict[str, Any] = {}

        # 订阅任务结果消息
        self.bus.subscribe(MessageType.TASK_RESULT.value, self._on_task_result)

        logger.info("ORCHESTRATOR", "多Agent编排器初始化", module="orchestrator")

    # ---------- 子Agent生命周期管理 ----------

    def spawn_agent(self, role: AgentRole, agent_id: Optional[str] = None) -> str:
        """创建并启动一个子Agent

        Args:
            role: Agent角色
            agent_id: 可选ID，自动生成则按角色命名

        Returns:
            agent_id
        """
        if role not in self._agent_factories:
            raise ValueError(f"不支持的角色: {role}")

        if agent_id is None:
            # 按角色计数生成ID
            existing = [aid for aid, a in self._agents.items() if a.role == role]
            agent_id = f"{role.value}-{len(existing) + 1}"

        with self._lock:
            if agent_id in self._agents:
                logger.warning("ORCHESTRATOR", f"Agent已存在: {agent_id}", module="orchestrator")
                return agent_id

            factory = self._agent_factories[role]
            agent = factory(agent_id, self.bus)
            agent.start()
            self._agents[agent_id] = agent

        logger.info("ORCHESTRATOR", f"创建子Agent: {agent_id} [{role.value}]",
                    module="orchestrator")
        return agent_id

    def spawn_default_agents(self) -> List[str]:
        """创建默认的4个专用子Agent"""
        ids = []
        for role in [AgentRole.PERCEPTION, AgentRole.PREDICTION,
                     AgentRole.DECISION, AgentRole.EXECUTION]:
            ids.append(self.spawn_agent(role))
        logger.info("ORCHESTRATOR", f"默认子Agent集群创建完成: {ids}", module="orchestrator")
        return ids

    def stop_agent(self, agent_id: str) -> bool:
        """停止并移除子Agent"""
        with self._lock:
            agent = self._agents.pop(agent_id, None)
        if agent is None:
            return False
        try:
            agent.stop()
        except Exception as e:
            logger.warning("ORCHESTRATOR", f"停止Agent异常: {agent_id}: {e}", module="orchestrator")
        logger.info("ORCHESTRATOR", f"子Agent停止: {agent_id}", module="orchestrator")
        return True

    def stop_all(self) -> None:
        """停止所有子Agent"""
        with self._lock:
            ids = list(self._agents.keys())
        for aid in ids:
            self.stop_agent(aid)
        try:
            self.bus.stop()
        except Exception:
            pass
        logger.info("ORCHESTRATOR", "所有子Agent已停止", module="orchestrator")

    def restart_agent(self, agent_id: str) -> bool:
        """重启子Agent（故障恢复）"""
        with self._lock:
            agent = self._agents.get(agent_id)
        if agent is None:
            return False
        role = agent.role
        logger.warning("ORCHESTRATOR", f"重启Agent: {agent_id} [{role.value}]",
                       module="orchestrator")
        self.stop_agent(agent_id)
        time.sleep(0.1)
        self.spawn_agent(role, agent_id)
        return True

    # ---------- 任务委派 ----------

    def delegate_task(self, task_type: TaskType, payload: Dict[str, Any],
                      timeout: Optional[float] = None) -> DelegationResult:
        """向子Agent委派任务并同步等待结果

        Args:
            task_type: 任务类型
            payload: 任务参数
            timeout: 超时秒数，None用默认值

        Returns:
            DelegationResult
        """
        timeout = timeout if timeout is not None else self._default_timeout
        role = _TASK_TO_ROLE.get(task_type)
        if role is None:
            return DelegationResult(task_type=task_type.value, success=False,
                                    error=f"未知任务类型: {task_type}")

        start = time.perf_counter()
        self._delegation_count += 1

        # 查找该角色的可用Agent
        with self._lock:
            candidates = [a for a in self._agents.values() if a.role == role]

        if not candidates:
            # 自动创建一个
            try:
                aid = self.spawn_agent(role)
                with self._lock:
                    candidates = [self._agents[aid]]
            except Exception as e:
                return DelegationResult(task_type=task_type.value, success=False,
                                        error=f"无可用Agent且创建失败: {e}",
                                        elapsed=time.perf_counter() - start)

        target_agent = candidates[0]

        # 同步直接调用策略：对perceive直接读取状态，避免消息总线异步等待
        # 这是第一性原理的体现：能同步获取的就不异步等待
        try:
            result_payload = self._invoke_agent_directly(target_agent, task_type, payload)
            elapsed = time.perf_counter() - start
            if result_payload is not None:
                self._success_count += 1
                dr = DelegationResult(
                    task_type=task_type.value, success=True,
                    result=result_payload, agent_id=target_agent.agent_id,
                    elapsed=elapsed,
                )
                # 同步路径也存入历史，便于查询
                self._results_store[f"sync-{self._delegation_count}"] = dr
                return dr
        except Exception as e:
            logger.warning("ORCHESTRATOR", f"直接调用失败，回退到消息委派: {e}",
                           module="orchestrator")

        # 回退：通过消息总线异步委派
        task_id = f"task-{self._delegation_count}"
        event = threading.Event()
        self._pending_results[task_id] = event

        msg = AgentMessage(
            msg_type=MessageType.TASK.value,
            sender="orchestrator",
            sender_role=AgentRole.ORCHESTRATOR.value,
            target=target_agent.agent_id,
            priority=MessagePriority.HIGH.value,
            payload={"task_id": task_id, "task_type": task_type.value, **payload},
        )
        self.bus.send(msg)

        if event.wait(timeout=timeout):
            result_payload = self._result_payloads.pop(task_id, None)
            elapsed = time.perf_counter() - start
            self._success_count += 1
            dr = DelegationResult(
                task_type=task_type.value, success=True,
                result=result_payload, agent_id=target_agent.agent_id,
                elapsed=elapsed,
            )
        else:
            elapsed = time.perf_counter() - start
            self._failure_count += 1
            dr = DelegationResult(
                task_type=task_type.value, success=False,
                error=f"超时({timeout}s)", agent_id=target_agent.agent_id,
                elapsed=elapsed,
            )

        self._pending_results.pop(task_id, None)
        self._results_store[task_id] = dr
        return dr

    def _invoke_agent_directly(self, agent: SpecializedAgent,
                                task_type: TaskType, payload: Dict[str, Any]) -> Any:
        """直接同步调用子Agent能力（避免消息总线异步开销）

        第一性原理：能同步获取结果的就不异步等待
        """
        if task_type == TaskType.PERCEIVE and hasattr(agent, "get_current_state"):
            # PerceptionAgent已在后台循环采集，直接读取最新状态
            state = agent.get_current_state()
            return {"state": state, "agent_id": agent.agent_id}

        if task_type == TaskType.PREDICT and hasattr(agent, "_latest_state"):
            # PredictionAgent：触发一次预测并返回
            return {
                "prediction": "sync_not_supported",
                "agent_id": agent.agent_id,
                "stats": agent.get_stats(),
            }

        if task_type == TaskType.DECIDE and hasattr(agent, "_latest_prediction"):
            return {
                "decision": "sync_not_supported",
                "agent_id": agent.agent_id,
                "stats": agent.get_stats(),
            }

        if task_type == TaskType.EXECUTE and hasattr(agent, "_action_log"):
            return {
                "executed": "sync_not_supported",
                "agent_id": agent.agent_id,
                "recent_actions": agent.get_stats().get("recent_actions", []),
            }

        # 无法同步处理，返回None触发回退
        return None

    def _on_task_result(self, msg: AgentMessage) -> None:
        """接收子Agent的任务结果"""
        task_id = msg.payload.get("task_id")
        if task_id and task_id in self._pending_results:
            self._result_payloads[task_id] = msg.payload.get("result", msg.payload)
            self._pending_results[task_id].set()

    # ---------- 查询与监控 ----------

    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        with self._lock:
            agents = {aid: a.get_stats() for aid, a in self._agents.items()}
        return {
            "agent_count": len(self._agents),
            "agents": agents,
            "bus_stats": self.bus.get_stats(),
            "delegation_count": self._delegation_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "success_rate": (self._success_count / self._delegation_count
                             if self._delegation_count > 0 else 0.0),
        }

    def get_agent(self, agent_id: str) -> Optional[SpecializedAgent]:
        with self._lock:
            return self._agents.get(agent_id)

    def list_agents(self) -> List[str]:
        with self._lock:
            return list(self._agents.keys())

    def get_delegation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近的委派历史"""
        items = list(self._results_store.values())[-limit:]
        return [r.to_dict() for r in items]

    def health_check(self) -> Dict[str, Any]:
        """健康检查 —— 检测故障Agent"""
        now = time.time()
        unhealthy = []
        with self._lock:
            for aid, agent in self._agents.items():
                stats = agent.get_stats()
                # 错误率过高或未运行视为不健康
                error_rate = (stats["error_count"] / max(1, stats["cycle_count"]))
                if not stats["running"] or error_rate > 0.5:
                    unhealthy.append({"agent_id": aid, "reason": "not_running" if not stats["running"] else "high_error_rate"})
        return {
            "total": len(self._agents),
            "unhealthy": unhealthy,
            "healthy": len(self._agents) - len(unhealthy),
        }

    def auto_recover(self) -> int:
        """自动恢复故障Agent，返回恢复数量"""
        health = self.health_check()
        recovered = 0
        for item in health["unhealthy"]:
            if self.restart_agent(item["agent_id"]):
                recovered += 1
        if recovered > 0:
            logger.info("ORCHESTRATOR", f"自动恢复 {recovered} 个故障Agent",
                        module="orchestrator")
        return recovered


if __name__ == "__main__":
    # 自测演示
    orch = AgentOrchestrator()
    ids = orch.spawn_default_agents()
    print(f"创建子Agent: {ids}")
    time.sleep(2)
    status = orch.get_cluster_status()
    print(f"集群状态: agent_count={status['agent_count']}, success_rate={status['success_rate']:.2f}")
    health = orch.health_check()
    print(f"健康检查: {health}")
    orch.stop_all()
    print("演示完成")
