"""
专用Agent小组 —— 感知/预测/决策/执行各一个Agent，模块分工协作

设计原则（第一性原理）：
1. 单一职责：每个Agent只做一件事，像工厂流水线
2. 故障隔离：一个Agent崩溃不影响其他Agent
3. 松耦合：通过消息总线通信，不直接调用
4. 可替换：任何Agent可随时升级或替换
"""

import os
import sys
import time
import threading
from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger
from core.multi_agent_bus import MultiAgentBus, AgentMessage, AgentRole, MessageType, MessagePriority


class SpecializedAgent(ABC):
    """专用Agent基类 —— 封装生命周期、心跳、消息处理"""

    def __init__(self, agent_id: str, role: AgentRole, bus: Optional[MultiAgentBus] = None):
        self.agent_id = agent_id
        self.role = role
        self.bus = bus or MultiAgentBus()
        self._running = False
        self._main_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_interval = 10.0
        self._cycle_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None

    def start(self) -> None:
        """启动Agent"""
        self._running = True
        self.bus.register_agent(self.agent_id, self.role)
        self.bus.start()

        # 订阅消息
        self._subscribe_messages()

        # 启动主循环线程
        self._main_thread = threading.Thread(target=self._main_loop, name=f"{self.agent_id}-main", daemon=True)
        self._main_thread.start()

        # 启动心跳线程
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name=f"{self.agent_id}-hb", daemon=True)
        self._heartbeat_thread.start()

        logger.info("SPEC_AGENT", f"Agent启动: {self.agent_id} [{self.role.value}]", module="specialized_agents")

    def stop(self) -> None:
        """停止Agent"""
        self._running = False
        if self._main_thread:
            self._main_thread.join(timeout=5.0)
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
        self.bus.unregister_agent(self.agent_id)
        self.bus.stop()
        logger.info("SPEC_AGENT", f"Agent停止: {self.agent_id}", module="specialized_agents")

    def _subscribe_messages(self) -> None:
        """订阅消息 —— 子类可覆盖"""
        pass

    def _main_loop(self) -> None:
        """主循环 —— 子类必须实现"""
        while self._running:
            try:
                self._cycle()
                self._cycle_count += 1
            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                logger.error("SPEC_AGENT", f"Agent {self.agent_id} 循环异常: {e}", module="specialized_agents")
                time.sleep(1.0)

    @abstractmethod
    def _cycle(self) -> None:
        """单次循环逻辑 —— 子类实现"""
        pass

    def _heartbeat_loop(self) -> None:
        """心跳循环"""
        while self._running:
            self.bus.heartbeat(self.agent_id, load_score=self._get_load_score())
            time.sleep(self._heartbeat_interval)

    def _get_load_score(self) -> float:
        """获取负载分数 —— 子类可覆盖"""
        return float(self._cycle_count)

    def get_stats(self) -> Dict[str, Any]:
        """获取Agent统计"""
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "cycle_count": self._cycle_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "running": self._running,
        }


class PerceptionAgent(SpecializedAgent):
    """感知Agent —— 负责环境状态采集和广播"""

    def __init__(self, agent_id: str = "perception-1", bus: Optional[MultiAgentBus] = None):
        super().__init__(agent_id, AgentRole.PERCEPTION, bus)
        self._state: Dict[str, Any] = {}

    def _subscribe_messages(self) -> None:
        pass  # 感知Agent只发送不接收

    def _cycle(self) -> None:
        """采集状态并广播"""
        import psutil
        self._state = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "timestamp": time.time(),
        }

        msg = AgentMessage(
            msg_type=MessageType.STATE.value,
            sender=self.agent_id,
            sender_role=self.role.value,
            priority=MessagePriority.NORMAL.value,
            payload=self._state,
        )
        self.bus.send(msg)
        time.sleep(1.0)

    def get_current_state(self) -> Dict[str, Any]:
        return self._state.copy()


class PredictionAgent(SpecializedAgent):
    """预测Agent —— 接收状态，输出预测"""

    def __init__(self, agent_id: str = "prediction-1", bus: Optional[MultiAgentBus] = None):
        super().__init__(agent_id, AgentRole.PREDICTION, bus)
        self._latest_state: Optional[Dict[str, Any]] = None
        self._prediction_count = 0

    def _subscribe_messages(self) -> None:
        self.bus.subscribe(MessageType.STATE.value, self._on_state)

    def _on_state(self, msg: AgentMessage) -> None:
        """接收状态消息"""
        self._latest_state = msg.payload

    def _cycle(self) -> None:
        """基于最新状态做预测"""
        if self._latest_state is None:
            time.sleep(0.5)
            return

        # 简单预测：当前趋势外推
        cpu = self._latest_state.get("cpu_percent", 0)
        mem = self._latest_state.get("memory_percent", 0)
        prediction = {
            "predicted_cpu_next": cpu * 1.05,  # 假设增长5%
            "predicted_mem_next": mem * 1.02,
            "confidence": 0.85,
            "based_on_state": self._latest_state.get("timestamp"),
        }

        msg = AgentMessage(
            msg_type=MessageType.TASK_RESULT.value,
            sender=self.agent_id,
            sender_role=self.role.value,
            priority=MessagePriority.NORMAL.value,
            payload={"type": "prediction", "data": prediction},
        )
        self.bus.send(msg)
        self._prediction_count += 1
        time.sleep(1.0)

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats["prediction_count"] = self._prediction_count
        stats["has_state"] = self._latest_state is not None
        return stats


class DecisionAgent(SpecializedAgent):
    """决策Agent —— 接收预测，做出决策"""

    def __init__(self, agent_id: str = "decision-1", bus: Optional[MultiAgentBus] = None):
        super().__init__(agent_id, AgentRole.DECISION, bus)
        self._latest_prediction: Optional[Dict[str, Any]] = None
        self._decision_count = 0

    def _subscribe_messages(self) -> None:
        self.bus.subscribe(MessageType.TASK_RESULT.value, self._on_prediction)

    def _on_prediction(self, msg: AgentMessage) -> None:
        """接收预测结果"""
        if msg.payload.get("type") == "prediction":
            self._latest_prediction = msg.payload.get("data")

    def _cycle(self) -> None:
        """基于预测做决策"""
        if self._latest_prediction is None:
            time.sleep(0.5)
            return

        pred_cpu = self._latest_prediction.get("predicted_cpu_next", 0)
        action = "noop"
        if pred_cpu > 80:
            action = "throttle"
        elif pred_cpu < 20:
            action = "boost"

        decision = {
            "action": action,
            "reason": f"predicted_cpu={pred_cpu:.1f}",
            "timestamp": time.time(),
        }

        msg = AgentMessage(
            msg_type=MessageType.TASK.value,
            sender=self.agent_id,
            sender_role=self.role.value,
            priority=MessagePriority.HIGH.value,
            payload={"type": "decision", "data": decision},
        )
        self.bus.send(msg)
        self._decision_count += 1
        time.sleep(1.0)

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats["decision_count"] = self._decision_count
        stats["has_prediction"] = self._latest_prediction is not None
        return stats


class ExecutionAgent(SpecializedAgent):
    """执行Agent —— 接收决策，执行动作"""

    def __init__(self, agent_id: str = "execution-1", bus: Optional[MultiAgentBus] = None):
        super().__init__(agent_id, AgentRole.EXECUTION, bus)
        self._latest_decision: Optional[Dict[str, Any]] = None
        self._execution_count = 0
        self._action_log: list = []

    def _subscribe_messages(self) -> None:
        self.bus.subscribe(MessageType.TASK.value, self._on_decision)

    def _on_decision(self, msg: AgentMessage) -> None:
        """接收决策消息"""
        if msg.payload.get("type") == "decision":
            self._latest_decision = msg.payload.get("data")

    def _cycle(self) -> None:
        """执行决策"""
        if self._latest_decision is None:
            time.sleep(0.5)
            return

        action = self._latest_decision.get("action", "noop")
        self._action_log.append({
            "action": action,
            "time": time.time(),
            "reason": self._latest_decision.get("reason", ""),
        })
        self._execution_count += 1

        logger.info("EXEC_AGENT", f"执行动作: {action}", module="specialized_agents",
                    extra={"reason": self._latest_decision.get("reason", "")})

        # 执行后清空，等待下一个决策
        self._latest_decision = None
        time.sleep(1.0)

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats["execution_count"] = self._execution_count
        stats["recent_actions"] = self._action_log[-5:]
        return stats
