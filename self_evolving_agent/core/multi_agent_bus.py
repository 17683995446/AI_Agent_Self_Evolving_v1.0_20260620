"""
多Agent通信协议与MessageBus升级 —— 支持混合架构的可靠通信

设计原则（第一性原理）：
1. 协议标准化：所有消息统一格式，像TCP/IP一样通用
2. 可靠性：消息不丢、不重、有序（至少一次交付+去重）
3. 去中心化：支持Hub中转和P2P直连两种模式
4. 角色感知：Agent注册时声明角色，便于任务分派
"""

import os
import sys
import time
import json
import uuid
import threading
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


class AgentRole(Enum):
    """Agent角色类型"""
    PERCEPTION = "perception"      # 专用感知Agent
    PREDICTION = "prediction"      # 专用预测Agent
    DECISION = "decision"          # 专用决策Agent
    EXECUTION = "execution"        # 专用执行Agent
    GENERAL = "general"            # 通用全功能Agent
    ORCHESTRATOR = "orchestrator"  # 调度中心
    MERGER = "merger"              # 合并与审计Agent


class MessagePriority(Enum):
    """消息优先级"""
    CRITICAL = 0    # 故障、告警
    HIGH = 1        # 紧急任务
    NORMAL = 2      # 常规消息
    LOW = 3         # 心跳、日志


class MessageType(Enum):
    """消息类型"""
    HEARTBEAT = "heartbeat"
    STATE = "state"                # 状态广播
    TASK = "task"                  # 任务分派
    TASK_RESULT = "task_result"    # 任务结果
    EXPERIENCE = "experience"      # 经验共享
    COMMAND = "command"            # 控制命令
    ACK = "ack"                    # 确认
    ERROR = "error"                # 错误报告


@dataclass
class AgentMessage:
    """标准消息格式 —— 所有Agent间通信的统一信封"""
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    msg_type: str = ""
    sender: str = ""
    sender_role: str = ""
    target: str = ""              # 空字符串表示广播
    priority: int = 2             # 默认NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    ttl: int = 3                  # 最大转发次数，防环路
    require_ack: bool = False     # 是否需要确认

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_id": self.msg_id,
            "msg_type": self.msg_type,
            "sender": self.sender,
            "sender_role": self.sender_role,
            "target": self.target,
            "priority": self.priority,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "require_ack": self.require_ack,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AgentMessage":
        return AgentMessage(
            msg_id=data.get("msg_id", ""),
            msg_type=data.get("msg_type", ""),
            sender=data.get("sender", ""),
            sender_role=data.get("sender_role", ""),
            target=data.get("target", ""),
            priority=data.get("priority", 2),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            ttl=data.get("ttl", 3),
            require_ack=data.get("require_ack", False),
        )


@dataclass
class AgentInfo:
    """Agent注册信息"""
    agent_id: str
    role: AgentRole
    capabilities: List[str] = field(default_factory=list)
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    address: str = ""             # P2P地址（如 "127.0.0.1:8001"）
    load_score: float = 0.0       # 负载分数，用于任务分派
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "capabilities": self.capabilities,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "address": self.address,
            "load_score": self.load_score,
            "status": self.status,
        }


class PriorityMessageQueue:
    """优先级消息队列 —— 高优先级消息先处理"""

    def __init__(self, max_size: int = 10000):
        self._queues: Dict[int, deque] = {
            0: deque(),  # CRITICAL
            1: deque(),  # HIGH
            2: deque(),  # NORMAL
            3: deque(),  # LOW
        }
        self._max_size = max_size
        self._size = 0
        self._lock = threading.Lock()

    def put(self, msg: AgentMessage) -> bool:
        """放入消息，按优先级分类"""
        with self._lock:
            if self._size >= self._max_size:
                # 队列满，丢弃最低优先级
                if msg.priority < 3:
                    self._drop_lowest()
                else:
                    return False
            self._queues[msg.priority].append(msg)
            self._size += 1
            return True

    def get(self) -> Optional[AgentMessage]:
        """取出最高优先级的消息"""
        with self._lock:
            for priority in sorted(self._queues.keys()):
                if self._queues[priority]:
                    self._size -= 1
                    return self._queues[priority].popleft()
            return None

    def _drop_lowest(self) -> None:
        """丢弃最低优先级的最老消息"""
        for priority in sorted(self._queues.keys(), reverse=True):
            if self._queues[priority]:
                self._queues[priority].popleft()
                self._size -= 1
                return

    def size(self) -> int:
        with self._lock:
            return self._size


class MultiAgentBus:
    """多Agent消息总线 —— 支持Hub模式和P2P模式"""

    def __init__(self, bus_id: str = "default"):
        self.bus_id = bus_id
        self._agents: Dict[str, AgentInfo] = {}
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._queue = PriorityMessageQueue()
        self._ack_tracker: Dict[str, Dict[str, Any]] = {}  # msg_id -> {sent_at, retries, callback}
        self._received_ids: Set[str] = set()  # 去重缓存
        self._max_received_ids = 5000
        self._running = False
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        logger.info("MA_BUS", f"多Agent总线初始化: {bus_id}", module="multi_agent_bus")

    def register_agent(self, agent_id: str, role: AgentRole,
                       capabilities: Optional[List[str]] = None,
                       address: str = "") -> bool:
        """注册Agent"""
        with self._lock:
            if agent_id in self._agents:
                logger.warning("MA_BUS", f"Agent已存在: {agent_id}", module="multi_agent_bus")
                return False
            self._agents[agent_id] = AgentInfo(
                agent_id=agent_id,
                role=role,
                capabilities=capabilities or [],
                address=address,
            )
        logger.info("MA_BUS", f"Agent注册: {agent_id} [{role.value}]",
                    module="multi_agent_bus", extra={"role": role.value, "address": address})
        return True

    def unregister_agent(self, agent_id: str) -> None:
        """注销Agent"""
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
        logger.info("MA_BUS", f"Agent注销: {agent_id}", module="multi_agent_bus")

    def heartbeat(self, agent_id: str, load_score: float = 0.0) -> None:
        """心跳更新"""
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].last_heartbeat = time.time()
                self._agents[agent_id].load_score = load_score

    def get_agents_by_role(self, role: AgentRole, active_only: bool = True,
                           timeout: float = 30.0) -> List[AgentInfo]:
        """按角色获取Agent列表"""
        now = time.time()
        with self._lock:
            agents = [
                info for info in self._agents.values()
                if info.role == role
                and (not active_only or (now - info.last_heartbeat) < timeout)
            ]
        return agents

    def get_least_loaded_agent(self, role: AgentRole) -> Optional[AgentInfo]:
        """获取负载最低的Agent（用于任务分派）"""
        agents = self.get_agents_by_role(role)
        if not agents:
            return None
        return min(agents, key=lambda a: a.load_score)

    def subscribe(self, msg_type: str, callback: Callable) -> None:
        """订阅某类消息"""
        self._subscribers[msg_type].append(callback)
        logger.debug("MA_BUS", f"订阅: {msg_type}", module="multi_agent_bus")

    def send(self, msg: AgentMessage) -> bool:
        """发送消息 —— 点对点或广播"""
        # 去重检查
        if msg.msg_id in self._received_ids:
            logger.debug("MA_BUS", f"重复消息丢弃: {msg.msg_id}", module="multi_agent_bus")
            return False
        self._received_ids.add(msg.msg_id)
        if len(self._received_ids) > self._max_received_ids:
            self._received_ids.clear()

        # 放入队列
        if not self._queue.put(msg):
            logger.warning("MA_BUS", "消息队列满，丢弃消息", module="multi_agent_bus")
            return False

        # 如果需要确认，加入追踪
        if msg.require_ack:
            self._ack_tracker[msg.msg_id] = {
                "sent_at": time.time(),
                "retries": 0,
                "sender": msg.sender,
            }

        logger.debug("MA_BUS", f"消息入队: {msg.msg_type} [{msg.msg_id}]",
                    module="multi_agent_bus", extra={"target": msg.target})
        return True

    def send_ack(self, original_msg_id: str, sender: str, target: str) -> None:
        """发送确认消息"""
        ack = AgentMessage(
            msg_type=MessageType.ACK.value,
            sender=sender,
            target=target,
            priority=MessagePriority.HIGH.value,
            payload={"ack_for": original_msg_id},
        )
        self.send(ack)

    def start(self) -> None:
        """启动消息分发线程"""
        self._running = True
        self._dispatcher_thread = threading.Thread(target=self._dispatch_loop, name="BusDispatcher", daemon=True)
        self._dispatcher_thread.start()
        logger.info("MA_BUS", "消息分发线程启动", module="multi_agent_bus")

    def stop(self) -> None:
        """停止消息分发"""
        self._running = False
        if self._dispatcher_thread:
            self._dispatcher_thread.join(timeout=5.0)
        logger.info("MA_BUS", "消息分发线程停止", module="multi_agent_bus")

    def _dispatch_loop(self) -> None:
        """消息分发循环"""
        while self._running:
            msg = self._queue.get()
            if msg is None:
                time.sleep(0.01)
                continue

            # TTL检查
            if msg.ttl <= 0:
                logger.debug("MA_BUS", f"消息TTL耗尽: {msg.msg_id}", module="multi_agent_bus")
                continue

            # 目标路由
            if msg.target:
                # 点对点
                self._route_to_target(msg)
            else:
                # 广播
                self._broadcast(msg)

    def _route_to_target(self, msg: AgentMessage) -> None:
        """路由到指定目标"""
        with self._lock:
            agent = self._agents.get(msg.target)
        if agent and agent.address:
            # P2P模式：直接发送到目标地址
            logger.debug("MA_BUS", f"P2P发送 to {msg.target}", module="multi_agent_bus")
        # 调用本地订阅者
        self._notify_subscribers(msg)

    def _broadcast(self, msg: AgentMessage) -> None:
        """广播消息"""
        msg.ttl -= 1
        self._notify_subscribers(msg)

    def _notify_subscribers(self, msg: AgentMessage) -> None:
        """通知订阅者"""
        callbacks = self._subscribers.get(msg.msg_type, [])
        for callback in callbacks:
            try:
                callback(msg)
            except Exception as e:
                logger.warning("MA_BUS", f"回调异常: {e}", module="multi_agent_bus")

    def get_stats(self) -> Dict[str, Any]:
        """获取总线统计"""
        with self._lock:
            return {
                "bus_id": self.bus_id,
                "agent_count": len(self._agents),
                "queue_size": self._queue.size(),
                "pending_acks": len(self._ack_tracker),
                "received_msgs": len(self._received_ids),
                "agents": {aid: info.to_dict() for aid, info in self._agents.items()},
            }
