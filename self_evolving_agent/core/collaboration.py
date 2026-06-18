"""
多Agent协作框架 —— 实现Agent间经验共享和分布式学习

设计原则：
1. 松耦合：Agent间不直接依赖，通过消息总线通信
2. 最终一致性：经验共享允许延迟和丢失
3. 防冲突：相同经验不重复积分，避免驱力计算偏差
"""

import os
import sys
import time
import json
import uuid
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


@dataclass
class SharedExperience:
    """共享经验 —— Agent间传递的学习成果"""
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_agent: str = ""
    timestamp: float = field(default_factory=time.time)
    state: Dict[str, Any] = field(default_factory=dict)
    action: str = ""
    reward: float = 0.0
    next_state: Dict[str, Any] = field(default_factory=dict)
    td_error: float = 0.0
    tags: List[str] = field(default_factory=list)
    # 防重复标记
    origin_fingerprint: str = ""  # 原始经验指纹，用于去重

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "source_agent": self.source_agent,
            "timestamp": self.timestamp,
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "td_error": self.td_error,
            "tags": self.tags,
            "origin_fingerprint": self.origin_fingerprint,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SharedExperience":
        return SharedExperience(
            experience_id=data.get("experience_id", ""),
            source_agent=data.get("source_agent", ""),
            timestamp=data.get("timestamp", time.time()),
            state=data.get("state", {}),
            action=data.get("action", ""),
            reward=data.get("reward", 0.0),
            next_state=data.get("next_state", {}),
            td_error=data.get("td_error", 0.0),
            tags=data.get("tags", []),
            origin_fingerprint=data.get("origin_fingerprint", ""),
        )


class MessageBus:
    """消息总线 —— 轻量级Agent间通信机制（内存模式，可扩展为网络模式）"""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._message_queue: deque = deque(maxlen=1000)
        self._agent_registry: Dict[str, Dict[str, Any]] = {}
        logger.info("COLLAB_BUS", "消息总线初始化", module="collaboration")

    def register_agent(self, agent_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """注册Agent到总线"""
        self._agent_registry[agent_id] = {
            "registered_at": time.time(),
            "metadata": metadata or {},
            "last_heartbeat": time.time(),
        }
        logger.info("COLLAB_BUS", f"Agent注册: {agent_id}", module="collaboration",
                    extra={"agent_id": agent_id, "metadata": metadata})

    def unregister_agent(self, agent_id: str) -> None:
        """注销Agent"""
        if agent_id in self._agent_registry:
            del self._agent_registry[agent_id]
            logger.info("COLLAB_BUS", f"Agent注销: {agent_id}", module="collaboration")

    def heartbeat(self, agent_id: str) -> None:
        """Agent心跳更新"""
        if agent_id in self._agent_registry:
            self._agent_registry[agent_id]["last_heartbeat"] = time.time()

    def get_active_agents(self, timeout: float = 30.0) -> List[str]:
        """获取活跃Agent列表"""
        now = time.time()
        return [
            aid for aid, info in self._agent_registry.items()
            if (now - info["last_heartbeat"]) < timeout
        ]

    def subscribe(self, topic: str, callback: Callable) -> None:
        """订阅主题"""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)
        logger.debug("COLLAB_BUS", f"订阅主题: {topic}", module="collaboration")

    def publish(self, topic: str, message: Any, sender: str = "") -> None:
        """发布消息到主题"""
        self._message_queue.append({"topic": topic, "message": message, "sender": sender, "time": time.time()})
        if topic in self._subscribers:
            for callback in self._subscribers[topic]:
                try:
                    callback(message, sender)
                except Exception as e:
                    logger.warning("COLLAB_BUS", f"消息处理异常: {e}", module="collaboration")
        logger.debug("COLLAB_BUS", f"发布消息: {topic}", module="collaboration", extra={"sender": sender})

    def get_messages(self, topic: Optional[str] = None, since: float = 0.0) -> List[Dict[str, Any]]:
        """获取消息历史"""
        messages = [m for m in self._message_queue if m["time"] >= since]
        if topic:
            messages = [m for m in messages if m["topic"] == topic]
        return list(messages)


class CollaborationHub:
    """协作中心 —— 管理Agent间的经验共享和冲突解决"""

    def __init__(self, agent_id: str, message_bus: Optional[MessageBus] = None):
        self.agent_id = agent_id
        self.bus = message_bus or MessageBus()
        self.bus.register_agent(agent_id)
        # 去重缓存：已接收的经验指纹
        self._received_fingerprints: set = set()
        self._received_fingerprints_limit = 1000
        # 共享经验缓冲区
        self._incoming_experiences: deque = deque(maxlen=100)
        # 统计
        self.shared_count = 0
        self.received_count = 0
        self.deduplicated_count = 0
        logger.info("COLLAB_HUB", f"协作中心初始化: {agent_id}", module="collaboration")

    def _compute_fingerprint(self, state: Dict[str, Any], action: str, reward: float,
                              next_state: Dict[str, Any]) -> str:
        """计算经验指纹，用于去重 —— 基于内容而非时间戳"""
        content = f"{json.dumps(state, sort_keys=True)}|{action}|{reward:.4f}|{json.dumps(next_state, sort_keys=True)}"
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def share_experience(self, state: Dict[str, Any], action: str, reward: float,
                         next_state: Dict[str, Any], td_error: float = 0.0,
                         tags: Optional[List[str]] = None) -> SharedExperience:
        """分享经验给其他Agent"""
        fingerprint = self._compute_fingerprint(state, action, reward, next_state)
        exp = SharedExperience(
            source_agent=self.agent_id,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            td_error=td_error,
            tags=tags or [],
            origin_fingerprint=fingerprint,
        )
        self.bus.publish("experience", exp.to_dict(), sender=self.agent_id)
        self.shared_count += 1
        logger.info("COLLAB_SHARE", f"分享经验: {exp.experience_id}", module="collaboration",
                    extra={"fingerprint": fingerprint, "action": action})
        return exp

    def on_receive_experience(self, message: Dict[str, Any], sender: str) -> Optional[SharedExperience]:
        """接收其他Agent的经验 —— 带去重和冲突解决"""
        if sender == self.agent_id:
            return None  # 忽略自己的消息

        try:
            exp = SharedExperience.from_dict(message)
        except Exception as e:
            logger.warning("COLLAB_RECV", f"经验解析失败: {e}", module="collaboration")
            return None

        # 去重检查
        if exp.origin_fingerprint in self._received_fingerprints:
            self.deduplicated_count += 1
            logger.debug("COLLAB_RECV", f"重复经验已忽略: {exp.experience_id}", module="collaboration")
            return None

        # 记录指纹
        self._received_fingerprints.add(exp.origin_fingerprint)
        if len(self._received_fingerprints) > self._received_fingerprints_limit:
            # 简单清理：清空后重新构建（实际可用LRU）
            self._received_fingerprints.clear()

        # 冲突解决：如果同一状态有多个不同动作，保留TD误差最小的
        self._resolve_conflicts(exp)

        self._incoming_experiences.append(exp)
        self.received_count += 1
        logger.info("COLLAB_RECV", f"接收经验: {exp.experience_id} 来自 {sender}", module="collaboration",
                    extra={"sender": sender, "action": exp.action, "td_error": exp.td_error})
        return exp

    def _resolve_conflicts(self, new_exp: SharedExperience) -> None:
        """冲突解决：同一状态下保留最优经验"""
        state_key = json.dumps(new_exp.state, sort_keys=True)
        # 查找同一状态的经验
        same_state_exps = [
            e for e in self._incoming_experiences
            if json.dumps(e.state, sort_keys=True) == state_key
        ]
        if same_state_exps:
            # 保留TD误差绝对值最小的（最可靠的经验）
            best = min(same_state_exps, key=lambda e: abs(e.td_error))
            if abs(new_exp.td_error) < abs(best.td_error):
                # 新经验更优，移除旧的
                try:
                    self._incoming_experiences.remove(best)
                    logger.debug("COLLAB_RESOLVE", f"冲突解决: 新经验更优，替换旧经验", module="collaboration",
                                extra={"old_action": best.action, "new_action": new_exp.action})
                except ValueError:
                    pass

    def get_shared_experiences(self, n: int = 10) -> List[SharedExperience]:
        """获取接收到的共享经验"""
        return list(self._incoming_experiences)[-n:]

    def get_stats(self) -> Dict[str, Any]:
        """获取协作统计"""
        return {
            "agent_id": self.agent_id,
            "shared_count": self.shared_count,
            "received_count": self.received_count,
            "deduplicated_count": self.deduplicated_count,
            "pending_experiences": len(self._incoming_experiences),
            "active_peers": len(self.bus.get_active_agents()),
        }

    def start_listening(self) -> None:
        """开始监听经验共享消息"""
        self.bus.subscribe("experience", self.on_receive_experience)
        logger.info("COLLAB_HUB", "开始监听经验共享", module="collaboration")

    def stop(self) -> None:
        """停止协作"""
        self.bus.unregister_agent(self.agent_id)
        logger.info("COLLAB_HUB", "协作中心停止", module="collaboration")
