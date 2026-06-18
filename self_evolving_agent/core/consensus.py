"""
经验共享、冲突解决与共识决策 —— 多Agent协作的核心协调机制

设计原则（第一性原理）：
1. 经验共享：学习成果在集群中流动，加速整体进化
2. 冲突解决：同一状态不同动作时，用TD误差投票，误差小的更可信
3. 共识决策：多个Agent独立决策，投票选出最优动作
4. 反脆弱：分歧是信息，冲突是改进机会
"""

import os
import sys
import time
import json
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger
from core.multi_agent_bus import MultiAgentBus, AgentMessage, MessageType, MessagePriority


@dataclass
class Experience:
    """经验 —— 状态-动作-奖励三元组"""
    state_key: str
    action: str
    reward: float
    td_error: float = 0.0
    source_agent: str = ""
    timestamp: float = field(default_factory=time.time)

    def fingerprint(self) -> str:
        """内容指纹"""
        content = f"{self.state_key}|{self.action}|{self.reward:.4f}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class ExperiencePool:
    """经验池 —— 收集、去重、冲突解决"""

    def __init__(self, max_size: int = 1000):
        self._experiences: Dict[str, List[Experience]] = defaultdict(list)  # state_key -> experiences
        self._fingerprints: set = set()
        self._max_size = max_size
        self._conflict_resolved = 0
        self._deduplicated = 0

    def add(self, exp: Experience) -> bool:
        """添加经验，自动去重和冲突解决"""
        fp = exp.fingerprint()
        if fp in self._fingerprints:
            self._deduplicated += 1
            return False

        self._fingerprints.add(fp)
        state_exps = self._experiences[exp.state_key]

        # 冲突解决：同一状态下，保留TD误差绝对值最小的经验
        same_action = [e for e in state_exps if e.action == exp.action]
        if same_action:
            # 已有相同动作，保留误差更小的
            best = min(same_action, key=lambda e: abs(e.td_error))
            if abs(exp.td_error) < abs(best.td_error):
                state_exps.remove(best)
                state_exps.append(exp)
                self._conflict_resolved += 1
                logger.debug("CONSENSUS", f"冲突解决: 替换更优经验 {exp.action}", module="consensus")
            return True

        # 不同动作，检查是否有冲突
        if state_exps:
            self._conflict_resolved += 1
            logger.debug("CONSENSUS", f"冲突检测: state={exp.state_key} 有新动作 {exp.action}", module="consensus")

        state_exps.append(exp)

        # 容量控制
        if len(self._fingerprints) > self._max_size * 2:
            self._cleanup()

        return True

    def get_best_action(self, state_key: str) -> Optional[str]:
        """获取某状态下的最优动作（TD误差最小）"""
        exps = self._experiences.get(state_key, [])
        if not exps:
            return None
        best = min(exps, key=lambda e: abs(e.td_error))
        return best.action

    def get_all_actions(self, state_key: str) -> List[Experience]:
        """获取某状态下的所有经验"""
        return list(self._experiences.get(state_key, []))

    def get_stats(self) -> Dict[str, Any]:
        total_exps = sum(len(v) for v in self._experiences.values())
        return {
            "states": len(self._experiences),
            "total_experiences": total_exps,
            "fingerprints": len(self._fingerprints),
            "conflict_resolved": self._conflict_resolved,
            "deduplicated": self._deduplicated,
        }

    def _cleanup(self) -> None:
        """清理旧经验"""
        # 简单策略：清空所有，实际可用LRU
        self._experiences.clear()
        self._fingerprints.clear()
        logger.info("CONSENSUS", "经验池清理完成", module="consensus")


class ConsensusVoter:
    """共识投票器 —— 收集多个Agent的决策，投票选出最优"""

    def __init__(self, min_votes: int = 2):
        self._votes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # topic -> votes
        self._min_votes = min_votes
        self._decisions: Dict[str, Dict[str, Any]] = {}

    def vote(self, topic: str, agent_id: str, proposal: str,
             confidence: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> None:
        """投票"""
        self._votes[topic].append({
            "agent_id": agent_id,
            "proposal": proposal,
            "confidence": confidence,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })

    def get_consensus(self, topic: str, timeout_sec: float = 5.0) -> Optional[Dict[str, Any]]:
        """获取共识结果"""
        votes = self._votes.get(topic, [])
        if len(votes) < self._min_votes:
            return None

        # 按proposal统计加权票数
        proposals: Dict[str, float] = defaultdict(float)
        for v in votes:
            proposals[v["proposal"]] += v["confidence"]

        if not proposals:
            return None

        winner = max(proposals, key=lambda p: proposals[p])
        total_votes = sum(proposals.values())
        agreement = proposals[winner] / total_votes if total_votes > 0 else 0

        result = {
            "topic": topic,
            "consensus": winner,
            "agreement_rate": round(agreement, 3),
            "total_votes": len(votes),
            "vote_distribution": dict(proposals),
        }
        self._decisions[topic] = result
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "topics": len(self._votes),
            "decisions": len(self._decisions),
            "min_votes_required": self._min_votes,
        }


class ConsensusManager:
    """共识管理器 —— 整合经验共享、冲突解决、共识投票"""

    def __init__(self, bus: Optional[MultiAgentBus] = None):
        self.bus = bus or MultiAgentBus()
        self.experience_pool = ExperiencePool()
        self.voter = ConsensusVoter()
        self._running = False

        if self.bus:
            self.bus.subscribe(MessageType.EXPERIENCE.value, self._on_experience)
            self.bus.subscribe(MessageType.TASK.value, self._on_vote_request)

    def start(self) -> None:
        self._running = True
        self.bus.start()
        logger.info("CONSENSUS", "共识管理器启动", module="consensus")

    def stop(self) -> None:
        self._running = False
        self.bus.stop()
        logger.info("CONSENSUS", "共识管理器停止", module="consensus")

    def _on_experience(self, msg: AgentMessage) -> None:
        """接收经验消息"""
        data = msg.payload
        exp = Experience(
            state_key=data.get("state_key", ""),
            action=data.get("action", ""),
            reward=data.get("reward", 0.0),
            td_error=data.get("td_error", 0.0),
            source_agent=msg.sender,
        )
        if self.experience_pool.add(exp):
            logger.debug("CONSENSUS", f"经验已接收: {exp.fingerprint()}", module="consensus")

    def _on_vote_request(self, msg: AgentMessage) -> None:
        """接收投票请求"""
        if msg.payload.get("type") == "vote_request":
            topic = msg.payload.get("topic", "")
            # 如果经验池中有该状态的最优动作，投一票
            best_action = self.experience_pool.get_best_action(topic)
            if best_action:
                self.voter.vote(topic, "consensus_manager", best_action, confidence=0.8)

    def share_experience(self, state_key: str, action: str, reward: float,
                         td_error: float = 0.0, sender: str = "") -> None:
        """主动分享经验"""
        msg = AgentMessage(
            msg_type=MessageType.EXPERIENCE.value,
            sender=sender,
            priority=MessagePriority.NORMAL.value,
            payload={
                "state_key": state_key,
                "action": action,
                "reward": reward,
                "td_error": td_error,
            },
        )
        self.bus.send(msg)

    def request_consensus(self, topic: str, proposals: List[str],
                          sender: str = "") -> Optional[Dict[str, Any]]:
        """请求共识决策"""
        # 广播投票请求
        msg = AgentMessage(
            msg_type=MessageType.TASK.value,
            sender=sender,
            priority=MessagePriority.HIGH.value,
            payload={"type": "vote_request", "topic": topic, "proposals": proposals},
        )
        self.bus.send(msg)

        # 等待收集投票
        time.sleep(0.5)
        return self.voter.get_consensus(topic)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "experience_pool": self.experience_pool.get_stats(),
            "voter": self.voter.get_stats(),
        }
