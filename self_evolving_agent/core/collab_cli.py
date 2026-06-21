"""
U35：协作与共识CLI —— 多Agent协作/共识投票CLI辅助

第一性原理：
- 协作与共识是多Agent系统的核心能力，但CLI完全空白
- 用户需要通过CLI查看协作统计、分享经验、发起共识投票
- 协作/共识管理应轻量级，直接访问CollaborationHub和ConsensusVoter

设计：
1. CollabCliHelper：协作CLI辅助器
   - 封装CollaborationHub的CLI访问
   - 支持stats/share/experiences
2. ConsensusCliHelper：共识CLI辅助器
   - 封装ConsensusVoter的CLI访问
   - 支持vote/result/stats
3. 复用既有模块，不重复造轮子
"""

import json
import time
from typing import Dict, List, Optional, Any


class CollabCliHelper:
    """协作CLI辅助器

    轻量级访问CollaborationHub，不启动完整Agent
    """

    def __init__(self, agent_id: str = "cli_agent"):
        from core.collaboration import CollaborationHub
        self.hub = CollaborationHub(agent_id=agent_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取协作统计"""
        return self.hub.get_stats()

    def share_experience(self, state: Dict[str, Any], action: str,
                         reward: float, next_state: Optional[Dict[str, Any]] = None,
                         td_error: float = 0.0, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """分享经验

        Args:
            state: 状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            td_error: TD误差
            tags: 标签

        Returns:
            分享的经验字典
        """
        exp = self.hub.share_experience(
            state=state, action=action, reward=reward,
            next_state=next_state or {}, td_error=td_error, tags=tags,
        )
        return exp.to_dict()

    def get_experiences(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取接收到的共享经验"""
        experiences = self.hub.get_shared_experiences(n)
        return [e.to_dict() for e in experiences]

    # ---------- 格式化方法 ----------

    def format_stats(self) -> str:
        """格式化统计"""
        stats = self.get_stats()
        lines = ["=== 协作统计 ==="]
        lines.append(f"  Agent ID: {stats.get('agent_id', 'N/A')}")
        lines.append(f"  分享经验数: {stats.get('shared_count', 0)}")
        lines.append(f"  接收经验数: {stats.get('received_count', 0)}")
        lines.append(f"  去重经验数: {stats.get('deduplicated_count', 0)}")
        lines.append(f"  待处理经验: {stats.get('pending_experiences', 0)}")
        lines.append(f"  活跃对端: {stats.get('active_peers', 0)}")
        return "\n".join(lines)

    def format_share_result(self, exp: Dict[str, Any]) -> str:
        """格式化分享结果"""
        lines = ["=== 经验分享成功 ==="]
        lines.append(f"  经验ID: {exp.get('experience_id', 'N/A')}")
        lines.append(f"  来源Agent: {exp.get('source_agent', 'N/A')}")
        lines.append(f"  动作: {exp.get('action', 'N/A')}")
        lines.append(f"  奖励: {exp.get('reward', 0)}")
        lines.append(f"  TD误差: {exp.get('td_error', 0)}")
        lines.append(f"  指纹: {exp.get('origin_fingerprint', 'N/A')}")
        tags = exp.get("tags", [])
        if tags:
            lines.append(f"  标签: {', '.join(tags)}")
        return "\n".join(lines)

    def format_experiences(self, n: int = 10) -> str:
        """格式化经验列表"""
        experiences = self.get_experiences(n)
        if not experiences:
            return "（无接收到的共享经验）"

        lines = [f"=== 共享经验（最近{len(experiences)}条）==="]
        lines.append(f"{'序号':>4} {'经验ID':<15} {'来源':<12} {'动作':<10} {'奖励':>8}")
        lines.append("-" * 60)
        for i, exp in enumerate(experiences, 1):
            lines.append(
                f"{i:>4} {exp.get('experience_id', 'N/A'):<15} "
                f"{exp.get('source_agent', 'N/A'):<12} "
                f"{exp.get('action', 'N/A'):<10} "
                f"{exp.get('reward', 0):>8.2f}"
            )
        return "\n".join(lines)


class ConsensusCliHelper:
    """共识CLI辅助器

    轻量级访问ConsensusVoter，不启动完整Agent
    """

    def __init__(self, min_votes: int = 2):
        from core.consensus import ConsensusVoter
        self.voter = ConsensusVoter(min_votes=min_votes)

    def vote(self, topic: str, agent_id: str, proposal: str,
             confidence: float = 1.0) -> None:
        """投票

        Args:
            topic: 议题
            agent_id: 投票者ID
            proposal: 提案
            confidence: 置信度
        """
        self.voter.vote(topic, agent_id, proposal, confidence)

    def get_consensus(self, topic: str) -> Optional[Dict[str, Any]]:
        """获取共识结果

        Args:
            topic: 议题

        Returns:
            共识结果，未达最小票数返回None
        """
        return self.voter.get_consensus(topic)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return self.voter.get_stats()

    # ---------- 格式化方法 ----------

    def format_vote_result(self, topic: str, agent_id: str, proposal: str) -> str:
        """格式化投票结果"""
        lines = ["=== 投票已提交 ==="]
        lines.append(f"  议题: {topic}")
        lines.append(f"  投票者: {agent_id}")
        lines.append(f"  提案: {proposal}")
        stats = self.get_stats()
        lines.append(f"  当前议题数: {stats.get('topics', 0)}")
        return "\n".join(lines)

    def format_consensus_result(self, topic: str) -> str:
        """格式化共识结果"""
        result = self.get_consensus(topic)
        if result is None:
            stats = self.get_stats()
            min_votes = stats.get("min_votes_required", 2)
            return (f"=== 议题 '{topic}' 未达成共识 ===\n"
                    f"  需要至少 {min_votes} 票，请继续投票")

        lines = [f"=== 议题 '{topic}' 共识结果 ==="]
        lines.append(f"  共识决策: {result.get('consensus', 'N/A')}")
        lines.append(f"  一致率: {result.get('agreement_rate', 0):.1%}")
        lines.append(f"  总票数: {result.get('total_votes', 0)}")
        distribution = result.get("vote_distribution", {})
        if distribution:
            lines.append(f"\n  投票分布:")
            for proposal, weight in distribution.items():
                lines.append(f"    {proposal}: {weight:.2f}")
        return "\n".join(lines)

    def format_stats(self) -> str:
        """格式化统计"""
        stats = self.get_stats()
        lines = ["=== 共识统计 ==="]
        lines.append(f"  议题数: {stats.get('topics', 0)}")
        lines.append(f"  已决策数: {stats.get('decisions', 0)}")
        lines.append(f"  最小票数: {stats.get('min_votes_required', 2)}")
        return "\n".join(lines)
