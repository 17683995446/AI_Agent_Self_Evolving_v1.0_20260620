"""
人机协作接口 —— 人类专家介入机制、人工审核、混合决策

设计原则（第一性原理）：
1. 人类是最终决策者：AI提供建议，人类做选择
2. 透明度：AI必须解释为什么给出这个建议
3. 可控性：人类可以随时接管、修正、停止
4. 渐进式信任：从辅助到自主，逐步建立信任
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


class HumanAction(Enum):
    """人类可执行的动作"""
    APPROVE = "approve"         # 批准
    REJECT = "reject"           # 拒绝
    MODIFY = "modify"           # 修改
    OVERRIDE = "override"       # 覆盖
    PAUSE = "pause"             # 暂停
    RESUME = "resume"           # 恢复
    EMERGENCY_STOP = "emergency_stop"  # 紧急停止


class ApprovalMode(Enum):
    """审批模式"""
    FULL_AUTO = "full_auto"         # 全自动，无需审批
    NOTIFY_ONLY = "notify_only"     # 仅通知，不阻塞
    REQUIRE_APPROVAL = "require_approval"  # 需要审批
    HUMAN_ONLY = "human_only"       # 仅人工


@dataclass
class DecisionProposal:
    """决策提案 —— AI向人类提交的决策请求"""
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str = ""
    action: str = ""              # 建议的动作
    context: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""         # AI的解释
    confidence: float = 0.0       # AI的置信度
    risk_level: str = "low"       # low/medium/high/critical
    created_at: float = field(default_factory=time.time)
    timeout_sec: float = 60.0     # 审批超时

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "agent_id": self.agent_id,
            "action": self.action,
            "context": self.context,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "created_at": self.created_at,
            "timeout_sec": self.timeout_sec,
        }


@dataclass
class HumanDecision:
    """人类决策结果"""
    proposal_id: str = ""
    action: HumanAction = HumanAction.APPROVE
    feedback: str = ""            # 人类反馈
    modified_action: Optional[str] = None  # 修改后的动作
    decided_at: float = field(default_factory=time.time)
    decider: str = ""             # 决策者标识

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "action": self.action.value,
            "feedback": self.feedback,
            "modified_action": self.modified_action,
            "decided_at": self.decided_at,
            "decider": self.decider,
        }


class HumanInTheLoop:
    """人机协作核心 —— 管理审批流程、人类反馈、信任度"""

    def __init__(self, default_mode: ApprovalMode = ApprovalMode.REQUIRE_APPROVAL):
        self.default_mode = default_mode
        self._mode_overrides: Dict[str, ApprovalMode] = {}  # action -> mode
        self._pending: Dict[str, DecisionProposal] = {}
        self._decisions: Dict[str, HumanDecision] = {}
        self._trust_scores: Dict[str, float] = {}  # agent_id -> trust score
        self._callbacks: Dict[str, Callable] = {}
        self._history: List[Dict[str, Any]] = []

    def set_mode(self, action_type: str, mode: ApprovalMode) -> None:
        """为特定动作类型设置审批模式"""
        self._mode_overrides[action_type] = mode
        logger.info("HITL", f"审批模式设置: {action_type} -> {mode.value}", module="human_in_the_loop")

    def get_mode(self, action_type: str) -> ApprovalMode:
        """获取动作类型的审批模式"""
        return self._mode_overrides.get(action_type, self.default_mode)

    def propose(self, proposal: DecisionProposal,
                callback: Optional[Callable] = None) -> Optional[HumanDecision]:
        """提交决策提案，根据模式决定是否需要审批"""
        mode = self.get_mode(proposal.action)

        if mode == ApprovalMode.FULL_AUTO:
            # 全自动，直接批准
            decision = HumanDecision(
                proposal_id=proposal.proposal_id,
                action=HumanAction.APPROVE,
                feedback="auto-approved",
                decider="system",
            )
            self._decisions[proposal.proposal_id] = decision
            self._history.append({
                "proposal": proposal.to_dict(),
                "decision": decision.to_dict(),
            })
            self._update_trust(proposal.agent_id, True)
            return decision

        if mode == ApprovalMode.HUMAN_ONLY:
            # 仅人工，拒绝AI建议
            decision = HumanDecision(
                proposal_id=proposal.proposal_id,
                action=HumanAction.REJECT,
                feedback="human-only mode",
                decider="system",
            )
            self._decisions[proposal.proposal_id] = decision
            return decision

        # 需要审批或仅通知
        self._pending[proposal.proposal_id] = proposal
        if callback:
            self._callbacks[proposal.proposal_id] = callback

        logger.info("HITL", f"决策提案待审批: {proposal.proposal_id} [{proposal.action}]",
                    module="human_in_the_loop", extra={"risk": proposal.risk_level})

        if mode == ApprovalMode.NOTIFY_ONLY:
            # 不阻塞，返回预批准
            decision = HumanDecision(
                proposal_id=proposal.proposal_id,
                action=HumanAction.APPROVE,
                feedback="notify-only mode",
                decider="system",
            )
            self._decisions[proposal.proposal_id] = decision
            return decision

        # REQUIRE_APPROVAL: 返回None，等待人类决策
        return None

    def decide(self, proposal_id: str, action: HumanAction,
               feedback: str = "", modified_action: Optional[str] = None,
               decider: str = "human") -> bool:
        """人类做出决策"""
        proposal = self._pending.pop(proposal_id, None)
        if not proposal:
            logger.warning("HITL", f"提案不存在或已处理: {proposal_id}", module="human_in_the_loop")
            return False

        decision = HumanDecision(
            proposal_id=proposal_id,
            action=action,
            feedback=feedback,
            modified_action=modified_action,
            decider=decider,
        )
        self._decisions[proposal_id] = decision
        self._history.append({
            "proposal": proposal.to_dict(),
            "decision": decision.to_dict(),
        })

        # 更新信任度
        approved = action in (HumanAction.APPROVE, HumanAction.OVERRIDE)
        self._update_trust(proposal.agent_id, approved)

        # 回调
        callback = self._callbacks.pop(proposal_id, None)
        if callback:
            try:
                callback(proposal, decision)
            except Exception as e:
                logger.warning("HITL", f"回调异常: {e}", module="human_in_the_loop")

        logger.info("HITL", f"人类决策: {proposal_id} -> {action.value}", module="human_in_the_loop")
        return True

    def _update_trust(self, agent_id: str, approved: bool) -> None:
        """更新对Agent的信任度"""
        current = self._trust_scores.get(agent_id, 0.5)
        if approved:
            new_trust = min(1.0, current + 0.05)
        else:
            new_trust = max(0.0, current - 0.1)
        self._trust_scores[agent_id] = new_trust

    def get_trust_score(self, agent_id: str) -> float:
        """获取对Agent的信任度"""
        return self._trust_scores.get(agent_id, 0.5)

    def get_pending(self) -> List[Dict[str, Any]]:
        """获取待审批列表"""
        now = time.time()
        result = []
        for pid, prop in list(self._pending.items()):
            if now - prop.created_at > prop.timeout_sec:
                # 超时，自动拒绝
                self.decide(pid, HumanAction.REJECT, feedback="timeout", decider="system")
                continue
            result.append(prop.to_dict())
        return result

    def get_decision(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """获取决策结果"""
        decision = self._decisions.get(proposal_id)
        return decision.to_dict() if decision else None

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        total = len(self._history)
        approved = sum(1 for h in self._history if h["decision"]["action"] == HumanAction.APPROVE.value)
        rejected = sum(1 for h in self._history if h["decision"]["action"] == HumanAction.REJECT.value)
        return {
            "total_decisions": total,
            "approved": approved,
            "rejected": rejected,
            "pending": len(self._pending),
            "trust_scores": self._trust_scores,
        }


class ExplainableAI:
    """可解释AI —— 为决策生成人类可理解的解释"""

    @staticmethod
    def explain_decision(proposal: DecisionProposal) -> str:
        """解释决策"""
        lines = [
            f"决策建议: {proposal.action}",
            f"置信度: {proposal.confidence:.1%}",
            f"风险等级: {proposal.risk_level}",
            f"解释: {proposal.explanation}",
            f"上下文: {proposal.context}",
        ]
        return "\n".join(lines)

    @staticmethod
    def explain_prediction(feature_importance: Dict[str, float]) -> str:
        """解释预测结果的关键特征"""
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        lines = ["关键影响因素:"]
        for feature, importance in sorted_features[:5]:
            direction = "增加" if importance > 0 else "减少"
            lines.append(f"  - {feature}: {direction}预测值 (权重: {abs(importance):.3f})")
        return "\n".join(lines)
