"""
人机协作接口测试 —— 验证审批流程、信任度、可解释AI

测试前置：
1. 拆解HumanInTheLoop接口：propose/decide/get_pending/get_trust_score/get_stats
2. 拆解ExplainableAI接口：explain_decision/explain_prediction
3. 边界条件：全自动模式、仅人工模式、超时、重复决策
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.human_in_the_loop import (
    HumanInTheLoop, ApprovalMode, HumanAction, DecisionProposal,
    ExplainableAI,
)


class TestHumanInTheLoop:
    """测试人机协作核心"""

    def setup_method(self):
        self.hitl = HumanInTheLoop(default_mode=ApprovalMode.REQUIRE_APPROVAL)

    def test_full_auto_mode(self):
        """全自动模式直接批准"""
        self.hitl.set_mode("restart", ApprovalMode.FULL_AUTO)
        prop = DecisionProposal(agent_id="agent1", action="restart", confidence=0.9)
        decision = self.hitl.propose(prop)
        assert decision is not None
        assert decision.action == HumanAction.APPROVE
        assert decision.decider == "system"

    def test_human_only_mode(self):
        """仅人工模式拒绝AI建议"""
        self.hitl.set_mode("delete", ApprovalMode.HUMAN_ONLY)
        prop = DecisionProposal(agent_id="agent1", action="delete")
        decision = self.hitl.propose(prop)
        assert decision is not None
        assert decision.action == HumanAction.REJECT

    def test_notify_only_mode(self):
        """仅通知模式不阻塞"""
        self.hitl.set_mode("scale", ApprovalMode.NOTIFY_ONLY)
        prop = DecisionProposal(agent_id="agent1", action="scale")
        decision = self.hitl.propose(prop)
        assert decision is not None
        assert decision.action == HumanAction.APPROVE

    def test_require_approval_blocks(self):
        """需要审批时阻塞"""
        prop = DecisionProposal(agent_id="agent1", action="shutdown")
        decision = self.hitl.propose(prop)
        assert decision is None  # 等待人类决策
        assert len(self.hitl.get_pending()) == 1

    def test_human_approve(self):
        """人类批准"""
        prop = DecisionProposal(agent_id="agent1", action="restart")
        self.hitl.propose(prop)
        result = self.hitl.decide(prop.proposal_id, HumanAction.APPROVE, feedback="looks good")
        assert result is True
        decision = self.hitl.get_decision(prop.proposal_id)
        assert decision["action"] == "approve"

    def test_human_reject(self):
        """人类拒绝"""
        prop = DecisionProposal(agent_id="agent1", action="restart")
        self.hitl.propose(prop)
        self.hitl.decide(prop.proposal_id, HumanAction.REJECT, feedback="too risky")
        decision = self.hitl.get_decision(prop.proposal_id)
        assert decision["action"] == "reject"

    def test_human_modify(self):
        """人类修改动作"""
        prop = DecisionProposal(agent_id="agent1", action="restart")
        self.hitl.propose(prop)
        self.hitl.decide(prop.proposal_id, HumanAction.MODIFY, modified_action="graceful_restart")
        decision = self.hitl.get_decision(prop.proposal_id)
        assert decision["action"] == "modify"
        assert decision["modified_action"] == "graceful_restart"

    def test_decide_nonexistent(self):
        """决策不存在的提案返回False"""
        assert self.hitl.decide("none", HumanAction.APPROVE) is False

    def test_trust_score_increase(self):
        """批准后信任度增加"""
        self.hitl.set_mode("action1", ApprovalMode.FULL_AUTO)
        prop = DecisionProposal(agent_id="agent1", action="action1")
        self.hitl.propose(prop)
        assert self.hitl.get_trust_score("agent1") > 0.5

    def test_trust_score_decrease(self):
        """拒绝后信任度减少"""
        prop = DecisionProposal(agent_id="agent2", action="shutdown")
        self.hitl.propose(prop)
        self.hitl.decide(prop.proposal_id, HumanAction.REJECT)
        assert self.hitl.get_trust_score("agent2") < 0.5

    def test_timeout(self):
        """超时自动拒绝"""
        prop = DecisionProposal(agent_id="agent1", action="restart", timeout_sec=0.1)
        self.hitl.propose(prop)
        time.sleep(0.2)
        pending = self.hitl.get_pending()
        assert len(pending) == 0
        decision = self.hitl.get_decision(prop.proposal_id)
        assert decision["action"] == "reject"
        assert decision["feedback"] == "timeout"

    def test_callback(self):
        """回调触发"""
        called = [False]
        def cb(prop, decision):
            called[0] = True

        prop = DecisionProposal(agent_id="agent1", action="restart")
        self.hitl.propose(prop, callback=cb)
        self.hitl.decide(prop.proposal_id, HumanAction.APPROVE)
        assert called[0] is True

    def test_stats(self):
        """统计信息"""
        self.hitl.set_mode("a1", ApprovalMode.FULL_AUTO)
        self.hitl.propose(DecisionProposal(agent_id="x", action="a1"))
        stats = self.hitl.get_stats()
        assert stats["total_decisions"] == 1
        assert stats["approved"] == 1


class TestExplainableAI:
    """测试可解释AI"""

    def test_explain_decision(self):
        """解释决策"""
        prop = DecisionProposal(
            action="throttle",
            confidence=0.85,
            risk_level="medium",
            explanation="CPU预测将超过阈值",
            context={"cpu": 85},
        )
        text = ExplainableAI.explain_decision(prop)
        assert "throttle" in text
        assert "85.0%" in text
        assert "CPU预测将超过阈值" in text

    def test_explain_prediction(self):
        """解释预测"""
        features = {"cpu": 0.5, "memory": -0.3, "disk": 0.1}
        text = ExplainableAI.explain_prediction(features)
        assert "cpu" in text
        assert "memory" in text
