"""
经验共享、冲突解决与共识决策测试 —— 验证经验池、投票器、共识管理器

测试前置：
1. 拆解ExperiencePool接口：add/get_best_action/get_all_actions/get_stats
2. 拆解ConsensusVoter接口：vote/get_consensus/get_stats
3. 边界条件：空池、重复经验、同状态不同动作、票数不足
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.consensus import (
    Experience, ExperiencePool, ConsensusVoter, ConsensusManager,
)
from core.multi_agent_bus import MultiAgentBus, AgentMessage, MessageType


class TestExperience:
    """测试经验数据类"""

    def test_fingerprint_consistency(self):
        """相同内容产生相同指纹"""
        e1 = Experience("state1", "action1", 1.0)
        e2 = Experience("state1", "action1", 1.0)
        assert e1.fingerprint() == e2.fingerprint()

    def test_fingerprint_difference(self):
        """不同内容产生不同指纹"""
        e1 = Experience("state1", "action1", 1.0)
        e2 = Experience("state1", "action2", 1.0)
        assert e1.fingerprint() != e2.fingerprint()


class TestExperiencePool:
    """测试经验池"""

    def setup_method(self):
        self.pool = ExperiencePool(max_size=10)

    def test_add_and_retrieve(self):
        """添加并获取经验"""
        exp = Experience("s1", "a1", 1.0, td_error=0.1)
        assert self.pool.add(exp) is True
        assert self.pool.get_best_action("s1") == "a1"

    def test_deduplication(self):
        """重复经验被去重"""
        exp = Experience("s1", "a1", 1.0)
        assert self.pool.add(exp) is True
        assert self.pool.add(exp) is False
        assert self.pool.get_stats()["deduplicated"] == 1

    def test_conflict_resolution_same_action(self):
        """同一动作保留TD误差更小的"""
        exp1 = Experience("s1", "a1", 1.0, td_error=0.5)
        exp2 = Experience("s1", "a1", 2.0, td_error=0.2)  # 不同reward使fingerprint不同
        self.pool.add(exp1)
        self.pool.add(exp2)
        actions = self.pool.get_all_actions("s1")
        assert len(actions) == 1
        assert actions[0].td_error == 0.2

    def test_conflict_detection_different_actions(self):
        """不同动作为冲突"""
        exp1 = Experience("s1", "a1", 1.0, td_error=0.1)
        exp2 = Experience("s1", "a2", 1.0, td_error=0.2)
        self.pool.add(exp1)
        self.pool.add(exp2)
        assert self.pool.get_stats()["conflict_resolved"] >= 1
        actions = self.pool.get_all_actions("s1")
        assert len(actions) == 2

    def test_best_action_selection(self):
        """最优动作是TD误差最小的"""
        self.pool.add(Experience("s1", "a1", 1.0, td_error=0.5))
        self.pool.add(Experience("s1", "a2", 1.0, td_error=0.1))
        self.pool.add(Experience("s1", "a3", 1.0, td_error=0.3))
        assert self.pool.get_best_action("s1") == "a2"

    def test_empty_state(self):
        """空状态返回None"""
        assert self.pool.get_best_action("nonexistent") is None
        assert self.pool.get_all_actions("nonexistent") == []

    def test_cleanup(self):
        """容量超限清理"""
        for i in range(25):
            self.pool.add(Experience(f"s{i}", "a", 1.0))
        # max_size=10, 触发清理阈值是20
        stats = self.pool.get_stats()
        assert stats["states"] <= 10


class TestConsensusVoter:
    """测试共识投票器"""

    def setup_method(self):
        self.voter = ConsensusVoter(min_votes=2)

    def test_insufficient_votes(self):
        """票数不足返回None"""
        self.voter.vote("topic1", "agent1", "proposal_a")
        assert self.voter.get_consensus("topic1") is None

    def test_simple_majority(self):
        """简单多数胜出"""
        self.voter.vote("topic1", "a1", "proposal_a", confidence=1.0)
        self.voter.vote("topic1", "a2", "proposal_a", confidence=1.0)
        self.voter.vote("topic1", "a3", "proposal_b", confidence=1.0)
        result = self.voter.get_consensus("topic1")
        assert result is not None
        assert result["consensus"] == "proposal_a"
        assert abs(result["agreement_rate"] - 2 / 3) < 0.01

    def test_weighted_votes(self):
        """加权投票"""
        self.voter.vote("topic1", "a1", "proposal_a", confidence=0.5)
        self.voter.vote("topic1", "a2", "proposal_b", confidence=1.0)
        result = self.voter.get_consensus("topic1")
        assert result["consensus"] == "proposal_b"

    def test_stats(self):
        """统计信息"""
        self.voter.vote("t1", "a1", "p1")
        stats = self.voter.get_stats()
        assert stats["topics"] == 1
        assert stats["min_votes_required"] == 2


class TestConsensusManager:
    """测试共识管理器"""

    def setup_method(self):
        self.bus = MultiAgentBus("test_consensus")
        self.manager = ConsensusManager(self.bus)

    def teardown_method(self):
        self.manager.stop()

    def test_share_experience(self):
        """分享经验"""
        self.bus.start()
        self.manager.start()
        self.manager.share_experience("state1", "action1", 1.0, td_error=0.1, sender="agent1")
        time.sleep(0.3)
        assert self.manager.experience_pool.get_best_action("state1") == "action1"

    def test_receive_experience_via_bus(self):
        """通过总线接收经验"""
        self.bus.start()
        self.manager.start()
        msg = AgentMessage(
            msg_type=MessageType.EXPERIENCE.value,
            sender="agent2",
            payload={"state_key": "s2", "action": "a2", "reward": 2.0, "td_error": 0.05},
        )
        self.bus.send(msg)
        time.sleep(0.3)
        assert self.manager.experience_pool.get_best_action("s2") == "a2"

    def test_request_consensus(self):
        """请求共识"""
        self.bus.start()
        self.manager.start()
        # 先添加一些经验
        self.manager.experience_pool.add(Experience("topic1", "action_a", 1.0, td_error=0.1))
        # 手动添加投票以满足min_votes=2
        self.manager.voter.vote("topic1", "agent1", "action_a", confidence=1.0)
        self.manager.voter.vote("topic1", "agent2", "action_a", confidence=1.0)
        result = self.manager.voter.get_consensus("topic1")
        assert result is not None
        assert result["consensus"] == "action_a"

    def test_stats(self):
        """统计信息"""
        self.manager.experience_pool.add(Experience("s1", "a1", 1.0))
        stats = self.manager.get_stats()
        assert "experience_pool" in stats
        assert stats["experience_pool"]["states"] == 1


class TestIntegration:
    """集成测试"""

    def test_full_workflow(self):
        """完整工作流：经验共享 -> 冲突解决 -> 共识决策"""
        bus = MultiAgentBus("integration")
        manager = ConsensusManager(bus)
        bus.start()
        manager.start()

        # 多个Agent分享经验
        manager.share_experience("server_cpu_high", "throttle", 1.0, td_error=0.1, sender="agent1")
        manager.share_experience("server_cpu_high", "throttle", 1.0, td_error=0.15, sender="agent2")
        manager.share_experience("server_cpu_high", "restart", 0.5, td_error=0.5, sender="agent3")
        time.sleep(0.3)

        # 验证冲突解决：throttle有两个，restart有一个
        actions = manager.experience_pool.get_all_actions("server_cpu_high")
        assert len(actions) == 2

        # 验证最优动作是throttle（TD误差更小）
        best = manager.experience_pool.get_best_action("server_cpu_high")
        assert best == "throttle"

        # 投票
        manager.voter.vote("what_to_do", "agent1", "throttle", confidence=1.0)
        manager.voter.vote("what_to_do", "agent2", "throttle", confidence=1.0)
        manager.voter.vote("what_to_do", "agent3", "restart", confidence=0.5)
        result = manager.voter.get_consensus("what_to_do")
        assert result["consensus"] == "throttle"

        manager.stop()
        bus.stop()
