"""
多Agent协作框架测试 —— 验证经验共享和冲突解决
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.collaboration import MessageBus, CollaborationHub, SharedExperience


class TestMessageBus:
    """测试消息总线"""

    def test_register_and_heartbeat(self):
        bus = MessageBus()
        bus.register_agent("agent1")
        assert "agent1" in bus.get_active_agents()
        bus.heartbeat("agent1")
        assert "agent1" in bus.get_active_agents()

    def test_unregister(self):
        bus = MessageBus()
        bus.register_agent("agent1")
        bus.unregister_agent("agent1")
        assert "agent1" not in bus.get_active_agents()

    def test_publish_and_subscribe(self):
        bus = MessageBus()
        received = []
        def handler(msg, sender):
            received.append((msg, sender))
        bus.subscribe("test_topic", handler)
        bus.publish("test_topic", "hello", sender="agent1")
        assert len(received) == 1
        assert received[0] == ("hello", "agent1")

    def test_get_messages(self):
        bus = MessageBus()
        bus.publish("t1", "m1")
        bus.publish("t2", "m2")
        msgs = bus.get_messages(topic="t1")
        assert len(msgs) == 1
        assert msgs[0]["message"] == "m1"


class TestCollaborationHub:
    """测试协作中心"""

    def test_share_and_receive(self):
        bus = MessageBus()
        hub1 = CollaborationHub("agent1", bus)
        hub2 = CollaborationHub("agent2", bus)
        hub2.start_listening()

        hub1.share_experience(
            state={"cpu": 0.5}, action="test_action", reward=1.0,
            next_state={"cpu": 0.6}, td_error=0.1
        )

        assert hub2.received_count == 1
        assert hub2.get_shared_experiences()[0].action == "test_action"

    def test_deduplication(self):
        bus = MessageBus()
        hub1 = CollaborationHub("agent1", bus)
        hub2 = CollaborationHub("agent2", bus)
        hub2.start_listening()

        # 分享两次完全相同的经验
        for _ in range(2):
            hub1.share_experience(
                state={"cpu": 0.5}, action="same", reward=1.0,
                next_state={"cpu": 0.6}, td_error=0.1
            )

        # 第二次应被去重
        assert hub2.received_count == 1
        assert hub2.deduplicated_count == 1

    def test_ignore_self_messages(self):
        bus = MessageBus()
        hub1 = CollaborationHub("agent1", bus)
        hub1.start_listening()

        hub1.share_experience(
            state={"cpu": 0.5}, action="self", reward=1.0,
            next_state={"cpu": 0.6}
        )

        assert hub1.received_count == 0

    def test_conflict_resolution(self):
        bus = MessageBus()
        hub1 = CollaborationHub("agent1", bus)
        hub2 = CollaborationHub("agent2", bus)
        hub2.start_listening()

        # 同一状态，不同动作，不同TD误差
        hub1.share_experience(
            state={"cpu": 0.5}, action="action_bad", reward=0.0,
            next_state={"cpu": 0.6}, td_error=10.0
        )
        hub1.share_experience(
            state={"cpu": 0.5}, action="action_good", reward=1.0,
            next_state={"cpu": 0.6}, td_error=0.1
        )

        # 应只保留最优的（td_error最小的）
        exps = hub2.get_shared_experiences()
        assert len(exps) == 1
        assert exps[0].action == "action_good"

    def test_stats(self):
        bus = MessageBus()
        hub = CollaborationHub("agent1", bus)
        stats = hub.get_stats()
        assert stats["agent_id"] == "agent1"
        assert stats["shared_count"] == 0
        assert stats["received_count"] == 0

    def test_shared_experience_serialization(self):
        exp = SharedExperience(
            source_agent="a1", state={"s": 1}, action="act",
            reward=1.0, next_state={"s": 2}, td_error=0.1
        )
        d = exp.to_dict()
        exp2 = SharedExperience.from_dict(d)
        assert exp2.source_agent == "a1"
        assert exp2.action == "act"
        assert exp2.reward == 1.0

    def test_multiple_agents_communication(self):
        bus = MessageBus()
        agents = [CollaborationHub(f"agent{i}", bus) for i in range(3)]
        for a in agents:
            a.start_listening()

        # agent0 广播经验
        agents[0].share_experience(
            state={"cpu": 0.5}, action="broadcast", reward=1.0,
            next_state={"cpu": 0.6}
        )

        # agent1 和 agent2 都应收到
        assert agents[1].received_count == 1
        assert agents[2].received_count == 1
