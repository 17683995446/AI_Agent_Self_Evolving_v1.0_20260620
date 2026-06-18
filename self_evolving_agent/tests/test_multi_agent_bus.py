"""
多Agent通信协议测试 —— 验证消息格式、优先级队列、角色注册、点对点/广播

测试前置：
1. 拆解MultiAgentBus接口：register_agent/unregister_agent/heartbeat/send/subscribe/start/stop/get_stats
2. 拆解PriorityMessageQueue接口：put/get/size
3. 边界条件：重复注册、未注册Agent发送、队列满、TTL耗尽、重复消息
"""
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.multi_agent_bus import (
    MultiAgentBus, AgentMessage, AgentRole, MessageType, MessagePriority,
    PriorityMessageQueue, AgentInfo,
)


class TestPriorityMessageQueue:
    """测试优先级消息队列"""

    def setup_method(self):
        self.queue = PriorityMessageQueue(max_size=10)

    def test_put_and_get_priority_order(self):
        """高优先级消息先出队"""
        low = AgentMessage(priority=MessagePriority.LOW.value)
        normal = AgentMessage(priority=MessagePriority.NORMAL.value)
        high = AgentMessage(priority=MessagePriority.HIGH.value)
        critical = AgentMessage(priority=MessagePriority.CRITICAL.value)

        self.queue.put(low)
        self.queue.put(normal)
        self.queue.put(high)
        self.queue.put(critical)

        assert self.queue.get().priority == MessagePriority.CRITICAL.value
        assert self.queue.get().priority == MessagePriority.HIGH.value
        assert self.queue.get().priority == MessagePriority.NORMAL.value
        assert self.queue.get().priority == MessagePriority.LOW.value

    def test_get_empty_returns_none(self):
        """空队列返回None"""
        assert self.queue.get() is None

    def test_queue_full_drop_lowest(self):
        """队列满时丢弃最低优先级"""
        for _ in range(10):
            self.queue.put(AgentMessage(priority=MessagePriority.LOW.value))
        # 第11个高优先级应能放入（触发丢弃最低）
        assert self.queue.put(AgentMessage(priority=MessagePriority.HIGH.value)) is True
        assert self.queue.size() == 10

    def test_queue_full_reject_low(self):
        """队列满时低优先级被拒绝"""
        for _ in range(10):
            self.queue.put(AgentMessage(priority=MessagePriority.NORMAL.value))
        assert self.queue.put(AgentMessage(priority=MessagePriority.LOW.value)) is False

    def test_size_consistency(self):
        """size与put/get一致"""
        assert self.queue.size() == 0
        self.queue.put(AgentMessage())
        assert self.queue.size() == 1
        self.queue.get()
        assert self.queue.size() == 0


class TestMultiAgentBusRegistration:
    """测试Agent注册与管理"""

    def setup_method(self):
        self.bus = MultiAgentBus("test_bus")

    def test_register_agent_success(self):
        """正常注册"""
        assert self.bus.register_agent("agent1", AgentRole.PERCEPTION) is True
        agents = self.bus.get_agents_by_role(AgentRole.PERCEPTION)
        assert len(agents) == 1
        assert agents[0].agent_id == "agent1"

    def test_register_duplicate_fails(self):
        """重复注册失败"""
        self.bus.register_agent("agent1", AgentRole.PERCEPTION)
        assert self.bus.register_agent("agent1", AgentRole.PREDICTION) is False

    def test_unregister_agent(self):
        """注销Agent"""
        self.bus.register_agent("agent1", AgentRole.PERCEPTION)
        self.bus.unregister_agent("agent1")
        agents = self.bus.get_agents_by_role(AgentRole.PERCEPTION)
        assert len(agents) == 0

    def test_heartbeat_updates(self):
        """心跳更新last_heartbeat"""
        self.bus.register_agent("agent1", AgentRole.GENERAL)
        before = self.bus._agents["agent1"].last_heartbeat
        time.sleep(0.05)
        self.bus.heartbeat("agent1", load_score=5.0)
        after = self.bus._agents["agent1"].last_heartbeat
        assert after > before
        assert self.bus._agents["agent1"].load_score == 5.0

    def test_get_agents_by_role_inactive_filtered(self):
        """不活跃Agent被过滤"""
        self.bus.register_agent("agent1", AgentRole.GENERAL)
        time.sleep(0.02)  # 等待超过timeout
        agents = self.bus.get_agents_by_role(AgentRole.GENERAL, active_only=True, timeout=0.01)
        assert len(agents) == 0  # 因为心跳时间太久远

    def test_get_least_loaded_agent(self):
        """获取负载最低的Agent"""
        self.bus.register_agent("a1", AgentRole.GENERAL)
        self.bus.register_agent("a2", AgentRole.GENERAL)
        self.bus.heartbeat("a1", load_score=10.0)
        self.bus.heartbeat("a2", load_score=5.0)
        best = self.bus.get_least_loaded_agent(AgentRole.GENERAL)
        assert best is not None
        assert best.agent_id == "a2"

    def test_get_least_loaded_empty(self):
        """无Agent时返回None"""
        assert self.bus.get_least_loaded_agent(AgentRole.ORCHESTRATOR) is None


class TestMultiAgentBusMessaging:
    """测试消息发送与接收"""

    def setup_method(self):
        self.bus = MultiAgentBus("msg_bus")
        self.bus.register_agent("sender1", AgentRole.GENERAL)
        self.bus.register_agent("receiver1", AgentRole.GENERAL)
        self.received: list = []

    def test_send_and_subscribe(self):
        """发送消息，订阅者收到"""
        self.bus.subscribe(MessageType.STATE.value, lambda msg: self.received.append(msg))
        self.bus.start()

        msg = AgentMessage(
            msg_type=MessageType.STATE.value,
            sender="sender1",
            payload={"cpu": 50.0},
        )
        self.bus.send(msg)
        time.sleep(0.05)

        assert len(self.received) == 1
        assert self.received[0].payload["cpu"] == 50.0
        self.bus.stop()

    def test_targeted_message(self):
        """点对点消息"""
        self.bus.subscribe(MessageType.TASK.value, lambda msg: self.received.append(msg))
        self.bus.start()

        msg = AgentMessage(
            msg_type=MessageType.TASK.value,
            sender="sender1",
            target="receiver1",
            payload={"action": "predict"},
        )
        self.bus.send(msg)
        time.sleep(0.05)

        assert len(self.received) == 1
        self.bus.stop()

    def test_duplicate_message_dropped(self):
        """重复消息被丢弃"""
        self.bus.subscribe(MessageType.STATE.value, lambda msg: self.received.append(msg))
        self.bus.start()

        msg = AgentMessage(
            msg_id="unique123",
            msg_type=MessageType.STATE.value,
            sender="sender1",
            payload={"x": 1},
        )
        self.bus.send(msg)
        self.bus.send(msg)  # 重复
        time.sleep(0.05)

        assert len(self.received) == 1
        self.bus.stop()

    def test_ttl_decrement_on_broadcast(self):
        """广播消息TTL递减"""
        self.bus.subscribe(MessageType.STATE.value, lambda msg: self.received.append(msg))
        self.bus.start()

        msg = AgentMessage(
            msg_type=MessageType.STATE.value,
            sender="sender1",
            ttl=1,
            payload={},
        )
        self.bus.send(msg)
        time.sleep(0.05)

        # 第一次广播ttl=1->0，订阅者收到
        assert len(self.received) == 1
        self.bus.stop()

    def test_send_ack(self):
        """发送确认消息"""
        self.bus.subscribe(MessageType.ACK.value, lambda msg: self.received.append(msg))
        self.bus.start()

        self.bus.send_ack("orig123", "sender1", "receiver1")
        time.sleep(0.05)

        assert len(self.received) == 1
        assert self.received[0].payload["ack_for"] == "orig123"
        self.bus.stop()

    def test_priority_respected(self):
        """高优先级消息先被处理"""
        received_order: list = []
        self.bus.subscribe(MessageType.STATE.value, lambda msg: received_order.append(msg.priority))
        self.bus.start()

        self.bus.send(AgentMessage(msg_type=MessageType.STATE.value, priority=MessagePriority.LOW.value))
        self.bus.send(AgentMessage(msg_type=MessageType.STATE.value, priority=MessagePriority.CRITICAL.value))
        self.bus.send(AgentMessage(msg_type=MessageType.STATE.value, priority=MessagePriority.HIGH.value))
        time.sleep(0.1)

        assert received_order[0] == MessagePriority.CRITICAL.value
        assert received_order[1] == MessagePriority.HIGH.value
        assert received_order[2] == MessagePriority.LOW.value
        self.bus.stop()


class TestAgentMessage:
    """测试消息序列化"""

    def test_to_dict_and_from_dict(self):
        """序列化与反序列化"""
        msg = AgentMessage(
            msg_id="test1",
            msg_type=MessageType.TASK.value,
            sender="a1",
            sender_role=AgentRole.DECISION.value,
            target="a2",
            priority=1,
            payload={"x": 1},
            ttl=5,
            require_ack=True,
        )
        d = msg.to_dict()
        restored = AgentMessage.from_dict(d)
        assert restored.msg_id == "test1"
        assert restored.msg_type == MessageType.TASK.value
        assert restored.sender_role == AgentRole.DECISION.value
        assert restored.target == "a2"
        assert restored.priority == 1
        assert restored.payload == {"x": 1}
        assert restored.ttl == 5
        assert restored.require_ack is True

    def test_default_values(self):
        """默认值正确"""
        msg = AgentMessage()
        assert msg.msg_id != ""
        assert msg.priority == MessagePriority.NORMAL.value
        assert msg.ttl == 3
        assert msg.require_ack is False


class TestBusStats:
    """测试总线统计"""

    def test_stats_basic(self):
        """统计信息基本字段"""
        bus = MultiAgentBus("stats_bus")
        bus.register_agent("a1", AgentRole.GENERAL)
        stats = bus.get_stats()
        assert stats["bus_id"] == "stats_bus"
        assert stats["agent_count"] == 1
        assert stats["queue_size"] == 0

    def test_stats_after_message(self):
        """发送消息后统计更新"""
        bus = MultiAgentBus("stats_bus2")
        bus.register_agent("a1", AgentRole.GENERAL)
        bus.start()
        bus.send(AgentMessage(msg_type=MessageType.STATE.value, sender="a1"))
        time.sleep(0.05)
        stats = bus.get_stats()
        assert stats["received_msgs"] == 1
        bus.stop()


class TestConcurrency:
    """测试并发安全"""

    def test_concurrent_send(self):
        """多线程并发发送"""
        bus = MultiAgentBus("concurrent")
        bus.register_agent("a1", AgentRole.GENERAL)
        received_count = [0]
        lock = threading.Lock()

        def counter(msg):
            with lock:
                received_count[0] += 1

        bus.subscribe(MessageType.STATE.value, counter)
        bus.start()

        def sender():
            for _ in range(50):
                bus.send(AgentMessage(msg_type=MessageType.STATE.value, sender="a1"))

        threads = [threading.Thread(target=sender) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        time.sleep(0.2)
        assert received_count[0] == 200
        bus.stop()
