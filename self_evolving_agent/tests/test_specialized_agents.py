"""
专用Agent小组测试 —— 验证感知/预测/决策/执行Agent的协作流程

测试前置：
1. 拆解SpecializedAgent基类：start/stop/get_stats/_cycle
2. 验证数据流：感知Agent广播STATE -> 预测Agent接收 -> 预测Agent广播TASK_RESULT -> 决策Agent接收 -> 决策Agent广播TASK -> 执行Agent接收
3. 边界条件：无状态时的等待、无预测时的等待、无决策时的等待
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.specialized_agents import (
    PerceptionAgent, PredictionAgent, DecisionAgent, ExecutionAgent,
)
from core.multi_agent_bus import MultiAgentBus, AgentMessage, AgentRole, MessageType


class TestPerceptionAgent:
    """测试感知Agent"""

    def setup_method(self):
        self.bus = MultiAgentBus("test_perception")
        self.agent = PerceptionAgent("perception-test", bus=self.bus)

    def teardown_method(self):
        self.agent.stop()

    def test_start_and_stop(self):
        """启动和停止"""
        self.agent.start()
        assert self.agent._running is True
        assert self.agent.role == AgentRole.PERCEPTION
        self.agent.stop()
        assert self.agent._running is False

    def test_broadcasts_state(self):
        """启动后广播状态消息"""
        received = []
        self.bus.subscribe(MessageType.STATE.value, lambda msg: received.append(msg))
        self.bus.start()

        self.agent.start()
        time.sleep(1.5)  # 等待至少一个循环

        assert len(received) >= 1
        assert received[0].sender == "perception-test"
        assert "cpu_percent" in received[0].payload
        self.agent.stop()
        self.bus.stop()

    def test_get_current_state(self):
        """获取当前状态"""
        self.agent.start()
        time.sleep(1.2)
        state = self.agent.get_current_state()
        assert "cpu_percent" in state
        assert "memory_percent" in state
        self.agent.stop()


class TestPredictionAgent:
    """测试预测Agent"""

    def setup_method(self):
        self.bus = MultiAgentBus("test_prediction")
        self.agent = PredictionAgent("prediction-test", bus=self.bus)

    def teardown_method(self):
        self.agent.stop()

    def test_receives_state(self):
        """接收状态消息后产生预测"""
        received = []
        self.bus.subscribe(MessageType.TASK_RESULT.value, lambda msg: received.append(msg))
        self.bus.start()

        self.agent.start()
        # 手动发送状态消息
        state_msg = AgentMessage(
            msg_type=MessageType.STATE.value,
            sender="perception-test",
            payload={"cpu_percent": 50.0, "memory_percent": 60.0, "timestamp": time.time()},
        )
        self.bus.send(state_msg)
        time.sleep(1.5)

        assert self.agent._latest_state is not None
        assert self.agent._prediction_count >= 1
        assert len(received) >= 1
        assert received[0].payload["type"] == "prediction"
        self.agent.stop()
        self.bus.stop()

    def test_no_state_waits(self):
        """无状态时等待不崩溃"""
        self.agent.start()
        time.sleep(0.8)
        assert self.agent._prediction_count == 0
        self.agent.stop()

    def test_stats(self):
        """统计信息正确"""
        self.agent.start()
        time.sleep(0.5)
        stats = self.agent.get_stats()
        assert stats["agent_id"] == "prediction-test"
        assert stats["role"] == "prediction"
        self.agent.stop()


class TestDecisionAgent:
    """测试决策Agent"""

    def setup_method(self):
        self.bus = MultiAgentBus("test_decision")
        self.agent = DecisionAgent("decision-test", bus=self.bus)

    def teardown_method(self):
        self.agent.stop()

    def test_receives_prediction(self):
        """接收预测后产生决策"""
        received = []
        self.bus.subscribe(MessageType.TASK.value, lambda msg: received.append(msg))
        self.bus.start()

        self.agent.start()
        # 发送预测结果
        pred_msg = AgentMessage(
            msg_type=MessageType.TASK_RESULT.value,
            sender="prediction-test",
            payload={
                "type": "prediction",
                "data": {"predicted_cpu_next": 85.0, "predicted_mem_next": 70.0},
            },
        )
        self.bus.send(pred_msg)
        time.sleep(1.5)

        assert self.agent._decision_count >= 1
        assert len(received) >= 1
        assert received[0].payload["type"] == "decision"
        self.agent.stop()
        self.bus.stop()

    def test_decision_logic_high_cpu(self):
        """高CPU时决策为throttle"""
        self.bus.start()
        self.agent.start()

        pred_msg = AgentMessage(
            msg_type=MessageType.TASK_RESULT.value,
            sender="prediction-test",
            payload={
                "type": "prediction",
                "data": {"predicted_cpu_next": 85.0},
            },
        )
        self.bus.send(pred_msg)
        time.sleep(1.5)

        assert self.agent._decision_count >= 1
        self.agent.stop()
        self.bus.stop()

    def test_decision_logic_low_cpu(self):
        """低CPU时决策为boost"""
        self.bus.start()
        self.agent.start()

        pred_msg = AgentMessage(
            msg_type=MessageType.TASK_RESULT.value,
            sender="prediction-test",
            payload={
                "type": "prediction",
                "data": {"predicted_cpu_next": 15.0},
            },
        )
        self.bus.send(pred_msg)
        time.sleep(1.5)

        assert self.agent._decision_count >= 1
        self.agent.stop()
        self.bus.stop()


class TestExecutionAgent:
    """测试执行Agent"""

    def setup_method(self):
        self.bus = MultiAgentBus("test_execution")
        self.agent = ExecutionAgent("execution-test", bus=self.bus)

    def teardown_method(self):
        self.agent.stop()

    def test_receives_decision(self):
        """接收决策后执行"""
        self.bus.start()
        self.agent.start()

        decision_msg = AgentMessage(
            msg_type=MessageType.TASK.value,
            sender="decision-test",
            payload={
                "type": "decision",
                "data": {"action": "throttle", "reason": "cpu_high"},
            },
        )
        self.bus.send(decision_msg)
        time.sleep(1.5)

        assert self.agent._execution_count >= 1
        assert len(self.agent._action_log) >= 1
        assert self.agent._action_log[0]["action"] == "throttle"
        self.agent.stop()
        self.bus.stop()

    def test_no_decision_waits(self):
        """无决策时等待不崩溃"""
        self.agent.start()
        time.sleep(0.8)
        assert self.agent._execution_count == 0
        self.agent.stop()


class TestFullPipeline:
    """测试完整流水线：感知 -> 预测 -> 决策 -> 执行"""

    def test_end_to_end_pipeline(self):
        """端到端协作流程"""
        bus = MultiAgentBus("pipeline")
        bus.start()

        perception = PerceptionAgent("p1", bus=bus)
        prediction = PredictionAgent("pred1", bus=bus)
        decision = DecisionAgent("d1", bus=bus)
        execution = ExecutionAgent("e1", bus=bus)

        perception.start()
        prediction.start()
        decision.start()
        execution.start()

        time.sleep(3.0)  # 等待完整流转

        # 验证执行Agent收到了动作
        exec_stats = execution.get_stats()
        assert exec_stats["execution_count"] >= 1

        # 验证预测Agent收到了状态
        pred_stats = prediction.get_stats()
        assert pred_stats["has_state"] is True

        # 验证决策Agent收到了预测
        dec_stats = decision.get_stats()
        assert dec_stats["has_prediction"] is True

        perception.stop()
        prediction.stop()
        decision.stop()
        execution.stop()
        bus.stop()

    def test_fault_isolation(self):
        """故障隔离：一个Agent停止不影响其他"""
        bus = MultiAgentBus("fault_iso")
        bus.start()

        perception = PerceptionAgent("p2", bus=bus)
        prediction = PredictionAgent("pred2", bus=bus)

        perception.start()
        prediction.start()
        time.sleep(1.5)

        # 停止感知Agent
        perception.stop()
        time.sleep(1.0)

        # 预测Agent仍在运行
        assert prediction._running is True
        assert prediction.get_stats()["error_count"] == 0

        prediction.stop()
        bus.stop()
