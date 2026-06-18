"""
通用Agent集群与任务分派测试 —— 验证负载均衡、任务执行、超时重试

测试前置：
1. 拆解GeneralAgent接口：start/stop/_on_task/_process/get_stats
2. 拆解TaskDispatcher接口：dispatch/start/stop/get_task_status/get_stats
3. 边界条件：无可用Agent、任务超时、Agent离线、重复结果
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.general_agent_cluster import (
    GeneralAgent, TaskDispatcher, Task, TaskStatus,
)
from core.multi_agent_bus import MultiAgentBus, AgentRole, MessagePriority, AgentMessage


class TestGeneralAgent:
    """测试通用Agent"""

    def setup_method(self):
        self.bus = MultiAgentBus("test_gen")
        self.agent = GeneralAgent("gen-1", self.bus)

    def teardown_method(self):
        self.agent.stop()

    def test_start_stop(self):
        """启动和停止"""
        self.agent.start()
        assert self.agent._running is True
        stats = self.agent.get_stats()
        assert stats["agent_id"] == "gen-1"
        self.agent.stop()
        assert self.agent._running is False

    def test_process_predict(self):
        """处理预测任务"""
        result = self.agent._process("predict", {"input": 100})
        assert result["prediction"] == 105.0

    def test_process_decide(self):
        """处理决策任务"""
        assert self.agent._process("decide", {"input": 85})["action"] == "throttle"
        assert self.agent._process("decide", {"input": 15})["action"] == "boost"
        assert self.agent._process("decide", {"input": 50})["action"] == "noop"

    def test_process_execute(self):
        """处理执行任务"""
        result = self.agent._process("execute", {"action": "restart"})
        assert result["executed"] == "restart"

    def test_process_perceive(self):
        """处理感知任务"""
        result = self.agent._process("perceive", {})
        assert "state" in result

    def test_process_unknown(self):
        """未知任务类型"""
        result = self.agent._process("unknown", {})
        assert result["result"] == "unknown_task_type"

    def test_task_execution_flow(self):
        """完整任务执行流程"""
        self.bus.start()
        self.agent.start()

        # 模拟发送任务消息
        task = Task(task_id="t1", task_type="predict", payload={"input": 200})
        msg = AgentMessage(
            msg_type="task",
            sender="orchestrator",
            target="gen-1",
            payload={"task": task.to_dict()},
        )
        self.bus.send(msg)
        time.sleep(0.5)

        assert self.agent._completed_tasks >= 1
        self.agent.stop()
        self.bus.stop()


class TestTaskDispatcher:
    """测试任务调度器"""

    def setup_method(self):
        self.bus = MultiAgentBus("test_dispatch")
        self.dispatcher = TaskDispatcher(self.bus)

    def teardown_method(self):
        self.dispatcher.stop()

    def test_dispatch_no_agents_fails(self):
        """无可用Agent时分派失败"""
        self.bus.start()
        self.dispatcher.start()
        task_id = self.dispatcher.dispatch("predict", {"input": 100})
        status = self.dispatcher.get_task_status(task_id)
        assert status is not None
        assert status["status"] == "failed"
        self.dispatcher.stop()
        self.bus.stop()

    def test_dispatch_with_agent(self):
        """有Agent时正常分派"""
        self.bus.start()
        agent = GeneralAgent("gen-2", self.bus)
        agent.start()
        self.dispatcher.start()

        task_id = self.dispatcher.dispatch("predict", {"input": 100})
        time.sleep(0.5)

        status = self.dispatcher.get_task_status(task_id)
        assert status is not None
        assert status["assigned_to"] == "gen-2"
        self.dispatcher.stop()
        agent.stop()
        self.bus.stop()

    def test_dispatch_callback(self):
        """任务完成回调"""
        self.bus.start()
        agent = GeneralAgent("gen-3", self.bus)
        agent.start()
        self.dispatcher.start()

        callback_called = [False]
        def cb(task):
            callback_called[0] = True

        task_id = self.dispatcher.dispatch("predict", {"input": 100}, callback=cb)
        time.sleep(0.5)

        assert callback_called[0] is True
        self.dispatcher.stop()
        agent.stop()
        self.bus.stop()

    def test_least_loaded_selection(self):
        """选择负载最低的Agent"""
        self.bus.start()
        agent1 = GeneralAgent("gen-a", self.bus)
        agent2 = GeneralAgent("gen-b", self.bus)
        agent1.start()
        agent2.start()

        # 让agent1忙
        agent1._current_task = Task(task_id="busy")
        self.bus.heartbeat("gen-a", load_score=1.0)
        self.bus.heartbeat("gen-b", load_score=0.0)
        time.sleep(0.1)

        self.dispatcher.start()
        task_id = self.dispatcher.dispatch("predict", {"input": 100})
        time.sleep(0.5)

        status = self.dispatcher.get_task_status(task_id)
        assert status["assigned_to"] == "gen-b"
        self.dispatcher.stop()
        agent1.stop()
        agent2.stop()
        self.bus.stop()

    def test_stats(self):
        """调度器统计"""
        self.bus.start()
        self.dispatcher.start()
        stats = self.dispatcher.get_stats()
        assert stats["total_tasks"] == 0
        assert stats["completed"] == 0
        self.dispatcher.stop()
        self.bus.stop()


class TestTask:
    """测试任务数据类"""

    def test_task_defaults(self):
        """默认值"""
        t = Task()
        assert t.task_id != ""
        assert t.status == TaskStatus.PENDING
        assert t.retry_count == 0
        assert t.max_retries == 2

    def test_task_to_dict(self):
        """序列化"""
        t = Task(task_id="abc", task_type="predict", payload={"x": 1})
        d = t.to_dict()
        assert d["task_id"] == "abc"
        assert d["task_type"] == "predict"
        assert d["payload"] == {"x": 1}


class TestClusterIntegration:
    """测试集群集成"""

    def test_multiple_agents_share_load(self):
        """多个Agent分担负载"""
        bus = MultiAgentBus("cluster")
        bus.start()

        agents = [GeneralAgent(f"gen-{i}", bus) for i in range(3)]
        for a in agents:
            a.start()

        dispatcher = TaskDispatcher(bus)
        dispatcher.start()

        task_ids = []
        for i in range(6):
            tid = dispatcher.dispatch("predict", {"input": i})
            task_ids.append(tid)

        time.sleep(1.0)

        stats = dispatcher.get_stats()
        assert stats["total_tasks"] == 6
        assert stats["completed"] == 6

        dispatcher.stop()
        for a in agents:
            a.stop()
        bus.stop()
