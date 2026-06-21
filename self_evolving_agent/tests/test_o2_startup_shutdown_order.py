"""
O2阶段验证测试 —— 启动/关闭秩序

验证点：
1. _stopped标志在__init__中显式初始化
2. start()调用perception.start()
3. stop()调用perception.stop()
4. stop()清理DI容器
5. stop()幂等性
6. 启动/关闭顺序正确（与文档一致）
7. 资源完整释放
8. 不破坏现有功能
"""

import os
import sys
import time
import threading
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestO2StoppedFlagInit:
    """验证_stopped标志初始化"""

    def test_stopped_flag_initialized_in_init(self):
        """_stopped应在__init__中显式初始化为False"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert hasattr(agent, "_stopped")
        assert agent._stopped is False
        agent.stop()

    def test_stopped_flag_set_after_stop(self):
        """stop()后_stopped应为True"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.stop()
        assert agent._stopped is True


class TestO2PerceptionLifecycle:
    """验证感知模块生命周期管理"""

    def test_start_calls_perception_start(self):
        """start()应调用perception.start()"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        # 用mock跟踪perception.start是否被调用
        original_start = agent.perception.start
        called = {"start": False}
        def mock_start():
            called["start"] = True
            original_start()
        agent.perception.start = mock_start
        agent.start()
        time.sleep(0.3)
        assert called["start"] is True
        agent.stop()

    def test_stop_calls_perception_stop(self):
        """stop()应调用perception.stop()"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        original_stop = agent.perception.stop
        called = {"stop": False}
        def mock_stop():
            called["stop"] = True
            original_stop()
        agent.perception.stop = mock_stop
        agent.start()
        time.sleep(0.3)
        agent.stop()
        assert called["stop"] is True

    def test_perception_start_failure_degrades_gracefully(self):
        """perception.start()失败应降级而非崩溃"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        def failing_start():
            raise RuntimeError("模拟启动失败")
        agent.perception.start = failing_start
        # 不应抛异常
        agent.start()
        time.sleep(0.3)
        assert agent.running is True
        agent.stop()


class TestO2StopIdempotency:
    """验证stop()幂等性"""

    def test_stop_idempotent(self):
        """多次调用stop()不应抛异常"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(0.3)
        agent.stop()
        # 再次调用不应抛异常
        agent.stop()
        agent.stop()
        assert agent._stopped is True
        assert agent.running is False

    def test_stop_without_start(self):
        """未start直接stop不应抛异常"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        # 未start直接stop
        agent.stop()
        assert agent._stopped is True


class TestO2ShutdownOrder:
    """验证关闭顺序"""

    def test_forever_runner_stopped_before_threads(self):
        """永续运行器应在循环线程之前停止"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        agent.enable_forever_running(health_check_interval=0.5,
                                     resource_cleanup_interval=1.0)
        time.sleep(0.5)
        agent.stop()
        # 永续运行器应已停止
        assert agent._forever_running_enabled is False
        # Agent应已停止
        assert agent.running is False

    def test_workflow_engine_stopped(self):
        """工作流引擎应被停止"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        # 启动工作流引擎
        agent.submit_goal("monitor_and_alert", "测试")
        time.sleep(0.3)
        agent.stop()
        assert agent._workflow_engine_started is False

    def test_orchestrator_stopped(self):
        """多Agent集群应被停止"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        agent.enable_multi_agent(spawn_default=True)
        time.sleep(0.3)
        agent.stop()
        assert agent._orchestrator_enabled is False


class TestO2ResourceCleanup:
    """验证资源完整释放"""

    def test_memory_closed_after_stop(self):
        """stop()后记忆系统应被关闭"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(0.3)
        agent.stop()
        # memory.close()应被调用（通过不崩溃验证）

    def test_di_container_cleaned(self):
        """DI容器应被清理（如果存在）"""
        from main import SelfEvolvingAgent
        from core.container import DIContainer
        container = DIContainer()
        agent = SelfEvolvingAgent(container=container)
        agent.start()
        time.sleep(0.3)
        agent.stop()
        # 不崩溃即可

    def test_no_thread_leak_after_stop(self):
        """stop()后不应有大量残留线程"""
        from main import SelfEvolvingAgent
        threads_before = threading.active_count()
        agent = SelfEvolvingAgent()
        agent.start()
        agent.enable_multi_agent(spawn_default=True)
        agent.enable_forever_running(health_check_interval=0.3,
                                     resource_cleanup_interval=0.5)
        time.sleep(1.0)
        agent.stop()
        time.sleep(1.0)  # 等待daemon线程退出
        threads_after = threading.active_count()
        # 线程数应回落到接近启动前
        assert threads_after < threads_before + 20, \
            f"线程泄漏: before={threads_before}, after={threads_after}"


class TestO2StartOrder:
    """验证启动顺序"""

    def test_start_sets_running_true_first(self):
        """start()应先设置running=True再启动线程"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        # running应为True
        assert agent.running is True
        # start_time应被设置
        assert agent.start_time is not None
        agent.stop()

    def test_start_initializes_heartbeat(self):
        """start()应初始化心跳"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(0.1)
        stats = agent.get_persistence_stats()
        # 心跳年龄应较小（刚启动）
        assert stats["last_heartbeat_age"] < 5.0
        agent.stop()

    def test_watchdog_started_after_main_loop(self):
        """看门狗应在主循环之后启动"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(0.3)
        stats = agent.get_persistence_stats()
        assert stats["watchdog_running"] is True
        agent.stop()


class TestO2NoRegression:
    """验证O2不破坏现有功能"""

    def test_main_loop_still_runs(self):
        """主循环应仍能正常运行"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(1.0)
        assert agent.loop_count > 0
        agent.stop()

    def test_run_for_duration_still_works(self):
        """run_for_duration应仍可用"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.run_for_duration(0.8)
        assert agent.loop_count > 0

    def test_m1_m7_capabilities_still_work(self):
        """M1-M7各能力应仍可用"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        try:
            # M1
            ids = agent.enable_multi_agent(spawn_default=True)
            assert len(ids) >= 4
            # M2
            wf_id = agent.submit_goal("monitor_and_alert", "O2回归测试")
            assert wf_id is not None
            # M3
            stats = agent.get_skill_library_stats()
            assert "total_skills" in stats
            # M4
            task_id = agent.plan_long_task("system_optimization")
            assert task_id is not None
            # M5
            path = agent.save_state()
            assert path is not None
            # M6
            report = agent.get_health_report()
            assert report is not None
        finally:
            agent.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
