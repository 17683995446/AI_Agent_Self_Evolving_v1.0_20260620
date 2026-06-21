"""
H7阶段验证测试 —— 端到端集成验证

验证目标：H1-H6所有特性在完整业务闭环中协同工作
1. 完整闭环：感知→预测→误差→优化→驱力→决策→执行→记忆→学习→告警
2. H1：动作真实执行（非time.sleep空壳）
3. H2：决策-执行回调接通
4. H3：告警通道分发
5. H4：增强预测引擎可接入
6. H5：场景化Agent可运行
7. H6：性能监控在循环中采集
8. 反应式自愈：低安全→告警，高误差→调参，高CPU→清缓存
9. 经验记忆持续累积
10. 在线学习持续进行
"""

import os
import sys
import time
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import SelfEvolvingAgent
from core.interfaces import StateKeys, MetricKeys, DriveKeys
from core.performance import perf_monitor
from core.config import config_manager
from modules.decision import ActionType, Action


@pytest.fixture
def agent():
    """基础Agent实例"""
    a = SelfEvolvingAgent()
    yield a
    # 清理：确保停止
    if a.running:
        a.stop()
    # 重置性能监控避免用例间污染
    perf_monitor.reset()


@pytest.fixture
def enhanced_agent():
    """使用增强预测引擎的Agent"""
    a = SelfEvolvingAgent(module_config={"prediction": "enhanced"})
    yield a
    if a.running:
        a.stop()
    perf_monitor.reset()


class TestH7FullLoopExecution:
    """验证完整闭环执行"""

    def test_agent_runs_multiple_loops(self, agent):
        """Agent能运行多轮主循环"""
        agent.run_for_duration(1.5)
        assert agent.loop_count >= 1
        assert agent.start_time is not None

    def test_perception_produces_state_vector(self, agent):
        """感知模块产生状态向量"""
        agent.run_for_duration(0.8)
        state = agent.perception.get_state_vector()
        assert isinstance(state, dict)
        assert len(state) > 0
        # 应包含关键状态键
        assert StateKeys.CPU_PERCENT in state or "cpu_percent" in state

    def test_prediction_has_history_after_loops(self, agent):
        """循环后预测引擎有历史数据"""
        agent.run_for_duration(0.8)
        # 至少一层预测器应有历史
        has_history = (
            len(agent.prediction.base_predictor.history) > 0 or
            len(agent.prediction.mid_predictor.history) > 0 or
            len(agent.prediction.high_predictor.history) > 0
        )
        assert has_history

    def test_memory_records_experiences(self, agent):
        """记忆系统记录了经验"""
        agent.run_for_duration(0.8)
        stats = agent.memory.get_stats()
        assert stats["short_term_size"] > 0 or stats.get("replay_pool_size", 0) > 0

    def test_loop_count_increments(self, agent):
        """循环计数递增"""
        initial = agent.loop_count
        agent.run_for_duration(0.8)
        assert agent.loop_count > initial


class TestH7RealActionExecution:
    """验证H1：动作真实执行"""

    def test_executor_has_adapter(self, agent):
        """执行器有真实适配器"""
        executor = agent.decision.executor
        assert executor.adapter is not None

    def test_executor_has_callbacks(self, agent):
        """执行器有回调注入"""
        executor = agent.decision.executor
        assert len(executor._callbacks) > 0
        # 应包含关键回调
        assert "predict" in executor._callbacks
        assert "learn" in executor._callbacks

    def test_new_action_types_registered(self, agent):
        """新动作类型已注册"""
        # ActionType应有H1新增的类型
        assert hasattr(ActionType, "CLEAN_CACHE")
        assert hasattr(ActionType, "RESTART_SERVICE")
        assert hasattr(ActionType, "ALERT_NOTIFY")
        assert hasattr(ActionType, "KILL_PROCESS")

    def test_alert_notify_action_executes(self, agent):
        """ALERT_NOTIFY动作真实执行（触发告警分发）"""
        executor = agent.decision.executor
        action = Action(
            ActionType.ALERT_NOTIFY, "测试告警", "H7端到端测试",
            params={"message": "test alert from H7", "level": "warning"},
        )
        result = executor.execute(action)
        # 应该成功执行（即使告警通道发送失败，动作本身完成）
        assert result is not None

    def test_clean_cache_action_executes(self, agent):
        """CLEAN_CACHE动作真实执行"""
        executor = agent.decision.executor
        action = Action(
            ActionType.CLEAN_CACHE, "测试清理", "H7测试",
            params={"target_dirs": ["/tmp/h7_test_cache"]},
        )
        # 创建临时目录供清理
        os.makedirs("/tmp/h7_test_cache", exist_ok=True)
        with open("/tmp/h7_test_cache/test.txt", "w") as f:
            f.write("test")
        result = executor.execute(action)
        assert result is not None


class TestH7WiringIntegration:
    """验证H2：决策-执行接通"""

    def test_predict_callback_works(self, agent):
        """predict回调真正调用prediction模块"""
        executor = agent.decision.executor
        cb = executor._callbacks.get("predict")
        assert cb is not None
        # 调用回调应返回预测结果
        result = cb({})
        # 可能返回None（未训练）或PredictionResult
        # 关键是不抛异常
        assert result is None or hasattr(result, "predicted_values")

    def test_alert_callback_dispatches(self, agent):
        """alert回调触发告警分发"""
        executor = agent.decision.executor
        cb = executor._callbacks.get("alert")
        assert cb is not None
        # 调用回调应不抛异常
        cb({"message": "H7 test alert", "level": "warning"})

    def test_reactive_actions_triggered_on_low_safety(self, agent):
        """低安全驱力触发反应式告警"""
        # 构造低安全驱力状态
        low_safety_state = MagicMock()
        low_safety_state.safety = 0.1  # 低于0.3阈值
        low_safety_state.gain = 0.5
        low_safety_state.efficiency = 0.5
        low_safety_state.intrinsic_motivation = 0.3

        state_vector = {StateKeys.CPU_PERCENT: 0.3}
        error_report = {"mae": 1.0}

        # 应不抛异常地触发告警动作
        agent._trigger_reactive_actions(low_safety_state, error_report, state_vector)

    def test_reactive_actions_triggered_on_high_error(self, agent):
        """高误差触发反应式模型调整"""
        high_error_state = MagicMock()
        high_error_state.safety = 0.8
        high_error_state.gain = 0.5
        high_error_state.efficiency = 0.5
        high_error_state.intrinsic_motivation = 0.5

        state_vector = {StateKeys.CPU_PERCENT: 0.3}
        error_report = {"mae": 15.0}  # 大于10阈值

        agent._trigger_reactive_actions(high_error_state, error_report, state_vector)

    def test_reactive_actions_triggered_on_high_cpu(self, agent):
        """高CPU触发反应式缓存清理"""
        normal_state = MagicMock()
        normal_state.safety = 0.8
        normal_state.gain = 0.5
        normal_state.efficiency = 0.5
        normal_state.intrinsic_motivation = 0.5

        state_vector = {StateKeys.CPU_PERCENT: 0.9}  # 90% > 80%阈值
        error_report = {"mae": 1.0}

        agent._trigger_reactive_actions(normal_state, error_report, state_vector)


class TestH7AlertChannels:
    """验证H3：告警通道在循环中工作"""

    def test_alert_manager_exists(self, agent):
        """Agent有告警管理器"""
        assert hasattr(agent, "alert_manager")
        assert agent.alert_manager is not None

    def test_alert_manager_has_log_channel(self, agent):
        """告警管理器至少有日志通道"""
        channels = agent.alert_manager._channels
        assert "log" in channels or len(channels) > 0

    def test_alert_dispatch_during_loop(self, agent):
        """循环中告警能分发"""
        agent.run_for_duration(0.8)
        # 告警历史应有记录（如果触发了告警规则）
        history = agent.alert_manager.get_history()
        # 不强制要求有告警（取决于状态），但接口应工作
        assert isinstance(history, list)

    def test_alert_dispatch_directly(self, agent):
        """直接调用dispatch能分发"""
        from core.alerting import AlertEvent, AlertLevel
        event = AlertEvent(
            rule_name="h7_test",
            level=AlertLevel.WARNING,
            message="H7 direct dispatch test",
            timestamp=time.time(),
            metric_value=0.9,
            threshold=0.8,
        )
        before = len(agent.alert_manager.get_history())
        agent.alert_manager.dispatch(event)
        after = len(agent.alert_manager.get_history())
        assert after == before + 1


class TestH7EnhancedPrediction:
    """验证H4：增强预测引擎接入"""

    def test_enhanced_agent_creates(self, enhanced_agent):
        """使用enhanced预测引擎的Agent能创建"""
        assert enhanced_agent is not None
        from modules.prediction import EnhancedPredictionEngine
        assert isinstance(enhanced_agent.prediction, EnhancedPredictionEngine)

    def test_enhanced_agent_runs_loops(self, enhanced_agent):
        """增强Agent能运行循环"""
        enhanced_agent.run_for_duration(0.8)
        assert enhanced_agent.loop_count >= 1

    def test_enhanced_engine_has_uncertainty(self, enhanced_agent):
        """增强引擎有不确定性估计器"""
        assert enhanced_agent.prediction.use_uncertainty is True
        assert enhanced_agent.prediction._uncertainty_estimator is not None


class TestH7ScenarioIntegration:
    """验证H5：场景化Agent运行"""

    def test_create_agent_from_system_monitor_scenario(self):
        """从system_monitor场景创建Agent"""
        from core.scenario_loader import create_agent_from_scenario
        agent = create_agent_from_scenario("system_monitor")
        assert agent is not None
        assert hasattr(agent, "decision")
        assert hasattr(agent, "alert_manager")
        if agent.running:
            agent.stop()
        perf_monitor.reset()

    def test_scenario_agent_runs(self):
        """场景Agent能运行"""
        from core.scenario_loader import create_agent_from_scenario
        agent = create_agent_from_scenario("dev_env_optimizer")
        agent.run_for_duration(0.8)
        assert agent.loop_count >= 1
        if agent.running:
            agent.stop()
        perf_monitor.reset()

    def test_scenario_applies_config(self):
        """场景配置应用到Agent"""
        from core.scenario_loader import create_agent_from_scenario
        agent = create_agent_from_scenario("system_monitor")
        # system_monitor设置max_cpu_percent=80
        assert config_manager.config.resource.max_cpu_percent == 80.0
        if agent.running:
            agent.stop()
        perf_monitor.reset()


class TestH7PerformanceMonitoring:
    """验证H6：性能监控在循环中采集"""

    def test_perf_monitor_enabled(self, agent):
        """性能监控已启用"""
        assert agent.config.logging.perf_monitor_enabled is True

    def test_performance_stats_collected_after_loops(self, agent):
        """循环后采集到性能统计"""
        perf_monitor.reset()
        agent.run_for_duration(0.8)
        stats = agent.get_performance_stats()
        # 应有操作记录
        assert stats["summary"]["operation_count"] > 0
        assert stats["summary"]["total_calls"] > 0

    def test_key_operations_tracked(self, agent):
        """关键操作被追踪"""
        perf_monitor.reset()
        agent.run_for_duration(0.8)
        operations = perf_monitor.get_stats()
        # 应至少追踪到部分关键操作
        key_ops = ["perceive", "predict", "error_eval", "optimize",
                   "drive_compute", "decide_execute"]
        tracked = [op for op in key_ops if op in operations]
        assert len(tracked) > 0

    def test_performance_report_generated(self, agent):
        """性能报告可生成"""
        agent.run_for_duration(0.8)
        report = agent.get_performance_report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_health_report_includes_performance(self, agent):
        """健康报告包含性能摘要"""
        agent.start_time = time.time()
        report = agent._generate_health_report()
        assert "performance_summary" in report
        assert "operation_count" in report["performance_summary"]


class TestH7ClosedLoopIntegrity:
    """验证闭环完整性 —— 所有模块协同"""

    def test_full_closed_loop_all_steps(self, agent):
        """完整闭环：所有10个步骤都执行"""
        agent.run_for_duration(1.0)
        # 验证各模块都有数据流动
        assert agent.loop_count >= 1

        # 感知：有状态向量
        state = agent.perception.get_state_vector()
        assert len(state) > 0

        # 预测：有历史数据
        assert (len(agent.prediction.base_predictor.history) > 0 or
                len(agent.prediction.mid_predictor.history) > 0)

        # 记忆：有经验记录
        mem_stats = agent.memory.get_stats()
        assert mem_stats["short_term_size"] > 0 or mem_stats.get("replay_pool_size", 0) > 0

        # 执行器：有执行统计
        exec_stats = agent.decision.executor.get_execution_stats()
        assert isinstance(exec_stats, dict)

    def test_drive_engine_produces_drives(self, agent):
        """驱力引擎产生驱力值"""
        agent.run_for_duration(0.8)
        summary = agent.drive_engine.get_drive_summary()
        assert isinstance(summary, dict)
        # 应有驱力相关字段
        assert len(summary) > 0

    def test_optimizer_has_history(self, agent):
        """优化器有优化历史"""
        agent.run_for_duration(0.8)
        history = agent.optimizer.get_history()
        assert isinstance(history, list)

    def test_error_eval_records_history(self, agent):
        """误差评估记录历史"""
        agent.run_for_duration(0.8)
        stats = agent.error_eval.get_history_stats()
        assert isinstance(stats, dict)

    def test_concurrent_loops_stable(self, agent):
        """多轮循环稳定运行无崩溃"""
        agent.run_for_duration(1.5)
        # 应运行多轮且无异常
        assert agent.loop_count >= 2
        # 系统应仍处于可响应状态
        assert agent.perception.get_state_vector() is not None


class TestH7LoggingConfigApplied:
    """验证H6：日志配置在运行时生效"""

    def test_default_log_level_info(self, agent):
        """默认控制台级别为INFO"""
        import logging
        assert logger._console_handler.level == logging.INFO

    def test_func_trace_disabled_by_default(self, agent):
        """默认禁用函数追踪"""
        assert logger.func_trace_enabled is False

    def test_sample_rates_applied(self, agent):
        """采样率已应用"""
        # FUNC_ENTER/FUNC_EXIT应有采样配置
        assert "FUNC_ENTER" in logger._sample_rates
        assert "FUNC_EXIT" in logger._sample_rates


class TestH7SystemStability:
    """验证系统稳定性"""

    def test_start_stop_cycle(self, agent):
        """启动-停止循环正常"""
        agent.start()
        time.sleep(0.3)
        assert agent.running is True
        agent.stop()
        assert agent.running is False

    def test_multiple_start_stop(self, agent):
        """多次启动停止"""
        for _ in range(2):
            agent.start()
            time.sleep(0.2)
            agent.stop()
        # 应正常完成无崩溃
        assert agent.running is False

    def test_stop_is_idempotent(self, agent):
        """stop幂等"""
        agent.start()
        time.sleep(0.2)
        agent.stop()
        # 再次stop不应抛异常
        agent.stop()

    def test_no_resource_leak_on_stop(self, agent):
        """停止后无资源泄漏（线程结束）"""
        agent.start()
        time.sleep(0.3)
        agent.stop()
        # 主线程应已结束
        if agent._main_thread:
            assert not agent._main_thread.is_alive()


# 导入logger用于TestH7LoggingConfigApplied
from core.logger import logger
