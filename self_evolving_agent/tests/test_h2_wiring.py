"""
H2阶段验证测试 —— 决策-执行接通

验证点：
1. SelfEvolvingAgent初始化时回调注入完成
2. 上下文注入机制工作
3. 反应式动作触发（低安全/高误差/高CPU）
4. 回调真实调用对应模块
5. 端到端：决策→执行→真实副作用
"""

import os
import sys
import time
import tempfile
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import SelfEvolvingAgent
from modules.decision import Action, ActionType


class TestH2CallbackWiring:
    """验证回调注入"""

    def test_agent_init_wires_callbacks(self):
        """Agent初始化时自动注入回调"""
        agent = SelfEvolvingAgent()
        executor = agent.decision.executor
        # 应至少注入predict/learn/adjust_model回调
        assert "predict" in executor._callbacks
        assert "learn" in executor._callbacks
        assert "adjust_model" in executor._callbacks

    def test_predict_callback_calls_real_module(self):
        """预测回调真实调用prediction模块"""
        agent = SelfEvolvingAgent()
        called = {"count": 0}
        original_predict = agent.prediction.predict

        def mock_predict():
            called["count"] += 1
            return original_predict()

        agent.prediction.predict = mock_predict
        agent.decision.executor._callbacks["predict"]({})
        assert called["count"] == 1


class TestH2ContextInjection:
    """验证上下文注入"""

    def test_inject_action_context_stores_data(self):
        """上下文注入存储到执行器"""
        agent = SelfEvolvingAgent()
        predicted = {"cpu": 50.0}
        actual = {"cpu": 55.0}
        error_report = {"mae": 5.0}
        state_vector = {"cpu_percent": 0.5}

        agent._inject_action_context(predicted, actual, error_report, state_vector)

        executor = agent.decision.executor
        assert hasattr(executor, "_last_context")
        assert executor._last_context["predicted"] == predicted
        assert executor._last_context["actual"] == actual
        assert executor._last_context["error_report"] == error_report

    def test_learn_callback_reads_context(self):
        """学习回调从上下文读取actual"""
        agent = SelfEvolvingAgent()
        called = {"actual": None}

        def mock_online_learn(actual):
            called["actual"] = actual

        agent.prediction.online_learn = mock_online_learn
        agent._inject_action_context({}, {"cpu": 60.0}, {}, {})
        agent.decision.executor._callbacks["learn"]({})
        assert called["actual"] == {"cpu": 60.0}


class TestH2ReactiveActions:
    """验证反应式动作触发"""

    def test_low_safety_triggers_alert(self):
        """低安全驱力触发告警动作"""
        agent = SelfEvolvingAgent()
        alert_called = {"count": 0}

        def mock_alert_cb(info):
            alert_called["count"] += 1

        agent.decision.executor.register_callback("alert", mock_alert_cb)

        # 构造低安全驱力
        from modules.drive_engine import DriveState
        drive_state = DriveState(
            safety=0.2,  # 低于0.3阈值
            gain=0.5,
            efficiency=0.5,
            intrinsic_motivation=0.4,
        )
        error_report = {"mae": 1.0}
        state_vector = {"cpu_percent": 0.3}

        agent._trigger_reactive_actions(drive_state, error_report, state_vector)
        assert alert_called["count"] == 1, "低安全应触发1次告警"

    def test_high_error_triggers_adjust(self):
        """高误差触发模型调整"""
        agent = SelfEvolvingAgent()
        adjust_called = {"count": 0}

        original_adjust = agent.decision.executor._callbacks.get("adjust_model")
        def mock_adjust(params):
            adjust_called["count"] += 1
        agent.decision.executor.register_callback("adjust_model", mock_adjust)

        from modules.drive_engine import DriveState
        drive_state = DriveState(safety=0.8, gain=0.5, efficiency=0.5, intrinsic_motivation=0.5)
        error_report = {"mae": 15.0}  # 大于10阈值
        state_vector = {"cpu_percent": 0.3}

        agent._trigger_reactive_actions(drive_state, error_report, state_vector)
        assert adjust_called["count"] == 1, "高误差应触发1次模型调整"

    def test_high_cpu_triggers_clean_cache(self):
        """高CPU触发缓存清理"""
        agent = SelfEvolvingAgent()
        clean_called = {"count": 0}

        # mock execute方法检测清理动作
        original_execute = agent.decision.executor.execute
        def mock_execute(action):
            if action.action_type == ActionType.CLEAN_CACHE:
                clean_called["count"] += 1
            return original_execute(action)
        agent.decision.executor.execute = mock_execute

        from modules.drive_engine import DriveState
        drive_state = DriveState(safety=0.8, gain=0.5, efficiency=0.5, intrinsic_motivation=0.5)
        error_report = {"mae": 1.0}
        state_vector = {"cpu_percent": 0.85}  # 85% > 80%

        agent._trigger_reactive_actions(drive_state, error_report, state_vector)
        assert clean_called["count"] == 1, "高CPU应触发1次缓存清理"

    def test_normal_conditions_no_trigger(self):
        """正常条件不触发反应式动作"""
        agent = SelfEvolvingAgent()
        triggered = {"count": 0}

        original_execute = agent.decision.executor.execute
        def mock_execute(action):
            triggered["count"] += 1
            return original_execute(action)
        agent.decision.executor.execute = mock_execute

        from modules.drive_engine import DriveState
        drive_state = DriveState(safety=0.9, gain=0.5, efficiency=0.5, intrinsic_motivation=0.5)
        error_report = {"mae": 1.0}
        state_vector = {"cpu_percent": 0.3}

        agent._trigger_reactive_actions(drive_state, error_report, state_vector)
        assert triggered["count"] == 0, "正常条件不应触发反应式动作"


class TestH2EndToEnd:
    """端到端验证：决策→执行→真实副作用"""

    def test_full_loop_produces_real_effects(self):
        """完整循环产生真实副作用"""
        agent = SelfEvolvingAgent()

        # 注入上下文
        agent._inject_action_context(
            predicted={"cpu": 50.0},
            actual={"cpu": 55.0},
            error_report={"mae": 5.0},
            state_vector={"cpu_percent": 0.5},
        )

        # 执行一个SAVE_SNAPSHOT动作，应真实生成文件
        action = Action(ActionType.SAVE_SNAPSHOT, "测试快照", "端到端验证")
        result = agent.decision.executor.execute(action)
        assert result.success

        import glob
        snapshots = glob.glob(os.path.join(agent.config.model_snapshot_dir, "action_snapshot_*.pkl"))
        assert len(snapshots) > 0, "应生成真实快照文件"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
