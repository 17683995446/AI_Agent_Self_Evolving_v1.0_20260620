"""
自动调参与超参优化测试 —— 验证自适应调参和探索平衡
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.auto_tune import AdaptiveTuner, ExplorationBalancer, ParameterSpace


class TestAdaptiveTuner:
    """测试自适应调参器"""

    def test_initialization(self):
        tuner = AdaptiveTuner()
        params = tuner.get_all_parameters()
        assert "prediction_window" in params
        assert "exploration_ratio" in params
        assert params["prediction_window"]["current"] == 10.0

    def test_record_performance(self):
        tuner = AdaptiveTuner()
        tuner.record_performance("prediction_error", 0.5)
        tuner.record_performance("prediction_error", 0.4)
        stats = tuner.get_stats()
        assert stats["performance_metrics"]["prediction_error"] == 2

    def test_tune_with_insufficient_data(self):
        tuner = AdaptiveTuner()
        result = tuner.tune(force=True)
        # 没有足够数据，不应调整
        assert result["tuned"] is False

    def test_tune_with_improving_performance(self):
        tuner = AdaptiveTuner()
        # 模拟性能持续改善（误差下降）
        for i in range(20):
            tuner.record_performance("prediction_error", 1.0 - i * 0.05)
        result = tuner.tune(force=True)
        # 应有参数调整
        assert "adjustments" in result

    def test_parameter_boundaries(self):
        tuner = AdaptiveTuner()
        # 尝试设置超出边界的值
        tuner.set_parameter("exploration_ratio", 999.0)
        assert tuner.get_parameter("exploration_ratio") <= 0.5
        tuner.set_parameter("exploration_ratio", -999.0)
        assert tuner.get_parameter("exploration_ratio") >= 0.05

    def test_set_unknown_parameter(self):
        tuner = AdaptiveTuner()
        assert tuner.set_parameter("unknown_param", 1.0) is False

    def test_interval_respected(self):
        tuner = AdaptiveTuner()
        tuner._last_tune_time = time.time()
        result = tuner.tune(force=False)
        assert result["tuned"] is False
        assert "间隔不足" in result["reason"]


class TestExplorationBalancer:
    """测试探索-利用平衡器"""

    def test_initial_epsilon(self):
        eb = ExplorationBalancer(initial_epsilon=0.3)
        assert eb.epsilon == 0.3

    def test_should_explore_distribution(self):
        eb = ExplorationBalancer(initial_epsilon=0.5)
        explores = sum(1 for _ in range(1000) if eb.should_explore())
        # 500次左右应为探索（允许较大误差范围）
        assert 400 < explores < 600

    def test_explore_success_decreases_epsilon(self):
        eb = ExplorationBalancer(initial_epsilon=0.3, adaptation_rate=0.01)
        initial = eb.epsilon
        for _ in range(25):
            eb.report_result(success=True, was_exploration=True)
        assert eb.epsilon < initial

    def test_exploit_failure_increases_epsilon(self):
        eb = ExplorationBalancer(initial_epsilon=0.1, adaptation_rate=0.01)
        initial = eb.epsilon
        for _ in range(25):
            eb.report_result(success=False, was_exploration=False)
        assert eb.epsilon > initial

    def test_epsilon_boundaries(self):
        eb = ExplorationBalancer(initial_epsilon=0.05, min_epsilon=0.05,
                                 max_epsilon=0.5, adaptation_rate=0.1)
        # 大量利用成功不应低于min
        for _ in range(100):
            eb.report_result(success=True, was_exploration=False)
        assert eb.epsilon >= 0.05

        eb2 = ExplorationBalancer(initial_epsilon=0.5, min_epsilon=0.05,
                                  max_epsilon=0.5, adaptation_rate=0.1)
        # 大量利用失败不应超过max
        for _ in range(100):
            eb2.report_result(success=False, was_exploration=False)
        assert eb2.epsilon <= 0.5

    def test_stats(self):
        eb = ExplorationBalancer()
        for _ in range(10):
            eb.should_explore()
            eb.report_result(success=True, was_exploration=False)
        stats = eb.get_stats()
        assert stats["total_actions"] == 10
        assert stats["success_rate"] == 1.0
        assert 0.0 <= stats["explore_rate"] <= 1.0


class TestParameterSpace:
    """测试参数空间"""

    def test_parameter_creation(self):
        p = ParameterSpace(name="test", current_value=5.0, min_value=0.0,
                           max_value=10.0, step_size=0.5)
        assert p.name == "test"
        assert p.current_value == 5.0
