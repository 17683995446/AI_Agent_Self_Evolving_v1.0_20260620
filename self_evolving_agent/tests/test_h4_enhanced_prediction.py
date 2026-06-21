"""
H4阶段验证测试 —— 高级预测模型接入

验证点：
1. EnhancedPredictionEngine类存在且可创建
2. 不确定性估计器初始化
3. 未训练时回退到父类预测
4. 训练后启用融合预测
5. 融合权重计算合理
6. 在线学习同时更新不确定性估计器
7. registry注册了enhanced插件
"""

import os
import sys
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.prediction import EnhancedPredictionEngine, PredictionEngine, PredictionResult
from core.interfaces import MetricKeys


class TestH4EnhancedEngineCreation:
    """验证增强引擎创建"""

    def test_enhanced_engine_exists(self):
        assert EnhancedPredictionEngine is not None

    def test_enhanced_engine_inherits_prediction_engine(self):
        """继承自PredictionEngine"""
        assert issubclass(EnhancedPredictionEngine, PredictionEngine)

    def test_create_with_uncertainty(self):
        """启用不确定性创建"""
        engine = EnhancedPredictionEngine(use_uncertainty=True)
        assert engine.use_uncertainty is True
        assert engine._uncertainty_estimator is not None

    def test_create_without_uncertainty(self):
        """禁用不确定性创建"""
        engine = EnhancedPredictionEngine(use_uncertainty=False)
        assert engine.use_uncertainty is False
        assert engine._uncertainty_estimator is None


class TestH4FallbackBehavior:
    """验证未训练时回退行为"""

    def test_predict_fallback_before_training(self):
        """未训练时回退到父类预测"""
        engine = EnhancedPredictionEngine(use_uncertainty=True)
        # 先update一些数据
        for i in range(15):
            engine.update({
                MetricKeys.CPU: 50.0 + i,
                MetricKeys.MEMORY: 60.0,
                MetricKeys.DISK: 70.0,
            })
        result = engine.predict()
        # 未训练时应回退到父类（layer不是enhanced_fusion）
        assert result.layer != "enhanced_fusion"

    def test_predict_uses_fusion_after_training(self):
        """训练后启用融合预测"""
        engine = EnhancedPredictionEngine(use_uncertainty=True)
        # 喂数据 + 在线学习
        for i in range(20):
            state = {
                MetricKeys.CPU: 50.0 + i * 0.5,
                MetricKeys.MEMORY: 60.0,
                MetricKeys.DISK: 70.0,
            }
            engine.update(state)
            engine.online_learn(state)

        assert engine._uncertainty_fitted is True
        result = engine.predict()
        assert result.layer == "enhanced_fusion"
        assert len(result.predicted_values) > 0


class TestH4FusionWeights:
    """验证融合权重计算"""

    def test_fusion_weights_sum_to_one(self):
        """融合权重归一化（和为1）"""
        engine = EnhancedPredictionEngine(use_uncertainty=True)
        for i in range(20):
            state = {
                MetricKeys.CPU: 50.0 + np.random.randn() * 5,
                MetricKeys.MEMORY: 60.0,
                MetricKeys.DISK: 70.0,
            }
            engine.update(state)
            engine.online_learn(state)

        result = engine.predict()
        report = engine.get_fusion_report()
        assert report["enabled"] is True
        assert report["uncertainty_fitted"] is True
        # 检查权重归一化
        for key, weights in report["latest_weights"].items():
            total = weights["base"] + weights["mid"] + weights["high"]
            assert abs(total - 1.0) < 0.01, f"{key}权重和应为1, 实际{total}"

    def test_confidence_in_range(self):
        """置信度在[0,1]范围"""
        engine = EnhancedPredictionEngine(use_uncertainty=True)
        for i in range(20):
            engine.update({MetricKeys.CPU: 50.0, MetricKeys.MEMORY: 60.0, MetricKeys.DISK: 70.0})
            engine.online_learn({MetricKeys.CPU: 50.0, MetricKeys.MEMORY: 60.0, MetricKeys.DISK: 70.0})

        result = engine.predict()
        assert 0.0 <= result.confidence <= 1.0


class TestH4OnlineLearning:
    """验证在线学习"""

    def test_online_learn_updates_uncertainty(self):
        """在线学习更新不确定性估计器"""
        engine = EnhancedPredictionEngine(use_uncertainty=True)
        assert engine._uncertainty_fitted is False

        for i in range(5):
            engine.update({MetricKeys.CPU: 50.0, MetricKeys.MEMORY: 60.0, MetricKeys.DISK: 70.0})
            engine.online_learn({MetricKeys.CPU: 50.0, MetricKeys.MEMORY: 60.0, MetricKeys.DISK: 70.0})

        assert engine._uncertainty_fitted is True

    def test_online_learn_handles_errors_gracefully(self):
        """在线学习处理异常不崩溃"""
        engine = EnhancedPredictionEngine(use_uncertainty=True)
        # 传入异常数据
        engine.online_learn({MetricKeys.CPU: float('inf'), MetricKeys.MEMORY: 60.0, MetricKeys.DISK: 70.0})
        # 不应抛异常


class TestH4Registry:
    """验证registry注册"""

    def test_enhanced_registered_in_registry(self):
        """enhanced插件已注册"""
        from core.registry import get_default_registry
        reg = get_default_registry()
        predictions = reg.list_predictions()
        assert "enhanced" in predictions

    def test_create_enhanced_from_registry(self):
        """从registry创建enhanced引擎"""
        from core.registry import get_default_registry
        reg = get_default_registry()
        engine = reg.create_prediction("enhanced")
        assert isinstance(engine, EnhancedPredictionEngine)


class TestH4AgentIntegration:
    """验证Agent可使用enhanced引擎"""

    def test_agent_with_enhanced_prediction(self):
        """Agent使用enhanced预测引擎"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent(module_config={"prediction": "enhanced"})
        assert isinstance(agent.prediction, EnhancedPredictionEngine)

    def test_enhanced_engine_works_in_loop(self):
        """enhanced引擎在模拟循环中工作"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent(module_config={"prediction": "enhanced"})

        # 模拟5轮在线循环的预测部分
        for i in range(5):
            state = {
                MetricKeys.CPU: 50.0 + i,
                MetricKeys.MEMORY: 60.0,
                MetricKeys.DISK: 70.0,
            }
            agent.prediction.update(state)
            result = agent.prediction.predict()
            agent.prediction.online_learn(state)
            assert isinstance(result, PredictionResult)


class TestH4FusionReport:
    """验证融合报告"""

    def test_get_fusion_report_empty(self):
        """无历史时报告正确"""
        engine = EnhancedPredictionEngine(use_uncertainty=True)
        report = engine.get_fusion_report()
        assert report["history_len"] == 0

    def test_get_fusion_report_after_predict(self):
        """预测后有报告"""
        engine = EnhancedPredictionEngine(use_uncertainty=True)
        for i in range(20):
            engine.update({MetricKeys.CPU: 50.0, MetricKeys.MEMORY: 60.0, MetricKeys.DISK: 70.0})
            engine.online_learn({MetricKeys.CPU: 50.0, MetricKeys.MEMORY: 60.0, MetricKeys.DISK: 70.0})
        engine.predict()
        report = engine.get_fusion_report()
        assert report["history_len"] > 0
        assert "latest_weights" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
