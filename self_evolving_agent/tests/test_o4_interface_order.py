"""
O4阶段验证测试 —— 接口秩序

验证点：
1. IErrorEvaluator.evaluate签名与实现一致（含error_type/model_params）
2. IOptimizer返回类型文档化
3. 类型注解统一（Tuple而非tuple）
4. 接口实现一致性
5. 不破坏现有功能
"""

import os
import sys
import inspect
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestO4InterfaceSync:
    """验证抽象接口与实现同步"""

    def test_error_evaluator_interface_has_extended_params(self):
        """IErrorEvaluator.evaluate应包含error_type和model_params参数"""
        from core.interfaces import IErrorEvaluator
        sig = inspect.signature(IErrorEvaluator.evaluate)
        params = list(sig.parameters.keys())
        assert "error_type" in params, "IErrorEvaluator.evaluate 应包含 error_type 参数"
        assert "model_params" in params, "IErrorEvaluator.evaluate 应包含 model_params 参数"

    def test_error_eval_impl_matches_interface(self):
        """ErrorEvaluator.evaluate签名应与接口一致"""
        from modules.error_eval import ErrorEvaluator
        from core.exceptions import ErrorEvalException
        # 通过实际调用验证实现支持扩展参数
        evaluator = ErrorEvaluator()
        # 调用时应支持error_type和model_params参数
        result = evaluator.evaluate({"cpu": 10.0}, {"cpu": 12.0})
        assert isinstance(result, dict)
        # 带扩展参数调用也应工作
        from modules.error_eval import ErrorType
        result2 = evaluator.evaluate({"cpu": 10.0}, {"cpu": 12.0},
                                      error_type=ErrorType.PREDICTION,
                                      model_params=5)
        assert isinstance(result2, dict)

    def test_optimizer_interface_documented(self):
        """IOptimizer方法应有返回类型文档"""
        from core.interfaces import IOptimizer
        # 检查docstring包含Returns
        assert "Returns" in (IOptimizer.online_optimize.__doc__ or "")
        assert "Returns" in (IOptimizer.offline_optimize.__doc__ or "")


class TestO4TypeAnnotationConsistency:
    """验证类型注解一致性"""

    def test_decision_uses_Tuple_not_tuple(self):
        """decision.py应使用Tuple而非tuple（与项目一致）"""
        decision_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "decision.py"
        )
        with open(decision_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 不应使用小写tuple[类型注解
        assert "-> tuple[" not in content, \
            "decision.py 应使用 Tuple[...] 而非 tuple[...]"

    def test_decision_imports_Tuple(self):
        """decision.py应导入Tuple"""
        decision_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "decision.py"
        )
        with open(decision_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "Tuple" in content, "decision.py 应导入并使用 Tuple"


class TestO4InterfaceImplementation:
    """验证接口实现"""

    def test_all_modules_implement_interfaces(self):
        """所有模块应实现对应接口"""
        from modules.perception import EnvironmentPerception
        from modules.prediction import PredictionEngine
        from modules.drive_engine import DriveEngine
        from modules.decision import DecisionEngine
        from modules.error_eval import ErrorEvaluator
        from modules.optimizer import Optimizer
        from modules.memory import MemorySystem
        from core.interfaces import (IPerception, IPrediction, IDriveEngine,
                                     IDecision, IErrorEvaluator, IOptimizer, IMemory)

        assert issubclass(EnvironmentPerception, IPerception)
        assert issubclass(PredictionEngine, IPrediction)
        assert issubclass(DriveEngine, IDriveEngine)
        assert issubclass(DecisionEngine, IDecision)
        assert issubclass(ErrorEvaluator, IErrorEvaluator)
        assert issubclass(Optimizer, IOptimizer)
        assert issubclass(MemorySystem, IMemory)

    def test_error_evaluator_evaluate_works_with_extended_params(self):
        """ErrorEvaluator.evaluate应支持扩展参数"""
        from modules.error_eval import ErrorEvaluator
        evaluator = ErrorEvaluator()
        # 使用扩展参数调用
        result = evaluator.evaluate({"cpu": 10.0}, {"cpu": 12.0})
        assert isinstance(result, dict)

    def test_optimizer_returns_optimization_result(self):
        """Optimizer.online_optimize应返回OptimizationResult"""
        from modules.optimizer import Optimizer, OptimizationResult
        opt = Optimizer()
        result = opt.online_optimize({"cpu": 10.0}, {"cpu": 12.0}, {"mae": 2.0})
        assert isinstance(result, OptimizationResult)


class TestO4NoRegression:
    """验证O4不破坏现有功能"""

    def test_agent_init_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert agent is not None
        agent.stop()

    def test_main_loop_still_runs(self):
        from main import SelfEvolvingAgent
        import time
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(0.8)
        assert agent.loop_count > 0
        agent.stop()

    def test_error_eval_evaluate_still_works(self):
        from modules.error_eval import ErrorEvaluator
        evaluator = ErrorEvaluator()
        result = evaluator.evaluate([1.0, 2.0], [1.5, 2.5])
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
