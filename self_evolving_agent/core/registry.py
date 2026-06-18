"""
模块注册中心 —— 实现乐高式插件化机制

用法：
    from core.registry import ModuleRegistry
    registry = ModuleRegistry()
    
    # 注册自定义预测器
    registry.register_prediction("my_plugin", MyPredictionPlugin)
    
    # 通过配置实例化
    agent = SelfEvolvingAgent(registry=registry, config={"prediction": "my_plugin"})
"""
from typing import Dict, Type, Any, Optional
from core.interfaces import (
    IPerception, IPrediction, IDriveEngine, IDecision,
    IErrorEvaluator, IOptimizer, IMemory,
)


class ModuleRegistry:
    """模块注册中心 —— 管理所有模块的插件注册和实例化"""

    def __init__(self):
        # 默认模块注册表
        self._perception: Dict[str, Type[IPerception]] = {}
        self._prediction: Dict[str, Type[IPrediction]] = {}
        self._drive_engine: Dict[str, Type[IDriveEngine]] = {}
        self._decision: Dict[str, Type[IDecision]] = {}
        self._error_eval: Dict[str, Type[IErrorEvaluator]] = {}
        self._optimizer: Dict[str, Type[IOptimizer]] = {}
        self._memory: Dict[str, Type[IMemory]] = {}

    # ------------------------------------------------------------------
    # 注册方法
    # ------------------------------------------------------------------
    def register_perception(self, name: str, cls: Type[IPerception]) -> None:
        self._perception[name] = cls

    def register_prediction(self, name: str, cls: Type[IPrediction]) -> None:
        self._prediction[name] = cls

    def register_drive_engine(self, name: str, cls: Type[IDriveEngine]) -> None:
        self._drive_engine[name] = cls

    def register_decision(self, name: str, cls: Type[IDecision]) -> None:
        self._decision[name] = cls

    def register_error_eval(self, name: str, cls: Type[IErrorEvaluator]) -> None:
        self._error_eval[name] = cls

    def register_optimizer(self, name: str, cls: Type[IOptimizer]) -> None:
        self._optimizer[name] = cls

    def register_memory(self, name: str, cls: Type[IMemory]) -> None:
        self._memory[name] = cls

    # ------------------------------------------------------------------
    # 实例化方法
    # ------------------------------------------------------------------
    def create_perception(self, name: str, **kwargs) -> IPerception:
        return self._perception[name](**kwargs)

    def create_prediction(self, name: str, **kwargs) -> IPrediction:
        return self._prediction[name](**kwargs)

    def create_drive_engine(self, name: str, **kwargs) -> IDriveEngine:
        return self._drive_engine[name](**kwargs)

    def create_decision(self, name: str, **kwargs) -> IDecision:
        return self._decision[name](**kwargs)

    def create_error_eval(self, name: str, **kwargs) -> IErrorEvaluator:
        return self._error_eval[name](**kwargs)

    def create_optimizer(self, name: str, **kwargs) -> IOptimizer:
        return self._optimizer[name](**kwargs)

    def create_memory(self, name: str, **kwargs) -> IMemory:
        return self._memory[name](**kwargs)

    # ------------------------------------------------------------------
    # 查询方法
    # ------------------------------------------------------------------
    def list_predictions(self) -> list:
        return list(self._prediction.keys())

    def list_drive_engines(self) -> list:
        return list(self._drive_engine.keys())

    def is_registered(self, module_type: str, name: str) -> bool:
        registry_map = {
            "perception": self._perception,
            "prediction": self._prediction,
            "drive_engine": self._drive_engine,
            "decision": self._decision,
            "error_eval": self._error_eval,
            "optimizer": self._optimizer,
            "memory": self._memory,
        }
        return name in registry_map.get(module_type, {})


# 全局默认注册表
_default_registry: Optional[ModuleRegistry] = None


def get_default_registry() -> ModuleRegistry:
    """获取全局默认注册表（懒加载）"""
    global _default_registry
    if _default_registry is None:
        _default_registry = ModuleRegistry()
        # 注册所有内置默认模块
        from modules.perception import EnvironmentPerception
        from modules.drive_engine import DriveEngine
        from modules.prediction import PredictionEngine
        from modules.decision import DecisionEngine
        from modules.error_eval import ErrorEvaluator
        from modules.optimizer import Optimizer
        from modules.memory import MemorySystem

        _default_registry.register_perception("default", EnvironmentPerception)
        _default_registry.register_drive_engine("default", DriveEngine)
        _default_registry.register_prediction("default", PredictionEngine)
        _default_registry.register_decision("default", DecisionEngine)
        _default_registry.register_error_eval("default", ErrorEvaluator)
        _default_registry.register_optimizer("default", Optimizer)
        _default_registry.register_memory("default", MemorySystem)
    return _default_registry
