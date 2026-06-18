"""
依赖注入容器 —— 实现模块解耦和显式依赖管理

设计原则：
1. 每个模块的依赖必须通过构造函数显式声明
2. 容器负责解析和组装依赖关系
3. 支持模块替换而无需修改其他代码
4. 生命周期由容器统一管理
"""

from typing import Dict, Any, Optional, Type, Callable
from core.interfaces import (
    IPerception, IPrediction, IDriveEngine, IDecision,
    IErrorEvaluator, IOptimizer, IMemory,
)
from core.registry import ModuleRegistry, get_default_registry
from core.config import config_manager
from core.logger import logger


class ModuleLifecycle:
    """模块生命周期管理 —— 统一控制模块的初始化、启动、停止、清理"""

    def __init__(self, name: str, instance: Any):
        self.name = name
        self.instance = instance
        self._initialized = False
        self._running = False

    def initialize(self) -> bool:
        """初始化模块"""
        if self._initialized:
            return True
        try:
            if hasattr(self.instance, 'initialize'):
                self.instance.initialize()
            self._initialized = True
            logger.debug("LIFECYCLE", f"模块 {self.name} 初始化完成", module="container")
            return True
        except Exception as e:
            logger.error("LIFECYCLE", f"模块 {self.name} 初始化失败: {e}", module="container")
            return False

    def start(self) -> bool:
        """启动模块"""
        if not self._initialized:
            if not self.initialize():
                return False
        if self._running:
            return True
        try:
            if hasattr(self.instance, 'start'):
                self.instance.start()
            self._running = True
            logger.debug("LIFECYCLE", f"模块 {self.name} 启动完成", module="container")
            return True
        except Exception as e:
            logger.error("LIFECYCLE", f"模块 {self.name} 启动失败: {e}", module="container")
            return False

    def stop(self) -> bool:
        """停止模块"""
        if not self._running:
            return True
        try:
            if hasattr(self.instance, 'stop'):
                self.instance.stop()
            self._running = False
            logger.debug("LIFECYCLE", f"模块 {self.name} 停止完成", module="container")
            return True
        except Exception as e:
            logger.error("LIFECYCLE", f"模块 {self.name} 停止失败: {e}", module="container")
            return False

    def cleanup(self) -> bool:
        """清理模块资源"""
        try:
            if hasattr(self.instance, 'cleanup'):
                self.instance.cleanup()
            elif hasattr(self.instance, 'close'):
                self.instance.close()
            logger.debug("LIFECYCLE", f"模块 {self.name} 清理完成", module="container")
            return True
        except Exception as e:
            logger.error("LIFECYCLE", f"模块 {self.name} 清理失败: {e}", module="container")
            return False


class DIContainer:
    """依赖注入容器 —— 管理模块的创建、依赖解析和生命周期"""

    def __init__(self, registry: Optional[ModuleRegistry] = None):
        self._registry = registry or get_default_registry()
        self._modules: Dict[str, ModuleLifecycle] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._config = config_manager.config
        logger.info("DI_CONTAINER", "依赖注入容器初始化", module="container")

    def register_factory(self, name: str, factory: Callable[[], Any]) -> None:
        """注册模块工厂函数"""
        self._factories[name] = factory
        logger.debug("DI_CONTAINER", f"注册工厂: {name}", module="container")

    def resolve(self, name: str, module_type: str = "") -> Any:
        """解析模块实例 —— 优先使用工厂，其次使用注册表"""
        cache_key = f"{module_type}:{name}" if module_type else name
        if cache_key in self._modules:
            return self._modules[cache_key].instance

        # 尝试工厂
        if name in self._factories:
            instance = self._factories[name]()
            self._modules[cache_key] = ModuleLifecycle(cache_key, instance)
            logger.debug("DI_CONTAINER", f"通过工厂解析模块: {name}", module="container")
            return instance

        # 尝试注册表
        if module_type and self._registry.is_registered(module_type, name):
            creator = getattr(self._registry, f"create_{module_type}", None)
            if creator:
                instance = creator(name)
                self._modules[cache_key] = ModuleLifecycle(cache_key, instance)
                logger.debug("DI_CONTAINER", f"通过注册表解析模块: {name}", module="container")
                return instance

        raise KeyError(f"无法解析模块: {name} (类型: {module_type})")

    def build_agent(self, module_config: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """构建Agent所需的所有模块，返回模块字典"""
        config = module_config or {}
        logger.info("DI_CONTAINER", "开始构建Agent模块", module="container",
                    extra={"config": config})

        modules = {}

        # 1. 环境感知 —— 无依赖
        modules["perception"] = self._resolve_for_build(
            "perception", config.get("perception", "default"))

        # 2. 预测引擎 —— 无依赖
        modules["prediction"] = self._resolve_for_build(
            "prediction", config.get("prediction", "default"))

        # 3. 误差评估 —— 无依赖
        modules["error_eval"] = self._resolve_for_build(
            "error_eval", config.get("error_eval", "default"))

        # 4. 优化器 —— 无依赖
        modules["optimizer"] = self._resolve_for_build(
            "optimizer", config.get("optimizer", "default"))

        # 5. 驱力引擎 —— 无依赖
        modules["drive_engine"] = self._resolve_for_build(
            "drive_engine", config.get("drive_engine", "default"))

        # 6. 决策引擎 —— 无依赖（内部已解耦）
        modules["decision"] = self._resolve_for_build(
            "decision", config.get("decision", "default"))

        # 7. 记忆系统 —— 无依赖
        modules["memory"] = self._resolve_for_build(
            "memory", config.get("memory", "default"))

        logger.info("DI_CONTAINER", "Agent模块构建完成", module="container",
                    extra={"modules": list(modules.keys())})
        return modules

    def _resolve_for_build(self, module_key: str, name: str) -> Any:
        """为build_agent解析模块，内部使用组合缓存键但返回实例"""
        cache_key = f"{module_key}:{name}"
        if cache_key in self._modules:
            return self._modules[cache_key].instance

        if self._registry.is_registered(module_key, name):
            creator = getattr(self._registry, f"create_{module_key}", None)
            if creator:
                instance = creator(name)
                self._modules[cache_key] = ModuleLifecycle(cache_key, instance)
                logger.debug("DI_CONTAINER", f"构建模块: {module_key}={name}", module="container")
                return instance

        raise KeyError(f"无法构建模块: {module_key}={name}")

    def initialize_all(self) -> Dict[str, bool]:
        """初始化所有已解析的模块"""
        results = {}
        for name, lifecycle in self._modules.items():
            results[name] = lifecycle.initialize()
        return results

    def start_all(self) -> Dict[str, bool]:
        """启动所有已解析的模块"""
        results = {}
        for name, lifecycle in self._modules.items():
            results[name] = lifecycle.start()
        return results

    def stop_all(self) -> Dict[str, bool]:
        """停止所有已解析的模块"""
        results = {}
        for name, lifecycle in self._modules.items():
            results[name] = lifecycle.stop()
        return results

    def cleanup_all(self) -> Dict[str, bool]:
        """清理所有已解析的模块"""
        results = {}
        for name, lifecycle in self._modules.items():
            results[name] = lifecycle.cleanup()
        return results

    def get_module_names(self) -> list:
        """获取所有已管理的模块名称"""
        return list(self._modules.keys())

    def get_module_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模块的状态 —— 键名为短名称（如 perception）"""
        status = {}
        for cache_key, lifecycle in self._modules.items():
            # 从缓存键提取短名称，如 "perception:default" -> "perception"
            short_name = cache_key.split(":")[0] if ":" in cache_key else cache_key
            status[short_name] = {
                "initialized": lifecycle._initialized,
                "running": lifecycle._running,
                "has_start": hasattr(lifecycle.instance, 'start'),
                "has_stop": hasattr(lifecycle.instance, 'stop'),
                "has_cleanup": hasattr(lifecycle.instance, 'cleanup') or hasattr(lifecycle.instance, 'close'),
            }
        return status
