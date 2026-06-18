"""
异常处理模块 —— 定义系统级异常层次结构和降级策略

设计原则：
1. 所有异常必须分类，便于针对性处理
2. 每个模块必须有异常降级路径，确保单点故障不会导致系统崩溃
3. 异常信息必须包含足够的上下文，便于排查
"""

from typing import Dict, Any, Optional


class AgentException(Exception):
    """Agent系统基础异常"""
    def __init__(self, message: str, module: str = "", context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.module = module
        self.context = context or {}
        self.degradable = True  # 是否可降级处理

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "message": str(self),
            "module": self.module,
            "context": self.context,
            "degradable": self.degradable,
        }


class PerceptionException(AgentException):
    """环境感知异常 —— 采集失败时使用默认值降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, module="perception", context=context)
        self.degradable = True


class PredictionException(AgentException):
    """预测异常 —— 使用滑动平均或默认值降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, module="prediction", context=context)
        self.degradable = True


class DriveEngineException(AgentException):
    """驱力引擎异常 —— 返回中性驱力状态降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, module="drive_engine", context=context)
        self.degradable = True


class DecisionException(AgentException):
    """决策异常 —— 返回安全动作（休眠）降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, module="decision", context=context)
        self.degradable = True


class ErrorEvalException(AgentException):
    """误差评估异常 —— 返回默认误差报告降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, module="error_eval", context=context)
        self.degradable = True


class OptimizerException(AgentException):
    """优化器异常 —— 跳过本次优化降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, module="optimizer", context=context)
        self.degradable = True


class MemoryException(AgentException):
    """记忆系统异常 —— 跳过记忆操作降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, module="memory", context=context)
        self.degradable = True


class ConfigurationException(AgentException):
    """配置异常 —— 不可降级，必须修复配置"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, module="config", context=context)
        self.degradable = False


class ResourceExhaustedException(AgentException):
    """资源耗尽异常 —— 触发保护性降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, module="resource", context=context)
        self.degradable = True


class ModuleInitException(AgentException):
    """模块初始化异常 —— 尝试使用备用模块或默认实现"""
    def __init__(self, message: str, module: str = "", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, module=module, context=context)
        self.degradable = True
