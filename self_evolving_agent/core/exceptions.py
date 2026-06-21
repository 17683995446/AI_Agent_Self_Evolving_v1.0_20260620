"""
异常处理模块 —— 定义系统级异常层次结构和降级策略

O3：异常秩序统一
U6：调试易用性增强 —— 增加 remediation 字段
设计原则：
1. 所有异常必须分类，便于针对性处理
2. 每个模块必须有异常降级路径，确保单点故障不会导致系统崩溃
3. 异常信息必须包含足够的上下文，便于排查
4. 构造签名统一：所有子类接受 (message, context) 或 (message, module, context)
5. U6：异常应提供 remediation（修复建议），告诉用户如何解决问题
"""

from typing import Dict, Any, Optional


class AgentException(Exception):
    """Agent系统基础异常

    U6：增加 remediation 字段 —— 告诉用户如何修复问题。
    第一性原理：异常不只是报错，更是指导用户行动的信号。
    """
    def __init__(self, message: str, module: str = "", context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message)
        self.module = module
        self.context = context or {}
        self.degradable = True  # 是否可降级处理
        # U6：修复建议 —— 告诉用户如何解决问题
        self.remediation = remediation or self._default_remediation()

    def _default_remediation(self) -> str:
        """子类可覆盖：返回该异常类型的默认修复建议"""
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "message": str(self),
            "module": self.module,
            "context": self.context,
            "degradable": self.degradable,
            "remediation": self.remediation,
        }


class PerceptionException(AgentException):
    """环境感知异常 —— 采集失败时使用默认值降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="perception", context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "检查 psutil 是否正确安装、系统权限是否足够。感知失败会自动降级为默认值。"


class PredictionException(AgentException):
    """预测异常 —— 使用滑动平均或默认值降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="prediction", context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "检查预测模型输入数据是否完整。预测失败会自动降级为滑动平均。"


class DriveEngineException(AgentException):
    """驱力引擎异常 —— 返回中性驱力状态降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="drive_engine", context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "检查驱力权重配置（safe+gain+eff=1.0）。驱力失败会自动降级为中性状态。"


class DecisionException(AgentException):
    """决策异常 —— 返回安全动作（休眠）降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="decision", context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "检查决策引擎输入状态向量。决策失败会自动降级为安全休眠动作。"


class ErrorEvalException(AgentException):
    """误差评估异常 —— 返回默认误差报告降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="error_eval", context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "检查预测值与真实值的数据类型是否匹配。误差评估失败会降级为默认报告。"


class OptimizerException(AgentException):
    """优化器异常 —— 跳过本次优化降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="optimizer", context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "检查优化器学习率与梯度。优化失败会跳过本次更新，下轮重试。"


class MemoryException(AgentException):
    """记忆系统异常 —— 跳过记忆操作降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="memory", context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "检查数据库路径权限与磁盘空间。记忆失败会跳过本次写入，不影响主循环。"


class ConfigurationException(AgentException):
    """配置异常 —— 不可降级，必须修复配置"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="config", context=context, remediation=remediation)
        self.degradable = False

    def _default_remediation(self) -> str:
        return (
            "请检查配置文件语法与取值范围。"
            "可运行 'python main.py config validate' 验证配置，"
            "或通过环境变量覆盖（如 AGENT_MAX_CPU、AGENT_LOG_LEVEL）。"
        )


class ResourceExhaustedException(AgentException):
    """资源耗尽异常 —— 触发保护性降级"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="resource", context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "降低主循环频率（AGENT_LOOP_FREQUENCY）或扩大资源约束（AGENT_MAX_CPU/AGENT_MAX_MEMORY）。"


class ModuleInitException(AgentException):
    """模块初始化异常 —— 尝试使用备用模块或默认实现"""
    def __init__(self, message: str, module: str = "", context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module=module, context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "检查模块依赖是否安装、构造参数是否合法。系统会尝试使用备用模块降级。"


# O3：扩展异常类覆盖 —— 为新增模块添加专用异常


class WorkflowException(AgentException):
    """工作流异常 —— 工作流执行失败时使用"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="workflow", context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "检查工作流步骤定义与依赖顺序。工作流失败会跳过当前步骤继续执行。"


class SkillLibraryException(AgentException):
    """技能库异常 —— 技能注册/检索/进化失败时使用"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="skill_library", context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "检查技能定义格式与依赖。技能库失败不影响主循环，下次重试。"


class PersistenceException(AgentException):
    """持久化异常 —— 状态保存/恢复失败时使用"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="persistence", context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "检查快照目录权限与磁盘空间。持久化失败不影响运行中状态。"


class ForeverRunnerException(AgentException):
    """永续运行异常 —— 健康检查/自愈失败时使用"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="forever_runner", context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "检查系统资源是否耗尽。永续运行异常会触发自愈逻辑，无需人工干预。"


class AlertingException(AgentException):
    """告警异常 —— 告警通道/规则处理失败时使用"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="alerting", context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "检查告警通道配置（webhook/email）。告警失败不影响主循环。"


class MultiAgentException(AgentException):
    """多Agent编排异常 —— Agent通信/委派失败时使用"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="multi_agent", context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "检查Agent总线连接与子Agent状态。编排失败会降级为单Agent模式。"


class LongTaskException(AgentException):
    """长任务异常 —— 长任务规划/推进失败时使用"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 remediation: str = ""):
        super().__init__(message, module="long_task", context=context, remediation=remediation)
        self.degradable = True

    def _default_remediation(self) -> str:
        return "检查长任务分解策略与检查点。长任务失败会从最近检查点恢复。"
