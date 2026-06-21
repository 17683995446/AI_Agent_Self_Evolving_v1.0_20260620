"""
核心日志模块 —— 为系统所有操作提供结构化日志记录
每一条代码执行都伴随日志输出，用于测试验证和问题排查

H6增强：
1. 运行时可调整控制台/文件日志级别（set_console_level/set_file_level）
2. 高频事件采样（避免FUNC_ENTER等噪声淹没关键信息）
3. 函数追踪开关（set_func_trace_enabled）支持配置驱动
4. 默认控制台级别提升至INFO，减少生产环境噪声
"""
import logging
import sys
import os
import json
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
import threading

# 全局日志锁，确保多线程安全
_log_lock = threading.Lock()

# 级别字符串到logging常量的映射
_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class StructuredLogger:
    """结构化日志记录器 —— 每条日志包含时间、模块、级别、消息和上下文

    H6新增能力：
    - set_console_level / set_file_level：运行时调整输出级别
    - set_func_trace_enabled：动态开关函数追踪装饰器
    - set_sample_rate：对指定event_type采样（如FUNC_ENTER每N次记录1次）
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name: str = "SelfEvolvingAgent", log_dir: str = "./logs"):
        if StructuredLogger._initialized:
            return
        StructuredLogger._initialized = True

        self.name = name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 创建 logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # 清除已有 handler

        # 控制台输出 —— H6: 默认INFO，减少生产噪声
        self._console_handler = logging.StreamHandler(sys.stdout)
        self._console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S.%f'[:-3]
        )
        self._console_handler.setFormatter(console_format)
        self.logger.addHandler(self._console_handler)

        # 文件输出 —— 结构化 JSON，保留DEBUG用于排查
        log_file = os.path.join(log_dir, f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self._file_handler = logging.FileHandler(log_file, encoding='utf-8')
        self._file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self._file_handler)

        # 审计日志 —— 单独文件记录关键操作
        audit_file = os.path.join(log_dir, f"audit_{datetime.now().strftime('%Y%m%d')}.log")
        self.audit_handler = logging.FileHandler(audit_file, encoding='utf-8')
        self.audit_handler.setLevel(logging.INFO)
        audit_format = logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.audit_handler.setFormatter(audit_format)

        # H6: 采样控制 —— event_type -> (已跳过次数, 每N次记录1次)
        self._sample_rates: Dict[str, int] = {}
        self._sample_counters: Dict[str, int] = {}
        self._sample_lock = threading.Lock()

        # H6: 函数追踪开关（默认根据环境变量）
        self._func_trace_enabled = os.environ.get("LOG_DISABLE_FUNC_TRACE", "0") != "1"

        self._log("LOGGER_INIT", "日志系统初始化完成", level="INFO",
                  extra={"log_file": log_file, "audit_file": audit_file,
                         "console_level": "INFO", "file_level": "DEBUG"})

    # ---------- H6: 运行时级别控制 ----------
    def set_console_level(self, level: str) -> None:
        """设置控制台日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）"""
        py_level = _LEVEL_MAP.get(level.upper())
        if py_level is None:
            logger.warning("LOGGER", f"未知日志级别: {level}", module="logger")
            return
        self._console_handler.setLevel(py_level)
        self._log("LOGGER", f"控制台日志级别设置为 {level}", level="INFO", module="logger")

    def set_file_level(self, level: str) -> None:
        """设置文件日志级别"""
        py_level = _LEVEL_MAP.get(level.upper())
        if py_level is None:
            logger.warning("LOGGER", f"未知日志级别: {level}", module="logger")
            return
        self._file_handler.setLevel(py_level)
        self._log("LOGGER", f"文件日志级别设置为 {level}", level="INFO", module="logger")

    def set_func_trace_enabled(self, enabled: bool) -> None:
        """动态开关函数追踪装饰器"""
        self._func_trace_enabled = enabled
        self._log("LOGGER", f"函数追踪 {'启用' if enabled else '禁用'}",
                  level="DEBUG", module="logger")

    @property
    def func_trace_enabled(self) -> bool:
        return self._func_trace_enabled

    # ---------- H6: 采样控制 ----------
    def set_sample_rate(self, event_type: str, every_n: int) -> None:
        """对指定event_type采样：每N次仅记录1次

        用于高频低价值日志（如FUNC_ENTER/FUNC_EXIT/CONFIG_UPDATE）
        Args:
            event_type: 事件类型
            every_n: 每N次记录1次（1=全记录，0或负=禁用该类型）
        """
        with self._sample_lock:
            self._sample_rates[event_type] = max(0, every_n)
            self._sample_counters.setdefault(event_type, 0)

    def clear_sample_rate(self, event_type: str) -> None:
        """清除某event_type的采样设置（恢复全记录）"""
        with self._sample_lock:
            self._sample_rates.pop(event_type, None)
            self._sample_counters.pop(event_type, None)

    def _should_sample(self, event_type: str) -> bool:
        """判断本次是否应该记录（True=记录，False=跳过）"""
        with self._sample_lock:
            n = self._sample_rates.get(event_type)
            if n is None:
                return True  # 未配置采样，全记录
            if n <= 0:
                return False  # 禁用
            counter = self._sample_counters.get(event_type, 0) + 1
            self._sample_counters[event_type] = counter
            if counter >= n:
                self._sample_counters[event_type] = 0
                return True
            return False
    
    def _log(self, event_type: str, message: str, level: str = "INFO", 
             module: str = "", extra: Optional[Dict[str, Any]] = None):
        """内部日志记录方法 —— 所有日志通过这里输出

        H6: 在输出前检查采样规则，被采样的日志直接跳过（不进锁、不构造entry）
        """
        # H6: 采样检查（在锁外快速判断，减少锁竞争）
        if not self._should_sample(event_type):
            return None

        with _log_lock:
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "event_type": event_type,
                "module": module or self.name,
                "level": level,
                "message": message,
                "thread_id": threading.current_thread().ident,
                "extra": extra or {}
            }
            
            # 控制台输出简化格式
            extra_str = json.dumps(extra, ensure_ascii=False) if extra else ""
            full_msg = f"[{event_type}] {message} {extra_str}"
            
            if level == "DEBUG":
                self.logger.debug(full_msg)
            elif level == "INFO":
                self.logger.info(full_msg)
            elif level == "WARNING":
                self.logger.warning(full_msg)
            elif level == "ERROR":
                self.logger.error(full_msg)
            elif level == "CRITICAL":
                self.logger.critical(full_msg)
            
            # 审计日志记录关键事件
            if event_type in ["CONSTRAINT_VIOLATION", "MODEL_ROLLBACK", 
                              "EMERGENCY_TRIGGERED", "SNAPSHOT_SAVED", "TASK_EXECUTED"]:
                self.audit_handler.emit(logging.LogRecord(
                    name="audit", level=logging.INFO, pathname="", lineno=0,
                    msg=f"{event_type} | {message} | {json.dumps(extra, ensure_ascii=False)}",
                    args=(), exc_info=None
                ))
            
            return log_entry
    
    def debug(self, event_type: str, message: str, module: str = "", extra: Optional[Dict] = None):
        return self._log(event_type, message, "DEBUG", module, extra)
    
    def info(self, event_type: str, message: str, module: str = "", extra: Optional[Dict] = None):
        return self._log(event_type, message, "INFO", module, extra)
    
    def warning(self, event_type: str, message: str, module: str = "", extra: Optional[Dict] = None):
        return self._log(event_type, message, "WARNING", module, extra)
    
    def error(self, event_type: str, message: str, module: str = "", extra: Optional[Dict] = None):
        return self._log(event_type, message, "ERROR", module, extra)
    
    def critical(self, event_type: str, message: str, module: str = "", extra: Optional[Dict] = None):
        return self._log(event_type, message, "CRITICAL", module, extra)

    # ---------- U6：调试易用性增强 ----------

    def exception(self, event_type: str, message: str, module: str = "",
                  extra: Optional[Dict] = None, exc_info: Optional[bool] = True) -> Any:
        """U6：记录异常日志，保留完整堆栈

        第一性原理：未知异常的堆栈是排查问题的关键线索，
        不能只记录 str(e) 而丢失调用链。

        Args:
            event_type: 事件类型
            message: 错误消息
            module: 模块名
            extra: 额外上下文
            exc_info: 是否附加完整堆栈（默认True，从当前异常上下文获取）
        """
        tb_str = ""
        if exc_info:
            tb_str = traceback.format_exc()
            if tb_str == "NoneType: None\n":
                tb_str = ""
        merged_extra = dict(extra or {})
        if tb_str:
            merged_extra["traceback"] = tb_str
        return self._log(event_type, message, "ERROR", module, merged_extra)

    @property
    def console_level(self) -> str:
        """U6：获取当前控制台日志级别（字符串形式）"""
        for name, level in _LEVEL_MAP.items():
            if level == self._console_handler.level:
                return name
        return "UNKNOWN"

    @console_level.setter
    def console_level(self, level: str) -> None:
        """U6：设置控制台日志级别（属性方式，委托给 set_console_level）"""
        self.set_console_level(level)


# 全局日志实例
logger = StructuredLogger()


def log_execution(func):
    """装饰器 —— 自动记录函数进入、退出和异常
    
    H6性能优化：
    1. 运行时开关：logger.set_func_trace_enabled(False) 可动态禁用
    2. 环境变量 LOG_DISABLE_FUNC_TRACE=1 仍兼容（初始化时读取）
    3. FUNC_ENTER/FUNC_EXIT 受采样规则控制（set_sample_rate）
    4. 异常日志始终记录（FUNC_EXCEPTION不采样）
    """
    func_name = func.__name__
    module_name = func.__module__ if hasattr(func, '__module__') else "unknown"

    def wrapper(*args, **kwargs):
        if not logger.func_trace_enabled:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error("FUNC_EXCEPTION", f"函数 {func_name} 异常: {str(e)}", module=module_name,
                            extra={"func": func_name, "exception": str(e), "exception_type": type(e).__name__})
                raise
        
        logger.debug("FUNC_ENTER", f"进入函数 {func_name}", module=module_name,
                    extra={"func": func_name, "args_count": len(args), "kwargs_keys": list(kwargs.keys())})
        
        try:
            result = func(*args, **kwargs)
            logger.debug("FUNC_EXIT", f"函数 {func_name} 正常退出", module=module_name,
                        extra={"func": func_name, "result_type": type(result).__name__})
            return result
        except Exception as e:
            logger.error("FUNC_EXCEPTION", f"函数 {func_name} 异常: {str(e)}", module=module_name,
                        extra={"func": func_name, "exception": str(e), "exception_type": type(e).__name__})
            raise
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
