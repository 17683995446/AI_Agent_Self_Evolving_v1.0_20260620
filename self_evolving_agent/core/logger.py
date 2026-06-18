"""
核心日志模块 —— 为系统所有操作提供结构化日志记录
每一条代码执行都伴随日志输出，用于测试验证和问题排查
"""
import logging
import sys
import os
import json
import time
from datetime import datetime
from typing import Any, Dict, Optional
import threading

# 全局日志锁，确保多线程安全
_log_lock = threading.Lock()

class StructuredLogger:
    """结构化日志记录器 —— 每条日志包含时间、模块、级别、消息和上下文"""
    
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
        
        # 控制台输出 —— 带颜色
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter(
            '[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S.%f'[:-3]
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # 文件输出 —— 结构化 JSON
        log_file = os.path.join(log_dir, f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        
        # 审计日志 —— 单独文件记录关键操作
        audit_file = os.path.join(log_dir, f"audit_{datetime.now().strftime('%Y%m%d')}.log")
        self.audit_handler = logging.FileHandler(audit_file, encoding='utf-8')
        self.audit_handler.setLevel(logging.INFO)
        audit_format = logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.audit_handler.setFormatter(audit_format)
        
        self._log("LOGGER_INIT", "日志系统初始化完成", level="INFO", 
                  extra={"log_file": log_file, "audit_file": audit_file})
    
    def _log(self, event_type: str, message: str, level: str = "INFO", 
             module: str = "", extra: Optional[Dict[str, Any]] = None):
        """内部日志记录方法 —— 所有日志通过这里输出"""
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


# 全局日志实例
logger = StructuredLogger()


def log_execution(func):
    """装饰器 —— 自动记录函数进入、退出和异常
    
    性能优化：
    1. 通过环境变量 LOG_DISABLE_FUNC_TRACE=1 可完全禁用函数级追踪（生产环境建议开启）
    2. 高频短函数建议手动控制日志，避免装饰器开销
    """
    _disable_trace = os.environ.get("LOG_DISABLE_FUNC_TRACE", "0") == "1"
    
    def wrapper(*args, **kwargs):
        if _disable_trace:
            return func(*args, **kwargs)
        
        func_name = func.__name__
        module_name = func.__module__ if hasattr(func, '__module__') else "unknown"
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
    
    return wrapper
