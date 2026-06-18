"""
链路追踪模块 —— 分布式追踪ID、结构化日志、耗时分析

设计原则（第一性原理）：
1. 每个请求/动作有唯一追踪ID，贯穿全链路
2. 日志是数据：结构化、可查询、可聚合
3. 性能可观测：关键路径自动计时
"""

import os
import sys
import time
import json
import uuid
import threading
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


class TraceContext:
    """追踪上下文 —— 保存当前链路的所有追踪信息"""

    _local = threading.local()

    @classmethod
    def get_trace_id(cls) -> str:
        """获取当前线程的追踪ID"""
        if not hasattr(cls._local, 'trace_id'):
            cls._local.trace_id = str(uuid.uuid4())[:12]
        return cls._local.trace_id

    @classmethod
    def set_trace_id(cls, trace_id: str) -> None:
        """设置追踪ID"""
        cls._local.trace_id = trace_id

    @classmethod
    def clear(cls) -> None:
        """清除当前线程的追踪信息"""
        if hasattr(cls._local, 'trace_id'):
            delattr(cls._local, 'trace_id')
        if hasattr(cls._local, 'span_stack'):
            delattr(cls._local, 'span_stack')

    @classmethod
    def get_span_stack(cls) -> List[str]:
        """获取当前跨度栈"""
        if not hasattr(cls._local, 'span_stack'):
            cls._local.span_stack = []
        return cls._local.span_stack

    @classmethod
    def push_span(cls, span_name: str) -> None:
        """压入跨度"""
        cls.get_span_stack().append(span_name)

    @classmethod
    def pop_span(cls) -> Optional[str]:
        """弹出跨度"""
        stack = cls.get_span_stack()
        if stack:
            return stack.pop()
        return None


class StructuredLogger:
    """结构化日志 —— JSON格式输出，支持动态级别调整"""

    LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

    def __init__(self, min_level: str = "INFO"):
        self.min_level = self.LEVELS.get(min_level, 20)
        self._dynamic_levels: Dict[str, int] = {}  # 模块 -> 级别
        self._json_mode = os.environ.get("AGENT_LOG_JSON", "0") == "1"

    def set_level(self, level: str, module: Optional[str] = None) -> None:
        """设置日志级别（全局或模块级）"""
        level_value = self.LEVELS.get(level, 20)
        if module:
            self._dynamic_levels[module] = level_value
            logger.info("LOG_LEVEL", f"模块 {module} 日志级别设为 {level}", module="tracing")
        else:
            self.min_level = level_value
            logger.info("LOG_LEVEL", f"全局日志级别设为 {level}", module="tracing")

    def _should_log(self, level: str, module: str = "") -> bool:
        """判断是否应记录该级别日志"""
        level_value = self.LEVELS.get(level, 20)
        # 模块级覆盖
        if module in self._dynamic_levels:
            return level_value >= self._dynamic_levels[module]
        return level_value >= self.min_level

    def log(self, level: str, event: str, message: str, module: str = "",
            extra: Optional[Dict[str, Any]] = None, duration_ms: Optional[float] = None) -> None:
        """输出结构化日志"""
        if not self._should_log(level, module):
            return

        trace_id = TraceContext.get_trace_id()
        span_stack = TraceContext.get_span_stack()

        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "event": event,
            "message": message,
            "module": module,
            "trace_id": trace_id,
            "span": ">".join(span_stack) if span_stack else "",
            "extra": extra or {},
        }
        if duration_ms is not None:
            log_entry["duration_ms"] = round(duration_ms, 3)

        if self._json_mode:
            print(json.dumps(log_entry, default=str))
        else:
            # 兼容原有格式
            extra_str = f" | extra={json.dumps(extra, default=str)}" if extra else ""
            dur_str = f" | dur={duration_ms:.2f}ms" if duration_ms is not None else ""
            span_str = f" | span={log_entry['span']}" if span_stack else ""
            print(f"[{level}] [{module}] [{trace_id}] {event}: {message}{span_str}{dur_str}{extra_str}")

    def debug(self, event: str, message: str, module: str = "",
              extra: Optional[Dict[str, Any]] = None, duration_ms: Optional[float] = None) -> None:
        self.log("DEBUG", event, message, module, extra, duration_ms)

    def info(self, event: str, message: str, module: str = "",
             extra: Optional[Dict[str, Any]] = None, duration_ms: Optional[float] = None) -> None:
        self.log("INFO", event, message, module, extra, duration_ms)

    def warning(self, event: str, message: str, module: str = "",
                extra: Optional[Dict[str, Any]] = None, duration_ms: Optional[float] = None) -> None:
        self.log("WARNING", event, message, module, extra, duration_ms)

    def error(self, event: str, message: str, module: str = "",
              extra: Optional[Dict[str, Any]] = None, duration_ms: Optional[float] = None) -> None:
        self.log("ERROR", event, message, module, extra, duration_ms)


class PerformanceTracer:
    """性能追踪器 —— 关键路径耗时分析"""

    def __init__(self):
        self._spans: Dict[str, List[float]] = defaultdict(list)
        self._active_spans: Dict[str, float] = {}

    @contextmanager
    def trace(self, span_name: str):
        """上下文管理器 —— 自动计时跨度"""
        TraceContext.push_span(span_name)
        start = time.perf_counter()
        self._active_spans[span_name] = start
        try:
            yield self
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self._spans[span_name].append(elapsed)
            TraceContext.pop_span()
            # 保留最近100个样本
            if len(self._spans[span_name]) > 100:
                self._spans[span_name] = self._spans[span_name][-100:]

    def get_stats(self, span_name: Optional[str] = None) -> Dict[str, Any]:
        """获取追踪统计"""
        if span_name:
            values = self._spans.get(span_name, [])
            if not values:
                return {"span": span_name, "count": 0}
            return {
                "span": span_name,
                "count": len(values),
                "avg_ms": round(sum(values) / len(values), 3),
                "min_ms": round(min(values), 3),
                "max_ms": round(max(values), 3),
                "p95_ms": round(sorted(values)[int(len(values) * 0.95)] if values else 0, 3),
            }

        return {name: self.get_stats(name) for name in self._spans.keys()}

    def get_slow_spans(self, threshold_ms: float = 100.0) -> List[Dict[str, Any]]:
        """获取慢跨度"""
        slow = []
        for name, values in self._spans.items():
            if values:
                avg = sum(values) / len(values)
                if avg > threshold_ms:
                    slow.append({
                        "span": name,
                        "avg_ms": round(avg, 3),
                        "max_ms": round(max(values), 3),
                    })
        return sorted(slow, key=lambda x: x["avg_ms"], reverse=True)

    def reset(self) -> None:
        """重置统计"""
        self._spans.clear()
        self._active_spans.clear()


class TracePropagator:
    """追踪传播器 —— 跨模块/跨线程传递追踪上下文"""

    @staticmethod
    def inject() -> Dict[str, str]:
        """提取当前上下文为可传播的字典"""
        return {
            "trace_id": TraceContext.get_trace_id(),
            "span_stack": ">".join(TraceContext.get_span_stack()),
        }

    @staticmethod
    def extract(headers: Dict[str, str]) -> None:
        """从字典恢复追踪上下文"""
        if "trace_id" in headers:
            TraceContext.set_trace_id(headers["trace_id"])
        if "span_stack" in headers and headers["span_stack"]:
            TraceContext._local.span_stack = headers["span_stack"].split(">")


# 全局实例
structured_logger = StructuredLogger()
performance_tracer = PerformanceTracer()
