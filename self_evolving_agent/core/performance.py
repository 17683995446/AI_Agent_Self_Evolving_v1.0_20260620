"""
H6：性能监控模块 —— 测量关键操作的执行时间，识别性能瓶颈

第一性原理：
1. 不可测量即不可优化 —— 必须先量化再优化
2. 性能数据应按操作维度聚合（次数/总时/均时/最大/P99）
3. 慢操作应主动暴露，而非埋没在日志中
4. 监控本身必须轻量，不能成为性能负担

用法：
    from core.performance import perf_monitor

    # 方式1：上下文管理器
    with perf_monitor.track("perceive"):
        state = agent.perception.perceive()

    # 方式2：直接记录
    perf_monitor.record("predict", 0.012)

    # 获取统计
    stats = perf_monitor.get_stats()
    slow = perf_monitor.get_slow_operations(threshold_sec=0.1)
"""

import os
import sys
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


@dataclass
class OperationStats:
    """单个操作的统计信息"""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    sum_sq: float = 0.0  # 用于计算标准差
    _samples: List[float] = field(default_factory=list)  # 保留最近样本用于分位数

    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.avg_time
        return max(0.0, self.sum_sq / self.count - mean * mean)

    @property
    def std_dev(self) -> float:
        return self.variance ** 0.5

    def percentile(self, p: float) -> float:
        """计算分位数（p∈[0,100]）"""
        if not self._samples:
            return 0.0
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * p / 100.0)
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "total_time": round(self.total_time, 6),
            "avg_time": round(self.avg_time, 6),
            "min_time": round(self.min_time, 6) if self.count > 0 else 0.0,
            "max_time": round(self.max_time, 6),
            "std_dev": round(self.std_dev, 6),
            "p50": round(self.percentile(50), 6),
            "p99": round(self.percentile(99), 6),
        }


class PerformanceMonitor:
    """性能监控器 —— 线程安全的操作耗时聚合

    设计要点：
    - 单例模式，全局共享一份统计
    - 细粒度锁，避免长时间持锁
    - 样本上限防止内存膨胀
    - track()上下文管理器自动计时
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_samples_per_op: int = 1000):
        if PerformanceMonitor._initialized:
            return
        PerformanceMonitor._initialized = True

        self._max_samples = max_samples_per_op
        self._stats: Dict[str, OperationStats] = {}
        self._lock = threading.RLock()
        self._enabled = True
        self._slow_threshold = 0.1  # 默认慢操作阈值100ms

        logger.debug("PERF_INIT", "性能监控器初始化",
                     module="performance",
                     extra={"max_samples": max_samples_per_op})

    def set_enabled(self, enabled: bool) -> None:
        """启用/禁用监控（禁用后track/record均为no-op）"""
        self._enabled = enabled
        logger.debug("PERF_CTRL", f"性能监控 {'启用' if enabled else '禁用'}",
                     module="performance")

    def set_slow_threshold(self, threshold_sec: float) -> None:
        """设置慢操作阈值（秒）"""
        self._slow_threshold = threshold_sec

    def record(self, name: str, duration: float) -> None:
        """记录一次操作耗时

        Args:
            name: 操作名（如 "perceive", "predict", "decide"）
            duration: 耗时（秒）
        """
        if not self._enabled or duration < 0:
            return

        with self._lock:
            if name not in self._stats:
                self._stats[name] = OperationStats(name=name)
            stats = self._stats[name]
            stats.count += 1
            stats.total_time += duration
            stats.min_time = min(stats.min_time, duration)
            stats.max_time = max(stats.max_time, duration)
            stats.sum_sq += duration * duration

            # 保留最近样本（环形截断）
            stats._samples.append(duration)
            if len(stats._samples) > self._max_samples:
                # 丢弃一半旧样本，保留趋势
                del stats._samples[:len(stats._samples) - self._max_samples]

        # 慢操作即时告警（不持锁，避免影响性能）
        if duration > self._slow_threshold:
            logger.warning(
                "PERF_SLOW",
                f"慢操作: {name} 耗时 {duration*1000:.1f}ms (阈值 {self._slow_threshold*1000:.1f}ms)",
                module="performance",
                extra={"op": name, "duration_ms": round(duration * 1000, 2)})

    def track(self, name: str) -> '_TrackContext':
        """上下文管理器：自动计时

        用法:
            with perf_monitor.track("perceive"):
                do_something()
        """
        return _TrackContext(self, name)

    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """获取统计

        Args:
            name: 指定操作名，None返回全部

        Returns:
            单操作返回其to_dict，全部返回{name: stats_dict}
        """
        with self._lock:
            if name is not None:
                if name not in self._stats:
                    return {}
                return self._stats[name].to_dict()
            return {n: s.to_dict() for n, s in self._stats.items()}

    def get_slow_operations(self, threshold_sec: Optional[float] = None) -> List[Dict[str, Any]]:
        """获取超过阈值的操作（按max_time降序）"""
        threshold = threshold_sec if threshold_sec is not None else self._slow_threshold
        with self._lock:
            slow = []
            for name, stats in self._stats.items():
                if stats.max_time > threshold:
                    d = stats.to_dict()
                    d["threshold_sec"] = threshold
                    slow.append(d)
            slow.sort(key=lambda x: x["max_time"], reverse=True)
            return slow

    def get_summary(self) -> Dict[str, Any]:
        """获取全局摘要"""
        with self._lock:
            total_ops = sum(s.count for s in self._stats.values())
            total_time = sum(s.total_time for s in self._stats.values())
            return {
                "operation_count": len(self._stats),
                "total_calls": total_ops,
                "total_time_tracked": round(total_time, 6),
                "enabled": self._enabled,
                "slow_threshold_sec": self._slow_threshold,
                "slow_operations": len(self.get_slow_operations()),
            }

    def reset(self) -> None:
        """清空所有统计"""
        with self._lock:
            self._stats.clear()
        logger.debug("PERF_RESET", "性能统计已清空", module="performance")

    def report(self) -> str:
        """生成可读的性能报告"""
        with self._lock:
            if not self._stats:
                return "[性能报告] 暂无数据"

            lines = ["[性能报告]"]
            summary = self.get_summary()
            lines.append(f"  操作种类: {summary['operation_count']}, 总调用: {summary['total_calls']}, "
                         f"总耗时: {summary['total_time_tracked']*1000:.1f}ms")

            # 按总耗时降序
            sorted_stats = sorted(self._stats.values(),
                                  key=lambda s: s.total_time, reverse=True)
            lines.append(f"  {'操作':<20} {'次数':>8} {'总ms':>10} {'均ms':>10} {'最大ms':>10} {'P99ms':>10}")
            lines.append("  " + "-" * 78)
            for s in sorted_stats:
                lines.append(
                    f"  {s.name:<20} {s.count:>8} {s.total_time*1000:>10.2f} "
                    f"{s.avg_time*1000:>10.2f} {s.max_time*1000:>10.2f} "
                    f"{s.percentile(99)*1000:>10.2f}"
                )

            slow = self.get_slow_operations()
            if slow:
                lines.append(f"\n  慢操作 (>{self._slow_threshold*1000:.0f}ms):")
                for op in slow[:5]:
                    lines.append(f"    - {op['name']}: max={op['max_time']*1000:.1f}ms, "
                                 f"count={op['count']}")

            return "\n".join(lines)


class _TrackContext:
    """track()返回的上下文管理器"""

    def __init__(self, monitor: PerformanceMonitor, name: str):
        self._monitor = monitor
        self._name = name
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self._start
        self._monitor.record(self._name, duration)
        return False  # 不吞异常


# 全局单例
perf_monitor = PerformanceMonitor()


def track_performance(name: str):
    """装饰器：自动追踪被装饰函数的执行耗时

    用法:
        @track_performance("predict")
        def predict(self):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with perf_monitor.track(name):
                return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


if __name__ == "__main__":
    # 自测演示
    import random

    for _ in range(100):
        with perf_monitor.track("test_op"):
            time.sleep(random.uniform(0.001, 0.005))

    perf_monitor.record("manual_op", 0.15)  # 触发慢操作告警

    print(perf_monitor.report())
    print("\n慢操作:", perf_monitor.get_slow_operations())
