"""
U24：性能分析 —— 性能CLI辅助

第一性原理：
- 性能数据只能被动查看，缺乏主动暴露入口
- 用户需要快速识别慢操作和性能瓶颈
- 性能统计应可格式化展示，支持单操作和全局视角
- 慢操作阈值应可动态调整，便于不同场景排障

设计：
1. PerfCliHelper：性能CLI辅助器
   - 封装PerformanceMonitor单例的CLI访问
   - 格式化单操作统计/全局摘要/慢操作列表
   - 支持重置统计、调整慢操作阈值
2. 复用PerformanceMonitor的核心能力，不重复造轮子
"""

from typing import Dict, List, Optional, Any


class PerfCliHelper:
    """性能CLI辅助器

    轻量级访问PerformanceMonitor单例，不启动完整Agent
    """

    def __init__(self):
        from core.performance import perf_monitor
        self.monitor = perf_monitor

    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """获取统计

        Args:
            name: 指定操作名，None返回全部

        Returns:
            单操作返回其统计字典，全部返回{name: stats_dict}
        """
        return self.monitor.get_stats(name)

    def get_slow_operations(self, threshold_sec: Optional[float] = None) -> List[Dict[str, Any]]:
        """获取超过阈值的操作（按max_time降序）"""
        return self.monitor.get_slow_operations(threshold_sec)

    def get_summary(self) -> Dict[str, Any]:
        """获取全局摘要"""
        return self.monitor.get_summary()

    def reset(self) -> None:
        """清空所有统计"""
        self.monitor.reset()

    def set_slow_threshold(self, threshold_sec: float) -> None:
        """设置慢操作阈值（秒）"""
        self.monitor.set_slow_threshold(threshold_sec)

    def set_enabled(self, enabled: bool) -> None:
        """启用/禁用监控"""
        self.monitor.set_enabled(enabled)

    def format_stats(self, name: Optional[str] = None) -> str:
        """格式化统计

        Args:
            name: 指定操作名，None格式化全部

        Returns:
            可读的统计文本
        """
        if name is not None:
            stats = self.monitor.get_stats(name)
            if not stats:
                return f"（操作 '{name}' 不存在）"
            return self._format_single_op(stats)

        all_stats = self.monitor.get_stats()
        if not all_stats:
            return "（暂无性能数据）"

        lines = [f"=== 性能统计（{len(all_stats)}个操作）==="]
        lines.append(
            f"{'操作':<20} {'次数':>8} {'总ms':>10} {'均ms':>10} "
            f"{'最小ms':>10} {'最大ms':>10} {'P50ms':>10} {'P99ms':>10}"
        )
        lines.append("-" * 100)
        # 按总耗时降序
        sorted_items = sorted(all_stats.items(), key=lambda kv: kv[1]["total_time"], reverse=True)
        for op_name, s in sorted_items:
            lines.append(
                f"{op_name:<20} {s['count']:>8} {s['total_time']*1000:>10.2f} "
                f"{s['avg_time']*1000:>10.2f} {s['min_time']*1000:>10.2f} "
                f"{s['max_time']*1000:>10.2f} {s['p50']*1000:>10.2f} {s['p99']*1000:>10.2f}"
            )
        return "\n".join(lines)

    def _format_single_op(self, stats: Dict[str, Any]) -> str:
        """格式化单个操作的统计"""
        lines = [f"=== 操作统计: {stats['name']} ==="]
        lines.append(f"  调用次数: {stats['count']}")
        lines.append(f"  总耗时: {stats['total_time']*1000:.2f} ms")
        lines.append(f"  平均耗时: {stats['avg_time']*1000:.2f} ms")
        lines.append(f"  最小耗时: {stats['min_time']*1000:.2f} ms")
        lines.append(f"  最大耗时: {stats['max_time']*1000:.2f} ms")
        lines.append(f"  标准差: {stats['std_dev']*1000:.2f} ms")
        lines.append(f"  P50: {stats['p50']*1000:.2f} ms")
        lines.append(f"  P99: {stats['p99']*1000:.2f} ms")
        return "\n".join(lines)

    def format_slow(self, threshold_sec: Optional[float] = None) -> str:
        """格式化慢操作列表

        Args:
            threshold_sec: 阈值（秒），None使用默认阈值

        Returns:
            可读的慢操作列表
        """
        slow = self.monitor.get_slow_operations(threshold_sec)
        if not slow:
            threshold = threshold_sec if threshold_sec is not None else self.monitor._slow_threshold
            return f"（无慢操作，阈值 {threshold*1000:.0f}ms）"

        threshold = slow[0].get("threshold_sec", self.monitor._slow_threshold)
        lines = [f"=== 慢操作（>{threshold*1000:.0f}ms，共{len(slow)}个）==="]
        lines.append(
            f"{'操作':<20} {'次数':>8} {'最大ms':>10} {'均ms':>10} {'P99ms':>10}"
        )
        lines.append("-" * 70)
        for op in slow:
            lines.append(
                f"{op['name']:<20} {op['count']:>8} {op['max_time']*1000:>10.2f} "
                f"{op['avg_time']*1000:>10.2f} {op['p99']*1000:>10.2f}"
            )
        return "\n".join(lines)

    def format_summary(self) -> str:
        """格式化全局摘要"""
        summary = self.monitor.get_summary()
        lines = ["=== 性能摘要 ==="]
        lines.append(f"  操作种类: {summary['operation_count']}")
        lines.append(f"  总调用次数: {summary['total_calls']}")
        lines.append(f"  总跟踪耗时: {summary['total_time_tracked']*1000:.2f} ms")
        lines.append(f"  监控状态: {'启用' if summary['enabled'] else '禁用'}")
        lines.append(f"  慢操作阈值: {summary['slow_threshold_sec']*1000:.0f} ms")
        lines.append(f"  慢操作数量: {summary['slow_operations']}")
        return "\n".join(lines)

    def format_report(self) -> str:
        """格式化完整报告（复用PerformanceMonitor.report）"""
        return self.monitor.report()
