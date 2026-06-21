"""
U32：永续运行与自愈控制 —— 永续CLI辅助

第一性原理：
- 系统名为"自驱完成长任务的永续运行Agent"，但永续运行/自愈控制CLI完全空白
- 永续运行=健康监控+资源回收+分级自愈，是M6能力的用户侧枢纽
- 用户需要通过CLI启用/禁用永续、查看健康历史、手动触发自愈/清理
- 永续管理应轻量级，直接访问ForeverRunner，不启动完整Agent

设计：
1. ForeverCliHelper：永续运行CLI辅助器
   - 封装ForeverRunner的CLI访问
   - 支持enable/disable/stats/history/health/self-heal/cleanup/auto-recovery
   - 格式化健康报告/统计/历史
2. 复用ForeverRunner的核心能力，不重复造轮子
3. auto_recovery作为本地配置标志，便于查询和设置
"""

import time
from typing import Dict, List, Optional, Any


class ForeverCliHelper:
    """永续运行CLI辅助器

    轻量级访问ForeverRunner，不启动完整Agent
    """

    # 健康等级到中文名的映射
    HEALTH_LEVEL_NAMES = {
        "excellent": "优秀",
        "good": "良好",
        "warning": "警告",
        "critical": "严重",
        "fatal": "致命",
        "unknown": "未知",
    }

    def __init__(self):
        from core.forever_runner import ForeverRunner
        self.runner = ForeverRunner()
        self._auto_recovery_enabled: bool = True

    # ---------- 永续运行控制 ----------

    def enable(self, health_check_interval: float = 30.0,
               resource_cleanup_interval: float = 300.0) -> bool:
        """启用永续运行

        Args:
            health_check_interval: 健康检查间隔（秒）
            resource_cleanup_interval: 资源回收间隔（秒）

        Returns:
            是否成功启用（已启用则返回False）
        """
        if self.runner._running:
            return False
        self.runner._health_interval = health_check_interval
        self.runner._cleanup_interval = resource_cleanup_interval
        self.runner.start()
        return True

    def disable(self) -> bool:
        """禁用永续运行

        Returns:
            是否成功禁用（未启用则返回False）
        """
        if not self.runner._running:
            return False
        self.runner.stop()
        return True

    def is_enabled(self) -> bool:
        """是否已启用永续运行"""
        return self.runner._running

    # ---------- 健康监控 ----------

    def get_health_report(self) -> Dict[str, Any]:
        """获取最新健康报告（主动触发一次检查）"""
        report = self.runner.check_health()
        return report.to_dict()

    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取健康历史记录"""
        return self.runner.get_health_history(limit)

    # ---------- 自愈与清理 ----------

    def trigger_self_heal(self) -> Dict[str, Any]:
        """手动触发自愈

        Returns:
            自愈前的健康报告
        """
        report = self.runner.check_health()
        self.runner._self_heal(report)
        return report.to_dict()

    def trigger_cleanup(self) -> int:
        """手动触发资源回收

        Returns:
            清理计数
        """
        self.runner._cleanup_resources()
        return self.runner._cleanup_count

    # ---------- 统计 ----------

    def get_stats(self) -> Dict[str, Any]:
        """获取永续运行统计"""
        stats = self.runner.get_stats()
        stats["auto_recovery_enabled"] = self._auto_recovery_enabled
        return stats

    # ---------- 自动恢复 ----------

    def set_auto_recovery(self, enabled: bool) -> None:
        """设置自动恢复开关"""
        self._auto_recovery_enabled = enabled

    def is_auto_recovery_enabled(self) -> bool:
        """是否启用自动恢复"""
        return self._auto_recovery_enabled

    # ---------- 格式化方法 ----------

    def format_stats(self) -> str:
        """格式化统计"""
        stats = self.get_stats()
        lines = ["=== 永续运行统计 ==="]
        lines.append(f"  运行状态: {'运行中' if stats.get('running') else '已停止'}")
        uptime = stats.get("uptime_seconds", 0)
        lines.append(f"  运行时长: {self._format_uptime(uptime)}")
        lines.append(f"  重启次数: {stats.get('restart_count', 0)}")
        lines.append(f"  自愈次数: {stats.get('self_heal_count', 0)}")
        lines.append(f"  清理次数: {stats.get('cleanup_count', 0)}")
        lines.append(f"  健康检查间隔: {stats.get('health_check_interval', 30)}秒")
        lines.append(f"  资源回收间隔: {stats.get('cleanup_interval', 300)}秒")
        level = stats.get("last_health_level", "unknown")
        level_name = self.HEALTH_LEVEL_NAMES.get(level, level)
        lines.append(f"  最近健康等级: {level_name} ({level})")
        lines.append(f"  Agent已附加: {'是' if stats.get('agent_attached') else '否'}")
        ar = "启用" if stats.get("auto_recovery_enabled") else "禁用"
        lines.append(f"  自动恢复: {ar}")
        return "\n".join(lines)

    def format_health_report(self) -> str:
        """格式化健康报告"""
        report = self.get_health_report()
        level = report.get("level", "unknown")
        level_name = self.HEALTH_LEVEL_NAMES.get(level, level)
        lines = ["=== 健康报告 ==="]
        lines.append(f"  健康等级: {level_name} ({level})")
        lines.append(f"  健康评分: {report.get('score', 0):.2f}")
        lines.append(f"  检查时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.get('timestamp', time.time())))}")

        issues = report.get("issues", [])
        if issues:
            lines.append(f"\n  问题（{len(issues)}个）:")
            for issue in issues:
                lines.append(f"    - {issue}")
        else:
            lines.append("\n  问题: 无")

        recommendations = report.get("recommendations", [])
        if recommendations:
            lines.append(f"\n  建议（{len(recommendations)}个）:")
            for rec in recommendations:
                lines.append(f"    - {rec}")

        metrics = report.get("metrics", {})
        if metrics:
            lines.append(f"\n  指标:")
            for key, value in metrics.items():
                lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def format_history(self, limit: int = 10) -> str:
        """格式化健康历史"""
        history = self.get_health_history(limit)
        if not history:
            return "（无健康历史记录，启用永续运行后将自动记录）"

        lines = [f"=== 健康历史（最近{len(history)}条）==="]
        lines.append(f"{'序号':>4} {'等级':<8} {'评分':>6} {'时间':<20} {'问题数':>6}")
        lines.append("-" * 60)
        for i, record in enumerate(history, 1):
            level = record.get("level", "unknown")
            level_name = self.HEALTH_LEVEL_NAMES.get(level, level)
            score = record.get("score", 0)
            ts = record.get("timestamp", 0)
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "N/A"
            issue_count = len(record.get("issues", []))
            lines.append(f"{i:>4} {level_name:<8} {score:>6.2f} {time_str:<20} {issue_count:>6}")
        return "\n".join(lines)

    def format_enable_result(self, success: bool, health_interval: float,
                             cleanup_interval: float) -> str:
        """格式化启用结果"""
        if success:
            lines = ["=== 永续运行已启用 ==="]
            lines.append(f"  健康检查间隔: {health_interval}秒")
            lines.append(f"  资源回收间隔: {cleanup_interval}秒")
            lines.append("  健康监控线程已启动")
            lines.append("  资源回收线程已启动")
        else:
            lines = ["=== 永续运行已在运行中 ==="]
            lines.append("  如需重新配置，请先执行 'forever disable'")
        return "\n".join(lines)

    def format_disable_result(self, success: bool) -> str:
        """格式化禁用结果"""
        if success:
            lines = ["=== 永续运行已禁用 ==="]
            lines.append("  健康监控线程已停止")
            lines.append("  资源回收线程已停止")
        else:
            lines = ["=== 永续运行未启用 ==="]
            lines.append("  无需禁用")
        return "\n".join(lines)

    def format_self_heal_result(self, report: Dict[str, Any]) -> str:
        """格式化自愈结果"""
        level = report.get("level", "unknown")
        level_name = self.HEALTH_LEVEL_NAMES.get(level, level)
        lines = ["=== 自愈已触发 ==="]
        lines.append(f"  自愈前健康等级: {level_name} ({level})")
        lines.append(f"  自愈前健康评分: {report.get('score', 0):.2f}")
        issues = report.get("issues", [])
        if issues:
            lines.append(f"  自愈前问题（{len(issues)}个）:")
            for issue in issues:
                lines.append(f"    - {issue}")
        else:
            lines.append("  自愈前问题: 无")
        lines.append(f"\n  当前自愈总次数: {self.runner._self_heal_count}")
        return "\n".join(lines)

    def format_cleanup_result(self, cleanup_count: int) -> str:
        """格式化清理结果"""
        lines = ["=== 资源回收已触发 ==="]
        lines.append(f"  累计清理次数: {cleanup_count}")
        lines.append("  已执行: Python垃圾回收")
        return "\n".join(lines)

    def format_auto_recovery(self) -> str:
        """格式化自动恢复状态"""
        enabled = self.is_auto_recovery_enabled()
        lines = ["=== 自动恢复状态 ==="]
        lines.append(f"  状态: {'启用' if enabled else '禁用'}")
        if enabled:
            lines.append("  说明: Agent运行时将自动恢复故障")
        else:
            lines.append("  说明: 故障需手动处理")
        return "\n".join(lines)

    def format_auto_recovery_set(self, enabled: bool) -> str:
        """格式化自动恢复设置结果"""
        lines = ["=== 自动恢复已设置 ==="]
        lines.append(f"  新状态: {'启用' if enabled else '禁用'}")
        return "\n".join(lines)

    # ---------- 内部工具 ----------

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """格式化运行时长"""
        if seconds <= 0:
            return "0秒"
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        parts = []
        if days > 0:
            parts.append(f"{days}天")
        if hours > 0:
            parts.append(f"{hours}小时")
        if minutes > 0:
            parts.append(f"{minutes}分钟")
        if secs > 0 or not parts:
            parts.append(f"{secs}秒")
        return "".join(parts)
