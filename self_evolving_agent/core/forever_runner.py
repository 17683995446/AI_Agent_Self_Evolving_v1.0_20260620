"""
M6：永续运行保障 —— 让Agent永远运行不停止

第一性原理：
1. 永续=故障可自愈+资源可回收+健康可监控
2. 资源泄漏是永续的最大敌人，必须主动回收
3. 健康监控应多维：主循环心跳+内存+线程+错误率
4. 自愈应分级：轻则重启线程，重则恢复状态，最重则全量重启

核心能力：
- ForeverRunner：永续运行器，封装start/stop/自愈循环
- 资源回收：定期清理快照/日志/内存
- 健康监控：多维健康评分
- 自愈策略：分级恢复
- 永续统计：运行时长/重启次数/自愈次数
"""

import os
import sys
import time
import threading
import gc
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


class HealthLevel(Enum):
    """健康等级"""
    EXCELLENT = "excellent"  # 一切正常
    GOOD = "good"           # 轻微问题
    WARNING = "warning"     # 需要关注
    CRITICAL = "critical"   # 需要自愈
    FATAL = "fatal"         # 需要全量重启


@dataclass
class HealthReport:
    """健康报告"""
    level: HealthLevel = HealthLevel.EXCELLENT
    score: float = 1.0  # 0-1
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "score": round(self.score, 4),
            "issues": self.issues,
            "recommendations": self.recommendations,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }


class ForeverRunner:
    """永续运行器 —— 保障Agent永远运行不停止

    设计要点：
    - 健康监控线程定期检查Agent健康
    - 资源回收线程定期清理
    - 自愈策略分级：重启线程→恢复状态→全量重启
    - 运行统计可查
    """

    def __init__(self, agent: Optional[Any] = None,
                 health_check_interval: float = 30.0,
                 resource_cleanup_interval: float = 300.0):
        self._agent = agent
        self._health_interval = health_check_interval
        self._cleanup_interval = resource_cleanup_interval

        self._running = False
        self._health_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None

        # 统计
        self._start_time: Optional[float] = None
        self._restart_count = 0
        self._self_heal_count = 0
        self._cleanup_count = 0
        self._last_health_report: Optional[HealthReport] = None
        self._health_history: List[HealthReport] = []

        # 阈值
        self._memory_warning_mb = 500
        self._memory_critical_mb = 1000
        self._error_rate_warning = 0.1
        self._error_rate_critical = 0.3
        self._heartbeat_critical_sec = 120

        logger.info("FOREVER", "永续运行器初始化", module="forever_runner")

    def attach(self, agent: Any) -> None:
        """附加Agent"""
        self._agent = agent
        logger.info("FOREVER", "已附加Agent", module="forever_runner")

    def start(self) -> None:
        """启动永续运行"""
        if self._running:
            return
        self._running = True
        self._start_time = time.time()

        # 启动健康监控线程
        self._health_thread = threading.Thread(
            target=self._health_loop, name="HealthMonitor", daemon=True
        )
        self._health_thread.start()

        # 启动资源回收线程
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, name="ResourceCleanup", daemon=True
        )
        self._cleanup_thread.start()

        logger.info("FOREVER", "永续运行已启动", module="forever_runner")

    def stop(self) -> None:
        """停止永续运行"""
        self._running = False
        if self._health_thread:
            self._health_thread.join(timeout=5.0)
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        logger.info("FOREVER", "永续运行已停止", module="forever_runner")

    # ---------- 健康监控 ----------

    def _health_loop(self) -> None:
        """健康监控循环"""
        while self._running:
            try:
                report = self.check_health()
                self._last_health_report = report
                self._health_history.append(report)
                # 保留最近100条
                if len(self._health_history) > 100:
                    self._health_history = self._health_history[-100:]

                if report.level in (HealthLevel.CRITICAL, HealthLevel.FATAL):
                    logger.warning("FOREVER",
                                   f"健康状态: {report.level.value}, 触发自愈",
                                   module="forever_runner")
                    self._self_heal(report)
            except Exception as e:
                logger.error("FOREVER", f"健康检查异常: {e}", module="forever_runner")
            time.sleep(self._health_interval)

    def check_health(self) -> HealthReport:
        """检查Agent健康状态"""
        report = HealthReport()
        score = 1.0
        issues = []
        recommendations = []

        if self._agent is None:
            report.level = HealthLevel.WARNING
            report.score = 0.5
            report.issues.append("未附加Agent")
            return report

        # 1. 主循环心跳检查
        try:
            stats = self._agent.get_persistence_stats()
            heartbeat_age = stats.get("last_heartbeat_age", 0)
            report.metrics["heartbeat_age"] = heartbeat_age
            if heartbeat_age > self._heartbeat_critical_sec:
                score -= 0.4
                issues.append(f"主循环心跳超时 {heartbeat_age:.0f}s")
                recommendations.append("重启主循环线程")
            elif heartbeat_age > 60:
                score -= 0.2
                issues.append(f"主循环心跳延迟 {heartbeat_age:.0f}s")
        except Exception as e:
            score -= 0.2
            issues.append(f"心跳检查失败: {e}")

        # 2. 内存使用检查
        try:
            import psutil
            proc = psutil.Process()
            mem_mb = proc.memory_info().rss / 1024 / 1024
            report.metrics["memory_mb"] = round(mem_mb, 1)
            if mem_mb > self._memory_critical_mb:
                score -= 0.3
                issues.append(f"内存使用过高 {mem_mb:.0f}MB")
                recommendations.append("触发垃圾回收和资源清理")
            elif mem_mb > self._memory_warning_mb:
                score -= 0.1
                issues.append(f"内存使用偏高 {mem_mb:.0f}MB")
        except Exception:
            pass

        # 3. 线程数检查
        try:
            thread_count = threading.active_count()
            report.metrics["thread_count"] = thread_count
            if thread_count > 50:
                score -= 0.2
                issues.append(f"线程数过多 {thread_count}")
                recommendations.append("检查线程泄漏")
        except Exception:
            pass

        # 4. 错误率检查（从性能监控获取）
        try:
            from core.performance import perf_monitor
            perf_stats = perf_monitor.get_summary()
            report.metrics["total_ops"] = perf_stats.get("total_calls", 0)
        except Exception:
            pass

        # 5. Agent运行状态
        try:
            if hasattr(self._agent, "running") and not self._agent.running:
                score -= 0.3
                issues.append("Agent未在运行")
                recommendations.append("重启Agent")
        except Exception:
            pass

        # 确定健康等级
        report.score = max(0.0, min(1.0, score))
        report.issues = issues
        report.recommendations = recommendations

        if report.score >= 0.9:
            report.level = HealthLevel.EXCELLENT
        elif report.score >= 0.7:
            report.level = HealthLevel.GOOD
        elif report.score >= 0.5:
            report.level = HealthLevel.WARNING
        elif report.score >= 0.3:
            report.level = HealthLevel.CRITICAL
        else:
            report.level = HealthLevel.FATAL

        return report

    # ---------- 资源回收 ----------

    def _cleanup_loop(self) -> None:
        """资源回收循环"""
        while self._running:
            try:
                self._cleanup_resources()
            except Exception as e:
                logger.error("FOREVER", f"资源回收异常: {e}", module="forever_runner")
            time.sleep(self._cleanup_interval)

    def _cleanup_resources(self) -> None:
        """执行资源回收"""
        cleaned = []

        # 1. Python垃圾回收
        collected = gc.collect()
        if collected > 0:
            cleaned.append(f"gc:{collected}")

        # 2. 清理旧快照
        if self._agent and hasattr(self._agent, "persistence"):
            try:
                snapshots = self._agent.persistence.list_snapshots()
                # 保留最新5个
                if len(snapshots) > 5:
                    for old_path in snapshots[:-5]:
                        try:
                            os.remove(old_path)
                            cleaned.append(f"snapshot:{os.path.basename(old_path)}")
                        except Exception:
                            pass
            except Exception:
                pass

        # 3. 清理旧日志（保留最近7天）
        if self._agent and hasattr(self._agent, "config"):
            try:
                log_dir = self._agent.config.log_dir
                if os.path.exists(log_dir):
                    now = time.time()
                    for fname in os.listdir(log_dir):
                        fpath = os.path.join(log_dir, fname)
                        if os.path.isfile(fpath) and fname.endswith(".log"):
                            age = now - os.path.getmtime(fpath)
                            if age > 7 * 24 * 3600:  # 7天
                                os.remove(fpath)
                                cleaned.append(f"log:{fname}")
            except Exception:
                pass

        self._cleanup_count += 1
        if cleaned:
            logger.info("FOREVER", f"资源回收: {', '.join(cleaned)}", module="forever_runner")

    # ---------- 自愈策略 ----------

    def _self_heal(self, report: HealthReport) -> None:
        """执行自愈 —— 分级恢复"""
        self._self_heal_count += 1
        logger.warning("FOREVER", f"执行自愈 #{self._self_heal_count}: {report.level.value}",
                       module="forever_runner")

        if self._agent is None:
            return

        try:
            # Level 1: 重启主循环线程
            if "重启主循环线程" in report.recommendations:
                self._restart_main_loop()

            # Level 2: 内存回收
            if "内存" in " ".join(report.issues):
                gc.collect()
                self._cleanup_resources()

            # Level 3: 状态恢复
            if report.level == HealthLevel.FATAL:
                self._full_restart()

        except Exception as e:
            logger.error("FOREVER", f"自愈失败: {e}", module="forever_runner")

    def _restart_main_loop(self) -> None:
        """重启主循环线程"""
        if self._agent is None:
            return
        try:
            if (self._agent._main_thread and
                    not self._agent._main_thread.is_alive() and
                    self._agent.running):
                logger.info("FOREVER", "重启主循环线程", module="forever_runner")
                self._agent._main_thread = threading.Thread(
                    target=self._agent._online_loop,
                    name="OnlineLoop-Healed", daemon=True
                )
                self._agent._main_thread.start()
                self._restart_count += 1
        except Exception as e:
            logger.error("FOREVER", f"重启主循环失败: {e}", module="forever_runner")

    def _full_restart(self) -> None:
        """全量重启"""
        if self._agent is None:
            return
        logger.warning("FOREVER", "执行全量重启", module="forever_runner")
        try:
            # 保存状态
            self._agent.save_state()
            # 停止
            self._agent.stop()
            time.sleep(1.0)
            # 恢复状态
            self._agent.load_state()
            # 重启
            self._agent.start()
            self._restart_count += 1
        except Exception as e:
            logger.error("FOREVER", f"全量重启失败: {e}", module="forever_runner")

    # ---------- 查询 ----------

    def get_last_health_report(self) -> Optional[Dict[str, Any]]:
        return self._last_health_report.to_dict() if self._last_health_report else None

    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._health_history[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "running": self._running,
            "uptime_seconds": round(uptime, 1),
            "restart_count": self._restart_count,
            "self_heal_count": self._self_heal_count,
            "cleanup_count": self._cleanup_count,
            "health_check_interval": self._health_interval,
            "cleanup_interval": self._cleanup_interval,
            "last_health_level": (self._last_health_report.level.value
                                  if self._last_health_report else "unknown"),
            "agent_attached": self._agent is not None,
        }


if __name__ == "__main__":
    runner = ForeverRunner()
    runner.start()
    time.sleep(2)
    print("统计:", runner.get_stats())
    print("健康:", runner.get_last_health_report())
    runner.stop()
