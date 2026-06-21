"""
U10：状态摘要优化

第一性原理：状态输出应可读可解析，机器与人类兼顾。
- 人类模式：分组显示、时间格式化、状态图标、颜色提示
- JSON模式：结构化字典，便于脚本与监控集成
- --watch模式：持续刷新

格式化逻辑与Agent分离，便于单元测试。
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


def format_duration(seconds: float) -> str:
    """将秒数格式化为人类友好的时长字符串

    Examples:
        45 -> "45秒"
        125 -> "2分5秒"
        3725 -> "1小时2分5秒"
        90061 -> "1天1小时1分1秒"
    """
    if seconds < 0:
        return "0秒"
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if days > 0:
        parts.append(f"{days}天")
    if hours > 0:
        parts.append(f"{hours}小时")
    if minutes > 0:
        parts.append(f"{minutes}分")
    if secs > 0 or not parts:
        parts.append(f"{secs}秒")
    return "".join(parts)


def format_bytes(num_bytes: int) -> str:
    """格式化字节数为人类友好字符串"""
    if num_bytes < 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def format_percent(value: float, total: float = 100.0) -> str:
    """格式化百分比，带状态图标"""
    pct = (value / total * 100) if total > 0 else 0
    if pct < 50:
        icon = "[OK]"
    elif pct < 80:
        icon = "[!!]"
    else:
        icon = "[XX]"
    return f"{icon} {pct:.1f}%"


@dataclass
class StatusSection:
    """状态报告的一个分组"""
    title: str
    items: List[Dict[str, str]]  # [{"label": "...", "value": "..."}]


class StatusFormatter:
    """状态格式化器

    将Agent状态字典格式化为人类友好文本或JSON。
    """

    @staticmethod
    def collect_status(agent) -> Dict[str, Any]:
        """从Agent实例收集完整状态"""
        uptime = time.time() - agent.start_time if getattr(agent, "start_time", None) else 0
        status: Dict[str, Any] = {
            "running": agent.running,
            "loop_count": getattr(agent, "loop_count", 0),
            "uptime_seconds": uptime,
            "uptime_human": format_duration(uptime),
        }
        # 配置信息
        try:
            cfg = agent.config
            status["config"] = {
                "max_cpu_percent": cfg.resource.max_cpu_percent,
                "max_memory_percent": cfg.resource.max_memory_percent,
                "main_loop_frequency": cfg.main_loop_frequency,
                "log_level": cfg.logging.console_level,
            }
        except Exception:
            status["config"] = {}
        # 驱力信息
        try:
            drives = agent.drive_engine.get_drive_state()
            status["drives"] = {
                "safe": round(drives.get("safe", 0), 3),
                "gain": round(drives.get("gain", 0), 3),
                "eff": round(drives.get("eff", 0), 3),
            }
        except Exception:
            status["drives"] = {}
        # 预测信息
        try:
            pred = agent.prediction.get_prediction()
            status["prediction"] = {
                "confidence": round(pred.get("confidence", 0), 3) if isinstance(pred, dict) else 0,
            }
        except Exception:
            status["prediction"] = {}
        # 资源使用
        try:
            state = agent.perception.get_state_vector()
            status["resources"] = {
                "cpu_percent": round(state.get("cpu_percent", 0), 1) if isinstance(state, dict) else 0,
                "memory_percent": round(state.get("memory_percent", 0), 1) if isinstance(state, dict) else 0,
            }
        except Exception:
            status["resources"] = {}
        # 多Agent
        try:
            if getattr(agent, "orchestrator", None):
                agents = agent.orchestrator.list_agents() if hasattr(agent.orchestrator, "list_agents") else []
                status["multi_agent"] = {
                    "enabled": True,
                    "agent_count": len(agents) if agents else 0,
                }
            else:
                status["multi_agent"] = {"enabled": False, "agent_count": 0}
        except Exception:
            status["multi_agent"] = {"enabled": False, "agent_count": 0}
        # 永续运行
        status["forever_running"] = getattr(agent, "_forever_enabled", False)
        # REST/Dashboard
        status["rest_api"] = {"enabled": agent.dashboard is not None and getattr(agent, "_dashboard_enabled", False)}
        return status

    @staticmethod
    def format_human(status: Dict[str, Any]) -> str:
        """格式化为人类友好文本"""
        lines: List[str] = []
        lines.append("=" * 50)
        lines.append("  自驱Agent系统 —— 状态")
        lines.append("=" * 50)

        # 运行状态
        running_icon = "[运行中]" if status.get("running") else "[已停止]"
        lines.append(f"\n{running_icon} 运行状态")
        lines.append(f"  循环数: {status.get('loop_count', 0)}")
        lines.append(f"  运行时长: {status.get('uptime_human', '0秒')}")
        lines.append(f"  永续运行: {'是' if status.get('forever_running') else '否'}")

        # 资源使用
        resources = status.get("resources", {})
        if resources:
            lines.append("\n[资源] 系统资源")
            cpu = resources.get("cpu_percent", 0)
            mem = resources.get("memory_percent", 0)
            lines.append(f"  CPU使用: {format_percent(cpu)}")
            lines.append(f"  内存使用: {format_percent(mem)}")

        # 驱力
        drives = status.get("drives", {})
        if drives:
            lines.append("\n[驱力] 内在驱力")
            lines.append(f"  安全: {drives.get('safe', 0)}")
            lines.append(f"  收益: {drives.get('gain', 0)}")
            lines.append(f"  能效: {drives.get('eff', 0)}")

        # 预测
        prediction = status.get("prediction", {})
        if prediction:
            lines.append("\n[预测] 预测引擎")
            conf = prediction.get("confidence", 0)
            lines.append(f"  置信度: {format_percent(conf, 1.0)}")

        # 配置
        config = status.get("config", {})
        if config:
            lines.append("\n[配置] 当前配置")
            lines.append(f"  CPU上限: {config.get('max_cpu_percent', '?')}%")
            lines.append(f"  内存上限: {config.get('max_memory_percent', '?')}%")
            lines.append(f"  循环频率: {config.get('main_loop_frequency', '?')}Hz")
            lines.append(f"  日志级别: {config.get('log_level', '?')}")

        # 多Agent
        multi = status.get("multi_agent", {})
        if multi:
            lines.append("\n[编排] 多Agent")
            lines.append(f"  已启用: {'是' if multi.get('enabled') else '否'}")
            lines.append(f"  子Agent数: {multi.get('agent_count', 0)}")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)

    @staticmethod
    def format_json(status: Dict[str, Any]) -> str:
        """格式化为JSON字符串"""
        import json
        return json.dumps(status, indent=2, ensure_ascii=False)

    @classmethod
    def format(cls, status: Dict[str, Any], output_format: str = "human") -> str:
        """根据格式类型输出

        Args:
            status: 状态字典
            output_format: "human" 或 "json"
        """
        if output_format == "json":
            return cls.format_json(status)
        return cls.format_human(status)
