"""
U25：多Agent控制台 —— 多Agent CLI辅助

第一性原理：
- 多Agent编排能力完整但CLI完全空白
- 用户需要查看集群状态、管理子Agent生命周期
- 需要能快速委派任务测试子Agent能力
- 故障Agent应可通过CLI重启恢复

设计：
1. AgentsCliHelper：多Agent CLI辅助器
   - 封装AgentOrchestrator的CLI访问
   - 格式化集群状态/Agent列表/健康检查/委派结果
   - 支持spawn/stop/restart/delegate/recover操作
2. 复用AgentOrchestrator的核心能力，不启动完整SelfEvolvingAgent
"""

import json
import time
from typing import Dict, List, Optional, Any


class AgentsCliHelper:
    """多Agent CLI辅助器

    轻量级访问AgentOrchestrator，不启动完整SelfEvolvingAgent
    """

    # 角色到中文名的映射
    ROLE_NAMES = {
        "perception": "感知",
        "prediction": "预测",
        "decision": "决策",
        "execution": "执行",
        "general": "通用",
        "orchestrator": "调度",
        "merger": "合并",
    }

    def __init__(self):
        from core.agent_orchestrator import AgentOrchestrator
        self.orchestrator = AgentOrchestrator()

    def spawn_agent(self, role: str, agent_id: Optional[str] = None) -> str:
        """创建子Agent

        Args:
            role: 角色名（perception/prediction/decision/execution）
            agent_id: 可选ID

        Returns:
            agent_id
        """
        from core.multi_agent_bus import AgentRole
        role_enum = self._parse_role(role)
        if role_enum is None:
            raise ValueError(f"不支持的角色: {role}（支持: perception/prediction/decision/execution）")
        return self.orchestrator.spawn_agent(role_enum, agent_id)

    def spawn_default_agents(self) -> List[str]:
        """创建默认4个专用子Agent"""
        return self.orchestrator.spawn_default_agents()

    def stop_agent(self, agent_id: str) -> bool:
        """停止子Agent"""
        return self.orchestrator.stop_agent(agent_id)

    def stop_all(self) -> None:
        """停止所有子Agent"""
        self.orchestrator.stop_all()

    def restart_agent(self, agent_id: str) -> bool:
        """重启子Agent"""
        return self.orchestrator.restart_agent(agent_id)

    def delegate_task(self, task_type: str, payload: Dict[str, Any],
                      timeout: Optional[float] = None) -> Dict[str, Any]:
        """委派任务

        Args:
            task_type: 任务类型（perceive/predict/decide/execute/analyze）
            payload: 任务参数
            timeout: 超时秒数

        Returns:
            委派结果字典
        """
        from core.agent_orchestrator import TaskType
        task_enum = self._parse_task_type(task_type)
        if task_enum is None:
            return {
                "task_type": task_type,
                "success": False,
                "error": f"未知任务类型: {task_type}（支持: perceive/predict/decide/execute/analyze）",
            }
        result = self.orchestrator.delegate_task(task_enum, payload, timeout)
        return result.to_dict()

    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        return self.orchestrator.get_cluster_status()

    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有Agent详情"""
        status = self.orchestrator.get_cluster_status()
        agents = []
        for agent_id, stats in status["agents"].items():
            agents.append({
                "agent_id": agent_id,
                "role": stats.get("role", "unknown"),
                "running": stats.get("running", False),
                "cycle_count": stats.get("cycle_count", 0),
                "error_count": stats.get("error_count", 0),
                "uptime": stats.get("uptime", 0),
            })
        return agents

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return self.orchestrator.health_check()

    def auto_recover(self) -> int:
        """自动恢复故障Agent"""
        return self.orchestrator.auto_recover()

    def get_delegation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取委派历史"""
        return self.orchestrator.get_delegation_history(limit)

    def _parse_role(self, role: str):
        """解析角色名"""
        from core.multi_agent_bus import AgentRole
        mapping = {
            "perception": AgentRole.PERCEPTION,
            "prediction": AgentRole.PREDICTION,
            "decision": AgentRole.DECISION,
            "execution": AgentRole.EXECUTION,
        }
        return mapping.get(role.lower())

    def _parse_task_type(self, task_type: str):
        """解析任务类型"""
        from core.agent_orchestrator import TaskType
        mapping = {
            "perceive": TaskType.PERCEIVE,
            "predict": TaskType.PREDICT,
            "decide": TaskType.DECIDE,
            "execute": TaskType.EXECUTE,
            "analyze": TaskType.ANALYZE,
        }
        return mapping.get(task_type.lower())

    # ---------- 格式化方法 ----------

    def format_status(self) -> str:
        """格式化集群状态"""
        status = self.get_cluster_status()
        lines = ["=== 多Agent集群状态 ==="]
        lines.append(f"  Agent数量: {status['agent_count']}")
        lines.append(f"  委派总数: {status['delegation_count']}")
        lines.append(f"  成功数: {status['success_count']}")
        lines.append(f"  失败数: {status['failure_count']}")
        lines.append(f"  成功率: {status['success_rate']*100:.1f}%")
        bus_stats = status.get("bus_stats", {})
        if bus_stats:
            lines.append(f"  消息总线: 已发送={bus_stats.get('sent_count', 0)}, "
                         f"已接收={bus_stats.get('received_count', 0)}")
        return "\n".join(lines)

    def format_list(self) -> str:
        """格式化Agent列表"""
        agents = self.list_agents()
        if not agents:
            return "（无子Agent，使用 'agents spawn <role>' 创建）"
        lines = [f"=== 子Agent列表（{len(agents)}个）==="]
        lines.append(f"{'Agent ID':<20} {'角色':<12} {'状态':<8} {'周期':>8} {'错误':>8} {'运行时长':>10}")
        lines.append("-" * 75)
        for a in agents:
            role_name = self.ROLE_NAMES.get(a["role"], a["role"])
            status = "运行中" if a["running"] else "已停止"
            uptime_str = f"{a['uptime']:.1f}s" if a["uptime"] > 0 else "-"
            lines.append(
                f"{a['agent_id']:<20} {role_name:<12} {status:<8} "
                f"{a['cycle_count']:>8} {a['error_count']:>8} {uptime_str:>10}"
            )
        return "\n".join(lines)

    def format_health(self) -> str:
        """格式化健康检查"""
        health = self.health_check()
        lines = ["=== 健康检查 ==="]
        lines.append(f"  总数: {health['total']}")
        lines.append(f"  健康: {health['healthy']}")
        lines.append(f"  不健康: {len(health['unhealthy'])}")
        if health["unhealthy"]:
            lines.append("  不健康Agent:")
            for item in health["unhealthy"]:
                reason_map = {"not_running": "未运行", "high_error_rate": "错误率过高"}
                reason = reason_map.get(item["reason"], item["reason"])
                lines.append(f"    - {item['agent_id']}: {reason}")
        return "\n".join(lines)

    def format_delegation_result(self, result: Dict[str, Any]) -> str:
        """格式化委派结果"""
        lines = ["=== 委派结果 ==="]
        lines.append(f"  任务类型: {result.get('task_type', 'unknown')}")
        lines.append(f"  成功: {'是' if result.get('success') else '否'}")
        lines.append(f"  执行Agent: {result.get('agent_id', '-')}")
        lines.append(f"  耗时: {result.get('elapsed', 0)*1000:.1f} ms")
        if result.get("error"):
            lines.append(f"  错误: {result['error']}")
        if result.get("result") is not None:
            result_str = json.dumps(result["result"], ensure_ascii=False, default=str)
            if len(result_str) > 200:
                result_str = result_str[:200] + "..."
            lines.append(f"  结果: {result_str}")
        return "\n".join(lines)

    def format_spawn_result(self, agent_id: str, role: str) -> str:
        """格式化创建结果"""
        role_name = self.ROLE_NAMES.get(role, role)
        return f"已创建子Agent: {agent_id} [{role_name}]"

    def format_history(self, limit: int = 20) -> str:
        """格式化委派历史"""
        history = self.get_delegation_history(limit)
        if not history:
            return "（无委派历史）"
        lines = [f"=== 委派历史（最近{len(history)}条）==="]
        lines.append(f"{'任务类型':<12} {'成功':<6} {'Agent':<20} {'耗时ms':>10}")
        lines.append("-" * 55)
        for h in history:
            success = "是" if h.get("success") else "否"
            elapsed = h.get("elapsed", 0) * 1000
            lines.append(
                f"{h.get('task_type', '-'):<12} {success:<6} "
                f"{h.get('agent_id', '-'):<20} {elapsed:>10.1f}"
            )
        return "\n".join(lines)
