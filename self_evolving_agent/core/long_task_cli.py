"""
U30：长任务管理 —— 长任务CLI辅助

第一性原理：
- 系统名称含"自驱完成长任务"，但用户无法通过CLI规划、推进、查看长任务
- 长任务是核心价值主张的直接缺失
- 长任务管理应轻量级，直接访问LongTaskPlanner，不启动完整Agent
- 支持规划/列表/状态/推进/暂停/恢复/统计全生命周期

设计：
1. LongTaskCliHelper：长任务CLI辅助器
   - 封装LongTaskPlanner的CLI访问
   - 格式化任务列表/详情/统计
   - 支持plan/list/status/advance/pause/resume/stats
2. 复用LongTaskPlanner的核心能力，不重复造轮子
"""

import time
from typing import Dict, List, Optional, Any


class LongTaskCliHelper:
    """长任务CLI辅助器

    轻量级访问LongTaskPlanner，不启动完整Agent
    """

    # 任务状态到中文名的映射
    STATE_NAMES = {
        "planned": "已规划",
        "running": "运行中",
        "completed": "已完成",
        "failed": "已失败",
        "paused": "已暂停",
    }

    # 子目标状态到中文名的映射
    SUBGOAL_STATE_NAMES = {
        "pending": "待执行",
        "in_progress": "执行中",
        "completed": "已完成",
        "failed": "已失败",
        "skipped": "已跳过",
    }

    def __init__(self):
        from core.long_task_planner import LongTaskPlanner
        self.planner = LongTaskPlanner()

    def plan_task(self, template_name: str, name: str = "",
                  description: str = "") -> Dict[str, Any]:
        """规划长任务

        Args:
            template_name: 模板名称
            name: 任务名称
            description: 任务描述

        Returns:
            任务信息字典
        """
        task = self.planner.plan_task(template_name, name, description)
        return task.to_dict()

    def list_tasks(self) -> List[Dict[str, Any]]:
        """列出所有任务"""
        tasks = self.planner.list_tasks()
        return [t.to_dict() for t in tasks]

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            任务信息字典，不存在返回None
        """
        task = self.planner.get_task(task_id)
        if task is None:
            return None
        return task.to_dict()

    def advance_task(self, task_id: str, drive_level: float = 0.8) -> Dict[str, Any]:
        """推进任务

        Args:
            task_id: 任务ID
            drive_level: 驱力水平 [0, 1]

        Returns:
            推进结果
        """
        return self.planner.advance_task(task_id, drive_level)

    def pause_task(self, task_id: str) -> bool:
        """暂停任务"""
        return self.planner.pause_task(task_id)

    def resume_task(self, task_id: str) -> bool:
        """恢复任务"""
        return self.planner.resume_task(task_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return self.planner.get_stats()

    def get_available_templates(self) -> List[str]:
        """获取可用模板列表"""
        return list(self.planner._templates.keys())

    # ---------- 格式化方法 ----------

    def format_list(self) -> str:
        """格式化任务列表"""
        tasks = self.list_tasks()
        if not tasks:
            return "（无长任务，使用 'long-task plan <template>' 创建）"

        lines = [f"=== 长任务列表（{len(tasks)}个）==="]
        lines.append(f"{'ID':<10} {'名称':<25} {'状态':<10} {'进度':>8} {'子目标':>8}")
        lines.append("-" * 70)
        for t in tasks:
            state_name = self.STATE_NAMES.get(t["state"], t["state"])
            subgoal_count = len(t["subgoals"])
            lines.append(
                f"{t['task_id']:<10} {t['name']:<25} {state_name:<10} "
                f"{t['progress']*100:>7.0f}% {subgoal_count:>8}"
            )
        return "\n".join(lines)

    def format_status(self, task_id: str) -> str:
        """格式化任务状态"""
        task = self.get_task_status(task_id)
        if task is None:
            return f"任务 '{task_id}' 不存在"

        state_name = self.STATE_NAMES.get(task["state"], task["state"])
        lines = [f"=== 长任务详情: {task['name']} ==="]
        lines.append(f"  ID: {task['task_id']}")
        lines.append(f"  名称: {task['name']}")
        lines.append(f"  描述: {task['description'] or '（无）'}")
        lines.append(f"  状态: {state_name} ({task['state']})")
        lines.append(f"  进度: {task['progress']*100:.0f}%")
        lines.append(f"  当前子目标索引: {task['current_subgoal_idx']}")
        lines.append(f"  创建时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task['created_at']))}")
        if task.get("started_at"):
            lines.append(f"  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task['started_at']))}")
        if task.get("completed_at"):
            lines.append(f"  完成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task['completed_at']))}")

        lines.append(f"\n  子目标列表（{len(task['subgoals'])}个）:")
        for i, sg in enumerate(task["subgoals"]):
            sg_state = self.SUBGOAL_STATE_NAMES.get(sg["state"], sg["state"])
            marker = ">" if i == task["current_subgoal_idx"] else " "
            lines.append(f"    {marker} [{i+1}] {sg['name']} ({sg_state}) - {sg['description']}")

        return "\n".join(lines)

    def format_stats(self) -> str:
        """格式化统计"""
        stats = self.get_stats()
        lines = ["=== 长任务统计 ==="]
        lines.append(f"  总任务数: {stats['total_tasks']}")
        lines.append(f"  运行中: {stats['running']}")
        lines.append(f"  已完成: {stats['completed']}")
        lines.append(f"  已失败: {stats['failed']}")
        lines.append(f"  可用模板: {', '.join(stats['templates'])}")
        return "\n".join(lines)

    def format_plan_result(self, result: Dict[str, Any]) -> str:
        """格式化规划结果"""
        lines = ["=== 长任务规划成功 ==="]
        lines.append(f"  任务ID: {result['task_id']}")
        lines.append(f"  名称: {result['name']}")
        lines.append(f"  描述: {result['description'] or '（无）'}")
        lines.append(f"  子目标数: {len(result['subgoals'])}")
        lines.append(f"  状态: {self.STATE_NAMES.get(result['state'], result['state'])}")
        return "\n".join(lines)

    def format_advance_result(self, result: Dict[str, Any]) -> str:
        """格式化推进结果"""
        lines = ["=== 任务推进结果 ==="]
        if result.get("completed"):
            lines.append(f"  状态: 任务已完成")
            lines.append(f"  进度: {result.get('progress', 1.0)*100:.0f}%")
        elif result.get("failed"):
            lines.append(f"  状态: 子目标失败")
            lines.append(f"  错误: {result.get('error', '未知')}")
        elif result.get("retrying"):
            lines.append(f"  状态: 重试中（第{result.get('retry_count', 0)}次）")
            lines.append(f"  错误: {result.get('error', '未知')}")
        elif result.get("skipped"):
            lines.append(f"  状态: 跳过")
            lines.append(f"  原因: {result.get('reason', '未知')}")
        elif result.get("error"):
            lines.append(f"  状态: 错误")
            lines.append(f"  错误: {result['error']}")
        else:
            lines.append(f"  状态: 子目标完成")
            lines.append(f"  进度: {result.get('progress', 0)*100:.0f}%")
            if result.get("next_subgoal"):
                lines.append(f"  下一个子目标: {result['next_subgoal']}")
        return "\n".join(lines)

    def format_templates(self) -> str:
        """格式化可用模板列表"""
        templates = self.get_available_templates()
        template_desc = {
            "system_optimization": "系统优化（监控→分析→优化→验证）",
            "continuous_monitoring": "持续监控（多轮监控循环）",
            "fault_recovery": "故障恢复（检测→清理→恢复→验证）",
            "capability_enhancement": "能力提升（评估→学习→进化→验证）",
        }
        lines = ["=== 可用长任务模板 ==="]
        for t in templates:
            desc = template_desc.get(t, "")
            lines.append(f"  {t:<30} {desc}")
        return "\n".join(lines)
