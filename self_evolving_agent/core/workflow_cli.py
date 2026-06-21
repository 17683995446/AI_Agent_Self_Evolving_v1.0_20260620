"""
U31：工作流管理 —— 工作流CLI辅助

第一性原理：
- 工作流引擎是任务分解与执行的枢纽，连接目标分解与多Agent执行
- 用户无法通过CLI提交目标、查询工作流状态、查看引擎统计
- 工作流管理应轻量级，直接访问WorkflowEngine
- 支持提交/状态/统计/列表全生命周期

设计：
1. WorkflowCliHelper：工作流CLI辅助器
   - 封装WorkflowEngine的CLI访问
   - 格式化工作流状态/统计/列表
   - 支持submit/status/stats/list
2. 复用WorkflowEngine的核心能力，不重复造轮子
"""

import time
from typing import Dict, List, Optional, Any


class WorkflowCliHelper:
    """工作流CLI辅助器

    轻量级访问WorkflowEngine，不启动完整Agent
    """

    # 工作流状态到中文名的映射
    STATE_NAMES = {
        "pending": "待执行",
        "running": "运行中",
        "completed": "已完成",
        "failed": "已失败",
        "partial": "部分完成",
    }

    # 任务状态到中文名的映射
    TASK_STATE_NAMES = {
        "pending": "待执行",
        "running": "执行中",
        "success": "成功",
        "failed": "失败",
        "retrying": "重试中",
        "skipped": "已跳过",
    }

    # 目标类型到描述的映射
    GOAL_TYPE_DESC = {
        "monitor_and_alert": "监控并告警",
        "optimize_resource": "资源优化",
        "clean_and_recover": "清理与恢复",
        "analyze_and_decide": "分析与决策",
        "custom": "自定义",
    }

    def __init__(self):
        from core.workflow_engine import WorkflowEngine
        self.engine = WorkflowEngine()

    def submit_goal(self, goal_type: str, description: str = "",
                    params: Optional[Dict[str, Any]] = None) -> str:
        """提交目标工作流

        Args:
            goal_type: 目标类型
            description: 描述
            params: 参数

        Returns:
            工作流ID
        """
        from core.workflow_engine import Workflow, WorkflowTask
        workflow = Workflow(name=description or f"goal_{goal_type}")
        # 创建单任务工作流
        task = WorkflowTask(
            name=goal_type,
            task_type=goal_type,
            payload=params or {},
        )
        workflow.add_task(task)
        return self.engine.submit_workflow(workflow)

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """获取工作流状态

        Args:
            workflow_id: 工作流ID

        Returns:
            工作流信息字典，不存在返回None
        """
        workflow = self.engine._workflows.get(workflow_id)
        if workflow is None:
            return None
        return workflow.to_dict()

    def list_workflows(self) -> List[Dict[str, Any]]:
        """列出所有工作流"""
        workflows = list(self.engine._workflows.values())
        return [w.to_dict() for w in workflows]

    def get_stats(self) -> Dict[str, Any]:
        """获取工作流引擎统计"""
        return self.engine.get_stats()

    def get_available_goal_types(self) -> List[str]:
        """获取可用目标类型"""
        return list(self.GOAL_TYPE_DESC.keys())

    # ---------- 格式化方法 ----------

    def format_list(self) -> str:
        """格式化工作流列表"""
        workflows = self.list_workflows()
        if not workflows:
            return "（无工作流，使用 'workflow submit <goal_type>' 创建）"

        lines = [f"=== 工作流列表（{len(workflows)}个）==="]
        lines.append(f"{'ID':<10} {'名称':<25} {'状态':<10} {'任务数':>8} {'创建时间'}")
        lines.append("-" * 80)
        for w in workflows:
            state_name = self.STATE_NAMES.get(w["state"], w["state"])
            task_count = len(w["tasks"])
            created_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(w["created_at"]))
            lines.append(
                f"{w['workflow_id']:<10} {w['name']:<25} {state_name:<10} "
                f"{task_count:>8} {created_str}"
            )
        return "\n".join(lines)

    def format_status(self, workflow_id: str) -> str:
        """格式化工作流状态"""
        workflow = self.get_workflow_status(workflow_id)
        if workflow is None:
            return f"工作流 '{workflow_id}' 不存在"

        state_name = self.STATE_NAMES.get(workflow["state"], workflow["state"])
        lines = [f"=== 工作流详情: {workflow['name']} ==="]
        lines.append(f"  ID: {workflow['workflow_id']}")
        lines.append(f"  名称: {workflow['name']}")
        lines.append(f"  状态: {state_name} ({workflow['state']})")
        lines.append(f"  创建时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(workflow['created_at']))}")
        if workflow.get("started_at"):
            lines.append(f"  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(workflow['started_at']))}")
        if workflow.get("completed_at"):
            lines.append(f"  完成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(workflow['completed_at']))}")

        lines.append(f"\n  任务列表（{len(workflow['tasks'])}个）:")
        for tid, task in workflow["tasks"].items():
            task_state = self.TASK_STATE_NAMES.get(task["state"], task["state"])
            lines.append(f"    - [{tid}] {task['name']} ({task_state})")
            if task.get("error"):
                lines.append(f"      错误: {task['error']}")

        return "\n".join(lines)

    def format_stats(self) -> str:
        """格式化统计"""
        stats = self.get_stats()
        lines = ["=== 工作流引擎统计 ==="]
        lines.append(f"  总工作流数: {stats.get('total_workflows', 0)}")
        lines.append(f"  运行中: {stats.get('running', 0)}")
        lines.append(f"  已完成: {stats.get('completed', 0)}")
        lines.append(f"  已失败: {stats.get('failed', 0)}")
        lines.append(f"  最大并发: {stats.get('max_workers', 0)}")
        return "\n".join(lines)

    def format_submit_result(self, workflow_id: str, goal_type: str) -> str:
        """格式化提交结果"""
        goal_desc = self.GOAL_TYPE_DESC.get(goal_type, goal_type)
        lines = ["=== 工作流提交成功 ==="]
        lines.append(f"  工作流ID: {workflow_id}")
        lines.append(f"  目标类型: {goal_type} ({goal_desc})")
        return "\n".join(lines)

    def format_goal_types(self) -> str:
        """格式化可用目标类型"""
        lines = ["=== 可用目标类型 ==="]
        for gt, desc in self.GOAL_TYPE_DESC.items():
            lines.append(f"  {gt:<25} {desc}")
        return "\n".join(lines)
