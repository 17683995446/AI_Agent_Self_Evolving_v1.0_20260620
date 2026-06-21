"""
U29：技能库管理 —— 技能CLI辅助

第一性原理：
- 系统名称含"增强技能"，但用户无法通过CLI查看、检索、进化技能
- 技能库是"自驱进化"的核心载体，必须可达
- 技能管理应轻量级，直接访问SkillLibrary，不启动完整Agent
- 技能数据应可格式化展示，支持按状态/任务类型过滤

设计：
1. SkillCliHelper：技能CLI辅助器
   - 封装SkillLibrary的CLI访问
   - 格式化技能列表/详情/统计
   - 支持按任务类型搜索、进化技能、清空技能库
2. 复用SkillLibrary的核心能力，不重复造轮子
"""

import time
from typing import Dict, List, Optional, Any


class SkillCliHelper:
    """技能CLI辅助器

    轻量级访问SkillLibrary，不启动完整Agent
    """

    # 技能状态到中文名的映射
    STATUS_NAMES = {
        "draft": "草稿",
        "active": "活跃",
        "deprecated": "已废弃",
        "evolving": "进化中",
    }

    def __init__(self):
        from core.skill_library import SkillLibrary
        self.library = SkillLibrary()

    def list_skills(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出技能

        Args:
            status: 状态过滤（draft/active/deprecated/evolving），None返回全部

        Returns:
            技能字典列表
        """
        from core.skill_library import SkillStatus
        status_enum = None
        if status is not None:
            status_map = {
                "draft": SkillStatus.DRAFT,
                "active": SkillStatus.ACTIVE,
                "deprecated": SkillStatus.DEPRECATED,
                "evolving": SkillStatus.EVOLVING,
            }
            status_enum = status_map.get(status.lower())
            if status_enum is None:
                return []

        skills = self.library.list_skills(status_enum)
        return [s.to_dict() for s in skills]

    def get_skill_info(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """获取技能详情

        Args:
            skill_id: 技能ID

        Returns:
            技能信息字典，不存在返回None
        """
        skill = self.library.get(skill_id)
        if skill is None:
            return None
        info = skill.to_dict()
        # 添加额外信息
        info["status_name"] = self.STATUS_NAMES.get(skill.status.value, skill.status.value)
        if skill.last_used_at:
            info["last_used_str"] = time.strftime("%Y-%m-%d %H:%M:%S",
                                                    time.localtime(skill.last_used_at))
        else:
            info["last_used_str"] = "（未使用）"
        info["created_str"] = time.strftime("%Y-%m-%d %H:%M:%S",
                                              time.localtime(skill.created_at))
        return info

    def search_skills(self, task_type: str) -> List[Dict[str, Any]]:
        """按任务类型搜索技能

        Args:
            task_type: 任务类型

        Returns:
            匹配的技能列表（按成功率降序）
        """
        skills = self.library.find_by_task_type(task_type, active_only=False)
        return [s.to_dict() for s in skills]

    def find_best_skill(self, task_type: str) -> Optional[Dict[str, Any]]:
        """找到最适合某任务类型的技能

        Args:
            task_type: 任务类型

        Returns:
            最佳技能信息，无返回None
        """
        skill = self.library.find_best_skill(task_type)
        if skill is None:
            return None
        return skill.to_dict()

    def evolve_skill(self, skill_id: str, steps: List[str],
                     params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """进化技能

        Args:
            skill_id: 技能ID
            steps: 新的执行步骤
            params: 新的参数（可选）

        Returns:
            进化后的技能信息，失败返回None
        """
        skill = self.library.evolve_skill(skill_id, steps, params)
        if skill is None:
            return None
        return skill.to_dict()

    def get_stats(self) -> Dict[str, Any]:
        """获取技能库统计"""
        return self.library.get_stats()

    def clear(self) -> None:
        """清空技能库"""
        self.library.clear()

    def register_skill(self, name: str, task_type: str, steps: List[str],
                       description: str = "", params: Optional[Dict[str, Any]] = None) -> str:
        """注册新技能

        Args:
            name: 技能名称
            task_type: 任务类型
            steps: 执行步骤
            description: 描述
            params: 参数

        Returns:
            技能ID
        """
        from core.skill_library import Skill
        skill = Skill(
            name=name,
            description=description,
            task_type=task_type,
            steps=steps,
            params=params or {},
        )
        return self.library.register(skill)

    # ---------- 格式化方法 ----------

    def format_list(self, status: Optional[str] = None) -> str:
        """格式化技能列表"""
        skills = self.list_skills(status)
        if not skills:
            return "（无技能）"

        status_filter = f"，状态={status}" if status else ""
        lines = [f"=== 技能列表（{len(skills)}个{status_filter}）==="]
        lines.append(f"{'ID':<10} {'名称':<20} {'任务类型':<15} {'状态':<10} {'成功率':>8} {'使用次数':>8} {'版本':>6}")
        lines.append("-" * 90)
        for s in skills:
            status_name = self.STATUS_NAMES.get(s["status"], s["status"])
            lines.append(
                f"{s['skill_id']:<10} {s['name']:<20} {s['task_type']:<15} "
                f"{status_name:<10} {s['success_rate']*100:>7.1f}% {s['usage_count']:>8} {s['version']:>6}"
            )
        return "\n".join(lines)

    def format_info(self, skill_id: str) -> str:
        """格式化技能详情"""
        info = self.get_skill_info(skill_id)
        if info is None:
            return f"技能 '{skill_id}' 不存在"

        lines = [f"=== 技能详情: {info['name']} ==="]
        lines.append(f"  ID: {info['skill_id']}")
        lines.append(f"  名称: {info['name']}")
        lines.append(f"  描述: {info['description'] or '（无）'}")
        lines.append(f"  任务类型: {info['task_type']}")
        lines.append(f"  状态: {info['status_name']} ({info['status']})")
        lines.append(f"  版本: v{info['version']}")
        lines.append(f"  成功次数: {info['success_count']}")
        lines.append(f"  失败次数: {info['failure_count']}")
        lines.append(f"  成功率: {info['success_rate']*100:.1f}%")
        lines.append(f"  使用次数: {info['usage_count']}")
        lines.append(f"  平均耗时: {info['avg_elapsed']*1000:.1f} ms")
        lines.append(f"  创建时间: {info['created_str']}")
        lines.append(f"  最后使用: {info['last_used_str']}")
        lines.append(f"  学习来源: {info['learned_from'] or '（手动创建）'}")
        lines.append(f"  执行步骤:")
        for i, step in enumerate(info['steps'], 1):
            lines.append(f"    {i}. {step}")
        if info['params']:
            lines.append(f"  参数: {info['params']}")
        return "\n".join(lines)

    def format_search(self, task_type: str) -> str:
        """格式化搜索结果"""
        skills = self.search_skills(task_type)
        if not skills:
            return f"（未找到任务类型 '{task_type}' 的技能）"

        lines = [f"=== 搜索结果: task_type='{task_type}'（{len(skills)}个）==="]
        lines.append(f"{'ID':<10} {'名称':<20} {'状态':<10} {'成功率':>8} {'版本':>6}")
        lines.append("-" * 60)
        for s in skills:
            status_name = self.STATUS_NAMES.get(s["status"], s["status"])
            lines.append(
                f"{s['skill_id']:<10} {s['name']:<20} {status_name:<10} "
                f"{s['success_rate']*100:>7.1f}% {s['version']:>6}"
            )
        return "\n".join(lines)

    def format_stats(self) -> str:
        """格式化统计"""
        stats = self.get_stats()
        lines = ["=== 技能库统计 ==="]
        lines.append(f"  总技能数: {stats['total_skills']}")
        lines.append(f"  活跃: {stats['active']}")
        lines.append(f"  草稿: {stats['draft']}")
        lines.append(f"  进化中: {stats['evolving']}")
        lines.append(f"  已废弃: {stats['deprecated']}")
        lines.append(f"  学习次数: {stats['learned_count']}")
        lines.append(f"  进化次数: {stats['evolved_count']}")
        if stats['task_types']:
            lines.append(f"  任务类型: {', '.join(stats['task_types'])}")
        return "\n".join(lines)

    def format_evolve_result(self, result: Optional[Dict[str, Any]]) -> str:
        """格式化进化结果"""
        if result is None:
            return "进化失败: 技能不存在"
        lines = ["=== 技能进化成功 ==="]
        lines.append(f"  技能: {result['name']}")
        lines.append(f"  新版本: v{result['version']}")
        lines.append(f"  状态: {self.STATUS_NAMES.get(result['status'], result['status'])}")
        lines.append(f"  新步骤数: {len(result['steps'])}")
        return "\n".join(lines)
