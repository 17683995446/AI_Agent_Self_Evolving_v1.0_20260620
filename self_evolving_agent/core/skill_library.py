"""
M3：技能库 —— 让Agent自主学会增强自己的技能

第一性原理：
1. 技能=可复用的动作序列/策略，从成功经验中提炼
2. 增强技能=评估技能效果+保留有效技能+淘汰无效技能
3. 技能应可检索、可复用、可进化
4. 技能学习是持续的：每次任务执行都是学习机会

核心能力：
- Skill：技能数据类（名称/步骤/成功率/来源）
- SkillLibrary：技能注册、检索、评估、淘汰
- 从成功工作流中自动提炼技能
- 技能效果追踪与自动增强
"""

import os
import sys
import time
import uuid
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


class SkillStatus(Enum):
    """技能状态"""
    DRAFT = "draft"          # 草稿（未验证）
    ACTIVE = "active"        # 活跃（验证有效）
    DEPRECATED = "deprecated"  # 废弃（效果差）
    EVOLVING = "evolving"    # 进化中（正在增强）


@dataclass
class Skill:
    """技能 —— 可复用的动作序列

    Attributes:
        skill_id: 唯一ID
        name: 技能名称
        description: 描述
        task_type: 适用任务类型
        steps: 执行步骤列表（每步是动作描述）
        params: 默认参数
        success_count: 成功次数
        failure_count: 失败次数
        total_elapsed: 总耗时
        status: 技能状态
        learned_from: 学习来源（工作流ID/经验ID）
        created_at: 创建时间
        last_used_at: 最后使用时间
        version: 版本号（每次增强+1）
    """
    skill_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    task_type: str = ""
    steps: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    success_count: int = 0
    failure_count: int = 0
    total_elapsed: float = 0.0
    status: SkillStatus = SkillStatus.DRAFT
    learned_from: str = ""
    created_at: float = field(default_factory=time.time)
    last_used_at: Optional[float] = None
    version: int = 1

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def avg_elapsed(self) -> float:
        total = self.success_count + self.failure_count
        return self.total_elapsed / total if total > 0 else 0.0

    @property
    def usage_count(self) -> int:
        return self.success_count + self.failure_count

    def record_use(self, success: bool, elapsed: float) -> None:
        """记录一次使用"""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.total_elapsed += elapsed
        self.last_used_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "steps": self.steps,
            "params": self.params,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 4),
            "avg_elapsed": round(self.avg_elapsed, 6),
            "usage_count": self.usage_count,
            "status": self.status.value,
            "learned_from": self.learned_from,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "version": self.version,
        }


class SkillLibrary:
    """技能库 —— 管理Agent的所有技能

    设计要点：
    - 技能按task_type索引，快速检索
    - 成功率低于阈值的技能自动降级
    - 从成功工作流中提炼新技能
    - 技能可进化（版本递增）
    """

    def __init__(self, deprecate_threshold: float = 0.3,
                 min_uses_before_deprecate: int = 3):
        self._skills: Dict[str, Skill] = {}
        self._by_task_type: Dict[str, List[str]] = {}  # task_type -> [skill_id]
        self._deprecate_threshold = deprecate_threshold
        self._min_uses = min_uses_before_deprecate
        self._skill_learned_count = 0
        self._skill_evolved_count = 0
        logger.info("SKILL_LIB", "技能库初始化", module="skill_library")

    def register(self, skill: Skill) -> str:
        """注册新技能"""
        self._skills[skill.skill_id] = skill
        if skill.task_type not in self._by_task_type:
            self._by_task_type[skill.task_type] = []
        self._by_task_type[skill.task_type].append(skill.skill_id)
        logger.info("SKILL_LIB", f"技能注册: {skill.name} [{skill.task_type}]",
                    module="skill_library")
        return skill.skill_id

    def get(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def find_by_task_type(self, task_type: str,
                          active_only: bool = True) -> List[Skill]:
        """按任务类型查找技能"""
        ids = self._by_task_type.get(task_type, [])
        skills = [self._skills[i] for i in ids if i in self._skills]
        if active_only:
            skills = [s for s in skills if s.status in (SkillStatus.ACTIVE, SkillStatus.EVOLVING)]
        # 按成功率降序
        skills.sort(key=lambda s: s.success_rate, reverse=True)
        return skills

    def find_best_skill(self, task_type: str) -> Optional[Skill]:
        """找到最适合某任务类型的技能"""
        skills = self.find_by_task_type(task_type, active_only=True)
        return skills[0] if skills else None

    def record_use(self, skill_id: str, success: bool, elapsed: float) -> None:
        """记录技能使用结果，并自动评估是否降级"""
        skill = self._skills.get(skill_id)
        if skill is None:
            return
        skill.record_use(success, elapsed)

        # 自动评估：使用次数足够且成功率低则降级
        if (skill.usage_count >= self._min_uses and
                skill.success_rate < self._deprecate_threshold and
                skill.status != SkillStatus.DEPRECATED):
            skill.status = SkillStatus.DEPRECATED
            logger.warning("SKILL_LIB",
                           f"技能降级: {skill.name} 成功率={skill.success_rate:.2f}",
                           module="skill_library")

        # 成功次数足够则激活
        if (skill.status == SkillStatus.DRAFT and
                skill.success_count >= 2):
            skill.status = SkillStatus.ACTIVE
            logger.info("SKILL_LIB", f"技能激活: {skill.name}", module="skill_library")

    def evolve_skill(self, skill_id: str, new_steps: List[str],
                     new_params: Optional[Dict[str, Any]] = None) -> Optional[Skill]:
        """进化技能（增强）—— 保留原技能，创建新版本"""
        skill = self._skills.get(skill_id)
        if skill is None:
            return None
        skill.version += 1
        skill.steps = new_steps
        if new_params:
            skill.params.update(new_params)
        skill.status = SkillStatus.EVOLVING
        # 重置统计以重新评估
        skill.success_count = 0
        skill.failure_count = 0
        skill.total_elapsed = 0.0
        self._skill_evolved_count += 1
        logger.info("SKILL_LIB", f"技能进化: {skill.name} v{skill.version}",
                    module="skill_library")
        return skill

    def learn_from_workflow(self, workflow_id: str, task_type: str,
                            steps: List[str], success: bool,
                            elapsed: float, params: Optional[Dict] = None) -> Skill:
        """从工作流执行中提炼技能

        第一性原理：成功的工作流模式应被提炼为可复用技能
        """
        skill = Skill(
            name=f"skill-{task_type}-{self._skill_learned_count + 1}",
            description=f"从工作流 {workflow_id} 提炼的 {task_type} 技能",
            task_type=task_type,
            steps=steps,
            params=params or {},
            status=SkillStatus.DRAFT if success else SkillStatus.DEPRECATED,
            learned_from=workflow_id,
        )
        if success:
            skill.success_count = 1
            skill.total_elapsed = elapsed
        else:
            skill.failure_count = 1
        self.register(skill)
        self._skill_learned_count += 1
        logger.info("SKILL_LIB", f"从工作流学习技能: {skill.name} (success={success})",
                    module="skill_library")
        return skill

    def list_skills(self, status: Optional[SkillStatus] = None) -> List[Skill]:
        """列出技能"""
        skills = list(self._skills.values())
        if status is not None:
            skills = [s for s in skills if s.status == status]
        return skills

    def get_stats(self) -> Dict[str, Any]:
        active = sum(1 for s in self._skills.values() if s.status == SkillStatus.ACTIVE)
        deprecated = sum(1 for s in self._skills.values() if s.status == SkillStatus.DEPRECATED)
        draft = sum(1 for s in self._skills.values() if s.status == SkillStatus.DRAFT)
        evolving = sum(1 for s in self._skills.values() if s.status == SkillStatus.EVOLVING)
        return {
            "total_skills": len(self._skills),
            "active": active,
            "deprecated": deprecated,
            "draft": draft,
            "evolving": evolving,
            "learned_count": self._skill_learned_count,
            "evolved_count": self._skill_evolved_count,
            "task_types": list(self._by_task_type.keys()),
        }

    def save(self) -> Dict[str, Any]:
        """保存技能库状态"""
        return {
            "skills": [s.to_dict() for s in self._skills.values()],
            "stats": self.get_stats(),
        }

    def clear(self) -> None:
        """清空技能库"""
        self._skills.clear()
        self._by_task_type.clear()
        self._skill_learned_count = 0
        self._skill_evolved_count = 0


if __name__ == "__main__":
    # 自测
    lib = SkillLibrary()
    s = Skill(name="test_skill", task_type="clean", steps=["step1", "step2"])
    lib.register(s)
    lib.record_use(s.skill_id, True, 0.5)
    lib.record_use(s.skill_id, True, 0.4)
    print("技能:", lib.get(s.skill_id).to_dict())
    print("统计:", lib.get_stats())
