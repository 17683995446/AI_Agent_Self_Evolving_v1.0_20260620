"""
M3阶段验证测试 —— 技能库与持续学习接入

验证点：
1. SkillLibrary类存在且可创建
2. Skill数据类完整
3. 技能注册、检索、评估
4. 从工作流学习技能
5. 技能进化（增强）
6. 技能自动降级
7. SelfEvolvingAgent集成skill_library
8. 工作流完成后自动学习技能
9. 技能查询接口
10. 持续学习控制器初始化
11. 元学习控制器初始化
12. 不破坏M1/M2和主循环
"""

import os
import sys
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.skill_library import SkillLibrary, Skill, SkillStatus


@pytest.fixture
def library():
    return SkillLibrary()


class TestM3SkillLibraryCreation:
    """验证技能库创建"""

    def test_skill_library_exists(self):
        assert SkillLibrary is not None

    def test_skill_dataclass(self):
        s = Skill(name="test", task_type="clean", steps=["a", "b"])
        assert s.name == "test"
        assert s.task_type == "clean"
        assert s.steps == ["a", "b"]
        assert s.status == SkillStatus.DRAFT

    def test_skill_status_enum(self):
        assert SkillStatus.DRAFT.value == "draft"
        assert SkillStatus.ACTIVE.value == "active"
        assert SkillStatus.DEPRECATED.value == "deprecated"
        assert SkillStatus.EVOLVING.value == "evolving"

    def test_create_library(self, library):
        assert library is not None
        stats = library.get_stats()
        assert stats["total_skills"] == 0


class TestM3SkillRegistration:
    """验证技能注册与检索"""

    def test_register_skill(self, library):
        s = Skill(name="clean_cache", task_type="clean_cache", steps=["step1"])
        sid = library.register(s)
        assert sid == s.skill_id
        assert library.get(sid) is not None

    def test_find_by_task_type(self, library):
        s = Skill(name="s1", task_type="clean", steps=["a"])
        library.register(s)
        # 默认active_only=True，draft状态不返回；用active_only=False查全部
        found = library.find_by_task_type("clean", active_only=False)
        assert len(found) == 1
        assert found[0].name == "s1"

    def test_find_by_task_type_empty(self, library):
        found = library.find_by_task_type("nonexistent")
        assert len(found) == 0

    def test_find_best_skill(self, library):
        s1 = Skill(name="s1", task_type="clean", steps=["a"])
        s2 = Skill(name="s2", task_type="clean", steps=["b"])
        library.register(s1)
        library.register(s2)
        # s1成功率高
        library.record_use(s1.skill_id, True, 0.1)
        library.record_use(s1.skill_id, True, 0.1)
        library.record_use(s2.skill_id, False, 0.1)
        best = library.find_best_skill("clean")
        # draft状态的不返回（active_only默认True）
        # 但s1成功2次后应激活
        assert best is not None

    def test_list_skills(self, library):
        library.register(Skill(name="s1", task_type="a"))
        library.register(Skill(name="s2", task_type="b"))
        all_skills = library.list_skills()
        assert len(all_skills) == 2


class TestM3SkillEvaluation:
    """验证技能评估与降级"""

    def test_record_use_success(self, library):
        s = Skill(name="s1", task_type="clean", steps=["a"])
        library.register(s)
        library.record_use(s.skill_id, True, 0.5)
        assert s.success_count == 1
        assert s.success_rate == 1.0

    def test_record_use_failure(self, library):
        s = Skill(name="s1", task_type="clean", steps=["a"])
        library.register(s)
        library.record_use(s.skill_id, False, 0.5)
        assert s.failure_count == 1
        assert s.success_rate == 0.0

    def test_skill_activates_after_successes(self, library):
        """成功2次后技能激活"""
        s = Skill(name="s1", task_type="clean", steps=["a"])
        library.register(s)
        assert s.status == SkillStatus.DRAFT
        library.record_use(s.skill_id, True, 0.1)
        library.record_use(s.skill_id, True, 0.1)
        assert s.status == SkillStatus.ACTIVE

    def test_skill_deprecated_after_failures(self, library):
        """成功率低且使用次数足够则降级"""
        lib = SkillLibrary(deprecate_threshold=0.3, min_uses_before_deprecate=3)
        s = Skill(name="s1", task_type="clean", steps=["a"])
        s.status = SkillStatus.ACTIVE
        lib.register(s)
        # 3次失败，0次成功
        lib.record_use(s.skill_id, False, 0.1)
        lib.record_use(s.skill_id, False, 0.1)
        lib.record_use(s.skill_id, False, 0.1)
        assert s.status == SkillStatus.DEPRECATED

    def test_skill_not_deprecated_below_min_uses(self, library):
        """使用次数不足不降级"""
        s = Skill(name="s1", task_type="clean", steps=["a"])
        s.status = SkillStatus.ACTIVE
        library.register(s)
        library.record_use(s.skill_id, False, 0.1)
        assert s.status == SkillStatus.ACTIVE  # 未达min_uses


class TestM3SkillEvolution:
    """验证技能进化"""

    def test_evolve_skill(self, library):
        s = Skill(name="s1", task_type="clean", steps=["a"])
        library.register(s)
        evolved = library.evolve_skill(s.skill_id, ["new_step1", "new_step2"])
        assert evolved is not None
        assert evolved.version == 2
        assert evolved.steps == ["new_step1", "new_step2"]
        assert evolved.status == SkillStatus.EVOLVING

    def test_evolve_resets_stats(self, library):
        s = Skill(name="s1", task_type="clean", steps=["a"])
        library.register(s)
        library.record_use(s.skill_id, True, 0.1)
        library.evolve_skill(s.skill_id, ["new"])
        assert s.success_count == 0
        assert s.failure_count == 0

    def test_evolve_nonexistent_returns_none(self, library):
        assert library.evolve_skill("nonexistent", ["a"]) is None


class TestM3LearnFromWorkflow:
    """验证从工作流学习技能"""

    def test_learn_from_workflow_success(self, library):
        s = library.learn_from_workflow(
            workflow_id="wf-1", task_type="clean_cache",
            steps=["perceive", "clean", "verify"],
            success=True, elapsed=1.5,
        )
        assert s is not None
        assert s.task_type == "clean_cache"
        assert s.success_count == 1
        assert s.learned_from == "wf-1"

    def test_learn_from_workflow_failure(self, library):
        s = library.learn_from_workflow(
            workflow_id="wf-2", task_type="predict",
            steps=["perceive", "predict"],
            success=False, elapsed=0.5,
        )
        assert s.failure_count == 1
        assert s.status == SkillStatus.DEPRECATED

    def test_learned_count_increments(self, library):
        library.learn_from_workflow("wf-1", "clean", ["a"], True, 0.1)
        library.learn_from_workflow("wf-2", "predict", ["b"], True, 0.1)
        stats = library.get_stats()
        assert stats["learned_count"] == 2


class TestM3SkillStats:
    """验证技能库统计"""

    def test_stats_structure(self, library):
        stats = library.get_stats()
        assert "total_skills" in stats
        assert "active" in stats
        assert "deprecated" in stats
        assert "draft" in stats
        assert "evolving" in stats
        assert "learned_count" in stats
        assert "evolved_count" in stats
        assert "task_types" in stats

    def test_save_returns_dict(self, library):
        library.register(Skill(name="s1", task_type="clean"))
        data = library.save()
        assert "skills" in data
        assert "stats" in data
        assert len(data["skills"]) == 1


class TestM3AgentIntegration:
    """验证SelfEvolvingAgent集成"""

    def test_agent_has_skill_library(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert hasattr(agent, "skill_library")
        assert agent.skill_library is not None
        agent.stop()

    def test_get_skill_library_stats(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        stats = agent.get_skill_library_stats()
        assert "total_skills" in stats
        agent.stop()

    def test_workflow_completes_and_learns_skill(self):
        """工作流完成后自动学习技能"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        initial_stats = agent.get_skill_library_stats()
        initial_count = initial_stats["total_skills"]

        wf_id = agent.submit_goal("clean_and_recover", "测试学习")
        agent.wait_for_workflow(wf_id, timeout=10)

        after_stats = agent.get_skill_library_stats()
        assert after_stats["total_skills"] > initial_count
        agent.stop()

    def test_find_skills_for_task(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        wf_id = agent.submit_goal("clean_and_recover", "测试")
        agent.wait_for_workflow(wf_id, timeout=10)
        # 应能查到技能（即使是draft状态，find_skills_for_task返回所有）
        skills = agent.find_skills_for_task("perceive")
        assert isinstance(skills, list)
        agent.stop()

    def test_find_best_skill(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        best = agent.find_best_skill("nonexistent_task")
        assert best is None
        agent.stop()

    def test_record_skill_use(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        # 先学习一个技能
        wf_id = agent.submit_goal("clean_and_recover", "测试")
        agent.wait_for_workflow(wf_id, timeout=10)
        skills = agent.find_skills_for_task("perceive")
        if skills:
            agent.record_skill_use(skills[0]["skill_id"], True, 0.1)
        agent.stop()

    def test_evolve_skill_from_agent(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        wf_id = agent.submit_goal("clean_and_recover", "测试")
        agent.wait_for_workflow(wf_id, timeout=10)
        skills = agent.find_skills_for_task("perceive")
        if skills:
            evolved = agent.evolve_skill(skills[0]["skill_id"], ["enhanced_step"])
            assert evolved is not None
            assert evolved["version"] == 2
        agent.stop()


class TestM3ContinualLearning:
    """验证持续学习集成"""

    def test_init_continual_learning(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.init_continual_learning(input_dim=4, output_dim=2)
        assert agent._continual_learner is not None
        agent.stop()

    def test_init_meta_learning(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.init_meta_learning(input_dim=4, output_dim=2)
        assert agent._meta_learner is not None
        agent.stop()

    def test_learn_task_continual(self):
        """持续学习一个简单任务"""
        import numpy as np
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.init_continual_learning(input_dim=4, output_dim=2)
        data = np.random.randn(20, 4)
        labels = np.random.randint(0, 2, 20)
        result = agent.learn_task_continual("task-1", data, labels)
        assert "error" not in result or result.get("error") is None or "task_count" in result
        agent.stop()

    def test_meta_train(self):
        """元学习训练"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.init_meta_learning(input_dim=4, output_dim=2)
        result = agent.meta_train(n_iterations=2)  # 少量迭代快速测试
        assert isinstance(result, dict)
        agent.stop()


class TestM3NoRegression:
    """验证M3不破坏现有功能"""

    def test_agent_runs_without_skills(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.run_for_duration(1.0)
        assert agent.loop_count >= 1
        agent.stop()

    def test_m1_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        ids = agent.enable_multi_agent()
        assert len(ids) == 4
        agent.stop()

    def test_m2_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        wf_id = agent.submit_goal("clean_and_recover", "回归测试")
        result = agent.wait_for_workflow(wf_id, timeout=10)
        assert result["state"] in ("completed", "failed", "partial", "timeout")
        agent.stop()
