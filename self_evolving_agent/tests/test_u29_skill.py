"""
U29：技能库管理 —— 测试

测试覆盖：
1. SkillCliHelper 的所有方法
2. list_skills / get_skill_info / search_skills / evolve_skill / get_stats / clear
3. format_list / format_info / format_search / format_stats / format_evolve_result
4. CLI集成（_cmd_skill 分发）
5. 无回归（U24-U28 仍可用）
"""

import os
import sys
import pytest

# 路径设置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def helper():
    """创建SkillCliHelper，预填充一些技能"""
    from core.skill_cli import SkillCliHelper
    h = SkillCliHelper()
    # 注册一些测试技能
    h.register_skill("clean_skill", "clean", ["step1", "step2"], "清理技能")
    h.register_skill("predict_skill", "predict", ["analyze", "forecast"], "预测技能")
    return h


@pytest.fixture
def empty_helper():
    """创建空的SkillCliHelper"""
    from core.skill_cli import SkillCliHelper
    return SkillCliHelper()


# ==================== TestU29SkillCliHelper ====================

class TestU29SkillCliHelper:
    """测试SkillCliHelper基础功能"""

    def test_helper_initialization(self, empty_helper):
        """测试初始化"""
        assert empty_helper is not None
        assert empty_helper.library is not None

    def test_list_skills_empty(self, empty_helper):
        """测试空技能列表"""
        skills = empty_helper.list_skills()
        assert skills == []

    def test_list_skills_with_data(self, helper):
        """测试有数据的技能列表"""
        skills = helper.list_skills()
        assert len(skills) == 2

    def test_list_skills_by_status(self, helper):
        """测试按状态过滤"""
        # 默认注册的技能是draft状态
        skills = helper.list_skills("draft")
        assert len(skills) == 2
        skills = helper.list_skills("active")
        assert len(skills) == 0

    def test_list_skills_invalid_status(self, helper):
        """测试无效状态"""
        skills = helper.list_skills("invalid")
        assert skills == []

    def test_get_skill_info(self, helper):
        """测试获取技能详情"""
        skills = helper.list_skills()
        skill_id = skills[0]["skill_id"]
        info = helper.get_skill_info(skill_id)
        assert info is not None
        assert info["skill_id"] == skill_id
        assert "status_name" in info
        assert "created_str" in info
        assert "last_used_str" in info

    def test_get_skill_info_nonexistent(self, helper):
        """测试获取不存在的技能"""
        info = helper.get_skill_info("nonexistent")
        assert info is None

    def test_search_skills(self, helper):
        """测试按任务类型搜索"""
        skills = helper.search_skills("clean")
        assert len(skills) == 1
        assert skills[0]["task_type"] == "clean"

    def test_search_skills_no_match(self, helper):
        """测试搜索无匹配"""
        skills = helper.search_skills("nonexistent_type")
        assert skills == []

    def test_find_best_skill(self, helper):
        """测试找最佳技能"""
        # 先记录一些使用结果使技能激活
        skills = helper.list_skills()
        skill_id = skills[0]["skill_id"]
        helper.library.record_use(skill_id, True, 0.1)
        helper.library.record_use(skill_id, True, 0.1)

        best = helper.find_best_skill(skills[0]["task_type"])
        assert best is not None
        assert best["task_type"] == skills[0]["task_type"]

    def test_find_best_skill_none(self, helper):
        """测试无最佳技能"""
        best = helper.find_best_skill("nonexistent_type")
        assert best is None

    def test_evolve_skill(self, helper):
        """测试进化技能"""
        skills = helper.list_skills()
        skill_id = skills[0]["skill_id"]
        result = helper.evolve_skill(skill_id, ["new_step1", "new_step2"])
        assert result is not None
        assert result["version"] == 2
        assert result["status"] == "evolving"

    def test_evolve_skill_nonexistent(self, helper):
        """测试进化不存在的技能"""
        result = helper.evolve_skill("nonexistent", ["step1"])
        assert result is None

    def test_evolve_skill_with_params(self, helper):
        """测试带参数进化技能"""
        skills = helper.list_skills()
        skill_id = skills[0]["skill_id"]
        result = helper.evolve_skill(skill_id, ["new_step"], {"key": "value"})
        assert result is not None
        assert "key" in result["params"]

    def test_get_stats(self, helper):
        """测试获取统计"""
        stats = helper.get_stats()
        assert stats["total_skills"] == 2
        assert stats["draft"] == 2
        assert stats["active"] == 0

    def test_clear(self, helper):
        """测试清空技能库"""
        assert helper.get_stats()["total_skills"] == 2
        helper.clear()
        assert helper.get_stats()["total_skills"] == 0

    def test_register_skill(self, empty_helper):
        """测试注册技能"""
        skill_id = empty_helper.register_skill("test", "test_type", ["step1"])
        assert skill_id is not None
        assert len(empty_helper.list_skills()) == 1


# ==================== TestU29FormatList ====================

class TestU29FormatList:
    """测试format_list格式化"""

    def test_format_list_empty(self, empty_helper):
        """测试空列表格式化"""
        result = empty_helper.format_list()
        assert "无技能" in result

    def test_format_list_with_data(self, helper):
        """测试有数据的列表格式化"""
        result = helper.format_list()
        assert "技能列表" in result
        assert "2个" in result

    def test_format_list_with_status(self, helper):
        """测试带状态过滤的列表"""
        result = helper.format_list("draft")
        assert "技能列表" in result
        assert "draft" in result

    def test_format_list_no_match_status(self, helper):
        """测试无匹配状态的列表"""
        result = helper.format_list("active")
        assert "无技能" in result


# ==================== TestU29FormatInfo ====================

class TestU29FormatInfo:
    """测试format_info格式化"""

    def test_format_info(self, helper):
        """测试技能详情格式化"""
        skills = helper.list_skills()
        result = helper.format_info(skills[0]["skill_id"])
        assert "技能详情" in result
        assert "ID" in result
        assert "名称" in result
        assert "执行步骤" in result

    def test_format_info_nonexistent(self, helper):
        """测试不存在的技能详情"""
        result = helper.format_info("nonexistent")
        assert "不存在" in result


# ==================== TestU29FormatSearch ====================

class TestU29FormatSearch:
    """测试format_search格式化"""

    def test_format_search_with_results(self, helper):
        """测试有结果的搜索格式化"""
        result = helper.format_search("clean")
        assert "搜索结果" in result
        assert "clean" in result

    def test_format_search_no_results(self, helper):
        """测试无结果的搜索格式化"""
        result = helper.format_search("nonexistent")
        assert "未找到" in result


# ==================== TestU29FormatStats ====================

class TestU29FormatStats:
    """测试format_stats格式化"""

    def test_format_stats(self, helper):
        """测试统计格式化"""
        result = helper.format_stats()
        assert "技能库统计" in result
        assert "总技能数" in result
        assert "活跃" in result


# ==================== TestU29FormatEvolveResult ====================

class TestU29FormatEvolveResult:
    """测试format_evolve_result格式化"""

    def test_format_evolve_result_success(self, helper):
        """测试成功的进化结果格式化"""
        skills = helper.list_skills()
        skill_id = skills[0]["skill_id"]
        result = helper.evolve_skill(skill_id, ["new_step"])
        formatted = helper.format_evolve_result(result)
        assert "进化成功" in formatted
        assert "v2" in formatted

    def test_format_evolve_result_failure(self, helper):
        """测试失败的进化结果格式化"""
        formatted = helper.format_evolve_result(None)
        assert "失败" in formatted


# ==================== TestU29CliIntegration ====================

class TestU29CliIntegration:
    """测试CLI集成"""

    def test_skill_parser_exists(self):
        """测试skill子命令parser存在"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["skill"])
        assert args.command == "skill"

    def test_skill_list_parser(self):
        """测试skill list parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["skill", "list"])
        assert args.skill_action == "list"
        assert args.status is None

    def test_skill_list_with_status_parser(self):
        """测试skill list带状态"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["skill", "list", "--status", "active"])
        assert args.status == "active"

    def test_skill_info_parser(self):
        """测试skill info parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["skill", "info", "abc123"])
        assert args.skill_action == "info"
        assert args.id == "abc123"

    def test_skill_search_parser(self):
        """测试skill search parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["skill", "search", "clean"])
        assert args.skill_action == "search"
        assert args.task_type == "clean"

    def test_skill_evolve_parser(self):
        """测试skill evolve parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["skill", "evolve", "abc123", "step1,step2"])
        assert args.skill_action == "evolve"
        assert args.id == "abc123"
        assert args.steps == "step1,step2"

    def test_skill_stats_parser(self):
        """测试skill stats parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["skill", "stats"])
        assert args.skill_action == "stats"

    def test_skill_clear_parser(self):
        """测试skill clear parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["skill", "clear"])
        assert args.skill_action == "clear"

    def test_cmd_skill_list_empty(self, capsys):
        """测试_cmd_skill list空"""
        from cli import _cmd_skill, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["skill", "list"])
        ret = _cmd_skill(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "无技能" in out

    def test_cmd_skill_stats(self, capsys):
        """测试_cmd_skill stats"""
        from cli import _cmd_skill, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["skill", "stats"])
        ret = _cmd_skill(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "技能库统计" in out

    def test_cmd_skill_info_nonexistent(self, capsys):
        """测试_cmd_skill info不存在"""
        from cli import _cmd_skill, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["skill", "info", "nonexistent"])
        ret = _cmd_skill(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "不存在" in out

    def test_cmd_skill_search_no_results(self, capsys):
        """测试_cmd_skill search无结果"""
        from cli import _cmd_skill, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["skill", "search", "nonexistent"])
        ret = _cmd_skill(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "未找到" in out

    def test_cmd_skill_evolve_nonexistent(self, capsys):
        """测试_cmd_skill evolve不存在"""
        from cli import _cmd_skill, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["skill", "evolve", "nonexistent", "step1,step2"])
        ret = _cmd_skill(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "失败" in out

    def test_cmd_skill_clear(self, capsys):
        """测试_cmd_skill clear"""
        from cli import _cmd_skill, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["skill", "clear"])
        ret = _cmd_skill(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "已清空" in out

    def test_cmd_skill_no_action(self, capsys):
        """测试_cmd_skill无操作"""
        from cli import _cmd_skill, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["skill"])
        ret = _cmd_skill(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "请指定操作" in out

    def test_cli_main_skill_dispatch(self, capsys):
        """测试cli_main分发skill命令"""
        from cli import cli_main
        ret = cli_main(["skill", "stats"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "技能库统计" in out


# ==================== TestU29NoRegression ====================

class TestU29NoRegression:
    """验证U29不破坏既有功能"""

    def test_u24_perf_still_works(self):
        """U24 perf_cli模块仍可导入"""
        from core.perf_cli import PerfCliHelper
        helper = PerfCliHelper()
        assert helper is not None

    def test_u25_agents_still_works(self):
        """U25 agents_cli模块仍可导入"""
        from core.agents_cli import AgentsCliHelper
        assert AgentsCliHelper is not None

    def test_u26_snapshot_still_works(self):
        """U26 snapshot_cli模块仍可导入"""
        from core.snapshot_cli import SnapshotCliHelper
        assert SnapshotCliHelper is not None

    def test_u27_export_still_works(self):
        """U27 export_cli模块仍可导入"""
        from core.export_cli import ExportCliHelper
        helper = ExportCliHelper()
        assert helper is not None

    def test_u28_plugins_still_works(self):
        """U28 plugins_cli模块仍可导入"""
        from core.plugins_cli import PluginsCliHelper
        helper = PluginsCliHelper()
        assert helper is not None

    def test_existing_commands_still_dispatch(self):
        """既有命令仍可正常分发"""
        from cli import _create_parser
        parser = _create_parser()
        for cmd in ["run", "status", "snapshot", "logs", "alerts", "perf",
                     "agents", "export", "plugins", "skill"]:
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_skill_library_unchanged(self):
        """SkillLibrary核心API未变"""
        from core.skill_library import SkillLibrary, Skill, SkillStatus
        lib = SkillLibrary()
        assert hasattr(lib, "register")
        assert hasattr(lib, "get")
        assert hasattr(lib, "find_by_task_type")
        assert hasattr(lib, "find_best_skill")
        assert hasattr(lib, "evolve_skill")
        assert hasattr(lib, "list_skills")
        assert hasattr(lib, "get_stats")
        assert hasattr(lib, "clear")
