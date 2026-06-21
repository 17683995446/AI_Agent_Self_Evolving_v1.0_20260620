"""
U34：持续学习与元学习 —— 测试

测试覆盖：
1. LearnCliHelper 的所有方法
2. init / train / stats / evaluate
3. CLI集成（_cmd_learn 分发）
4. 无回归（U29-U33 仍可用）
"""

import os
import sys
import pytest

# 路径设置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def helper():
    """创建LearnCliHelper"""
    from core.learn_cli import LearnCliHelper
    return LearnCliHelper()


@pytest.fixture
def initialized_helper():
    """创建已初始化的LearnCliHelper"""
    from core.learn_cli import LearnCliHelper
    h = LearnCliHelper()
    h.init_all(input_dim=4, output_dim=2)
    return h


# ==================== TestU34LearnCliHelper ====================

class TestU34LearnCliHelper:
    """测试LearnCliHelper基础功能"""

    def test_helper_initialization(self, helper):
        """测试初始化"""
        assert helper is not None
        assert helper._continual_learner is None
        assert helper._meta_learner is None

    def test_init_continual(self, helper):
        """测试初始化持续学习"""
        success = helper.init_continual(input_dim=4, output_dim=2)
        assert success is True
        assert helper.is_continual_initialized() is True

    def test_init_meta(self, helper):
        """测试初始化元学习"""
        success = helper.init_meta(input_dim=4, output_dim=2)
        assert success is True
        assert helper.is_meta_initialized() is True

    def test_init_all(self, helper):
        """测试初始化所有学习器"""
        results = helper.init_all(input_dim=4, output_dim=2)
        assert results["continual"] is True
        assert results["meta"] is True
        assert helper.is_continual_initialized() is True
        assert helper.is_meta_initialized() is True

    def test_init_all_custom_dims(self, helper):
        """测试自定义维度初始化"""
        results = helper.init_all(input_dim=8, output_dim=3)
        assert results["continual"] is True
        assert results["meta"] is True
        assert helper._input_dim == 8
        assert helper._output_dim == 3

    def test_is_continual_initialized_default_false(self, helper):
        """测试默认未初始化"""
        assert helper.is_continual_initialized() is False

    def test_is_meta_initialized_default_false(self, helper):
        """测试默认未初始化"""
        assert helper.is_meta_initialized() is False

    def test_get_stats_uninitialized(self, helper):
        """测试未初始化统计"""
        stats = helper.get_stats()
        assert stats["continual_initialized"] is False
        assert stats["meta_initialized"] is False

    def test_get_stats_initialized(self, initialized_helper):
        """测试已初始化统计"""
        stats = initialized_helper.get_stats()
        assert stats["continual_initialized"] is True
        assert stats["meta_initialized"] is True
        assert "continual" in stats
        assert "meta" in stats

    def test_get_stats_continual_fields(self, initialized_helper):
        """测试持续学习统计字段"""
        stats = initialized_helper.get_stats()
        cl = stats["continual"]
        assert "strategy" in cl
        assert "task_count" in cl

    def test_get_stats_meta_fields(self, initialized_helper):
        """测试元学习统计字段"""
        stats = initialized_helper.get_stats()
        ml = stats["meta"]
        assert "meta_steps" in ml


# ==================== TestU34Train ====================

class TestU34Train:
    """测试训练功能"""

    def test_train_continual_uninitialized(self, helper):
        """测试未初始化持续学习训练"""
        result = helper.train_continual("task1")
        assert "error" in result

    def test_train_meta_uninitialized(self, helper):
        """测试未初始化元学习训练"""
        result = helper.train_meta(n_iterations=2)
        assert "error" in result

    def test_train_continual(self, initialized_helper):
        """测试持续学习训练"""
        result = initialized_helper.train_continual("task1", n_samples=20, n_steps=10)
        assert "error" not in result

    def test_train_meta(self, initialized_helper):
        """测试元学习训练"""
        result = initialized_helper.train_meta(n_iterations=2, batch_size=2)
        assert "error" not in result

    def test_train_continual_increases_task_count(self, initialized_helper):
        """测试持续学习训练增加任务数"""
        initial_count = initialized_helper.get_stats()["continual"]["task_count"]
        initialized_helper.train_continual("task1", n_samples=20, n_steps=5)
        new_count = initialized_helper.get_stats()["continual"]["task_count"]
        assert new_count == initial_count + 1


# ==================== TestU34Evaluate ====================

class TestU34Evaluate:
    """测试评估功能"""

    def test_evaluate_continual_uninitialized(self, helper):
        """测试未初始化持续学习评估"""
        result = helper.evaluate_continual("task1")
        assert "error" in result

    def test_evaluate_meta_uninitialized(self, helper):
        """测试未初始化元学习评估"""
        result = helper.evaluate_meta()
        assert "error" in result

    def test_evaluate_continual(self, initialized_helper):
        """测试持续学习评估"""
        # 先训练一个任务
        initialized_helper.train_continual("task1", n_samples=20, n_steps=5)
        result = initialized_helper.evaluate_continual("task1", n_samples=20)
        assert "error" not in result
        assert "accuracy" in result

    def test_evaluate_meta(self, initialized_helper):
        """测试元学习评估"""
        result = initialized_helper.evaluate_meta(n_tasks=3)
        assert "error" not in result


# ==================== TestU34Format ====================

class TestU34Format:
    """测试格式化方法"""

    def test_format_init_result_success(self, helper):
        """测试初始化成功格式化"""
        results = {"continual": True, "meta": True}
        result = helper.format_init_result(results, 4, 2)
        assert "学习系统初始化" in result
        assert "成功" in result
        assert "EWC" in result
        assert "MAML" in result

    def test_format_init_result_failure(self, helper):
        """测试初始化失败格式化"""
        results = {"continual": False, "meta": False}
        result = helper.format_init_result(results, 4, 2)
        assert "失败" in result

    def test_format_train_result_success(self, helper):
        """测试训练成功格式化"""
        result = helper.format_train_result("元学习", {"loss": 0.5, "steps": 10})
        assert "元学习训练完成" in result
        assert "loss" in result

    def test_format_train_result_error(self, helper):
        """测试训练错误格式化"""
        result = helper.format_train_result("元学习", {"error": "未初始化"})
        assert "失败" in result
        assert "未初始化" in result

    def test_format_stats_uninitialized(self, helper):
        """测试未初始化统计格式化"""
        result = helper.format_stats()
        assert "学习系统统计" in result
        assert "未初始化" in result

    def test_format_stats_initialized(self, initialized_helper):
        """测试已初始化统计格式化"""
        result = initialized_helper.format_stats()
        assert "学习系统统计" in result
        assert "已初始化" in result
        assert "持续学习详情" in result
        assert "元学习详情" in result

    def test_format_evaluate_result_success(self, helper):
        """测试评估成功格式化"""
        result = helper.format_evaluate_result("元学习", {"accuracy": 0.85})
        assert "元学习评估结果" in result
        assert "accuracy" in result

    def test_format_evaluate_result_error(self, helper):
        """测试评估错误格式化"""
        result = helper.format_evaluate_result("元学习", {"error": "未初始化"})
        assert "失败" in result


# ==================== TestU34CliIntegration ====================

class TestU34CliIntegration:
    """测试CLI集成"""

    def test_learn_parser_exists(self):
        """测试learn子命令解析器存在"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["learn"])
        assert args.command == "learn"

    def test_learn_init_parser(self):
        """测试learn init parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["learn", "init"])
        assert args.learn_action == "init"
        assert args.input_dim == 4
        assert args.output_dim == 2

    def test_learn_init_parser_custom_dims(self):
        """测试learn init自定义维度"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["learn", "init", "--input-dim", "8", "--output-dim", "3"])
        assert args.input_dim == 8
        assert args.output_dim == 3

    def test_learn_train_parser_meta(self):
        """测试learn train meta parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["learn", "train", "--type", "meta", "--iterations", "20"])
        assert args.learn_action == "train"
        assert args.type == "meta"
        assert args.iterations == 20

    def test_learn_train_parser_continual(self):
        """测试learn train continual parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["learn", "train", "--type", "continual", "--task", "mytask"])
        assert args.type == "continual"
        assert args.task == "mytask"

    def test_learn_stats_parser(self):
        """测试learn stats parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["learn", "stats"])
        assert args.learn_action == "stats"

    def test_learn_evaluate_parser_meta(self):
        """测试learn evaluate meta parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["learn", "evaluate", "--type", "meta", "--n-tasks", "5"])
        assert args.learn_action == "evaluate"
        assert args.type == "meta"
        assert args.n_tasks == 5

    def test_learn_evaluate_parser_continual(self):
        """测试learn evaluate continual parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["learn", "evaluate", "--type", "continual", "--task", "t1"])
        assert args.type == "continual"
        assert args.task == "t1"

    def test_cmd_learn_stats(self, capsys):
        """测试_cmd_learn stats"""
        from cli import _cmd_learn, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["learn", "stats"])
        ret = _cmd_learn(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "学习系统统计" in out

    def test_cmd_learn_init(self, capsys):
        """测试_cmd_learn init"""
        from cli import _cmd_learn, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["learn", "init"])
        ret = _cmd_learn(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "学习系统初始化" in out

    def test_cmd_learn_train_meta_uninitialized(self, capsys):
        """测试_cmd_learn train meta未初始化"""
        from cli import _cmd_learn, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["learn", "train", "--type", "meta", "--iterations", "2"])
        ret = _cmd_learn(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "失败" in out or "error" in out.lower()

    def test_cmd_learn_no_action(self, capsys):
        """测试_cmd_learn无操作"""
        from cli import _cmd_learn, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["learn"])
        ret = _cmd_learn(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "请指定操作" in out

    def test_cli_main_learn_dispatch(self, capsys):
        """测试cli_main分发learn命令"""
        from cli import cli_main
        ret = cli_main(["learn", "stats"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "学习系统统计" in out


# ==================== TestU34NoRegression ====================

class TestU34NoRegression:
    """验证U34不破坏既有功能"""

    def test_u29_skill_still_works(self):
        """U29 skill_cli模块仍可导入"""
        from core.skill_cli import SkillCliHelper
        assert SkillCliHelper is not None

    def test_u30_long_task_still_works(self):
        """U30 long_task_cli模块仍可导入"""
        from core.long_task_cli import LongTaskCliHelper
        assert LongTaskCliHelper is not None

    def test_u31_workflow_still_works(self):
        """U31 workflow_cli模块仍可导入"""
        from core.workflow_cli import WorkflowCliHelper
        assert WorkflowCliHelper is not None

    def test_u32_forever_still_works(self):
        """U32 forever_cli模块仍可导入"""
        from core.forever_cli import ForeverCliHelper
        assert ForeverCliHelper is not None

    def test_u33_presets_still_works(self):
        """U33 presets_cli模块仍可导入"""
        from core.presets_cli import PresetsCliHelper, ScenariosCliHelper
        assert PresetsCliHelper is not None
        assert ScenariosCliHelper is not None

    def test_learn_cli_module_importable(self):
        """U34 learn_cli模块可导入"""
        from core.learn_cli import LearnCliHelper
        assert LearnCliHelper is not None

    def test_learn_subcommand_in_help(self, capsys):
        """learn子命令出现在帮助中"""
        from cli import _create_parser
        parser = _create_parser()
        parser.print_help()
        out = capsys.readouterr().out
        assert "learn" in out
