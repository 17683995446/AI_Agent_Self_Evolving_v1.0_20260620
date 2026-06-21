"""
U35：交互式Shell帮助同步 + 协作/共识CLI —— 测试

测试覆盖：
1. CollabCliHelper 的所有方法
2. ConsensusCliHelper 的所有方法
3. InteractiveShell 帮助动态同步
4. CLI集成（_cmd_collab / _cmd_consensus 分发）
5. 无回归（U30-U34 仍可用）
"""

import os
import sys
import io
import pytest

# 路径设置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def collab_helper():
    """创建CollabCliHelper"""
    from core.collab_cli import CollabCliHelper
    return CollabCliHelper()


@pytest.fixture
def consensus_helper():
    """创建ConsensusCliHelper"""
    from core.collab_cli import ConsensusCliHelper
    return ConsensusCliHelper(min_votes=2)


# ==================== TestU35CollabCliHelper ====================

class TestU35CollabCliHelper:
    """测试CollabCliHelper基础功能"""

    def test_helper_initialization(self, collab_helper):
        """测试初始化"""
        assert collab_helper is not None
        assert collab_helper.hub is not None

    def test_get_stats(self, collab_helper):
        """测试获取统计"""
        stats = collab_helper.get_stats()
        assert "agent_id" in stats
        assert "shared_count" in stats
        assert "received_count" in stats
        assert "active_peers" in stats

    def test_share_experience(self, collab_helper):
        """测试分享经验"""
        exp = collab_helper.share_experience(
            state={"cpu": 80}, action="optimize", reward=0.8,
        )
        assert "experience_id" in exp
        assert exp["action"] == "optimize"
        assert exp["reward"] == 0.8

    def test_share_experience_with_tags(self, collab_helper):
        """测试带标签分享"""
        exp = collab_helper.share_experience(
            state={"cpu": 80}, action="optimize", reward=0.8,
            tags=["performance", "cpu"],
        )
        assert "performance" in exp["tags"]

    def test_share_experience_increases_count(self, collab_helper):
        """测试分享增加计数"""
        initial = collab_helper.get_stats()["shared_count"]
        collab_helper.share_experience({"cpu": 80}, "optimize", 0.8)
        assert collab_helper.get_stats()["shared_count"] == initial + 1

    def test_get_experiences_empty(self, collab_helper):
        """测试空经验列表"""
        experiences = collab_helper.get_experiences()
        assert experiences == []

    def test_get_experiences_with_limit(self, collab_helper):
        """测试经验列表限制"""
        experiences = collab_helper.get_experiences(n=5)
        assert len(experiences) <= 5


# ==================== TestU35CollabFormat ====================

class TestU35CollabFormat:
    """测试CollabCliHelper格式化"""

    def test_format_stats(self, collab_helper):
        """测试统计格式化"""
        result = collab_helper.format_stats()
        assert "协作统计" in result
        assert "Agent ID" in result
        assert "分享经验数" in result

    def test_format_share_result(self, collab_helper):
        """测试分享结果格式化"""
        exp = collab_helper.share_experience({"cpu": 80}, "optimize", 0.8)
        result = collab_helper.format_share_result(exp)
        assert "经验分享成功" in result
        assert "optimize" in result

    def test_format_experiences_empty(self, collab_helper):
        """测试空经验列表格式化"""
        result = collab_helper.format_experiences()
        assert "无接收到的共享经验" in result


# ==================== TestU35ConsensusCliHelper ====================

class TestU35ConsensusCliHelper:
    """测试ConsensusCliHelper基础功能"""

    def test_helper_initialization(self, consensus_helper):
        """测试初始化"""
        assert consensus_helper is not None
        assert consensus_helper.voter is not None

    def test_vote(self, consensus_helper):
        """测试投票"""
        consensus_helper.vote("topic1", "agent1", "yes")
        stats = consensus_helper.get_stats()
        assert stats["topics"] >= 1

    def test_vote_with_confidence(self, consensus_helper):
        """测试带置信度投票"""
        consensus_helper.vote("topic1", "agent1", "yes", confidence=0.9)
        # 投票后应该能查询
        result = consensus_helper.get_consensus("topic1")
        # 只有一票，未达最小票数2，应为None
        assert result is None

    def test_get_consensus_no_votes(self, consensus_helper):
        """测试无投票时获取共识"""
        result = consensus_helper.get_consensus("nonexistent")
        assert result is None

    def test_get_consensus_insufficient_votes(self, consensus_helper):
        """测试票数不足"""
        consensus_helper.vote("topic1", "agent1", "yes")
        result = consensus_helper.get_consensus("topic1")
        assert result is None  # 需要至少2票

    def test_get_consensus_sufficient_votes(self, consensus_helper):
        """测试票数足够达成共识"""
        consensus_helper.vote("topic1", "agent1", "yes")
        consensus_helper.vote("topic1", "agent2", "yes")
        result = consensus_helper.get_consensus("topic1")
        assert result is not None
        assert result["consensus"] == "yes"
        assert result["total_votes"] == 2

    def test_get_consensus_disagreement(self, consensus_helper):
        """测试分歧投票"""
        consensus_helper.vote("topic1", "agent1", "yes", confidence=1.0)
        consensus_helper.vote("topic1", "agent2", "no", confidence=1.0)
        result = consensus_helper.get_consensus("topic1")
        assert result is not None
        assert result["agreement_rate"] == 0.5

    def test_get_stats(self, consensus_helper):
        """测试获取统计"""
        stats = consensus_helper.get_stats()
        assert "topics" in stats
        assert "decisions" in stats
        assert "min_votes_required" in stats


# ==================== TestU35ConsensusFormat ====================

class TestU35ConsensusFormat:
    """测试ConsensusCliHelper格式化"""

    def test_format_vote_result(self, consensus_helper):
        """测试投票结果格式化"""
        consensus_helper.vote("topic1", "agent1", "yes")
        result = consensus_helper.format_vote_result("topic1", "agent1", "yes")
        assert "投票已提交" in result
        assert "topic1" in result
        assert "yes" in result

    def test_format_consensus_result_no_consensus(self, consensus_helper):
        """测试未达成共识格式化"""
        result = consensus_helper.format_consensus_result("topic1")
        assert "未达成共识" in result

    def test_format_consensus_result_with_consensus(self, consensus_helper):
        """测试达成共识格式化"""
        consensus_helper.vote("topic1", "agent1", "yes")
        consensus_helper.vote("topic1", "agent2", "yes")
        result = consensus_helper.format_consensus_result("topic1")
        assert "共识结果" in result
        assert "yes" in result

    def test_format_stats(self, consensus_helper):
        """测试统计格式化"""
        result = consensus_helper.format_stats()
        assert "共识统计" in result
        assert "议题数" in result


# ==================== TestU35ShellHelpSync ====================

class TestU35ShellHelpSync:
    """测试交互式Shell帮助动态同步"""

    def test_shell_help_includes_new_commands(self):
        """测试Shell帮助包含新命令"""
        from core.interactive_shell import InteractiveShell
        output = io.StringIO()
        shell = InteractiveShell(
            input_fn=lambda _: "exit",
            output=output,
        )
        shell.show_help()
        help_text = output.getvalue()
        # 应包含U29-U35的新命令
        assert "skill" in help_text or "forever" in help_text or "learn" in help_text

    def test_shell_help_dynamic_commands(self):
        """测试Shell帮助动态获取命令"""
        from core.interactive_shell import InteractiveShell
        shell = InteractiveShell(
            input_fn=lambda _: "exit",
        )
        commands = shell._get_cli_commands()
        assert len(commands) > 0
        # 应包含一些已知命令
        cmd_names = [c[0] for c in commands]
        assert "run" in cmd_names
        assert "status" in cmd_names

    def test_shell_help_includes_collab_consensus(self):
        """测试Shell帮助包含collab/consensus"""
        from core.interactive_shell import InteractiveShell
        shell = InteractiveShell(
            input_fn=lambda _: "exit",
        )
        commands = shell._get_cli_commands()
        cmd_names = [c[0] for c in commands]
        assert "collab" in cmd_names
        assert "consensus" in cmd_names

    def test_shell_help_includes_learn(self):
        """测试Shell帮助包含learn"""
        from core.interactive_shell import InteractiveShell
        shell = InteractiveShell(
            input_fn=lambda _: "exit",
        )
        commands = shell._get_cli_commands()
        cmd_names = [c[0] for c in commands]
        assert "learn" in cmd_names

    def test_shell_help_includes_presets_scenarios(self):
        """测试Shell帮助包含presets/scenarios"""
        from core.interactive_shell import InteractiveShell
        shell = InteractiveShell(
            input_fn=lambda _: "exit",
        )
        commands = shell._get_cli_commands()
        cmd_names = [c[0] for c in commands]
        assert "presets" in cmd_names
        assert "scenarios" in cmd_names


# ==================== TestU35CliIntegration ====================

class TestU35CliIntegration:
    """测试CLI集成"""

    def test_collab_parser_exists(self):
        """测试collab子命令解析器存在"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["collab"])
        assert args.command == "collab"

    def test_collab_stats_parser(self):
        """测试collab stats parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["collab", "stats"])
        assert args.collab_action == "stats"

    def test_collab_share_parser(self):
        """测试collab share parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["collab", "share", "--action", "optimize", "--reward", "0.8"])
        assert args.collab_action == "share"
        assert args.action == "optimize"
        assert args.reward == 0.8

    def test_collab_experiences_parser(self):
        """测试collab experiences parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["collab", "experiences", "--n", "20"])
        assert args.collab_action == "experiences"
        assert args.n == 20

    def test_consensus_parser_exists(self):
        """测试consensus子命令解析器存在"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["consensus"])
        assert args.command == "consensus"

    def test_consensus_vote_parser(self):
        """测试consensus vote parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["consensus", "vote", "--topic", "deploy", "--agent", "a1", "--proposal", "yes"])
        assert args.consensus_action == "vote"
        assert args.topic == "deploy"
        assert args.agent == "a1"
        assert args.proposal == "yes"

    def test_consensus_result_parser(self):
        """测试consensus result parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["consensus", "result", "--topic", "deploy"])
        assert args.consensus_action == "result"
        assert args.topic == "deploy"

    def test_consensus_stats_parser(self):
        """测试consensus stats parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["consensus", "stats"])
        assert args.consensus_action == "stats"

    def test_cmd_collab_stats(self, capsys):
        """测试_cmd_collab stats"""
        from cli import _cmd_collab, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["collab", "stats"])
        ret = _cmd_collab(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "协作统计" in out

    def test_cmd_collab_share(self, capsys):
        """测试_cmd_collab share"""
        from cli import _cmd_collab, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["collab", "share", "--action", "optimize", "--reward", "0.8"])
        ret = _cmd_collab(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "经验分享成功" in out

    def test_cmd_collab_share_with_state(self, capsys):
        """测试_cmd_collab share带状态"""
        from cli import _cmd_collab, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["collab", "share", "--action", "optimize", "--reward", "0.8",
                                  "--state", '{"cpu": 80}'])
        ret = _cmd_collab(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "经验分享成功" in out

    def test_cmd_collab_share_invalid_state(self, capsys):
        """测试_cmd_collab share无效状态JSON"""
        from cli import _cmd_collab, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["collab", "share", "--action", "optimize",
                                  "--state", "invalid_json"])
        ret = _cmd_collab(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "JSON解析失败" in out

    def test_cmd_collab_experiences(self, capsys):
        """测试_cmd_collab experiences"""
        from cli import _cmd_collab, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["collab", "experiences"])
        ret = _cmd_collab(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "共享经验" in out or "无接收到" in out

    def test_cmd_collab_no_action(self, capsys):
        """测试_cmd_collab无操作"""
        from cli import _cmd_collab, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["collab"])
        ret = _cmd_collab(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "请指定操作" in out

    def test_cmd_consensus_vote(self, capsys):
        """测试_cmd_consensus vote"""
        from cli import _cmd_consensus, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["consensus", "vote", "--topic", "deploy",
                                  "--agent", "a1", "--proposal", "yes"])
        ret = _cmd_consensus(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "投票已提交" in out

    def test_cmd_consensus_result_no_consensus(self, capsys):
        """测试_cmd_consensus result未达成"""
        from cli import _cmd_consensus, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["consensus", "result", "--topic", "deploy"])
        ret = _cmd_consensus(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "未达成共识" in out

    def test_cmd_consensus_stats(self, capsys):
        """测试_cmd_consensus stats"""
        from cli import _cmd_consensus, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["consensus", "stats"])
        ret = _cmd_consensus(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "共识统计" in out

    def test_cmd_consensus_no_action(self, capsys):
        """测试_cmd_consensus无操作"""
        from cli import _cmd_consensus, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["consensus"])
        ret = _cmd_consensus(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "请指定操作" in out

    def test_cli_main_collab_dispatch(self, capsys):
        """测试cli_main分发collab命令"""
        from cli import cli_main
        ret = cli_main(["collab", "stats"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "协作统计" in out

    def test_cli_main_consensus_dispatch(self, capsys):
        """测试cli_main分发consensus命令"""
        from cli import cli_main
        ret = cli_main(["consensus", "stats"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "共识统计" in out

    def test_consensus_vote_and_result_flow(self, capsys):
        """测试投票+查看结果完整流程"""
        from cli import _cmd_consensus, _create_parser
        from core.collab_cli import ConsensusCliHelper
        parser = _create_parser()
        # 使用同一helper实例投票两次
        helper = ConsensusCliHelper(min_votes=2)
        helper.vote("deploy", "agent1", "yes")
        helper.vote("deploy", "agent2", "yes")
        result = helper.format_consensus_result("deploy")
        assert "共识结果" in result
        assert "yes" in result


# ==================== TestU35NoRegression ====================

class TestU35NoRegression:
    """验证U35不破坏既有功能"""

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
        from core.presets_cli import PresetsCliHelper
        assert PresetsCliHelper is not None

    def test_u34_learn_still_works(self):
        """U34 learn_cli模块仍可导入"""
        from core.learn_cli import LearnCliHelper
        assert LearnCliHelper is not None

    def test_collab_cli_module_importable(self):
        """U35 collab_cli模块可导入"""
        from core.collab_cli import CollabCliHelper, ConsensusCliHelper
        assert CollabCliHelper is not None
        assert ConsensusCliHelper is not None

    def test_interactive_shell_still_works(self):
        """交互式Shell仍可导入"""
        from core.interactive_shell import InteractiveShell
        assert InteractiveShell is not None

    def test_collab_subcommand_in_help(self, capsys):
        """collab子命令出现在帮助中"""
        from cli import _create_parser
        parser = _create_parser()
        parser.print_help()
        out = capsys.readouterr().out
        assert "collab" in out

    def test_consensus_subcommand_in_help(self, capsys):
        """consensus子命令出现在帮助中"""
        from cli import _create_parser
        parser = _create_parser()
        parser.print_help()
        out = capsys.readouterr().out
        assert "consensus" in out
