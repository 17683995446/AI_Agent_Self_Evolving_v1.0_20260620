"""
U26：快照管理增强 —— 测试

测试覆盖：
1. SnapshotCliHelper 的所有方法
2. resolve_snapshot_id 的多种ID格式解析
3. get_snapshot_info / delete_snapshot / diff_snapshots / export_snapshot
4. format_list / format_info / format_diff 格式化
5. CLI集成（_cmd_snapshot 分发）
6. 修复restore --id bug的验证
7. 无回归（U21-U25 仍可用）
"""

import os
import sys
import time
import json
import tempfile
import pytest

# 路径设置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_snapshot_dir():
    """创建临时快照目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def helper(temp_snapshot_dir):
    """创建SnapshotCliHelper，使用临时目录"""
    from core.snapshot_cli import SnapshotCliHelper
    return SnapshotCliHelper(snapshot_dir=temp_snapshot_dir)


@pytest.fixture
def sample_snapshots(temp_snapshot_dir):
    """创建示例快照文件"""
    from core.persistence import Snapshot, PersistenceManager
    manager = PersistenceManager(snapshot_dir=temp_snapshot_dir)

    paths = []
    for i in range(3):
        snapshot = Snapshot()
        snapshot.timestamp = time.time() - (3 - i) * 100  # 递增时间戳
        snapshot.set_data(f"key_{i}", f"value_{i}")
        snapshot.set_data("common_key", f"common_{i}")
        path = manager.save_snapshot(snapshot)
        paths.append(path)
        time.sleep(0.01)  # 确保时间戳不同

    return paths


# ==================== TestU26SnapshotCliHelper ====================

class TestU26SnapshotCliHelper:
    """测试SnapshotCliHelper基础功能"""

    def test_helper_initialization(self, helper):
        """测试初始化"""
        assert helper is not None
        assert helper.manager is not None

    def test_list_snapshots_empty(self, helper):
        """测试空快照列表"""
        assert helper.list_snapshots() == []

    def test_list_snapshots_with_data(self, helper, sample_snapshots):
        """测试有快照的列表"""
        snapshots = helper.list_snapshots()
        assert len(snapshots) == 3

    def test_resolve_snapshot_id_by_full_path(self, helper, sample_snapshots):
        """测试按完整路径解析"""
        path = helper.resolve_snapshot_id(sample_snapshots[0])
        assert path == sample_snapshots[0]

    def test_resolve_snapshot_id_by_filename(self, helper, sample_snapshots):
        """测试按文件名解析"""
        filename = os.path.basename(sample_snapshots[0])
        path = helper.resolve_snapshot_id(filename)
        assert path == sample_snapshots[0]

    def test_resolve_snapshot_id_by_index(self, helper, sample_snapshots):
        """测试按序号解析（1=最新）"""
        path = helper.resolve_snapshot_id("1")
        # 最新的是sample_snapshots[-1]（时间戳最大）
        assert path == sample_snapshots[-1]

    def test_resolve_snapshot_id_by_index_2(self, helper, sample_snapshots):
        """测试按序号解析（2=第二新）"""
        path = helper.resolve_snapshot_id("2")
        assert path == sample_snapshots[-2]

    def test_resolve_snapshot_id_nonexistent(self, helper):
        """测试解析不存在的ID"""
        path = helper.resolve_snapshot_id("nonexistent")
        assert path is None

    def test_resolve_snapshot_id_invalid_index(self, helper, sample_snapshots):
        """测试无效序号"""
        path = helper.resolve_snapshot_id("999")
        assert path is None

    def test_get_snapshot_info(self, helper, sample_snapshots):
        """测试获取快照详情"""
        info = helper.get_snapshot_info("1")
        assert info is not None
        assert "path" in info
        assert "filename" in info
        assert "size_bytes" in info
        assert "checksum" in info
        assert "data_keys" in info

    def test_get_snapshot_info_nonexistent(self, helper):
        """测试获取不存在的快照详情"""
        info = helper.get_snapshot_info("nonexistent")
        assert info is None

    def test_delete_snapshot(self, helper, sample_snapshots):
        """测试删除快照"""
        initial_count = len(helper.list_snapshots())
        success = helper.delete_snapshot("1")
        assert success is True
        assert len(helper.list_snapshots()) == initial_count - 1

    def test_delete_snapshot_nonexistent(self, helper):
        """测试删除不存在的快照"""
        success = helper.delete_snapshot("nonexistent")
        assert success is False

    def test_diff_snapshots(self, helper, sample_snapshots):
        """测试对比快照差异"""
        diff = helper.diff_snapshots("1", "2")
        assert diff is not None
        assert "snapshot1" in diff
        assert "snapshot2" in diff
        assert "keys_only_in_1" in diff
        assert "keys_only_in_2" in diff
        assert "common_keys" in diff

    def test_diff_snapshots_nonexistent(self, helper, sample_snapshots):
        """测试对比不存在的快照"""
        diff = helper.diff_snapshots("1", "nonexistent")
        assert diff is None

    def test_export_snapshot_to_string(self, helper, sample_snapshots):
        """测试导出快照为字符串"""
        result = helper.export_snapshot("1")
        assert result is not None
        data = json.loads(result)
        assert "version" in data
        assert "timestamp" in data
        assert "data" in data

    def test_export_snapshot_to_file(self, helper, sample_snapshots, tmp_path):
        """测试导出快照到文件"""
        output_file = str(tmp_path / "export.json")
        result = helper.export_snapshot("1", output_file)
        assert result == output_file
        assert os.path.exists(output_file)
        with open(output_file, 'r') as f:
            data = json.load(f)
        assert "version" in data

    def test_export_snapshot_nonexistent(self, helper):
        """测试导出不存在的快照"""
        result = helper.export_snapshot("nonexistent")
        assert result is None


# ==================== TestU26FormatList ====================

class TestU26FormatList:
    """测试format_list格式化"""

    def test_format_list_empty(self, helper):
        """测试空列表格式化"""
        result = helper.format_list()
        assert "无快照" in result

    def test_format_list_with_data(self, helper, sample_snapshots):
        """测试有快照的列表格式化"""
        result = helper.format_list()
        assert "快照列表" in result
        assert "3个" in result
        assert "snapshot_" in result


# ==================== TestU26FormatInfo ====================

class TestU26FormatInfo:
    """测试format_info格式化"""

    def test_format_info(self, helper, sample_snapshots):
        """测试快照详情格式化"""
        result = helper.format_info("1")
        assert "快照详情" in result
        assert "路径" in result
        assert "版本" in result
        assert "大小" in result
        assert "校验和" in result

    def test_format_info_nonexistent(self, helper):
        """测试不存在的快照详情格式化"""
        result = helper.format_info("nonexistent")
        assert "不存在" in result


# ==================== TestU26FormatDiff ====================

class TestU26FormatDiff:
    """测试format_diff格式化"""

    def test_format_diff(self, helper, sample_snapshots):
        """测试快照差异格式化"""
        result = helper.format_diff("1", "2")
        assert "快照差异" in result
        assert "快照1" in result
        assert "快照2" in result
        assert "时间差" in result

    def test_format_diff_nonexistent(self, helper, sample_snapshots):
        """测试不存在的快照差异格式化"""
        result = helper.format_diff("1", "nonexistent")
        assert "失败" in result


# ==================== TestU26CliIntegration ====================

class TestU26CliIntegration:
    """测试CLI集成"""

    def test_snapshot_info_parser(self):
        """测试snapshot info parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["snapshot", "info", "1"])
        assert args.command == "snapshot"
        assert args.snapshot_action == "info"
        assert args.id == "1"

    def test_snapshot_diff_parser(self):
        """测试snapshot diff parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["snapshot", "diff", "1", "2"])
        assert args.snapshot_action == "diff"
        assert args.id1 == "1"
        assert args.id2 == "2"

    def test_snapshot_delete_parser(self):
        """测试snapshot delete parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["snapshot", "delete", "1"])
        assert args.snapshot_action == "delete"
        assert args.id == "1"

    def test_snapshot_export_parser(self):
        """测试snapshot export parser"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["snapshot", "export", "1"])
        assert args.snapshot_action == "export"
        assert args.id == "1"
        assert args.output is None

    def test_snapshot_export_with_output_parser(self):
        """测试snapshot export带输出路径"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["snapshot", "export", "1", "--output", "/tmp/out.json"])
        assert args.output == "/tmp/out.json"

    def test_snapshot_restore_parser_uses_id(self):
        """测试snapshot restore parser正确解析--id"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["snapshot", "restore", "--id", "1"])
        assert args.snapshot_action == "restore"
        assert args.id == "1"

    def test_cmd_snapshot_info(self, capsys, temp_snapshot_dir, sample_snapshots, monkeypatch):
        """测试_cmd_snapshot info"""
        from cli import _cmd_agents  # noqa: ensure import works
        from cli import _cmd_snapshot, _create_parser
        # monkeypatch SnapshotCliHelper使用临时目录
        from core import snapshot_cli
        original_init = snapshot_cli.SnapshotCliHelper.__init__
        def patched_init(self, snapshot_dir=None):
            original_init(self, snapshot_dir=temp_snapshot_dir)
        monkeypatch.setattr(snapshot_cli.SnapshotCliHelper, "__init__", patched_init)

        parser = _create_parser()
        args = parser.parse_args(["snapshot", "info", "1"])
        ret = _cmd_snapshot(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "快照详情" in out

    def test_cmd_snapshot_delete(self, capsys, temp_snapshot_dir, sample_snapshots, monkeypatch):
        """测试_cmd_snapshot delete"""
        from cli import _cmd_snapshot, _create_parser
        from core import snapshot_cli
        original_init = snapshot_cli.SnapshotCliHelper.__init__
        def patched_init(self, snapshot_dir=None):
            original_init(self, snapshot_dir=temp_snapshot_dir)
        monkeypatch.setattr(snapshot_cli.SnapshotCliHelper, "__init__", patched_init)

        initial_count = len(snapshot_cli.SnapshotCliHelper().list_snapshots())
        parser = _create_parser()
        args = parser.parse_args(["snapshot", "delete", "1"])
        ret = _cmd_snapshot(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "已删除" in out

    def test_cmd_snapshot_export(self, capsys, temp_snapshot_dir, sample_snapshots, monkeypatch):
        """测试_cmd_snapshot export"""
        from cli import _cmd_snapshot, _create_parser
        from core import snapshot_cli
        original_init = snapshot_cli.SnapshotCliHelper.__init__
        def patched_init(self, snapshot_dir=None):
            original_init(self, snapshot_dir=temp_snapshot_dir)
        monkeypatch.setattr(snapshot_cli.SnapshotCliHelper, "__init__", patched_init)

        parser = _create_parser()
        args = parser.parse_args(["snapshot", "export", "1"])
        ret = _cmd_snapshot(args)
        assert ret == 0
        out = capsys.readouterr().out
        # 应该输出JSON
        assert "version" in out

    def test_cmd_snapshot_diff(self, capsys, temp_snapshot_dir, sample_snapshots, monkeypatch):
        """测试_cmd_snapshot diff"""
        from cli import _cmd_snapshot, _create_parser
        from core import snapshot_cli
        original_init = snapshot_cli.SnapshotCliHelper.__init__
        def patched_init(self, snapshot_dir=None):
            original_init(self, snapshot_dir=temp_snapshot_dir)
        monkeypatch.setattr(snapshot_cli.SnapshotCliHelper, "__init__", patched_init)

        parser = _create_parser()
        args = parser.parse_args(["snapshot", "diff", "1", "2"])
        ret = _cmd_snapshot(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "快照差异" in out


# ==================== TestU26RestoreBugFix ====================

class TestU26RestoreBugFix:
    """验证restore --id bug修复"""

    def test_restore_passes_id_to_load_state(self, monkeypatch, temp_snapshot_dir):
        """测试restore将--id传递给load_state"""
        from cli import _cmd_snapshot, _create_parser

        # Mock SnapshotCliHelper
        from core import snapshot_cli
        original_init = snapshot_cli.SnapshotCliHelper.__init__
        def patched_init(self, snapshot_dir=None):
            original_init(self, snapshot_dir=temp_snapshot_dir)
        def patched_resolve(self, snapshot_id):
            return f"/resolved/path/{snapshot_id}"
        monkeypatch.setattr(snapshot_cli.SnapshotCliHelper, "__init__", patched_init)
        monkeypatch.setattr(snapshot_cli.SnapshotCliHelper, "resolve_snapshot_id", patched_resolve)

        # Mock SelfEvolvingAgent - 在main模块中替换
        captured_path = []

        class MockAgent:
            def load_state(self, path=None):
                captured_path.append(path)
                return True
            def stop(self):
                pass

        import sys
        import types
        # 创建或获取main模块的mock
        if 'main' not in sys.modules:
            mock_main = types.ModuleType('main')
            mock_main.SelfEvolvingAgent = MockAgent
            sys.modules['main'] = mock_main
        else:
            monkeypatch.setattr(sys.modules['main'], 'SelfEvolvingAgent', MockAgent)

        parser = _create_parser()
        args = parser.parse_args(["snapshot", "restore", "--id", "my_snapshot"])
        ret = _cmd_snapshot(args)

        assert ret == 0
        # 验证load_state被调用时传入了解析后的路径
        assert len(captured_path) == 1
        assert captured_path[0] is not None
        assert "my_snapshot" in captured_path[0]


# ==================== TestU26NoRegression ====================

class TestU26NoRegression:
    """验证U26不破坏既有功能"""

    def test_u21_context_help_still_works(self):
        """U21 context_help模块仍可导入"""
        from core.context_help import ContextualHelpProvider
        assert ContextualHelpProvider is not None

    def test_u22_log_viewer_still_works(self):
        """U22 log_viewer模块仍可导入"""
        from core.log_viewer import LogManager
        assert LogManager is not None

    def test_u23_alerts_still_works(self):
        """U23 alert_cli模块仍可导入"""
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        assert helper is not None

    def test_u24_perf_still_works(self):
        """U24 perf_cli模块仍可导入"""
        from core.perf_cli import PerfCliHelper
        helper = PerfCliHelper()
        assert helper is not None

    def test_u25_agents_still_works(self):
        """U25 agents_cli模块仍可导入"""
        from core.agents_cli import AgentsCliHelper
        assert AgentsCliHelper is not None

    def test_existing_snapshot_commands_still_dispatch(self):
        """既有snapshot命令仍可正常解析"""
        from cli import _create_parser
        parser = _create_parser()
        for action in ["save", "list", "restore", "info", "diff", "delete", "export"]:
            if action == "restore":
                args = parser.parse_args(["snapshot", "restore", "--id", "1"])
            elif action == "diff":
                args = parser.parse_args(["snapshot", "diff", "1", "2"])
            elif action in ("info", "delete", "export"):
                args = parser.parse_args(["snapshot", action, "1"])
            else:
                args = parser.parse_args(["snapshot", action])
            assert args.command == "snapshot"

    def test_persistence_manager_unchanged(self):
        """PersistenceManager核心API未变"""
        from core.persistence import PersistenceManager, Snapshot
        assert hasattr(PersistenceManager, "save_snapshot")
        assert hasattr(PersistenceManager, "load_snapshot")
        assert hasattr(PersistenceManager, "list_snapshots")
        assert hasattr(Snapshot, "to_dict")
        assert hasattr(Snapshot, "verify_checksum")
