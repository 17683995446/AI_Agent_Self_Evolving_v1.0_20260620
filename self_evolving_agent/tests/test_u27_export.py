"""
U27：数据导出 —— 测试

测试覆盖：
1. ExportCliHelper 的所有方法
2. collect_status/health/alerts/perf/snapshot 数据采集
3. to_json/to_csv 格式化
4. export 导出（JSON/CSV/文件/stdout）
5. CLI集成（_cmd_export 分发）
6. 无回归（U22-U26 仍可用）
"""

import os
import sys
import json
import csv
import io
import tempfile
import pytest

# 路径设置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def helper():
    """创建ExportCliHelper"""
    from core.export_cli import ExportCliHelper
    return ExportCliHelper()


@pytest.fixture
def fresh_perf_monitor():
    """重置性能监控器"""
    from core.performance import perf_monitor
    perf_monitor.reset()
    perf_monitor.set_enabled(True)
    yield perf_monitor
    perf_monitor.reset()


# ==================== TestU27ExportCliHelper ====================

class TestU27ExportCliHelper:
    """测试ExportCliHelper基础功能"""

    def test_helper_initialization(self, helper):
        """测试初始化"""
        assert helper is not None

    def test_lazy_loading_alert_helper(self, helper):
        """测试延迟加载alert_helper"""
        assert helper._alert_helper is None
        h = helper.alert_helper
        assert h is not None
        assert helper._alert_helper is not None

    def test_lazy_loading_perf_helper(self, helper):
        """测试延迟加载perf_helper"""
        assert helper._perf_helper is None
        h = helper.perf_helper
        assert h is not None

    def test_get_available_sources(self, helper):
        """测试获取可用数据源"""
        sources = helper.get_available_sources()
        assert "status" in sources
        assert "health" in sources
        assert "alerts" in sources
        assert "perf" in sources
        assert "snapshot" in sources

    def test_format_sources_list(self, helper):
        """测试格式化数据源列表"""
        result = helper.format_sources_list()
        assert "可用导出数据源" in result
        assert "status" in result
        assert "health" in result
        assert "alerts" in result
        assert "perf" in result
        assert "snapshot" in result


# ==================== TestU27CollectData ====================

class TestU27CollectData:
    """测试数据采集"""

    def test_collect_status(self, helper):
        """测试采集状态数据"""
        data = helper.collect_status()
        assert "export_time" in data
        assert "export_time_str" in data
        assert "perf_summary" in data
        assert "agents_status" in data

    def test_collect_health(self, helper):
        """测试采集健康数据"""
        data = helper.collect_health()
        assert "export_time" in data
        assert "agents_health" in data
        assert "perf_slow_operations" in data
        assert "alert_stats" in data

    def test_collect_alerts(self, helper):
        """测试采集告警数据"""
        data = helper.collect_alerts()
        assert "export_time" in data
        assert "rules" in data
        assert "channels" in data
        assert "history" in data
        assert "stats" in data

    def test_collect_perf(self, helper, fresh_perf_monitor):
        """测试采集性能数据"""
        fresh_perf_monitor.record("test_op", 0.01)
        data = helper.collect_perf()
        assert "export_time" in data
        assert "summary" in data
        assert "stats" in data
        assert "slow_operations" in data

    def test_collect_snapshot_list(self, helper):
        """测试采集快照列表数据"""
        data = helper.collect_snapshot_list()
        assert "export_time" in data
        assert "snapshot_count" in data
        assert "snapshots" in data
        assert isinstance(data["snapshots"], list)


# ==================== TestU27FormatJson ====================

class TestU27FormatJson:
    """测试JSON格式化"""

    def test_to_json_simple(self, helper):
        """测试简单JSON格式化"""
        data = {"key": "value", "num": 123}
        result = helper.to_json(data)
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["num"] == 123

    def test_to_json_nested(self, helper):
        """测试嵌套JSON格式化"""
        data = {"outer": {"inner": "value"}, "list": [1, 2, 3]}
        result = helper.to_json(data)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == "value"
        assert parsed["list"] == [1, 2, 3]

    def test_to_json_with_non_serializable(self, helper):
        """测试包含不可序列化对象的JSON"""
        class Custom:
            def __str__(self):
                return "custom"
        data = {"obj": Custom()}
        result = helper.to_json(data)
        parsed = json.loads(result)
        assert "custom" in str(parsed["obj"])


# ==================== TestU27FormatCsv ====================

class TestU27FormatCsv:
    """测试CSV格式化"""

    def test_to_csv_list_of_dicts(self, helper):
        """测试List[Dict]的CSV格式化"""
        data = [{"name": "a", "value": 1}, {"name": "b", "value": 2}]
        result = helper.to_csv(data)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["name"] == "a"
        assert rows[0]["value"] == "1"

    def test_to_csv_dict(self, helper):
        """测试Dict的CSV格式化"""
        data = {"key1": "value1", "key2": "value2"}
        result = helper.to_csv(data)
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert rows[0] == ["key", "value"]
        assert rows[1] == ["key1", "value1"]

    def test_to_csv_nested_data(self, helper):
        """测试嵌套数据的CSV格式化"""
        data = {
            "rules": [{"name": "rule1", "level": "warning"}],
            "stats": {"count": 5},
        }
        result = helper.to_csv(data)
        # 应该提取rules列表
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["name"] == "rule1"

    def test_to_csv_with_nested_values(self, helper):
        """测试包含嵌套值的CSV格式化"""
        data = [{"name": "a", "config": {"x": 1}}]
        result = helper.to_csv(data)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 1
        # 嵌套值应被JSON序列化
        assert "x" in rows[0]["config"]


# ==================== TestU27Export ====================

class TestU27Export:
    """测试导出功能"""

    def test_export_json_to_string(self, helper):
        """测试导出JSON到字符串"""
        result = helper.export("perf", "json")
        assert result is not None
        data = json.loads(result)
        assert "summary" in data

    def test_export_csv_to_string(self, helper):
        """测试导出CSV到字符串"""
        result = helper.export("alerts", "csv")
        assert result is not None
        assert isinstance(result, str)

    def test_export_json_to_file(self, helper, tmp_path):
        """测试导出JSON到文件"""
        output_file = str(tmp_path / "export.json")
        result = helper.export("perf", "json", output_file)
        assert result == output_file
        assert os.path.exists(output_file)
        with open(output_file, 'r') as f:
            data = json.load(f)
        assert "summary" in data

    def test_export_csv_to_file(self, helper, tmp_path):
        """测试导出CSV到文件"""
        output_file = str(tmp_path / "export.csv")
        result = helper.export("alerts", "csv", output_file)
        assert result == output_file
        assert os.path.exists(output_file)

    def test_export_invalid_source(self, helper):
        """测试无效数据源"""
        result = helper.export("invalid_source", "json")
        assert result is None

    def test_export_all_sources(self, helper):
        """测试所有数据源都能导出"""
        for source in ["status", "health", "alerts", "perf", "snapshot"]:
            result = helper.export(source, "json")
            assert result is not None
            data = json.loads(result)
            assert "export_time" in data

    def test_export_all_sources_csv(self, helper):
        """测试所有数据源都能导出CSV"""
        for source in ["status", "health", "alerts", "perf", "snapshot"]:
            result = helper.export(source, "csv")
            assert result is not None
            assert isinstance(result, str)


# ==================== TestU27CliIntegration ====================

class TestU27CliIntegration:
    """测试CLI集成"""

    def test_export_parser_exists(self):
        """测试export子命令parser存在"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["export"])
        assert args.command == "export"

    def test_export_with_source_parser(self):
        """测试export带数据源"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["export", "perf"])
        assert args.source == "perf"
        assert args.format == "json"

    def test_export_with_format_parser(self):
        """测试export带格式"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["export", "alerts", "--format", "csv"])
        assert args.source == "alerts"
        assert args.format == "csv"

    def test_export_with_output_parser(self):
        """测试export带输出路径"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["export", "perf", "--output", "/tmp/out.json"])
        assert args.output == "/tmp/out.json"

    def test_export_list_parser(self):
        """测试export --list"""
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["export", "--list"])
        assert args.list is True

    def test_cmd_export_list(self, capsys):
        """测试_cmd_export --list"""
        from cli import _cmd_export, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["export", "--list"])
        ret = _cmd_export(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "可用导出数据源" in out

    def test_cmd_export_no_source(self, capsys):
        """测试_cmd_export无数据源"""
        from cli import _cmd_export, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["export"])
        ret = _cmd_export(args)
        assert ret == 1
        out = capsys.readouterr().out
        assert "请指定数据源" in out

    def test_cmd_export_perf_json(self, capsys, fresh_perf_monitor):
        """测试_cmd_export perf JSON"""
        from cli import _cmd_export, _create_parser
        fresh_perf_monitor.record("test_op", 0.01)
        parser = _create_parser()
        args = parser.parse_args(["export", "perf"])
        ret = _cmd_export(args)
        assert ret == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "summary" in data

    def test_cmd_export_alerts_csv(self, capsys):
        """测试_cmd_export alerts CSV"""
        from cli import _cmd_export, _create_parser
        parser = _create_parser()
        args = parser.parse_args(["export", "alerts", "--format", "csv"])
        ret = _cmd_export(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert isinstance(out, str)

    def test_cmd_export_to_file(self, capsys, tmp_path, fresh_perf_monitor):
        """测试_cmd_export到文件"""
        from cli import _cmd_export, _create_parser
        output_file = str(tmp_path / "export.json")
        fresh_perf_monitor.record("test_op", 0.01)
        parser = _create_parser()
        args = parser.parse_args(["export", "perf", "--output", output_file])
        ret = _cmd_export(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "已导出到" in out
        assert os.path.exists(output_file)

    def test_cli_main_export_dispatch(self, capsys, fresh_perf_monitor):
        """测试cli_main分发export命令"""
        from cli import cli_main
        fresh_perf_monitor.record("test_op", 0.01)
        ret = cli_main(["export", "perf"])
        assert ret == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "summary" in data


# ==================== TestU27NoRegression ====================

class TestU27NoRegression:
    """验证U27不破坏既有功能"""

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

    def test_u26_snapshot_still_works(self):
        """U26 snapshot_cli模块仍可导入"""
        from core.snapshot_cli import SnapshotCliHelper
        assert SnapshotCliHelper is not None

    def test_existing_commands_still_dispatch(self):
        """既有命令仍可正常分发"""
        from cli import _create_parser
        parser = _create_parser()
        for cmd in ["run", "status", "snapshot", "logs", "alerts", "perf", "agents", "export"]:
            args = parser.parse_args([cmd])
            assert args.command == cmd
