"""
真实场景接入测试 —— 验证系统可对接真实业务场景
"""
import sys
import os
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.log_adapter import LogFileAdapter
from adapters.action_adapter import ExternalActionAdapter


class TestLogAdapter:
    """测试日志适配器"""

    def test_analyze_sample_log(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("INFO 服务启动\n")
            f.write("ERROR 数据库连接失败\n")
            f.write("WARNING 响应时间超过阈值\n")
            f.write("INFO 请求处理完成\n")
            f.write("ERROR NullPointerException at line 42\n")
            path = f.name

        try:
            adapter = LogFileAdapter(path)
            error_rate, entropy, errors = adapter.analyze()
            assert error_rate == 0.4  # 2/5
            assert len(errors) == 2
            assert "数据库连接失败" in errors[0]

            # 第二次分析应无新内容
            error_rate2, entropy2, errors2 = adapter.analyze()
            assert error_rate2 == 0.0
            assert len(errors2) == 0
        finally:
            os.unlink(path)

    def test_analyze_nonexistent_file(self):
        adapter = LogFileAdapter("/nonexistent/path.log")
        error_rate, entropy, errors = adapter.analyze()
        assert error_rate == 0.0
        assert len(errors) == 0

    def test_entropy_calculation(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            # 高度规律的日志
            for _ in range(100):
                f.write("INFO 正常请求\n")
            path = f.name

        try:
            adapter = LogFileAdapter(path)
            _, entropy, _ = adapter.analyze()
            assert entropy < 1.0  # 规律日志熵低
        finally:
            os.unlink(path)

    def test_custom_error_patterns(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("CUSTOM_ERROR 特殊错误\n")
            f.write("INFO 正常\n")
            path = f.name

        try:
            adapter = LogFileAdapter(path, error_patterns=[r"CUSTOM_ERROR"])
            error_rate, _, errors = adapter.analyze()
            assert error_rate == 0.5
            assert len(errors) == 1
        finally:
            os.unlink(path)


class TestActionAdapter:
    """测试外部动作适配器"""

    def test_execute_command_echo(self):
        adapter = ExternalActionAdapter(timeout=5.0)
        result = adapter.execute_command("echo hello")
        assert result["success"] is True
        assert "hello" in result["stdout"]

    def test_execute_command_failure(self):
        adapter = ExternalActionAdapter(timeout=5.0)
        result = adapter.execute_command("false")
        assert result["success"] is False
        assert result["returncode"] == 1

    def test_execute_command_timeout(self):
        adapter = ExternalActionAdapter(timeout=0.1)
        result = adapter.execute_command("sleep 10")
        assert result["success"] is False
        assert "超时" in result["error"]

    def test_script_whitelist(self):
        adapter = ExternalActionAdapter(timeout=5.0, script_whitelist=["/allowed/script.sh"])
        result = adapter.execute_script("/not/allowed.sh")
        assert result["success"] is False
        assert "白名单" in result["error"]

    def test_stats(self):
        adapter = ExternalActionAdapter(timeout=5.0)
        adapter.execute_command("echo test1")
        adapter.execute_command("echo test2")
        stats = adapter.get_stats()
        assert stats["total_executions"] == 2
        assert stats["success_count"] == 2
        assert stats["success_rate"] == 1.0

    def test_http_callback_mock(self):
        # 使用一个不可达的URL测试错误处理
        adapter = ExternalActionAdapter(timeout=1.0)
        result = adapter.execute_http("http://127.0.0.1:1/test", {"key": "value"})
        assert result["success"] is False
        assert "error" in result


class TestScenarioConfig:
    """测试场景配置"""

    def test_scenarios_yaml_exists(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "scenarios.yaml")
        assert os.path.exists(path), f"场景配置文件不存在: {path}"

    def test_scenarios_yaml_loadable(self):
        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "scenarios.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "scenarios" in data
        assert "system_monitor" in data["scenarios"]
        assert "log_anomaly_detection" in data["scenarios"]

    def test_scenario_structure(self):
        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "scenarios.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        for name, scenario in data["scenarios"].items():
            assert "description" in scenario
            assert "modules" in scenario
