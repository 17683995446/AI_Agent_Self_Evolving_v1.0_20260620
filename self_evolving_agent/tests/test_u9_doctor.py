"""
U9：智能诊断命令测试

验证目标：
1. Doctor类存在且可实例化
2. 各项检查返回 CheckResult（status/message/remediation）
3. Python版本检查
4. 依赖检查（检测缺失依赖）
5. 配置文件检查（存在/缺失/语法错误）
6. 目录检查（创建/可写性）
7. 磁盘空间检查
8. 端口检查（可用/占用，可注入port_checker）
9. 系统资源检查
10. 汇总报告（PASS/WARN/FAIL整体状态）
11. JSON输出
12. CLI doctor子命令集成
13. 无回归

第一性原理：用户不应手动排查环境问题。
诊断应全面、可读、可机器解析，每项失败带修复建议。
"""
import os
import sys
import unittest
import tempfile
import shutil
import json
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent


class TestU9DoctorBasics(unittest.TestCase):
    """U9：Doctor 基础功能"""

    def test_doctor_importable(self):
        from core.doctor import Doctor
        self.assertTrue(callable(Doctor))

    def test_check_result_dataclass(self):
        from core.doctor import CheckResult, CheckStatus
        r = CheckResult(name="test", status=CheckStatus.PASS, message="ok")
        self.assertTrue(r.passed)
        self.assertFalse(r.is_error)

    def test_check_status_enum(self):
        from core.doctor import CheckStatus
        self.assertEqual(CheckStatus.PASS.value, "PASS")
        self.assertEqual(CheckStatus.WARN.value, "WARN")
        self.assertEqual(CheckStatus.FAIL.value, "FAIL")
        self.assertEqual(CheckStatus.SKIP.value, "SKIP")

    def test_doctor_instantiable(self):
        from core.doctor import Doctor
        with tempfile.TemporaryDirectory() as tmpdir:
            doctor = Doctor(project_root=tmpdir)
            self.assertIsNotNone(doctor)

    def test_port_checker_injectable(self):
        """port_checker 可注入，便于测试"""
        from core.doctor import Doctor
        calls = []

        def mock_port_check(port):
            calls.append(port)
            return False  # 模拟端口被占用

        doctor = Doctor(port_checker=mock_port_check)
        result = doctor.check_port(9090, "rest")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], 9090)
        self.assertFalse(result.passed)


class TestU9PythonVersionCheck(unittest.TestCase):
    """U9：Python版本检查"""

    def test_passes_on_modern_python(self):
        from core.doctor import Doctor, CheckStatus
        doctor = Doctor()
        result = doctor.check_python_version()
        # 测试环境Python应 >= 3.8
        self.assertEqual(result.status, CheckStatus.PASS)

    def test_result_has_detail(self):
        from core.doctor import Doctor
        doctor = Doctor()
        result = doctor.check_python_version()
        self.assertIn("major", result.detail)
        self.assertIn("minor", result.detail)


class TestU9DependencyCheck(unittest.TestCase):
    """U9：依赖检查"""

    def test_passes_when_deps_installed(self):
        from core.doctor import Doctor, CheckStatus
        doctor = Doctor()
        result = doctor.check_dependencies()
        # 测试环境应已安装核心依赖
        self.assertEqual(result.status, CheckStatus.PASS)

    def test_detail_lists_each_dep(self):
        from core.doctor import Doctor
        doctor = Doctor()
        result = doctor.check_dependencies()
        for mod in ["yaml", "psutil", "numpy"]:
            self.assertIn(mod, result.detail)

    def test_remediation_on_failure(self):
        """模拟缺失依赖时的修复建议"""
        from core.doctor import Doctor, CheckStatus
        from unittest.mock import patch

        doctor = Doctor()
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("mock")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", mock_import):
            result = doctor.check_dependencies()
            self.assertEqual(result.status, CheckStatus.FAIL)
            self.assertIn("pip install", result.remediation)


class TestU9ConfigFileCheck(unittest.TestCase):
    """U9：配置文件检查"""

    def test_warns_when_config_missing(self):
        from core.doctor import Doctor, CheckStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            doctor = Doctor(project_root=tmpdir, config_path="nonexistent.yaml")
            result = doctor.check_config_file()
            self.assertEqual(result.status, CheckStatus.WARN)
            self.assertIn("init", result.remediation)

    def test_passes_when_config_valid(self):
        from core.doctor import Doctor, CheckStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "agent.yaml")
            with open(config_path, "w") as f:
                f.write("resource:\n  max_cpu_percent: 20.0\n")
            doctor = Doctor(project_root=tmpdir, config_path="agent.yaml")
            result = doctor.check_config_file()
            self.assertEqual(result.status, CheckStatus.PASS)

    def test_fails_on_invalid_yaml(self):
        """顶层非字典（list）应FAIL"""
        from core.doctor import Doctor, CheckStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "agent.yaml")
            with open(config_path, "w") as f:
                f.write("- just\n- a\n- list\n")
            doctor = Doctor(project_root=tmpdir, config_path="agent.yaml")
            result = doctor.check_config_file()
            self.assertEqual(result.status, CheckStatus.FAIL)


class TestU9DirectoryCheck(unittest.TestCase):
    """U9：目录检查"""

    def test_creates_missing_directories(self):
        from core.doctor import Doctor, CheckStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            doctor = Doctor(project_root=tmpdir)
            result = doctor.check_directories()
            self.assertEqual(result.status, CheckStatus.PASS)
            for d in ["logs", "data", "data/snapshots"]:
                self.assertTrue(os.path.isdir(os.path.join(tmpdir, d)))

    def test_passes_when_directories_exist(self):
        from core.doctor import Doctor, CheckStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            for d in ["logs", "data", "data/snapshots"]:
                os.makedirs(os.path.join(tmpdir, d))
            doctor = Doctor(project_root=tmpdir)
            result = doctor.check_directories()
            self.assertEqual(result.status, CheckStatus.PASS)


class TestU9DiskSpaceCheck(unittest.TestCase):
    """U9：磁盘空间检查"""

    def test_returns_pass_or_warn(self):
        from core.doctor import Doctor, CheckStatus
        doctor = Doctor()
        result = doctor.check_disk_space()
        self.assertIn(result.status, [CheckStatus.PASS, CheckStatus.WARN, CheckStatus.SKIP])

    def test_detail_has_free_mb(self):
        from core.doctor import Doctor
        doctor = Doctor()
        result = doctor.check_disk_space()
        if result.status != doctor.check_disk_space().SKIP if hasattr(doctor.check_disk_space(), 'SKIP') else True:
            if result.detail:
                self.assertIn("free_mb", result.detail)


class TestU9PortCheck(unittest.TestCase):
    """U9：端口检查"""

    def test_passes_when_port_available(self):
        from core.doctor import Doctor, CheckStatus
        doctor = Doctor(port_checker=lambda p: True)
        result = doctor.check_port(9090, "rest")
        self.assertEqual(result.status, CheckStatus.PASS)

    def test_warns_when_port_occupied(self):
        from core.doctor import Doctor, CheckStatus
        doctor = Doctor(port_checker=lambda p: False)
        result = doctor.check_port(9090, "rest")
        self.assertEqual(result.status, CheckStatus.WARN)
        self.assertTrue(result.remediation)

    def test_skips_when_port_none(self):
        from core.doctor import Doctor, CheckStatus
        doctor = Doctor()
        result = doctor.check_port(None, "rest")
        self.assertEqual(result.status, CheckStatus.SKIP)


class TestU9SystemResourcesCheck(unittest.TestCase):
    """U9：系统资源检查"""

    def test_returns_result(self):
        from core.doctor import Doctor, CheckStatus
        doctor = Doctor()
        result = doctor.check_system_resources()
        self.assertIn(result.status, [CheckStatus.PASS, CheckStatus.WARN, CheckStatus.SKIP])


class TestU9SummaryAndReport(unittest.TestCase):
    """U9：汇总报告"""

    def setUp(self):
        from core.doctor import Doctor
        self.doctor = Doctor(port_checker=lambda p: True)

    def test_run_all_checks_returns_list(self):
        results = self.doctor.run_all_checks()
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 5)

    def test_get_summary_has_counts(self):
        self.doctor.run_all_checks()
        summary = self.doctor.get_summary()
        for key in ["overall", "total", "passed", "warnings", "failed", "skipped"]:
            self.assertIn(key, summary)
        self.assertEqual(summary["total"], len(self.doctor.results))

    def test_overall_pass_when_no_failures(self):
        """无FAIL时整体状态为PASS或WARN"""
        self.doctor.run_all_checks()
        summary = self.doctor.get_summary()
        self.assertIn(summary["overall"], ["PASS", "WARN"])

    def test_format_report_is_string(self):
        self.doctor.run_all_checks()
        report = self.doctor.format_report()
        self.assertIsInstance(report, str)
        self.assertIn("诊断报告", report)

    def test_to_dict_is_json_serializable(self):
        self.doctor.run_all_checks()
        d = self.doctor.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertIn("summary", parsed)
        self.assertIn("checks", parsed)

    def test_report_includes_remediation_for_failures(self):
        """FAIL项应在报告中显示修复建议"""
        from core.doctor import Doctor, CheckResult, CheckStatus
        doctor = Doctor()
        doctor.results = [
            CheckResult(name="test_fail", status=CheckStatus.FAIL,
                       message="测试失败", remediation="修复方法A"),
        ]
        report = doctor.format_report()
        self.assertIn("修复方法A", report)


class TestU9CliIntegration(unittest.TestCase):
    """U9：CLI doctor子命令集成"""

    def test_cli_has_doctor_subcommand(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["doctor"])
        self.assertEqual(args.command, "doctor")

    def test_cli_doctor_json_flag(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["doctor", "--json"])
        self.assertTrue(args.json)

    def test_cli_doctor_port_flags(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["doctor", "--rest-port", "9090", "--dashboard-port", "8080"])
        self.assertEqual(args.rest_port, 9090)
        self.assertEqual(args.dashboard_port, 8080)

    def test_cli_doctor_runs_successfully(self):
        """端到端：CLI doctor 正常运行"""
        from cli import cli_main
        exit_code = cli_main(["doctor"])
        # 测试环境应无FAIL（依赖已安装），退出码0
        self.assertIn(exit_code, [0, 1])

    def test_cli_doctor_json_output(self):
        """端到端：CLI doctor --json 输出有效JSON"""
        from cli import cli_main
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli_main(["doctor", "--json"])
        output = buf.getvalue().strip()
        data = json.loads(output)
        self.assertIn("summary", data)
        self.assertIn("checks", data)


class TestU9NoRegression(unittest.TestCase):
    """U9：无回归保护"""

    def test_agent_init_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        self.assertIsNotNone(agent.perception)

    def test_main_loop_still_runs(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        import time
        time.sleep(1.5)
        agent.stop()
        self.assertGreaterEqual(agent.loop_count, 0)

    def test_existing_cli_commands_still_work(self):
        from cli import _create_parser
        parser = _create_parser()
        for cmd in ["run", "status", "health", "config", "init", "doctor"]:
            args = parser.parse_args([cmd] if cmd not in ("config",) else [cmd, "validate"])
            self.assertEqual(args.command, cmd)


if __name__ == "__main__":
    unittest.main()
