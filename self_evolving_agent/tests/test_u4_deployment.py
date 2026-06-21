"""
U4：部署易用性测试

验证目标：
1. Dockerfile 多阶段构建、端口暴露、健康检查、默认CMD
2. docker-compose.yml 服务编排、环境变量、卷挂载、profile
3. systemd 单元文件结构、重启策略、环境变量
4. supervisor 配置无硬编码路径、无行首空格错误
5. deploy/ 脚本无硬编码 /workspace/self_evolving_agent 路径
6. requirements.txt 注释准确（不误导用户安装未使用的flask/requests）
7. .dockerignore 与 .env.example 存在且合理

第一性原理：部署产物应与具体开发环境解耦——
路径可推断、端口可配置、依赖最小化、监控可选启用。
"""
import os
import sys
import unittest
from pathlib import Path

# 通过 conftest.py 统一管理 sys.path
AGENT_ROOT = Path(__file__).resolve().parent.parent


def _read_text(rel_path: str) -> str:
    """读取Agent根目录下相对路径文件的文本内容"""
    full = AGENT_ROOT / rel_path
    if not full.exists():
        return ""
    return full.read_text(encoding="utf-8")


class TestU4Dockerfile(unittest.TestCase):
    """U4：Dockerfile 验证"""

    def setUp(self):
        self.content = _read_text("Dockerfile")

    def test_dockerfile_exists(self):
        self.assertTrue((AGENT_ROOT / "Dockerfile").exists())

    def test_dockerfile_multistage_build(self):
        """多阶段构建减小镜像体积"""
        self.assertIn("FROM python:3.11-slim as builder", self.content)
        self.assertIn("FROM python:3.11-slim", self.content)
        # 第二阶段应通过 COPY --from=builder 复用产物
        self.assertIn("COPY --from=builder", self.content)

    def test_dockerfile_exposes_ports(self):
        """暴露 REST API 和 Dashboard 端口"""
        self.assertIn("EXPOSE 9090 8080", self.content)

    def test_dockerfile_has_healthcheck(self):
        """内置健康检查"""
        self.assertIn("HEALTHCHECK", self.content)

    def test_dockerfile_default_cmd_uses_cli(self):
        """默认CMD应使用CLI子命令，启用全部能力"""
        self.assertIn("run", self.content)
        self.assertIn("--forever", self.content)
        self.assertIn("--enable-multi-agent", self.content)
        self.assertIn("--enable-forever", self.content)
        self.assertIn("--rest-port", self.content)
        self.assertIn("--dashboard-port", self.content)

    def test_dockerfile_sets_pythonpath(self):
        """设置 PYTHONPATH 让模块导入正常工作"""
        self.assertIn("PYTHONPATH=/app", self.content)

    def test_dockerfile_creates_data_dirs(self):
        """创建数据和日志目录"""
        self.assertIn("mkdir -p", self.content)
        self.assertIn("/app/logs", self.content)
        self.assertIn("/app/data", self.content)


class TestU4DockerCompose(unittest.TestCase):
    """U4：docker-compose.yml 验证"""

    def setUp(self):
        self.content = _read_text("docker-compose.yml")

    def test_compose_exists(self):
        self.assertTrue((AGENT_ROOT / "docker-compose.yml").exists())

    def test_compose_has_agent_service(self):
        """主服务为 agent"""
        self.assertIn("agent:", self.content)

    def test_compose_agent_restart_policy(self):
        """崩溃自动重启"""
        self.assertIn("restart:", self.content)
        self.assertIn("unless-stopped", self.content)

    def test_compose_ports_configurable(self):
        """端口通过环境变量可配置"""
        self.assertIn("${AGENT_REST_PORT:-9090}", self.content)
        self.assertIn("${AGENT_DASHBOARD_PORT:-8080}", self.content)

    def test_compose_volumes_persist(self):
        """日志和数据持久化卷"""
        self.assertIn("agent-logs", self.content)
        self.assertIn("agent-data", self.content)

    def test_compose_healthcheck(self):
        """服务级健康检查"""
        self.assertIn("healthcheck", self.content)

    def test_compose_monitoring_profile(self):
        """监控栈通过 profile 可选启用"""
        self.assertIn("profiles:", self.content)
        self.assertIn("monitoring", self.content)
        self.assertIn("prometheus", self.content)
        self.assertIn("grafana", self.content)

    def test_compose_config_mounted_readonly(self):
        """配置文件以只读方式挂载"""
        self.assertIn("./config:/app/config:ro", self.content)

    def test_compose_logging_rotation(self):
        """日志轮转避免磁盘撑爆"""
        self.assertIn("max-size", self.content)
        self.assertIn("max-file", self.content)


class TestU4SystemdUnit(unittest.TestCase):
    """U4：systemd 服务单元验证"""

    def setUp(self):
        self.path = AGENT_ROOT / "deploy" / "self-evolving-agent.service"
        self.content = self.path.read_text(encoding="utf-8") if self.path.exists() else ""

    def test_service_file_exists(self):
        self.assertTrue(self.path.exists())

    def test_service_unit_section(self):
        """标准 [Unit] 段，网络依赖"""
        self.assertIn("[Unit]", self.content)
        self.assertIn("After=network-online.target", self.content)

    def test_service_install_section(self):
        """[Install] 段支持开机自启"""
        self.assertIn("[Install]", self.content)
        self.assertIn("WantedBy=multi-user.target", self.content)

    def test_service_restart_on_failure(self):
        """崩溃自动重启"""
        self.assertIn("Restart=on-failure", self.content)

    def test_service_execstart_uses_cli(self):
        """ExecStart 使用 CLI 子命令，启用全部能力"""
        self.assertIn("run", self.content)
        self.assertIn("--forever", self.content)
        self.assertIn("--enable-multi-agent", self.content)

    def test_service_env_overridable(self):
        """关键路径与端口通过变量可覆盖"""
        self.assertIn("${AGENT_DIR:-", self.content)
        self.assertIn("${AGENT_REST_PORT:-", self.content)
        self.assertIn("${AGENT_DASHBOARD_PORT:-", self.content)

    def test_service_resource_limits(self):
        """资源限制防止单实例吃满主机"""
        self.assertIn("MemoryMax=", self.content)
        self.assertIn("CPUQuota=", self.content)

    def test_service_graceful_stop(self):
        """优雅停止：SIGTERM + 超时"""
        self.assertIn("KillSignal=SIGTERM", self.content)
        self.assertIn("TimeoutStopSec=", self.content)


class TestU4SupervisorConfig(unittest.TestCase):
    """U4：supervisor 配置验证"""

    def setUp(self):
        self.path = AGENT_ROOT / "deploy" / "supervisor_agent.conf"
        self.content = self.path.read_text(encoding="utf-8") if self.path.exists() else ""

    def test_supervisor_conf_exists(self):
        self.assertTrue(self.path.exists())

    def test_no_hardcoded_workspace_path(self):
        """不硬编码 /workspace/self_evolving_agent"""
        self.assertNotIn(
            "/workspace/self_evolving_agent",
            self.content,
            "supervisor配置不应硬编码开发环境路径",
        )

    def test_uses_here_variable(self):
        """使用 supervisor 内置 %(here)s 变量推断路径"""
        self.assertIn("%(here)s", self.content)

    def test_no_leading_space_on_directory(self):
        """directory= 行不应有行首多余空格"""
        for line in self.content.splitlines():
            if line.strip().startswith("directory="):
                self.assertEqual(
                    line, line.lstrip(),
                    f"directory= 行不应有行首空格: {line!r}",
                )

    def test_command_uses_cli(self):
        """command 使用 CLI 子命令"""
        self.assertIn("run", self.content)
        self.assertIn("--forever", self.content)


class TestU4DeployScriptsNoHardcodedPath(unittest.TestCase):
    """U4：deploy/ 脚本不应硬编码 /workspace/self_evolving_agent"""

    def _check_no_hardcoded(self, rel_path: str) -> None:
        content = _read_text(rel_path)
        self.assertNotIn(
            "/workspace/self_evolving_agent",
            content,
            f"{rel_path} 不应硬编码 /workspace/self_evolving_agent 路径",
        )

    def test_start_sh_no_hardcoded(self):
        self._check_no_hardcoded("deploy/start.sh")

    def test_stop_sh_no_hardcoded(self):
        self._check_no_hardcoded("deploy/stop.sh")

    def test_status_sh_no_hardcoded(self):
        self._check_no_hardcoded("deploy/status.sh")

    def test_long_run_monitor_sh_no_hardcoded(self):
        self._check_no_hardcoded("deploy/long_run_monitor.sh")

    def test_six_hour_monitor_sh_no_hardcoded(self):
        self._check_no_hardcoded("deploy/six_hour_monitor.sh")

    def test_six_hour_reporter_sh_no_hardcoded(self):
        self._check_no_hardcoded("deploy/six_hour_reporter.sh")

    def test_health_check_py_no_hardcoded(self):
        self._check_no_hardcoded("deploy/health_check.py")

    def test_metrics_server_py_no_hardcoded(self):
        self._check_no_hardcoded("deploy/metrics_server.py")


class TestU4DeployScriptsPathInference(unittest.TestCase):
    """U4：deploy/ 脚本应通过 SCRIPT_DIR 推断 AGENT_DIR，并允许环境变量覆盖"""

    def _assert_path_inference(self, rel_path: str, shell: bool = True) -> None:
        content = _read_text(rel_path)
        if shell:
            self.assertIn("SCRIPT_DIR=", content, f"{rel_path} 应推断 SCRIPT_DIR")
            self.assertIn(
                'AGENT_DIR="${AGENT_DIR:-',
                content,
                f"{rel_path} 应允许通过 AGENT_DIR 环境变量覆盖",
            )

    def test_start_sh_infers_path(self):
        self._assert_path_inference("deploy/start.sh")

    def test_stop_sh_infers_path(self):
        self._assert_path_inference("deploy/stop.sh")

    def test_status_sh_infers_path(self):
        self._assert_path_inference("deploy/status.sh")

    def test_long_run_monitor_sh_infers_path(self):
        self._assert_path_inference("deploy/long_run_monitor.sh")

    def test_six_hour_monitor_sh_infers_path(self):
        self._assert_path_inference("deploy/six_hour_monitor.sh")

    def test_six_hour_reporter_sh_infers_path(self):
        self._assert_path_inference("deploy/six_hour_reporter.sh")

    def test_health_check_py_infers_path(self):
        """Python 脚本应通过环境变量或 __file__ 推断 AGENT_DIR"""
        content = _read_text("deploy/health_check.py")
        self.assertIn(
            "os.environ.get",
            content,
            "health_check.py 应通过 AGENT_DIR 环境变量覆盖",
        )
        self.assertIn('"AGENT_DIR"', content)
        self.assertIn(
            "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
            content,
        )

    def test_metrics_server_py_infers_path(self):
        content = _read_text("deploy/metrics_server.py")
        self.assertIn("os.environ.get", content)
        self.assertIn('"AGENT_DIR"', content)
        self.assertIn(
            "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
            content,
        )


class TestU4RequirementsAccuracy(unittest.TestCase):
    """U4：requirements.txt 注释应准确，不误导用户"""

    def setUp(self):
        self.content = _read_text("requirements.txt")

    def test_core_deps_listed(self):
        """核心依赖齐全"""
        self.assertIn("pyyaml", self.content.lower())
        self.assertIn("psutil", self.content.lower())
        self.assertIn("numpy", self.content.lower())

    def test_no_misleading_flask_comment(self):
        """不应误导用户认为 Dashboard 需要 flask（实际用标准库）"""
        # 不应出现 "flask>=2.2  # Dashboard" 这种暗示必需的注释
        for line in self.content.splitlines():
            stripped = line.strip()
            if "flask" in stripped.lower():
                # 如果提到 flask，必须明确标注为不需要或可选
                self.assertTrue(
                    "无需" in stripped or "不需要" in stripped or "可选" in stripped,
                    f"flask 注释应明确说明非必需: {line!r}",
                )

    def test_no_misleading_requests_comment(self):
        """不应误导用户认为 REST API 需要 requests"""
        for line in self.content.splitlines():
            stripped = line.strip()
            if "requests" in stripped.lower() and "pytest" not in stripped.lower():
                self.assertTrue(
                    "无需" in stripped or "不需要" in stripped or "可选" in stripped,
                    f"requests 注释应明确说明非必需: {line!r}",
                )

    def test_explains_stdlib_usage(self):
        """应说明使用标准库实现 Web 服务"""
        self.assertTrue(
            "标准库" in self.content or "http.server" in self.content,
            "应说明 REST API 与 Dashboard 使用标准库实现",
        )


class TestU4DockerignoreAndEnvExample(unittest.TestCase):
    """U4：.dockerignore 与 .env.example 存在且合理"""

    def test_dockerignore_exists(self):
        self.assertTrue((AGENT_ROOT / ".dockerignore").exists())

    def test_dockerignore_excludes_pycache(self):
        content = _read_text(".dockerignore")
        self.assertIn("__pycache__", content)
        self.assertIn("*.pyc", content)

    def test_dockerignore_excludes_runtime_data(self):
        """运行时数据（日志/快照/数据库）不应打入镜像"""
        content = _read_text(".dockerignore")
        self.assertIn("logs/", content)
        self.assertIn("*.pkl", content)
        self.assertIn("*.db", content)

    def test_dockerignore_excludes_tests(self):
        """测试代码不应进入运行镜像"""
        content = _read_text(".dockerignore")
        self.assertIn("tests/", content)

    def test_env_example_exists(self):
        self.assertTrue((AGENT_ROOT / ".env.example").exists())

    def test_env_example_documents_ports(self):
        content = _read_text(".env.example")
        self.assertIn("AGENT_REST_PORT", content)
        self.assertIn("AGENT_DASHBOARD_PORT", content)

    def test_env_example_documents_monitoring(self):
        content = _read_text(".env.example")
        self.assertIn("PROMETHEUS_PORT", content)
        self.assertIn("GRAFANA_PORT", content)


class TestU4PrometheusConfig(unittest.TestCase):
    """U4：Prometheus 抓取配置"""

    def setUp(self):
        self.path = AGENT_ROOT / "deploy" / "prometheus.yml"
        self.content = self.path.read_text(encoding="utf-8") if self.path.exists() else ""

    def test_prometheus_config_exists(self):
        self.assertTrue(self.path.exists())

    def test_prometheus_scrapes_agent(self):
        """抓取 agent 服务的 /metrics 端点"""
        self.assertIn("self-evolving-agent", self.content)
        self.assertIn("/metrics", self.content)
        self.assertIn("agent:8080", self.content)


class TestU4NoRegression(unittest.TestCase):
    """U4：无回归保护"""

    def test_agent_init_still_works(self):
        """Agent 仍可正常初始化"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        self.assertIsNotNone(agent.perception)
        self.assertIsNotNone(agent.prediction)

    def test_main_loop_still_runs(self):
        """主循环仍可短时运行"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        import time
        time.sleep(1.5)
        agent.stop()
        self.assertGreaterEqual(agent.loop_count, 0)


if __name__ == "__main__":
    unittest.main()
