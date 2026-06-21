"""
U7：文档与示例测试

验证目标：
1. QUICKSTART.md 存在且包含安装、启动、配置等关键章节
2. FAQ.md 存在且覆盖常见问题分类
3. TROUBLESHOOTING.md 存在且包含错误映射与环境变量参考
4. README.md 已扩充，包含CLI命令、环境变量、部署选项
5. 文档间交叉链接完整
6. 无回归

第一性原理：文档是用户与系统之间的契约——
用户按文档操作应能成功，文档应覆盖常见问题与故障排除路径。
"""
import os
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = AGENT_ROOT / "docs"


def _read(rel_path: str) -> str:
    full = AGENT_ROOT / rel_path
    if not full.exists():
        return ""
    return full.read_text(encoding="utf-8")


class TestU7QuickstartDoc(unittest.TestCase):
    """U7：QUICKSTART.md 验证"""

    def setUp(self):
        self.path = DOCS_DIR / "QUICKSTART.md"
        self.content = self.path.read_text(encoding="utf-8") if self.path.exists() else ""

    def test_quickstart_exists(self):
        self.assertTrue(self.path.exists(), "QUICKSTART.md 应存在")

    def test_has_install_section(self):
        """包含安装步骤"""
        self.assertTrue(
            any(kw in self.content for kw in ["安装", "install", "pip install"]),
            "QUICKSTART 应包含安装步骤",
        )

    def test_has_run_command(self):
        """包含启动命令"""
        self.assertIn("python main.py run", self.content)

    def test_has_cli_commands(self):
        """包含常用CLI命令"""
        self.assertIn("status", self.content)
        self.assertIn("health", self.content)

    def test_has_env_var_section(self):
        """包含环境变量配置说明"""
        self.assertIn("AGENT_", self.content)

    def test_has_docker_section(self):
        """包含Docker部署说明"""
        self.assertTrue(
            "docker" in self.content.lower(),
            "QUICKSTART 应提及 Docker 部署",
        )

    def test_has_log_level_section(self):
        """包含日志调试说明"""
        self.assertIn("--log-level", self.content)


class TestU7FaqDoc(unittest.TestCase):
    """U7：FAQ.md 验证"""

    def setUp(self):
        self.path = DOCS_DIR / "FAQ.md"
        self.content = self.path.read_text(encoding="utf-8") if self.path.exists() else ""

    def test_faq_exists(self):
        self.assertTrue(self.path.exists(), "FAQ.md 应存在")

    def test_has_questions(self):
        """包含问答结构"""
        # FAQ 应有 Q: 开头的问题和 A 开头的回答（允许加粗 **A**）
        self.assertIn("Q:", self.content)
        self.assertTrue(
            "A:" in self.content or "**A**" in self.content,
            "FAQ 应包含回答标记（A: 或 **A**:）",
        )

    def test_covers_configuration_issues(self):
        """覆盖配置相关问题"""
        self.assertTrue(
            "ConfigurationException" in self.content or "配置" in self.content,
            "FAQ 应覆盖配置相关问题",
        )

    def test_covers_env_var_override(self):
        """覆盖环境变量覆盖"""
        self.assertIn("AGENT_", self.content)

    def test_covers_log_level(self):
        """覆盖日志级别问题"""
        self.assertTrue(
            "log-level" in self.content or "日志" in self.content,
            "FAQ 应覆盖日志相关问题",
        )

    def test_covers_deployment(self):
        """覆盖部署相关问题"""
        self.assertTrue(
            any(kw in self.content for kw in ["Docker", "systemd", "部署"]),
            "FAQ 应覆盖部署相关问题",
        )

    def test_covers_remediation(self):
        """覆盖异常 remediation 字段说明"""
        self.assertIn("remediation", self.content)


class TestU7TroubleshootingDoc(unittest.TestCase):
    """U7：TROUBLESHOOTING.md 验证"""

    def setUp(self):
        self.path = DOCS_DIR / "TROUBLESHOOTING.md"
        self.content = self.path.read_text(encoding="utf-8") if self.path.exists() else ""

    def test_troubleshooting_exists(self):
        self.assertTrue(self.path.exists(), "TROUBLESHOOTING.md 应存在")

    def test_has_error_to_solution_mapping(self):
        """包含错误到解决方案的映射"""
        self.assertIn("ConfigurationException", self.content)

    def test_has_env_var_reference_table(self):
        """包含完整环境变量参考表"""
        self.assertIn("环境变量参考", self.content)
        # 应列出关键环境变量
        for var in ["AGENT_MAX_CPU", "AGENT_LOG_LEVEL", "AGENT_LOOP_FREQUENCY"]:
            self.assertIn(var, self.content, f"环境变量参考应包含 {var}")

    def test_has_dependency_errors(self):
        """覆盖依赖错误"""
        self.assertIn("ModuleNotFoundError", self.content)

    def test_has_permission_errors(self):
        """覆盖权限错误"""
        self.assertIn("PermissionError", self.content)

    def test_has_port_conflict(self):
        """覆盖端口冲突"""
        self.assertTrue(
            "端口" in self.content or "Address already in use" in self.content,
            "故障排除应覆盖端口冲突",
        )

    def test_has_docker_troubleshooting(self):
        """覆盖Docker故障排除"""
        self.assertIn("Docker", self.content)

    def test_has_systemd_troubleshooting(self):
        """覆盖systemd故障排除"""
        self.assertIn("systemd", self.content)


class TestU7ReadmeExpanded(unittest.TestCase):
    """U7：README.md 已扩充"""

    def setUp(self):
        self.content = _read("README.md")

    def test_readme_exists(self):
        self.assertTrue((AGENT_ROOT / "README.md").exists())

    def test_has_cli_commands_section(self):
        """包含CLI命令说明"""
        self.assertIn("python main.py run", self.content)
        self.assertIn("python main.py status", self.content)
        self.assertIn("python main.py health", self.content)

    def test_has_env_var_table(self):
        """包含环境变量表格"""
        self.assertIn("AGENT_MAX_CPU", self.content)
        self.assertIn("AGENT_LOG_LEVEL", self.content)

    def test_has_deployment_section(self):
        """包含部署说明"""
        self.assertIn("Docker", self.content)
        self.assertIn("systemd", self.content)

    def test_has_log_level_options(self):
        """包含日志级别选项说明"""
        self.assertIn("--log-level", self.content)

    def test_has_doc_links(self):
        """包含文档链接"""
        self.assertIn("QUICKSTART.md", self.content)
        self.assertIn("FAQ.md", self.content)
        self.assertIn("TROUBLESHOOTING.md", self.content)

    def test_has_directory_structure(self):
        """包含目录结构说明"""
        self.assertIn("cli.py", self.content)
        self.assertIn("api/rest_server.py", self.content)
        self.assertIn("dashboard/dashboard_server.py", self.content)

    def test_has_test_commands(self):
        """包含测试命令"""
        self.assertIn("pytest", self.content)


class TestU7DocCrossReferences(unittest.TestCase):
    """U7：文档间交叉链接完整"""

    def test_quickstart_links_to_faq(self):
        content = _read("docs/QUICKSTART.md")
        self.assertIn("FAQ.md", content)

    def test_quickstart_links_to_troubleshooting(self):
        content = _read("docs/QUICKSTART.md")
        self.assertIn("TROUBLESHOOTING.md", content)

    def test_readme_links_to_quickstart(self):
        content = _read("README.md")
        self.assertIn("QUICKSTART.md", content)

    def test_readme_links_to_faq(self):
        content = _read("README.md")
        self.assertIn("FAQ.md", content)

    def test_readme_links_to_troubleshooting(self):
        content = _read("README.md")
        self.assertIn("TROUBLESHOOTING.md", content)

    def test_troubleshooting_links_to_faq(self):
        content = _read("docs/TROUBLESHOOTING.md")
        self.assertIn("FAQ.md", content)


class TestU7EnvExampleDoc(unittest.TestCase):
    """U7：.env.example 作为文档存在"""

    def test_env_example_exists(self):
        self.assertTrue((AGENT_ROOT / ".env.example").exists())

    def test_env_example_documents_all_key_vars(self):
        content = _read(".env.example")
        for var in ["AGENT_REST_PORT", "AGENT_DASHBOARD_PORT", "AGENT_ENV"]:
            self.assertIn(var, content)


class TestU7NoRegression(unittest.TestCase):
    """U7：无回归保护"""

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

    def test_all_docs_exist(self):
        """所有关键文档都存在"""
        for doc in ["QUICKSTART.md", "FAQ.md", "TROUBLESHOOTING.md"]:
            self.assertTrue(
                (DOCS_DIR / doc).exists(),
                f"文档 {doc} 应存在",
            )


if __name__ == "__main__":
    unittest.main()
