"""
O7阶段验证测试 —— 文档与目录秩序

验证点：
1. README.md存在并包含必要内容
2. requirements.txt存在并声明依赖
3. docs/ARCHITECTURE.md存在
4. docs/API_REFERENCE.md存在
5. 目录结构规范
6. 配置文件存在
7. 不破坏现有功能
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestO7ReadmeExists:
    """验证README.md存在"""

    def test_readme_exists(self):
        """README.md应存在"""
        readme_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "README.md"
        )
        assert os.path.exists(readme_path), "README.md 应存在"

    def test_readme_contains_quick_start(self):
        """README应包含快速开始指南"""
        readme_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "README.md"
        )
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "快速开始" in content or "Quick Start" in content, \
            "README 应包含快速开始指南"

    def test_readme_contains_directory_structure(self):
        """README应包含目录结构说明"""
        readme_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "README.md"
        )
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "目录结构" in content or "Directory" in content, \
            "README 应包含目录结构说明"

    def test_readme_contains_core_capabilities(self):
        """README应包含核心能力说明"""
        readme_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "README.md"
        )
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "多Agent" in content or "Multi-Agent" in content
        assert "永续运行" in content or "Forever" in content


class TestO7RequirementsExists:
    """验证requirements.txt存在"""

    def test_requirements_exists(self):
        """requirements.txt应存在"""
        req_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "requirements.txt"
        )
        assert os.path.exists(req_path), "requirements.txt 应存在"

    def test_requirements_contains_core_deps(self):
        """requirements.txt应包含核心依赖"""
        req_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "requirements.txt"
        )
        with open(req_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "pyyaml" in content.lower() or "yaml" in content.lower(), \
            "requirements.txt 应包含 pyyaml"
        assert "psutil" in content.lower(), \
            "requirements.txt 应包含 psutil"
        assert "numpy" in content.lower(), \
            "requirements.txt 应包含 numpy"
        assert "pytest" in content.lower(), \
            "requirements.txt 应包含 pytest"


class TestO7DocsExist:
    """验证文档存在"""

    def test_architecture_doc_exists(self):
        """ARCHITECTURE.md应存在"""
        arch_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs", "ARCHITECTURE.md"
        )
        assert os.path.exists(arch_path), "docs/ARCHITECTURE.md 应存在"

    def test_api_reference_doc_exists(self):
        """API_REFERENCE.md应存在"""
        api_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs", "API_REFERENCE.md"
        )
        assert os.path.exists(api_path), "docs/API_REFERENCE.md 应存在"


class TestO7DirectoryStructure:
    """验证目录结构规范"""

    def test_core_directory_exists(self):
        """core/目录应存在"""
        core_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "core"
        )
        assert os.path.isdir(core_path), "core/ 目录应存在"

    def test_modules_directory_exists(self):
        """modules/目录应存在"""
        modules_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules"
        )
        assert os.path.isdir(modules_path), "modules/ 目录应存在"

    def test_tests_directory_exists(self):
        """tests/目录应存在"""
        tests_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests"
        )
        assert os.path.isdir(tests_path), "tests/ 目录应存在"

    def test_config_directory_exists(self):
        """config/目录应存在"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config"
        )
        assert os.path.isdir(config_path), "config/ 目录应存在"

    def test_main_py_exists(self):
        """main.py应存在"""
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main.py"
        )
        assert os.path.exists(main_path), "main.py 应存在"


class TestO7ConfigFiles:
    """验证配置文件存在"""

    def test_agent_yaml_exists(self):
        """agent.yaml应存在"""
        yaml_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "agent.yaml"
        )
        assert os.path.exists(yaml_path), "config/agent.yaml 应存在"

    def test_scenarios_yaml_exists(self):
        """scenarios.yaml应存在"""
        yaml_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "scenarios.yaml"
        )
        assert os.path.exists(yaml_path), "config/scenarios.yaml 应存在"

    def test_conftest_py_exists(self):
        """tests/conftest.py应存在"""
        conftest_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests", "conftest.py"
        )
        assert os.path.exists(conftest_path), "tests/conftest.py 应存在"


class TestO7NoRegression:
    """验证O7不破坏现有功能"""

    def test_agent_init_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert agent is not None
        agent.stop()

    def test_main_loop_still_runs(self):
        from main import SelfEvolvingAgent
        import time
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(0.8)
        assert agent.loop_count > 0
        agent.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
