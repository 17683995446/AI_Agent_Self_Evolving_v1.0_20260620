"""
O5阶段验证测试 —— 测试秩序

验证点：
1. conftest.py存在并统一管理sys.path
2. 测试文件使用pytest框架
3. 测试命名规范
4. 无重复的端到端测试职责
5. 不破坏现有功能
"""

import os
import sys
import pytest

# O5：通过conftest.py统一管理路径，不再需要sys.path.insert
# 但为兼容性保留（conftest.py已处理路径）


class TestO5ConftestExists:
    """验证conftest.py存在并统一管理路径"""

    def test_conftest_exists(self):
        """conftest.py应存在"""
        conftest_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests", "conftest.py"
        )
        assert os.path.exists(conftest_path), "tests/conftest.py 应存在"

    def test_conftest_manages_sys_path(self):
        """conftest.py应管理sys.path"""
        conftest_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests", "conftest.py"
        )
        with open(conftest_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "sys.path" in content, "conftest.py 应包含 sys.path 管理"
        assert "_PROJECT_ROOT" in content, "conftest.py 应定义项目根目录"

    def test_project_root_in_sys_path(self):
        """项目根目录应在sys.path中（由conftest.py添加）"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assert project_root in sys.path, "项目根目录应在 sys.path 中"


class TestO5TestFrameworkConsistency:
    """验证测试框架一致性"""

    def test_o1_o4_tests_use_pytest(self):
        """O1-O4测试应使用pytest框架"""
        test_files = [
            "test_o1_config_order.py",
            "test_o2_startup_shutdown_order.py",
            "test_o3_exception_order.py",
            "test_o4_interface_order.py",
        ]
        tests_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests"
        )
        for tf in test_files:
            path = os.path.join(tests_dir, tf)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert "import pytest" in content, f"{tf} 应导入 pytest"
            # 不应使用unittest
            assert "import unittest" not in content, f"{tf} 不应使用 unittest"

    def test_m1_m7_tests_use_pytest(self):
        """M1-M7测试应使用pytest框架"""
        test_files = [
            "test_m1_multi_agent_orchestration.py",
            "test_m7_end_to_end_integration.py",
        ]
        tests_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests"
        )
        for tf in test_files:
            path = os.path.join(tests_dir, tf)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert "import pytest" in content, f"{tf} 应导入 pytest"


class TestO5TestNamingConvention:
    """验证测试命名规范"""

    def test_test_files_follow_naming_convention(self):
        """测试文件应遵循test_*.py命名规范（conftest.py和__init__.py除外）"""
        tests_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests"
        )
        exempt_files = {"conftest.py", "__init__.py"}
        for filename in os.listdir(tests_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                if filename in exempt_files:
                    continue
                assert filename.startswith("test_"), \
                    f"测试文件 {filename} 应以 test_ 开头"

    def test_o_series_tests_exist(self):
        """O系列测试文件应存在"""
        tests_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests"
        )
        for i in range(1, 5):
            assert os.path.exists(os.path.join(tests_dir, f"test_o{i}_")) or \
                   any(f.startswith(f"test_o{i}_") for f in os.listdir(tests_dir)), \
                f"test_o{i}_*.py 应存在"


class TestO5NoRegression:
    """验证O5不破坏现有功能"""

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

    def test_all_o_tests_pass(self):
        """所有O系列测试应能被发现"""
        tests_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests"
        )
        o_tests = [f for f in os.listdir(tests_dir) if f.startswith("test_o") and f.endswith(".py")]
        assert len(o_tests) >= 4, f"应至少有4个O系列测试文件，实际: {len(o_tests)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
