"""
O6阶段验证测试 —— 代码风格秩序

验证点：
1. core/config.py使用if __name__保护sys.path
2. core/config_center.py使用if __name__保护sys.path
3. modules/文件统一使用if __name__保护sys.path
4. 类型注解一致性（Tuple而非tuple）
5. 导入顺序规范（标准库→第三方→项目内）
6. 不破坏现有功能
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestO6SysPathProtection:
    """验证sys.path保护策略"""

    def test_config_py_uses_path_protection(self):
        """config.py应使用if __name__保护sys.path"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "core", "config.py"
        )
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 应包含if __name__保护
        assert 'if __name__ == "__main__":' in content, \
            "config.py 应使用 if __name__ == '__main__' 保护 sys.path"
        # 不应有未保护的sys.path.insert（在模块级别）
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'sys.path.insert' in line and 'if __name__' not in lines[max(0, i-1)]:
                # 允许在if __name__块内
                if i > 0 and 'if __name__' in lines[i-1]:
                    continue
                pytest.fail(f"config.py 第{i+1}行有未保护的 sys.path.insert")

    def test_config_center_py_uses_path_protection(self):
        """config_center.py应使用if __name__保护sys.path"""
        config_center_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "core", "config_center.py"
        )
        with open(config_center_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'if __name__ == "__main__":' in content, \
            "config_center.py 应使用 if __name__ == '__main__' 保护 sys.path"

    def test_modules_use_path_protection(self):
        """modules/目录文件应使用if __name__保护sys.path"""
        modules_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules"
        )
        for filename in os.listdir(modules_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                filepath = os.path.join(modules_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'sys.path.insert' in content:
                    assert 'if __name__ == "__main__":' in content, \
                        f"modules/{filename} 应使用 if __name__ 保护 sys.path"


class TestO6TypeAnnotationConsistency:
    """验证类型注解一致性"""

    def test_no_lowercase_tuple_annotation_in_modules(self):
        """modules/不应使用小写tuple[类型注解"""
        modules_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules"
        )
        for filename in os.listdir(modules_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                filepath = os.path.join(modules_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 不应使用 -> tuple[（小写tuple类型注解）
                assert "-> tuple[" not in content, \
                    f"modules/{filename} 应使用 Tuple[...] 而非 tuple[...]"

    def test_no_lowercase_tuple_annotation_in_core(self):
        """core/不应使用小写tuple[类型注解"""
        core_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "core"
        )
        for filename in os.listdir(core_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                filepath = os.path.join(core_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 不应使用 -> tuple[（小写tuple类型注解）
                assert "-> tuple[" not in content, \
                    f"core/{filename} 应使用 Tuple[...] 而非 tuple[...]"


class TestO6ImportOrder:
    """验证导入顺序规范"""

    def test_main_imports_standard_first(self):
        """main.py应先导入标准库再导入项目内模块"""
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main.py"
        )
        with open(main_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 找到第一个from core导入的位置
        first_core_import = None
        first_std_import = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("from core.") or stripped.startswith("import core"):
                if first_core_import is None:
                    first_core_import = i
            if stripped.startswith("import os") or stripped.startswith("import sys") or \
               stripped.startswith("import time") or stripped.startswith("import threading"):
                if first_std_import is None:
                    first_std_import = i

        # 标准库导入应在core导入之前
        if first_core_import and first_std_import:
            assert first_std_import < first_core_import, \
                "标准库导入应在项目内导入之前"


class TestO6NoMagicNumbers:
    """验证魔法数字清理"""

    def test_main_no_hardcoded_test_intervals(self):
        """main.py不应包含硬编码的测试间隔"""
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main.py"
        )
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 不应包含硬编码的测试模式缩短
        assert "min(interval_seconds, 60.0)" not in content
        assert "为了测试" not in content


class TestO6NoRegression:
    """验证O6不破坏现有功能"""

    def test_agent_init_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert agent is not None
        agent.stop()

    def test_config_loads_correctly(self):
        """配置应能正常加载"""
        from core.config import config_manager
        assert config_manager.config is not None
        assert config_manager.config.drive_weights.safe > 0

    def test_config_center_loads_correctly(self):
        """配置中心应能正常加载"""
        from core.config_center import ConfigCenter
        cc = ConfigCenter()
        assert cc.config is not None

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
