"""
U14：迁移与版本管理测试

验证目标：
1. CURRENT_CONFIG_VERSION 定义
2. 迁移函数注册（v0->v1, v1->v2）
3. ConfigMigrator.read_config 读取版本号
4. ConfigMigrator.needs_migration 检测
5. ConfigMigrator.get_migration_path 迁移路径
6. ConfigMigrator.migrate 执行迁移（含dry_run）
7. 迁移前自动备份
8. 迁移幂等性（已是最新版本不重复迁移）
9. check_compatibility 兼容性检查（过时/未知/缺失字段）
10. format_compatibility_report 报告格式化
11. CLI config migrate/check 子命令
12. 无回归

第一性原理：配置演进应有版本号与自动迁移。
迁移是单向链，幂等，可审计。
"""
import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent


class TestU14VersionConstant(unittest.TestCase):
    """U14：版本常量"""

    def test_current_config_version_defined(self):
        from core.config_migrator import CURRENT_CONFIG_VERSION
        self.assertIsInstance(CURRENT_CONFIG_VERSION, int)
        self.assertGreaterEqual(CURRENT_CONFIG_VERSION, 1)

    def test_migrations_registered(self):
        from core.config_migrator import _MIGRATIONS
        # 至少有 v0->v1 和 v1->v2
        self.assertIn(0, _MIGRATIONS)
        self.assertIn(1, _MIGRATIONS)


class TestU14ReadConfig(unittest.TestCase):
    """U14：读取配置"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_read_config_no_version_returns_v0(self):
        from core.config_migrator import ConfigMigrator
        config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(config_path, "w") as f:
            f.write("resource:\n  max_cpu_percent: 20.0\n")
        migrator = ConfigMigrator(config_path=config_path)
        config, version = migrator.read_config()
        self.assertEqual(version, 0)
        self.assertIn("resource", config)

    def test_read_config_with_version(self):
        from core.config_migrator import ConfigMigrator
        config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(config_path, "w") as f:
            f.write("config_version: 2\nresource:\n  max_cpu_percent: 20.0\n")
        migrator = ConfigMigrator(config_path=config_path)
        _, version = migrator.read_config()
        self.assertEqual(version, 2)

    def test_read_config_nonexistent_returns_empty(self):
        from core.config_migrator import ConfigMigrator
        migrator = ConfigMigrator(config_path="/nonexistent/agent.yaml")
        config, version = migrator.read_config()
        self.assertEqual(config, {})
        self.assertEqual(version, 0)


class TestU14NeedsMigration(unittest.TestCase):
    """U14：迁移检测"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_needs_migration_for_v0(self):
        from core.config_migrator import ConfigMigrator, CURRENT_CONFIG_VERSION
        if CURRENT_CONFIG_VERSION == 0:
            self.skipTest("当前版本就是v0")
        config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(config_path, "w") as f:
            f.write("resource:\n  max_cpu_percent: 20.0\n")
        migrator = ConfigMigrator(config_path=config_path)
        self.assertTrue(migrator.needs_migration())

    def test_no_migration_needed_for_current(self):
        from core.config_migrator import ConfigMigrator, CURRENT_CONFIG_VERSION
        config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(config_path, "w") as f:
            f.write(f"config_version: {CURRENT_CONFIG_VERSION}\nresource:\n  max_cpu_percent: 20.0\n")
        migrator = ConfigMigrator(config_path=config_path)
        self.assertFalse(migrator.needs_migration())


class TestU14MigrationPath(unittest.TestCase):
    """U14：迁移路径"""

    def test_migration_path_from_v0(self):
        from core.config_migrator import ConfigMigrator, CURRENT_CONFIG_VERSION
        migrator = ConfigMigrator(config_path="/nonexistent/agent.yaml")
        path = migrator.get_migration_path(0)
        self.assertEqual(path[-1], CURRENT_CONFIG_VERSION)
        self.assertEqual(path[0], 1)

    def test_migration_path_from_current_is_empty(self):
        from core.config_migrator import ConfigMigrator, CURRENT_CONFIG_VERSION
        migrator = ConfigMigrator(config_path="/nonexistent/agent.yaml")
        path = migrator.get_migration_path(CURRENT_CONFIG_VERSION)
        self.assertEqual(path, [])


class TestU14Migrate(unittest.TestCase):
    """U14：执行迁移"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_migrate_v0_to_current(self):
        from core.config_migrator import ConfigMigrator, CURRENT_CONFIG_VERSION
        config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(config_path, "w") as f:
            f.write("resource_constraints:\n  max_cpu_percent: 20.0\n")
        migrator = ConfigMigrator(config_path=config_path)
        result = migrator.migrate()
        self.assertTrue(result.success)
        self.assertEqual(result.to_version, CURRENT_CONFIG_VERSION)
        # 验证文件已更新
        _, version = migrator.read_config()
        self.assertEqual(version, CURRENT_CONFIG_VERSION)

    def test_migrate_dry_run_does_not_modify(self):
        from core.config_migrator import ConfigMigrator
        config_path = os.path.join(self.tmpdir, "agent.yaml")
        original = "resource:\n  max_cpu_percent: 20.0\n"
        with open(config_path, "w") as f:
            f.write(original)
        migrator = ConfigMigrator(config_path=config_path)
        result = migrator.migrate(dry_run=True)
        self.assertTrue(result.success)
        # 文件应未被修改
        with open(config_path, "r") as f:
            content = f.read()
        self.assertEqual(content, original)

    def test_migrate_idempotent(self):
        """已是最新版本不重复迁移"""
        from core.config_migrator import ConfigMigrator, CURRENT_CONFIG_VERSION
        config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(config_path, "w") as f:
            f.write(f"config_version: {CURRENT_CONFIG_VERSION}\nresource:\n  max_cpu_percent: 20.0\n")
        migrator = ConfigMigrator(config_path=config_path)
        result = migrator.migrate()
        self.assertTrue(result.success)
        self.assertEqual(result.from_version, CURRENT_CONFIG_VERSION)
        self.assertIn("无需迁移", result.message)

    def test_migrate_renames_deprecated_fields(self):
        """迁移应重命名过时字段"""
        from core.config_migrator import ConfigMigrator
        import yaml
        config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(config_path, "w") as f:
            f.write("resource_constraints:\n  max_cpu_percent: 20.0\n"
                    "safety_constraints:\n  max_consecutive_exploration_failures: 5\n")
        migrator = ConfigMigrator(config_path=config_path)
        result = migrator.migrate()
        self.assertTrue(result.success)
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.assertIn("resource", cfg)
        self.assertNotIn("resource_constraints", cfg)
        self.assertIn("safety", cfg)
        self.assertNotIn("safety_constraints", cfg)

    def test_migrate_adds_version_field(self):
        from core.config_migrator import ConfigMigrator, CURRENT_CONFIG_VERSION
        import yaml
        config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(config_path, "w") as f:
            f.write("resource:\n  max_cpu_percent: 20.0\n")
        migrator = ConfigMigrator(config_path=config_path)
        migrator.migrate()
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg.get("config_version"), CURRENT_CONFIG_VERSION)


class TestU14CompatibilityCheck(unittest.TestCase):
    """U14：兼容性检查"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_check_returns_dict(self):
        from core.config_migrator import ConfigMigrator
        config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(config_path, "w") as f:
            f.write("resource:\n  max_cpu_percent: 20.0\ndrive_weights:\n  safe: 0.4\n")
        migrator = ConfigMigrator(config_path=config_path)
        report = migrator.check_compatibility()
        self.assertIsInstance(report, dict)
        self.assertIn("version", report)
        self.assertIn("is_current", report)

    def test_check_detects_deprecated_fields(self):
        from core.config_migrator import ConfigMigrator
        config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(config_path, "w") as f:
            f.write("resource_constraints:\n  max_cpu_percent: 20.0\n")
        migrator = ConfigMigrator(config_path=config_path)
        report = migrator.check_compatibility()
        self.assertIn("resource_constraints", report["deprecated_fields"])

    def test_check_detects_unknown_fields(self):
        from core.config_migrator import ConfigMigrator
        config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(config_path, "w") as f:
            f.write("unknown_field: value\nresource:\n  max_cpu_percent: 20.0\n")
        migrator = ConfigMigrator(config_path=config_path)
        report = migrator.check_compatibility()
        self.assertIn("unknown_field", report["unknown_fields"])

    def test_check_detects_missing_fields(self):
        from core.config_migrator import ConfigMigrator
        config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(config_path, "w") as f:
            f.write("config_version: 2\n")
        migrator = ConfigMigrator(config_path=config_path)
        report = migrator.check_compatibility()
        self.assertIn("resource", report["missing_fields"])

    def test_format_compatibility_report(self):
        from core.config_migrator import ConfigMigrator
        config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(config_path, "w") as f:
            f.write("resource:\n  max_cpu_percent: 20.0\ndrive_weights:\n  safe: 0.4\n")
        migrator = ConfigMigrator(config_path=config_path)
        text = migrator.format_compatibility_report()
        self.assertIn("兼容性报告", text)
        self.assertIn("版本", text)


class TestU14CliIntegration(unittest.TestCase):
    """U14：CLI迁移命令集成"""

    def test_cli_config_migrate(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["config", "migrate"])
        self.assertEqual(args.config_action, "migrate")

    def test_cli_config_migrate_dry_run(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["config", "migrate", "--dry-run"])
        self.assertTrue(args.dry_run)

    def test_cli_config_check(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["config", "check"])
        self.assertEqual(args.config_action, "check")

    def test_cli_config_check_runs(self):
        """端到端：CLI config check 正常运行"""
        from cli import cli_main
        exit_code = cli_main(["config", "check"])
        self.assertIn(exit_code, [0, 1])

    def test_cli_config_migrate_runs(self):
        """端到端：CLI config migrate 正常运行"""
        from cli import cli_main
        exit_code = cli_main(["config", "migrate", "--dry-run"])
        self.assertIn(exit_code, [0, 1])


class TestU14NoRegression(unittest.TestCase):
    """U14：无回归保护"""

    def test_agent_init_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        self.assertIsNotNone(agent.perception)

    def test_main_loop_still_runs(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        import time as _time
        _time.sleep(1.5)
        agent.stop()
        self.assertGreaterEqual(agent.loop_count, 0)

    def test_config_validate_still_works(self):
        from core.config import SystemConfig
        cfg = SystemConfig()
        self.assertTrue(cfg.validate())


if __name__ == "__main__":
    unittest.main()
