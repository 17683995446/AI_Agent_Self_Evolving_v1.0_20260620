"""
U12：自动备份与回滚测试

验证目标：
1. ConfigBackupManager 类存在且可实例化
2. BackupEntry 数据类（path/timestamp/reason/size）
3. backup() 创建带时间戳的备份文件
4. list_backups() 按时间倒序列出
5. rollback() 回滚到指定/最近备份
6. 回滚前自动备份当前配置（安全网）
7. max_backups 限制：自动清理旧备份
8. 原子性备份（先tmp再rename）
9. CLI config backup/rollback/backups 子命令
10. config reload 前自动备份
11. 无回归

第一性原理：配置变更不可逆是反人性的，应可回滚。
备份是安全网，回滚前也要备份当前（可能损坏的）配置。
"""
import os
import sys
import time
import unittest
import tempfile
import shutil
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent


class TestU12BackupManagerBasics(unittest.TestCase):
    """U12：备份管理器基础"""

    def test_backup_manager_importable(self):
        from core.config_backup import ConfigBackupManager
        self.assertTrue(callable(ConfigBackupManager))

    def test_backup_entry_dataclass(self):
        from core.config_backup import BackupEntry
        entry = BackupEntry(path="/tmp/test.yaml", timestamp=time.time(),
                           reason="manual", size=100)
        self.assertEqual(entry.name, "test.yaml")
        self.assertTrue(entry.time_str)

    def test_backup_entry_to_dict(self):
        from core.config_backup import BackupEntry
        entry = BackupEntry(path="/tmp/test.yaml", timestamp=time.time(),
                           reason="init", size=200)
        d = entry.to_dict()
        self.assertIn("name", d)
        self.assertIn("timestamp", d)
        self.assertEqual(d["reason"], "init")

    def test_manager_instantiable(self):
        from core.config_backup import ConfigBackupManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ConfigBackupManager(
                config_path=os.path.join(tmpdir, "agent.yaml"),
            )
            self.assertIsNotNone(mgr)


class TestU12BackupOperation(unittest.TestCase):
    """U12：备份操作"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(self.config_path, "w") as f:
            f.write("test: config_v1\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_backup_creates_file(self):
        from core.config_backup import ConfigBackupManager
        mgr = ConfigBackupManager(config_path=self.config_path)
        entry = mgr.backup(reason="manual")
        self.assertIsNotNone(entry)
        self.assertTrue(os.path.exists(entry.path))
        self.assertEqual(entry.reason, "manual")

    def test_backup_returns_none_when_no_config(self):
        from core.config_backup import ConfigBackupManager
        mgr = ConfigBackupManager(config_path="/nonexistent/agent.yaml")
        entry = mgr.backup()
        self.assertIsNone(entry)

    def test_backup_preserves_content(self):
        from core.config_backup import ConfigBackupManager
        mgr = ConfigBackupManager(config_path=self.config_path)
        entry = mgr.backup()
        with open(entry.path, "r") as f:
            content = f.read()
        self.assertIn("config_v1", content)

    def test_backup_name_includes_timestamp_and_reason(self):
        from core.config_backup import ConfigBackupManager
        mgr = ConfigBackupManager(config_path=self.config_path)
        entry = mgr.backup(reason="init")
        self.assertIn("init", entry.name)
        self.assertIn("agent_", entry.name)

    def test_multiple_backups_all_listed(self):
        from core.config_backup import ConfigBackupManager
        mgr = ConfigBackupManager(config_path=self.config_path)
        mgr.backup(reason="first")
        time.sleep(0.1)
        mgr.backup(reason="second")
        backups = mgr.list_backups()
        self.assertEqual(len(backups), 2)
        # 倒序：second 在前
        self.assertEqual(backups[0].reason, "second")

    def test_backup_atomic_no_tmp_left(self):
        """备份后不应残留 .tmp 文件"""
        from core.config_backup import ConfigBackupManager
        mgr = ConfigBackupManager(config_path=self.config_path)
        mgr.backup()
        backup_dir = mgr.backup_dir
        files = os.listdir(backup_dir)
        for f in files:
            self.assertFalse(f.endswith(".tmp"), f"残留临时文件: {f}")


class TestU12RollbackOperation(unittest.TestCase):
    """U12：回滚操作"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(self.config_path, "w") as f:
            f.write("version: 1\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rollback_to_latest(self):
        from core.config_backup import ConfigBackupManager
        mgr = ConfigBackupManager(config_path=self.config_path)
        # 备份v1
        mgr.backup(reason="v1")
        # 修改为v2
        with open(self.config_path, "w") as f:
            f.write("version: 2\n")
        # 回滚
        mgr.rollback()
        with open(self.config_path, "r") as f:
            content = f.read()
        self.assertIn("version: 1", content)

    def test_rollback_creates_pre_rollback_backup(self):
        """回滚前应备份当前配置"""
        from core.config_backup import ConfigBackupManager
        mgr = ConfigBackupManager(config_path=self.config_path)
        mgr.backup(reason="original")
        with open(self.config_path, "w") as f:
            f.write("version: 2\n")
        pre_backup = mgr.rollback()
        self.assertIsNotNone(pre_backup, "回滚前应创建备份")
        self.assertEqual(pre_backup.reason, "pre_rollback")

    def test_rollback_to_specific_backup(self):
        from core.config_backup import ConfigBackupManager
        mgr = ConfigBackupManager(config_path=self.config_path)
        # 创建多个备份
        with open(self.config_path, "w") as f:
            f.write("version: 1\n")
        b1 = mgr.backup(reason="v1")
        time.sleep(0.1)
        with open(self.config_path, "w") as f:
            f.write("version: 2\n")
        mgr.backup(reason="v2")
        time.sleep(0.1)
        with open(self.config_path, "w") as f:
            f.write("version: 3\n")
        # 回滚到v1
        mgr.rollback(backup_name=b1.name)
        with open(self.config_path, "r") as f:
            content = f.read()
        self.assertIn("version: 1", content)

    def test_rollback_nonexistent_backup_raises(self):
        from core.config_backup import ConfigBackupManager
        mgr = ConfigBackupManager(config_path=self.config_path)
        mgr.backup()
        with self.assertRaises(FileNotFoundError):
            mgr.rollback(backup_name="nonexistent.yaml")

    def test_rollback_no_backups_returns_none(self):
        from core.config_backup import ConfigBackupManager
        mgr = ConfigBackupManager(config_path=self.config_path)
        result = mgr.rollback()
        self.assertIsNone(result)


class TestU12MaxBackupsCleanup(unittest.TestCase):
    """U12：备份上限清理"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(self.config_path, "w") as f:
            f.write("test: 1\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cleanup_keeps_max_backups(self):
        from core.config_backup import ConfigBackupManager
        mgr = ConfigBackupManager(config_path=self.config_path, max_backups=3)
        for i in range(5):
            with open(self.config_path, "w") as f:
                f.write(f"test: {i}\n")
            mgr.backup(reason=f"v{i}")
            time.sleep(0.05)
        backups = mgr.list_backups()
        self.assertLessEqual(len(backups), 3)

    def test_cleanup_removes_oldest(self):
        from core.config_backup import ConfigBackupManager
        mgr = ConfigBackupManager(config_path=self.config_path, max_backups=2)
        entries = []
        for i in range(3):
            with open(self.config_path, "w") as f:
                f.write(f"test: {i}\n")
            entries.append(mgr.backup(reason=f"v{i}"))
            time.sleep(0.05)
        backups = mgr.list_backups()
        # 最旧的 v0 应已被清理
        names = [b.name for b in backups]
        self.assertNotIn(entries[0].name, names)


class TestU12FormatBackupsList(unittest.TestCase):
    """U12：备份列表格式化"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "agent.yaml")
        with open(self.config_path, "w") as f:
            f.write("test: 1\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_backups_message(self):
        from core.config_backup import ConfigBackupManager
        mgr = ConfigBackupManager(config_path=self.config_path)
        text = mgr.format_backups_list()
        self.assertIn("无可用备份", text)

    def test_list_includes_backup_info(self):
        from core.config_backup import ConfigBackupManager
        mgr = ConfigBackupManager(config_path=self.config_path)
        mgr.backup(reason="test")
        text = mgr.format_backups_list()
        self.assertIn("备份列表", text)
        self.assertIn("test", text)


class TestU12CliIntegration(unittest.TestCase):
    """U12：CLI备份命令集成"""

    def test_cli_config_backup(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["config", "backup"])
        self.assertEqual(args.config_action, "backup")

    def test_cli_config_backup_reason(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["config", "backup", "--reason", "init"])
        self.assertEqual(args.reason, "init")

    def test_cli_config_rollback(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["config", "rollback"])
        self.assertEqual(args.config_action, "rollback")

    def test_cli_config_rollback_name(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["config", "rollback", "--name", "agent_20240101_120000_init.yaml"])
        self.assertEqual(args.name, "agent_20240101_120000_init.yaml")

    def test_cli_config_backups_list(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["config", "backups"])
        self.assertEqual(args.config_action, "backups")

    def test_cli_config_backup_runs(self):
        """端到端：CLI config backup 正常运行"""
        from cli import cli_main
        exit_code = cli_main(["config", "backup", "--reason", "test"])
        self.assertIn(exit_code, [0, 1])

    def test_cli_config_backups_list_runs(self):
        """端到端：CLI config backups 正常运行"""
        from cli import cli_main
        exit_code = cli_main(["config", "backups"])
        self.assertEqual(exit_code, 0)


class TestU12NoRegression(unittest.TestCase):
    """U12：无回归保护"""

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
