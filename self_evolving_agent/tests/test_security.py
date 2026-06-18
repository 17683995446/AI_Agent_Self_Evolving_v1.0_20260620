"""
安全加固测试 —— 验证权限控制、审计日志、配置加密
"""
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.security import (
    PermissionLevel, PermissionManager, AuditLogger,
    SecureConfigStore, ActionGuard, AuditRecord,
)


class TestPermissionManager:
    """测试权限管理器"""

    def test_register_and_check(self):
        pm = PermissionManager()
        pm.register_token("token_read", PermissionLevel.READ)
        pm.register_token("token_admin", PermissionLevel.ADMIN)

        assert pm.check_permission("token_read", "get_state") is True
        assert pm.check_permission("token_read", "execute_action") is False
        assert pm.check_permission("token_admin", "stop_agent") is True
        assert pm.check_permission("unknown_token", "get_state") is False

    def test_revoke_token(self):
        pm = PermissionManager()
        pm.register_token("token1", PermissionLevel.READ)
        assert pm.check_permission("token1", "get_state") is True
        pm.revoke_token("token1")
        assert pm.check_permission("token1", "get_state") is False

    def test_hierarchy(self):
        pm = PermissionManager()
        pm.register_token("system", PermissionLevel.SYSTEM)
        # SYSTEM 应能执行所有操作
        assert pm.check_permission("system", "get_state") is True
        assert pm.check_permission("system", "execute_action") is True
        assert pm.check_permission("system", "stop_agent") is True
        assert pm.check_permission("system", "execute_shell") is True

    def test_get_token_level(self):
        pm = PermissionManager()
        pm.register_token("t1", PermissionLevel.EXECUTE)
        assert pm.get_token_level("t1") == PermissionLevel.EXECUTE
        assert pm.get_token_level("unknown") is None


class TestAuditLogger:
    """测试审计日志"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.audit = AuditLogger(log_dir=self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_and_query(self):
        self.audit.log("test_action", "user1", PermissionLevel.READ, result="OK")
        records = self.audit.query()
        assert len(records) == 1
        assert records[0].action == "test_action"
        assert records[0].actor == "user1"

    def test_query_with_filters(self):
        self.audit.log("action_a", "user1", PermissionLevel.READ, result="OK")
        self.audit.log("action_b", "user2", PermissionLevel.READ, result="OK", risk_score=0.8)

        assert len(self.audit.query(action_filter="action_a")) == 1
        assert len(self.audit.query(actor_filter="user2")) == 1
        assert len(self.audit.query(min_risk=0.7)) == 1

    def test_integrity_verification(self):
        self.audit.log("action1", "user1", PermissionLevel.READ, result="OK")
        self.audit.log("action2", "user2", PermissionLevel.EXECUTE, result="OK")
        assert self.audit.verify_integrity() is True

    def test_persistence(self):
        self.audit.log("persist_test", "user1", PermissionLevel.READ, result="OK")
        # 检查文件是否写入
        import glob
        files = glob.glob(os.path.join(self.temp_dir, "audit_*.log"))
        assert len(files) >= 1
        with open(files[0]) as f:
            content = f.read()
            assert "persist_test" in content

    def test_stats(self):
        self.audit.log("low", "u1", PermissionLevel.READ, result="OK", risk_score=0.1)
        self.audit.log("high", "u2", PermissionLevel.READ, result="OK", risk_score=0.8)
        stats = self.audit.get_stats()
        assert stats["total_records"] == 2
        assert stats["high_risk_records"] == 1
        assert stats["integrity_verified"] is True


class TestSecureConfigStore:
    """测试安全配置存储"""

    def test_encrypt_decrypt(self):
        store = SecureConfigStore(key="test_key_12345")
        store.set("password", "secret123")
        retrieved = store.get("password")
        assert retrieved == "secret123"

    def test_plain_storage(self):
        store = SecureConfigStore()
        store.set("public", "visible", encrypt=False)
        assert store.get("public") == "visible"

    def test_default_value(self):
        store = SecureConfigStore()
        assert store.get("nonexistent", "default") == "default"

    def test_delete(self):
        store = SecureConfigStore(key="k")
        store.set("tmp", "val")
        assert store.get("tmp") == "val"
        store.delete("tmp")
        assert store.get("tmp") == ""

    def test_tamper_detection(self):
        store = SecureConfigStore(key="k")
        store.set("key1", "value1")
        # 篡改存储值
        store._store["key1"] = store._store["key1"].replace("a", "b")
        result = store.get("key1")
        # 篡改后应返回默认值
        assert result == ""

    def test_list_keys(self):
        store = SecureConfigStore(key="k")
        store.set("a", "1")
        store.set("b", "2")
        keys = store.list_keys()
        assert "a" in keys
        assert "b" in keys


class TestActionGuard:
    """测试动作守卫"""

    def test_low_risk_allowed(self):
        pm = PermissionManager()
        pm.register_token("token1", PermissionLevel.EXECUTE)
        guard = ActionGuard(pm)
        result = guard.check_and_log("token1", "execute_action", {"param": 1}, actor="user1")
        assert result["allowed"] is True
        assert result["risk_score"] < 0.5

    def test_high_risk_requires_confirm(self):
        pm = PermissionManager()
        pm.register_token("token1", PermissionLevel.SYSTEM)
        guard = ActionGuard(pm)
        result = guard.check_and_log("token1", "execute_shell", {"cmd": "rm -rf /"},
                                     actor="user1", require_confirm=True)
        assert result["allowed"] is False
        assert result["require_confirm"] is True
        assert result["risk_score"] >= 0.5

    def test_confirm_execution(self):
        pm = PermissionManager()
        pm.register_token("token1", PermissionLevel.SYSTEM)
        guard = ActionGuard(pm)
        result = guard.confirm_execution("token1", "execute_shell", {"cmd": "ls"}, actor="user1")
        assert result["allowed"] is True
        assert result["confirmed"] is True

    def test_permission_denied(self):
        pm = PermissionManager()
        pm.register_token("token1", PermissionLevel.READ)
        guard = ActionGuard(pm)
        result = guard.check_and_log("token1", "stop_agent", {}, actor="user1")
        assert result["allowed"] is False
        assert result["reason"] == "权限不足"

    def test_risk_evaluation(self):
        guard = ActionGuard()
        # shell命令风险高
        r1 = guard.evaluate_risk("execute_shell", {"cmd": "rm -rf /"})
        assert r1 >= 0.5
        # 普通操作风险低
        r2 = guard.evaluate_risk("get_state", {})
        assert r2 < 0.5

    def test_unauthorized_token(self):
        guard = ActionGuard()
        result = guard.check_and_log("no_such_token", "get_state", {}, actor="anon")
        assert result["allowed"] is False

    def test_audit_records_created(self):
        pm = PermissionManager()
        pm.register_token("token1", PermissionLevel.EXECUTE)
        guard = ActionGuard(pm)
        guard.check_and_log("token1", "execute_action", {}, actor="user1")
        assert len(guard.audit._records) >= 1
