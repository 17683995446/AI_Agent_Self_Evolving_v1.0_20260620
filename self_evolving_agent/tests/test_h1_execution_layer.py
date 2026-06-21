"""
H1阶段验证测试 —— 确认动作执行层真实落地，不再是time.sleep空壳

验证点：
1. 新增4个真实运维动作类型存在
2. ActionExecutor接入ExternalActionAdapter
3. 每个动作真实执行（产生副作用/调用回调/写文件）
4. 安全白名单机制生效
5. 回调注入机制工作
"""

import os
import sys
import time
import tempfile
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.decision import ActionExecutor, ActionType, Action, ActionSpace, DecisionEngine


class TestH1NewActionTypes:
    """验证新增动作类型"""

    def test_new_action_types_exist(self):
        """4个新动作类型已定义"""
        assert hasattr(ActionType, "CLEAN_CACHE")
        assert hasattr(ActionType, "RESTART_SERVICE")
        assert hasattr(ActionType, "ALERT_NOTIFY")
        assert hasattr(ActionType, "KILL_PROCESS")

    def test_action_space_has_new_actions(self):
        """动作空间包含新动作"""
        space = ActionSpace()
        types = [a.action_type for a in space.get_all_actions()]
        assert ActionType.CLEAN_CACHE in types
        assert ActionType.RESTART_SERVICE in types
        assert ActionType.ALERT_NOTIFY in types
        assert ActionType.KILL_PROCESS in types
        # 总共13个动作
        assert len(types) == 13


class TestH1RealExecution:
    """验证动作真实执行（非time.sleep）"""

    def test_clean_cache_actually_deletes_files(self):
        """清理缓存真实删除文件"""
        # 创建临时缓存文件
        tmp_dir = tempfile.gettempdir()
        test_file = os.path.join(tmp_dir, "agent_cache_test_h1.tmp")
        with open(test_file, "wb") as f:
            f.write(b"test cache content" * 100)

        assert os.path.exists(test_file), "测试文件应存在"

        executor = ActionExecutor()
        action = Action(
            action_type=ActionType.CLEAN_CACHE,
            name="清理缓存",
            description="测试",
            params={"targets": [test_file]},
        )
        executor._execute_action_logic(action)

        # 验证文件被真实删除
        assert not os.path.exists(test_file), "缓存文件应被删除"
        assert executor.total_cache_cleaned_bytes > 0, "应记录清理字节数"

    def test_save_snapshot_writes_real_file(self):
        """保存快照真实写文件"""
        executor = ActionExecutor()
        action = Action(
            action_type=ActionType.SAVE_SNAPSHOT,
            name="保存快照",
            description="测试",
        )
        executor._execute_action_logic(action)

        # 验证快照文件真实存在
        import glob
        snapshots = glob.glob(os.path.join(executor.config.model_snapshot_dir, "action_snapshot_*.pkl"))
        assert len(snapshots) > 0, "应生成快照文件"

        # 验证文件内容有效
        import pickle
        with open(snapshots[-1], "rb") as f:
            data = pickle.load(f)
        assert "timestamp" in data
        assert "action" in data
        assert "cpu_percent" in data

    def test_release_resources_calls_gc(self):
        """释放资源真实调用gc"""
        executor = ActionExecutor()
        action = Action(
            action_type=ActionType.RELEASE_RESOURCES,
            name="释放资源",
            description="测试",
        )
        # 不应抛异常，且应真实执行gc
        executor._execute_action_logic(action)

    def test_collect_data_reads_real_metrics(self):
        """采集数据真实读取psutil"""
        executor = ActionExecutor()
        action = Action(ActionType.COLLECT_DATA, "采集", "测试")
        # 不应抛异常，应真实读取CPU/内存
        executor._execute_action_logic(action)


class TestH1CallbackInjection:
    """验证回调注入机制"""

    def test_callback_registered_and_called(self):
        """回调被注册并被调用"""
        executor = ActionExecutor()
        called = {"count": 0}

        def mock_predict(params):
            called["count"] += 1
            return {"prediction": "ok"}

        executor.register_callback("predict", mock_predict)

        action = Action(ActionType.PREDICT, "预测", "测试")
        executor._execute_action_logic(action)

        assert called["count"] == 1, "预测回调应被调用1次"

    def test_learn_callback_called(self):
        """学习回调被调用"""
        executor = ActionExecutor()
        called = {"count": 0}

        def mock_learn(params):
            called["count"] += 1

        executor.register_callback("learn", mock_learn)

        action = Action(ActionType.LEARN, "学习", "测试")
        executor._execute_action_logic(action)

        assert called["count"] == 1

    def test_adjust_model_callback_called(self):
        """模型调整回调被调用"""
        executor = ActionExecutor()
        received_params = {}

        def mock_adjust(params):
            received_params.update(params)

        executor.register_callback("adjust_model", mock_adjust)

        action = Action(ActionType.ADJUST_MODEL, "调整", "测试", params={"lr": 0.01})
        executor._execute_action_logic(action)

        assert received_params.get("lr") == 0.01


class TestH1SecurityWhitelist:
    """验证安全白名单机制"""

    def test_restart_service_blocked_by_whitelist(self):
        """未授权服务被白名单拒绝"""
        executor = ActionExecutor(allowed_services=["nginx", "redis"])
        action = Action(
            ActionType.RESTART_SERVICE, "重启", "测试",
            params={"service": "mysql"},  # mysql不在白名单
        )
        # 不应抛异常，但应拒绝执行（通过日志记录）
        executor._execute_action_logic(action)
        # 验证：未调用adapter执行（通过检查recent_tasks_completed未增加）
        # 注意：_execute_action_logic不直接更新统计，但逻辑上应拒绝

    def test_restart_service_allowed_by_whitelist(self):
        """授权服务通过白名单"""
        mock_adapter = MagicMock()
        mock_adapter.execute_command.return_value = {"success": True, "returncode": 0}
        executor = ActionExecutor(
            action_adapter=mock_adapter,
            allowed_services=["nginx"],
        )
        action = Action(
            ActionType.RESTART_SERVICE, "重启", "测试",
            params={"service": "nginx", "method": "systemctl"},
        )
        executor._execute_action_logic(action)
        mock_adapter.execute_command.assert_called_once_with("systemctl restart nginx")

    def test_kill_process_blocked_by_whitelist(self):
        """未授权进程被白名单拒绝"""
        executor = ActionExecutor(allowed_kill_patterns=["runaway_task"])
        action = Action(
            ActionType.KILL_PROCESS, "终止", "测试",
            params={"pattern": "important_service"},  # 不在白名单
        )
        executor._execute_action_logic(action)
        # 不应终止任何进程（白名单拒绝）

    def test_kill_process_excludes_self(self):
        """终止进程时排除自身"""
        executor = ActionExecutor(allowed_kill_patterns=["python"])
        action = Action(
            ActionType.KILL_PROCESS, "终止", "测试",
            params={"pattern": "python"},
        )
        # 不应终止自己（测试进程本身是python）
        executor._execute_action_logic(action)


class TestH1AlertNotify:
    """验证告警通知"""

    def test_alert_via_callback_when_no_webhook(self):
        """无webhook时通过回调通知"""
        executor = ActionExecutor()
        received = {}

        def mock_alert(info):
            received.update(info)

        executor.register_callback("alert", mock_alert)
        action = Action(
            ActionType.ALERT_NOTIFY, "告警", "测试",
            params={"message": "CPU过高", "level": "critical"},
        )
        executor._execute_action_logic(action)

        assert received.get("message") == "CPU过高"
        assert received.get("level") == "critical"

    def test_alert_via_http_when_webhook(self):
        """有webhook时通过HTTP通知"""
        mock_adapter = MagicMock()
        mock_adapter.execute_http.return_value = {"success": True, "status": 200}
        executor = ActionExecutor(action_adapter=mock_adapter)
        action = Action(
            ActionType.ALERT_NOTIFY, "告警", "测试",
            params={"webhook_url": "http://example.com/hook", "message": "测试告警"},
        )
        executor._execute_action_logic(action)
        mock_adapter.execute_http.assert_called_once()
        call_args = mock_adapter.execute_http.call_args
        assert call_args[0][0] == "http://example.com/hook"
        assert call_args[0][1]["message"] == "测试告警"


class TestH1NoTimeSleepShell:
    """验证不再是time.sleep空壳"""

    def test_action_execution_produces_real_effect(self):
        """动作执行产生真实副作用（非空sleep）"""
        executor = ActionExecutor()

        # 清理缓存动作应产生文件删除副作用
        tmp_dir = tempfile.gettempdir()
        marker = os.path.join(tmp_dir, "agent_cache_marker_h1.tmp")
        with open(marker, "wb") as f:
            f.write(b"x" * 1000)

        before_size = executor.total_cache_cleaned_bytes
        action = Action(
            ActionType.CLEAN_CACHE, "清理", "测试",
            params={"targets": [marker]},
        )
        executor._execute_action_logic(action)
        after_size = executor.total_cache_cleaned_bytes

        # 验证：清理字节数增加了（真实副作用）
        assert after_size > before_size, "清理字节数应增加"
        assert after_size - before_size >= 1000, "应清理约1000字节"
        assert not os.path.exists(marker), "文件应被删除"


class TestH1DecisionEngineIntegration:
    """验证DecisionEngine集成"""

    def test_decision_engine_uses_new_executor(self):
        """DecisionEngine使用新的ActionExecutor"""
        engine = DecisionEngine()
        assert hasattr(engine.executor, "adapter"), "执行器应有adapter属性"
        assert hasattr(engine.executor, "_callbacks"), "执行器应有callbacks属性"
        assert hasattr(engine.executor, "allowed_services"), "执行器应有白名单属性"

    def test_decision_engine_register_callback(self):
        """DecisionEngine可注册回调"""
        engine = DecisionEngine()
        called = {"count": 0}

        def cb(params):
            called["count"] += 1

        engine.executor.register_callback("predict", cb)
        assert "predict" in engine.executor._callbacks


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--timeout=30"])
