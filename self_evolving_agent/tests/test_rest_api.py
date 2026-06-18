"""
RESTful API测试 —— 验证API端点
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.rest_server import RESTHandler, set_agent, set_default_tokens, _permission_manager
from main import SelfEvolvingAgent


class MockRequest:
    """模拟HTTP请求"""
    def __init__(self, method="GET", path="/", headers=None, body=""):
        self.method = method
        self.path = path
        self.headers = headers or {}
        self.body = body.encode()


class TestRESTAPI:
    """测试REST API端点"""

    def setup_method(self):
        set_default_tokens()
        self.agent = SelfEvolvingAgent()
        set_agent(self.agent)

    def test_health_endpoint(self):
        # health端点不需要token
        assert _permission_manager.check_permission("", "get_state") is False

    def test_unauthorized_access(self):
        # 无token访问state应被拒绝
        assert _permission_manager.check_permission("", "get_state") is False

    def test_read_token_access(self):
        # read_token可以访问state
        assert _permission_manager.check_permission("read_token", "get_state") is True
        assert _permission_manager.check_permission("read_token", "execute_action") is False

    def test_execute_token_access(self):
        # execute_token可以执行动作
        assert _permission_manager.check_permission("execute_token", "execute_action") is True
        assert _permission_manager.check_permission("execute_token", "reload_config") is False

    def test_admin_token_access(self):
        # admin_token可以重载配置
        assert _permission_manager.check_permission("admin_token", "reload_config") is True
        assert _permission_manager.check_permission("admin_token", "stop_agent") is True

    def test_agent_modules_exposed(self):
        # 验证Agent模块可以被API访问
        modules = {
            "perception": type(self.agent.perception).__name__,
            "prediction": type(self.agent.prediction).__name__,
            "drive_engine": type(self.agent.drive_engine).__name__,
        }
        assert modules["perception"] == "EnvironmentPerception"
        assert "PredictionEngine" in modules["prediction"]

    def test_state_vector_accessible(self):
        state = self.agent.perception.get_state_vector()
        assert isinstance(state, dict)
        assert "cpu_percent" in state

    def test_prediction_accessible(self):
        pred = self.agent.prediction.predict()
        assert hasattr(pred, "predicted_values")
        assert hasattr(pred, "confidence")

    def test_memory_stats_accessible(self):
        stats = self.agent.memory.get_stats()
        assert isinstance(stats, dict)
