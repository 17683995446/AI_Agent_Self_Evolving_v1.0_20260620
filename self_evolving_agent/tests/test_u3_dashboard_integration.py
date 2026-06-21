"""
U3阶段验证测试 —— Dashboard集成

验证点：
1. DashboardServer类存在
2. enable_dashboard/disable_dashboard接口
3. 状态推送（update_state被调用）
4. Dashboard在stop()中被停止
5. DashboardServer非阻塞启动
6. 不破坏现有功能
"""

import os
import sys
import time
import json
import urllib.request
import socket
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_free_port():
    """获取空闲端口"""
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestU3DashboardServer:
    """验证DashboardServer类"""

    def test_dashboard_server_class_exists(self):
        """DashboardServer类应存在"""
        from dashboard.dashboard_server import DashboardServer
        assert DashboardServer is not None

    def test_dashboard_server_start_stop(self):
        """DashboardServer应能启停"""
        from dashboard.dashboard_server import DashboardServer
        port = _get_free_port()
        server = DashboardServer(port=port)
        server.start()
        assert server.running is True
        time.sleep(0.3)
        server.stop()
        assert server.running is False

    def test_dashboard_server_idempotent_start(self):
        """重复启动应幂等"""
        from dashboard.dashboard_server import DashboardServer
        port = _get_free_port()
        server = DashboardServer(port=port)
        server.start()
        server.start()  # 不应抛异常
        server.stop()

    def test_dashboard_http_accessible(self):
        """Dashboard应可通过HTTP访问"""
        from dashboard.dashboard_server import DashboardServer
        port = _get_free_port()
        server = DashboardServer(port=port)
        server.start()
        time.sleep(0.3)
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/", timeout=2)
            assert resp.status == 200
            content = resp.read().decode()
            assert "监控面板" in content
        finally:
            server.stop()


class TestU3AgentIntegration:
    """验证Agent集成Dashboard"""

    def test_agent_has_enable_dashboard(self):
        """Agent应有enable_dashboard方法"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert hasattr(agent, "enable_dashboard")
        assert hasattr(agent, "disable_dashboard")
        agent.stop()

    def test_enable_dashboard(self):
        """enable_dashboard应启动Dashboard"""
        from main import SelfEvolvingAgent
        port = _get_free_port()
        agent = SelfEvolvingAgent()
        agent.enable_dashboard(port=port)
        assert agent._dashboard_enabled is True
        assert agent.dashboard is not None
        assert agent.dashboard.running is True
        agent.disable_dashboard()
        agent.stop()

    def test_disable_dashboard(self):
        """disable_dashboard应停止Dashboard"""
        from main import SelfEvolvingAgent
        port = _get_free_port()
        agent = SelfEvolvingAgent()
        agent.enable_dashboard(port=port)
        agent.disable_dashboard()
        assert agent._dashboard_enabled is False
        agent.stop()

    def test_dashboard_stopped_on_agent_stop(self):
        """Agent.stop()应停止Dashboard"""
        from main import SelfEvolvingAgent
        port = _get_free_port()
        agent = SelfEvolvingAgent()
        agent.start()
        agent.enable_dashboard(port=port)
        time.sleep(0.3)
        agent.stop()
        assert agent._dashboard_enabled is False


class TestU3StatePush:
    """验证状态推送"""

    def test_update_state_function_exists(self):
        """update_state函数应存在"""
        from dashboard.dashboard_server import update_state
        assert callable(update_state)

    def test_update_state_updates_values(self):
        """update_state应更新状态值"""
        from dashboard.dashboard_server import update_state, _dashboard_state
        update_state(cpu_percent=0.75, loop_count=42)
        assert _dashboard_state["cpu_percent"] == 0.75
        assert _dashboard_state["loop_count"] == 42

    def test_state_pushed_during_loop(self):
        """主循环运行时状态应被推送"""
        from main import SelfEvolvingAgent
        from dashboard.dashboard_server import _dashboard_state
        port = _get_free_port()
        agent = SelfEvolvingAgent()
        agent.start()
        agent.enable_dashboard(port=port)
        time.sleep(1.5)  # 等待几轮循环
        try:
            # loop_count应被更新
            assert _dashboard_state["loop_count"] > 0
        finally:
            agent.stop()


class TestU3NoRegression:
    """验证U3不破坏现有功能"""

    def test_agent_init_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert agent is not None
        agent.stop()

    def test_main_loop_still_runs(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(0.8)
        assert agent.loop_count > 0
        agent.stop()

    def test_dashboard_with_forever_running(self):
        """Dashboard与永续运行应能共存"""
        from main import SelfEvolvingAgent
        port = _get_free_port()
        agent = SelfEvolvingAgent()
        agent.start()
        agent.enable_forever_running(health_check_interval=0.5)
        agent.enable_dashboard(port=port)
        time.sleep(1.0)
        assert agent._dashboard_enabled is True
        assert agent._forever_running_enabled is True
        agent.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
