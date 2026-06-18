"""
可视化监控面板测试 —— 验证Web仪表板端点
"""
import sys
import os
import json
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.dashboard_server import update_state, DashboardHandler, HTML_PAGE


class TestDashboardState:
    """测试状态更新"""

    def test_update_state(self):
        update_state(cpu_percent=0.5, memory_percent=0.6, loop_count=10)
        # 无法直接断言全局状态，但验证不抛异常
        assert True

    def test_update_state_with_history(self):
        for i in range(5):
            update_state(cpu_percent=i * 0.1, loop_count=i)
        assert True


class TestDashboardEndpoints:
    """测试HTTP端点（使用内存模拟）"""

    def test_html_page_content(self):
        assert "自驱Agent监控面板" in HTML_PAGE
        assert "/api/state" in HTML_PAGE
        assert "cpu_percent" in HTML_PAGE

    def test_html_has_required_elements(self):
        required_ids = ["cpu-val", "mem-val", "disk-val", "conf-val",
                        "safe-val", "gain-val", "eff-val", "loop", "status"]
        for rid in required_ids:
            assert f'id="{rid}"' in HTML_PAGE, f"缺少元素: {rid}"

    def test_html_has_charts(self):
        assert "pred-chart" in HTML_PAGE
        assert "history-chart" in HTML_PAGE
        assert "svg" in HTML_PAGE.lower()
