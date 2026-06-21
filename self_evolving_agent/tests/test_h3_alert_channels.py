"""
H3阶段验证测试 —— 真实告警通道

验证点：
1. 4个新通道类存在（钉钉/飞书/企业微信/邮件）
2. 各通道payload构造正确
3. AlertManager.dispatch公开方法工作
4. create_channel_from_config工厂方法
5. setup_channels_from_config批量创建
6. Agent初始化时alert_manager就绪
7. 主循环告警检查接入
"""

import os
import sys
import time
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.alerting import (
    AlertManager, AlertEvent, AlertLevel, AlertRule,
    DingTalkChannel, FeishuChannel, WeComChannel, EmailChannel,
    HTTPChannel, LogChannel,
)


class TestH3NewChannels:
    """验证新通道类"""

    def test_dingtalk_channel_exists(self):
        assert DingTalkChannel is not None

    def test_feishu_channel_exists(self):
        assert FeishuChannel is not None

    def test_wecom_channel_exists(self):
        assert WeComChannel is not None

    def test_email_channel_exists(self):
        assert EmailChannel is not None

    def test_dingtalk_payload_format(self):
        """钉钉payload格式正确"""
        ch = DingTalkChannel(webhook_url="http://example.com/hook")
        event = AlertEvent(
            rule_name="cpu_high", level=AlertLevel.WARNING,
            message="CPU过高", timestamp=time.time(),
            metric_value=0.9, threshold=0.8,
        )
        payload = ch._build_payload(event)
        assert payload["msgtype"] == "markdown"
        assert "markdown" in payload
        assert "CPU过高" in payload["markdown"]["text"]

    def test_feishu_payload_format(self):
        """飞书payload格式正确"""
        ch = FeishuChannel(webhook_url="http://example.com/hook")
        event = AlertEvent(
            rule_name="mem_high", level=AlertLevel.ERROR,
            message="内存告警", timestamp=time.time(),
            metric_value=0.95, threshold=0.8,
        )
        payload = ch._build_payload(event)
        assert payload["msg_type"] == "interactive"
        assert "card" in payload
        assert payload["card"]["header"]["template"] == "red"

    def test_wecom_payload_format(self):
        """企业微信payload格式正确"""
        ch = WeComChannel(webhook_url="http://example.com/hook")
        event = AlertEvent(
            rule_name="disk_full", level=AlertLevel.CRITICAL,
            message="磁盘满", timestamp=time.time(),
            metric_value=0.99, threshold=0.9,
        )
        payload = ch._build_payload(event)
        assert payload["msgtype"] == "markdown"
        assert "磁盘满" in payload["markdown"]["content"]


class TestH3Dispatch:
    """验证公开dispatch方法"""

    def test_dispatch_sends_to_all_channels(self):
        """dispatch向所有通道分发"""
        am = AlertManager()
        mock_channel1 = MagicMock()
        mock_channel1.send.return_value = True
        mock_channel2 = MagicMock()
        mock_channel2.send.return_value = True
        am.add_channel("ch1", mock_channel1)
        am.add_channel("ch2", mock_channel2)

        event = AlertEvent(
            rule_name="test", level=AlertLevel.WARNING,
            message="测试", timestamp=time.time(),
            metric_value=1.0, threshold=0.5,
        )
        am.dispatch(event)

        mock_channel1.send.assert_called_once_with(event, None)
        mock_channel2.send.assert_called_once_with(event, None)

    def test_dispatch_does_not_require_rule(self):
        """dispatch不依赖rule存在"""
        am = AlertManager()
        event = AlertEvent(
            rule_name="nonexistent", level=AlertLevel.INFO,
            message="直接触发", timestamp=time.time(),
            metric_value=0.0, threshold=0.0,
        )
        # 不应抛异常
        am.dispatch(event)
        assert len(am.get_history()) == 1

    def test_dispatch_handles_channel_failure(self):
        """dispatch处理通道失败"""
        am = AlertManager()
        bad_channel = MagicMock()
        bad_channel.send.side_effect = Exception("连接失败")
        am.add_channel("bad", bad_channel)

        event = AlertEvent(
            rule_name="test", level=AlertLevel.WARNING,
            message="测试", timestamp=time.time(),
            metric_value=1.0, threshold=0.5,
        )
        # 不应抛异常
        am.dispatch(event)


class TestH3ChannelFactory:
    """验证通道工厂方法"""

    def test_create_dingtalk_from_config(self):
        ch = AlertManager.create_channel_from_config({
            "type": "dingtalk",
            "webhook_url": "http://example.com",
            "secret": "abc",
        })
        assert isinstance(ch, DingTalkChannel)
        assert ch.webhook_url == "http://example.com"

    def test_create_feishu_from_config(self):
        ch = AlertManager.create_channel_from_config({
            "type": "feishu",
            "webhook_url": "http://example.com",
        })
        assert isinstance(ch, FeishuChannel)

    def test_create_wecom_from_config(self):
        ch = AlertManager.create_channel_from_config({
            "type": "wecom",
            "webhook_url": "http://example.com",
        })
        assert isinstance(ch, WeComChannel)

    def test_create_email_from_config(self):
        ch = AlertManager.create_channel_from_config({
            "type": "email",
            "smtp_host": "smtp.example.com",
            "smtp_port": 465,
            "username": "user",
            "password": "pass",
            "from_addr": "from@example.com",
            "to_addrs": ["to@example.com"],
        })
        assert isinstance(ch, EmailChannel)
        assert ch.smtp_host == "smtp.example.com"

    def test_create_unknown_type_returns_none(self):
        ch = AlertManager.create_channel_from_config({"type": "unknown"})
        assert ch is None

    def test_setup_channels_from_config_batch(self):
        """批量创建通道"""
        am = AlertManager()
        configs = [
            {"type": "dingtalk", "name": "dt", "webhook_url": "http://dt.com"},
            {"type": "feishu", "name": "fs", "webhook_url": "http://fs.com"},
            {"type": "wecom", "name": "wc", "webhook_url": "http://wc.com"},
        ]
        count = am.setup_channels_from_config(configs)
        assert count == 3
        assert "dt" in am._channels
        assert "fs" in am._channels
        assert "wc" in am._channels


class TestH3AgentIntegration:
    """验证Agent集成告警管理器"""

    def test_agent_has_alert_manager(self):
        """Agent初始化后拥有alert_manager"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert hasattr(agent, "alert_manager")
        assert agent.alert_manager is not None

    def test_agent_has_default_alert_rules(self):
        """Agent有默认告警规则"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        rules = agent.alert_manager._rules
        assert "cpu_high" in rules
        assert "memory_high" in rules

    def test_agent_alert_callback_wired(self):
        """Agent的alert回调已注入"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert "alert" in agent.decision.executor._callbacks

    def test_alert_via_action_triggers_dispatch(self):
        """通过ALERT_NOTIFY动作触发dispatch"""
        from main import SelfEvolvingAgent
        from modules.decision import Action, ActionType
        agent = SelfEvolvingAgent()

        dispatched = {"count": 0}
        original_dispatch = agent.alert_manager.dispatch
        def mock_dispatch(event, context=None):
            dispatched["count"] += 1
        agent.alert_manager.dispatch = mock_dispatch

        action = Action(
            ActionType.ALERT_NOTIFY, "测试告警", "测试",
            params={"message": "测试消息", "level": "warning"},
        )
        agent.decision.executor.execute(action)
        assert dispatched["count"] == 1


class TestH3ChannelSendMocked:
    """验证通道发送（mock网络）"""

    def test_dingtalk_send_success(self):
        """钉钉发送成功（mock）"""
        ch = DingTalkChannel(webhook_url="http://example.com")
        event = AlertEvent(
            rule_name="test", level=AlertLevel.WARNING,
            message="测试", timestamp=time.time(),
            metric_value=1.0, threshold=0.5,
        )
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"errcode": 0}).encode()
            mock_resp.status = 200
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            result = ch.send(event)
        assert result is True

    def test_feishu_send_success(self):
        """飞书发送成功（mock）"""
        ch = FeishuChannel(webhook_url="http://example.com")
        event = AlertEvent(
            rule_name="test", level=AlertLevel.ERROR,
            message="测试", timestamp=time.time(),
            metric_value=1.0, threshold=0.5,
        )
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"code": 0}).encode()
            mock_resp.status = 200
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            result = ch.send(event)
        assert result is True

    def test_channel_send_failure_returns_false(self):
        """通道发送失败返回False"""
        ch = DingTalkChannel(webhook_url="http://invalid.url")
        event = AlertEvent(
            rule_name="test", level=AlertLevel.WARNING,
            message="测试", timestamp=time.time(),
            metric_value=1.0, threshold=0.5,
        )
        with patch("urllib.request.urlopen", side_effect=Exception("网络错误")):
            result = ch.send(event)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
