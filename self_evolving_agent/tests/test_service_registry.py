"""
分布式Agent部署与发现测试 —— 验证服务注册、发现、心跳、健康检查

测试前置：
1. 拆解ServiceRegistry接口：register/deregister/heartbeat/discover/select_least_loaded/get_stats
2. 拆解ServiceRegistryClient接口：start/stop/discover
3. 边界条件：重复注册、未注册心跳、超时离线、空发现、负载选择
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.service_registry import (
    ServiceRegistry, ServiceInstance, ServiceStatus, ServiceRegistryClient,
)


class TestServiceRegistry:
    """测试服务注册中心"""

    def setup_method(self):
        self.registry = ServiceRegistry(heartbeat_timeout=0.5, check_interval=0.2)
        self.registry.start()

    def teardown_method(self):
        self.registry.stop()

    def test_register_and_discover(self):
        """注册并发现服务"""
        inst = ServiceInstance("i1", "agent1", "perception", "127.0.0.1", 8001)
        self.registry.register(inst)
        found = self.registry.discover(role="perception")
        assert len(found) == 1
        assert found[0].instance_id == "i1"

    def test_deregister(self):
        """注销服务"""
        inst = ServiceInstance("i2", "agent2", "prediction", "127.0.0.1", 8002)
        self.registry.register(inst)
        self.registry.deregister("i2")
        found = self.registry.discover(role="prediction")
        assert len(found) == 0

    def test_discover_by_capability(self):
        """按能力发现"""
        inst = ServiceInstance("i3", "agent3", "general", "127.0.0.1", 8003,
                               capabilities=["predict", "decide"])
        self.registry.register(inst)
        found = self.registry.discover(capability="predict")
        assert len(found) == 1

    def test_heartbeat_updates(self):
        """心跳更新负载"""
        inst = ServiceInstance("i4", "agent4", "general", "127.0.0.1", 8004)
        self.registry.register(inst)
        assert self.registry.heartbeat("i4", load_score=5.0) is True
        found = self.registry.get_instance("i4")
        assert found.load_score == 5.0

    def test_heartbeat_unregistered(self):
        """未注册实例心跳失败"""
        assert self.registry.heartbeat("unknown", 0.0) is False

    def test_health_check_offline(self):
        """健康检查标记离线"""
        inst = ServiceInstance("i5", "agent5", "general", "127.0.0.1", 8005,
                               heartbeat_interval=0.1)
        self.registry.register(inst)
        # 不发送心跳，等待超时
        time.sleep(0.7)
        found = self.registry.get_instance("i5")
        assert found.status == ServiceStatus.OFFLINE

    def test_health_check_recovery(self):
        """心跳恢复在线状态"""
        inst = ServiceInstance("i6", "agent6", "general", "127.0.0.1", 8006,
                               heartbeat_interval=0.1)
        self.registry.register(inst)
        time.sleep(0.7)
        assert self.registry.get_instance("i6").status == ServiceStatus.OFFLINE

        # 发送心跳恢复
        self.registry.heartbeat("i6")
        assert self.registry.get_instance("i6").status == ServiceStatus.HEALTHY

    def test_discover_healthy_only(self):
        """只发现健康服务"""
        inst = ServiceInstance("i7", "agent7", "general", "127.0.0.1", 8007,
                               heartbeat_interval=0.1)
        self.registry.register(inst)
        time.sleep(0.7)
        found = self.registry.discover(role="general", healthy_only=True)
        assert len(found) == 0
        found_all = self.registry.discover(role="general", healthy_only=False)
        assert len(found_all) == 1

    def test_select_least_loaded(self):
        """选择负载最低"""
        inst1 = ServiceInstance("i8", "agent8", "general", "127.0.0.1", 8008)
        inst2 = ServiceInstance("i9", "agent9", "general", "127.0.0.1", 8009)
        self.registry.register(inst1)
        self.registry.register(inst2)
        self.registry.heartbeat("i8", load_score=10.0)
        self.registry.heartbeat("i9", load_score=5.0)
        best = self.registry.select_least_loaded(role="general")
        assert best.instance_id == "i9"

    def test_select_least_loaded_empty(self):
        """无服务返回None"""
        assert self.registry.select_least_loaded(role="none") is None

    def test_watcher(self):
        """监听器触发"""
        events = []
        def watcher(event, inst):
            events.append((event, inst.instance_id))
        self.registry.add_watcher(watcher)

        inst = ServiceInstance("i10", "agent10", "general", "127.0.0.1", 8010)
        self.registry.register(inst)
        assert events[0] == ("register", "i10")

    def test_stats(self):
        """统计信息"""
        self.registry.register(ServiceInstance("i11", "a11", "general", "127.0.0.1", 8011))
        stats = self.registry.get_stats()
        assert stats["total_services"] == 1
        assert stats["healthy"] == 1


class TestServiceRegistryClient:
    """测试服务注册客户端"""

    def setup_method(self):
        self.registry = ServiceRegistry(heartbeat_timeout=1.0)
        self.registry.start()
        self.instance = ServiceInstance("client1", "agent-c1", "general", "127.0.0.1", 9001,
                                        heartbeat_interval=0.2)
        self.client = ServiceRegistryClient(self.registry, self.instance)

    def teardown_method(self):
        self.client.stop()
        self.registry.stop()

    def test_start_stop(self):
        """启动和停止"""
        self.client.start()
        assert self.registry.get_instance("client1") is not None
        self.client.stop()
        assert self.registry.get_instance("client1") is None

    def test_heartbeat_keeps_alive(self):
        """心跳保持服务在线"""
        self.client.start()
        time.sleep(0.5)
        inst = self.registry.get_instance("client1")
        assert inst.status == ServiceStatus.HEALTHY
        self.client.stop()

    def test_discover_via_client(self):
        """通过客户端发现服务"""
        self.client.start()
        other = ServiceInstance("other1", "agent-o1", "perception", "127.0.0.1", 9002)
        self.registry.register(other)
        found = self.client.discover(role="perception")
        assert len(found) == 1
        self.client.stop()
