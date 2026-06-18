"""
分布式Agent部署与发现 —— 服务注册中心，支持跨机器部署

设计原则（第一性原理）：
1. 位置透明：Agent在哪里运行不重要，重要的是它能做什么
2. 服务注册：Agent启动时向注册中心注册自己的地址和能力
3. 心跳保活：定期心跳，超时自动剔除，防止调用已死亡节点
4. 服务发现：按角色/能力/负载查找可用Agent
"""

import os
import sys
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


class ServiceStatus(Enum):
    """服务状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class ServiceInstance:
    """服务实例信息"""
    instance_id: str
    agent_id: str
    role: str
    host: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    heartbeat_interval: float = 10.0
    load_score: float = 0.0
    status: ServiceStatus = ServiceStatus.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "agent_id": self.agent_id,
            "role": self.role,
            "host": self.host,
            "port": self.port,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "load_score": self.load_score,
            "status": self.status.value,
        }


class ServiceRegistry:
    """服务注册中心 —— 管理所有Agent实例的注册、发现、健康检查"""

    def __init__(self, heartbeat_timeout: float = 30.0, check_interval: float = 5.0):
        self._services: Dict[str, ServiceInstance] = {}
        self._by_role: Dict[str, List[str]] = {}      # role -> [instance_id]
        self._by_capability: Dict[str, List[str]] = {} # capability -> [instance_id]
        self._heartbeat_timeout = heartbeat_timeout
        self._check_interval = check_interval
        self._lock = threading.Lock()
        self._running = False
        self._health_check_thread: Optional[threading.Thread] = None
        self._watchers: List[Callable] = []

    def start(self) -> None:
        """启动健康检查线程"""
        self._running = True
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()
        logger.info("SVC_REG", "服务注册中心启动", module="service_registry")

    def stop(self) -> None:
        """停止健康检查"""
        self._running = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=2.0)
        logger.info("SVC_REG", "服务注册中心停止", module="service_registry")

    def register(self, instance: ServiceInstance) -> bool:
        """注册服务实例"""
        with self._lock:
            self._services[instance.instance_id] = instance

            # 按角色索引
            if instance.role not in self._by_role:
                self._by_role[instance.role] = []
            if instance.instance_id not in self._by_role[instance.role]:
                self._by_role[instance.role].append(instance.instance_id)

            # 按能力索引
            for cap in instance.capabilities:
                if cap not in self._by_capability:
                    self._by_capability[cap] = []
                if instance.instance_id not in self._by_capability[cap]:
                    self._by_capability[cap].append(instance.instance_id)

        logger.info("SVC_REG", f"服务注册: {instance.instance_id} @ {instance.host}:{instance.port}",
                    module="service_registry", extra={"role": instance.role})
        self._notify_watchers("register", instance)
        return True

    def deregister(self, instance_id: str) -> bool:
        """注销服务实例"""
        with self._lock:
            instance = self._services.pop(instance_id, None)
            if not instance:
                return False

            # 清理索引
            if instance.role in self._by_role:
                self._by_role[instance.role] = [i for i in self._by_role[instance.role] if i != instance_id]
            for cap in instance.capabilities:
                if cap in self._by_capability:
                    self._by_capability[cap] = [i for i in self._by_capability[cap] if i != instance_id]

        logger.info("SVC_REG", f"服务注销: {instance_id}", module="service_registry")
        self._notify_watchers("deregister", instance)
        return True

    def heartbeat(self, instance_id: str, load_score: float = 0.0) -> bool:
        """接收心跳"""
        with self._lock:
            instance = self._services.get(instance_id)
            if not instance:
                return False
            instance.last_heartbeat = time.time()
            instance.load_score = load_score
            if instance.status == ServiceStatus.OFFLINE:
                instance.status = ServiceStatus.HEALTHY
                logger.info("SVC_REG", f"服务恢复: {instance_id}", module="service_registry")
        return True

    def discover(self, role: Optional[str] = None,
                 capability: Optional[str] = None,
                 healthy_only: bool = True) -> List[ServiceInstance]:
        """发现服务 —— 按角色或能力过滤"""
        with self._lock:
            if role:
                ids = self._by_role.get(role, [])
            elif capability:
                ids = self._by_capability.get(capability, [])
            else:
                ids = list(self._services.keys())

            result = []
            for iid in ids:
                inst = self._services.get(iid)
                if not inst:
                    continue
                if healthy_only and inst.status in (ServiceStatus.OFFLINE, ServiceStatus.CRITICAL):
                    continue
                result.append(inst)
            return result

    def get_instance(self, instance_id: str) -> Optional[ServiceInstance]:
        """获取单个实例"""
        with self._lock:
            return self._services.get(instance_id)

    def select_least_loaded(self, role: Optional[str] = None,
                            capability: Optional[str] = None) -> Optional[ServiceInstance]:
        """选择负载最低的实例"""
        instances = self.discover(role=role, capability=capability, healthy_only=True)
        if not instances:
            return None
        return min(instances, key=lambda i: i.load_score)

    def _health_check_loop(self) -> None:
        """健康检查循环"""
        while self._running:
            now = time.time()
            with self._lock:
                for instance in list(self._services.values()):
                    elapsed = now - instance.last_heartbeat
                    if elapsed > self._heartbeat_timeout:
                        if instance.status != ServiceStatus.OFFLINE:
                            instance.status = ServiceStatus.OFFLINE
                            logger.warning("SVC_REG", f"服务离线: {instance.instance_id} (未心跳{elapsed:.0f}秒)",
                                           module="service_registry")
                            self._notify_watchers("offline", instance)
                    elif elapsed > self._heartbeat_timeout * 0.6:
                        instance.status = ServiceStatus.WARNING
                    else:
                        instance.status = ServiceStatus.HEALTHY
            time.sleep(self._check_interval)

    def add_watcher(self, callback: Callable) -> None:
        """添加状态变更监听器"""
        self._watchers.append(callback)

    def _notify_watchers(self, event: str, instance: ServiceInstance) -> None:
        """通知监听器"""
        for watcher in self._watchers:
            try:
                watcher(event, instance)
            except Exception as e:
                logger.warning("SVC_REG", f"监听器异常: {e}", module="service_registry")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            total = len(self._services)
            healthy = sum(1 for s in self._services.values() if s.status == ServiceStatus.HEALTHY)
            offline = sum(1 for s in self._services.values() if s.status == ServiceStatus.OFFLINE)
            return {
                "total_services": total,
                "healthy": healthy,
                "offline": offline,
                "roles": len(self._by_role),
                "capabilities": len(self._by_capability),
            }


class ServiceRegistryClient:
    """服务注册客户端 —— Agent使用此客户端与注册中心交互"""

    def __init__(self, registry: ServiceRegistry, instance: ServiceInstance):
        self.registry = registry
        self.instance = instance
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """启动注册和心跳"""
        self.registry.register(self.instance)
        self._running = True
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        logger.info("SVC_CLIENT", f"客户端启动: {self.instance.instance_id}", module="service_registry")

    def stop(self) -> None:
        """停止心跳并注销"""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
        self.registry.deregister(self.instance.instance_id)
        logger.info("SVC_CLIENT", f"客户端停止: {self.instance.instance_id}", module="service_registry")

    def _heartbeat_loop(self) -> None:
        """心跳循环"""
        while self._running:
            self.registry.heartbeat(self.instance.instance_id, load_score=self._get_load())
            time.sleep(self.instance.heartbeat_interval)

    def _get_load(self) -> float:
        """获取当前负载 —— 子类可覆盖"""
        return 0.0

    def discover(self, role: Optional[str] = None,
                 capability: Optional[str] = None) -> List[ServiceInstance]:
        """发现其他服务"""
        return self.registry.discover(role=role, capability=capability, healthy_only=True)
