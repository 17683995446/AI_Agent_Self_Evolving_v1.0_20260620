"""
环境感知模块 —— 采集本机全维度状态数据
包含系统层、日志层、文件层、自身层四大采集维度
"""
import os
import sys
import time
import json
import math
import psutil
import asyncio
import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# 仅在直接运行时添加路径（避免副作用）
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger, log_execution
from core.config import config_manager
from core.interfaces import IPerception, StateKeys


@dataclass
class SystemMetrics:
    """系统层指标"""
    timestamp: float
    cpu_percent: float
    cpu_count: int
    cpu_freq_mhz: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_percent: float
    disk_read_mb: float
    disk_write_mb: float
    net_sent_mb: float
    net_recv_mb: float
    process_count: int
    load_avg_1m: float
    load_avg_5m: float
    load_avg_15m: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LogMetrics:
    """日志层指标"""
    timestamp: float
    error_count: int
    warning_count: int
    error_rate: float
    recent_errors: List[str]
    log_entropy: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FileMetrics:
    """文件层指标"""
    timestamp: float
    file_change_count: int
    new_files: int
    deleted_files: int
    modified_files: int
    total_files: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelfMetrics:
    """自身层指标"""
    timestamp: float
    model_accuracy: float
    prediction_error: float
    model_entropy: float
    agent_cpu_percent: float
    agent_memory_mb: float
    recent_tasks_completed: int
    recent_tasks_failed: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnvironmentPerception(IPerception):
    """环境感知器 —— 全维度状态采集"""
    
    def __init__(self) -> None:
        logger.info("PERCEPTION_INIT", "环境感知模块初始化", module="perception")
        
        self.config = config_manager.config
        self.running = False
        self._thread = None
        
        # 历史数据缓冲区
        self.system_history = deque(maxlen=1000)
        self.log_history = deque(maxlen=1000)
        self.file_history = deque(maxlen=1000)
        self.self_history = deque(maxlen=1000)
        
        # 文件监控状态
        self._file_snapshot = {}
        self._last_disk_io = psutil.disk_io_counters()
        self._last_net_io = psutil.net_io_counters()
        self._last_check_time = time.time()
        
        # 日志监控
        self._error_buffer = deque(maxlen=100)
        self._warning_buffer = deque(maxlen=100)
        
        # 状态向量缓存（减少同一周期内重复计算）
        self._state_cache: Optional[Dict[str, float]] = None
        self._state_cache_time: float = 0.0
        self._state_cache_ttl: float = 0.05  # 50ms缓存有效期
        
        logger.info("PERCEPTION_INIT", "环境感知模块初始化完成", module="perception",
                   extra={"buffer_size": 1000})
    
    @log_execution
    def collect_system_metrics(self) -> SystemMetrics:
        """采集系统层指标"""
        logger.debug("PERCEPTION_COLLECT", "开始采集系统层指标", module="perception")
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count() or 0
            cpu_freq = psutil.cpu_freq()
            cpu_freq_mhz = 0.0
            if cpu_freq is not None:
                cpu_freq_mhz = getattr(cpu_freq, 'current', 0.0) or 0.0
            
            # 内存
            mem = psutil.virtual_memory()
            memory_percent = mem.percent
            memory_used_gb = mem.used / (1024**3)
            memory_available_gb = mem.available / (1024**3)
            
            # 磁盘
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # 磁盘 IO
            current_disk_io = psutil.disk_io_counters()
            current_time = time.time()
            time_delta = current_time - self._last_check_time
            
            disk_read_mb = 0.0
            disk_write_mb = 0.0
            if self._last_disk_io and current_disk_io and time_delta > 0:
                try:
                    disk_read_mb = (current_disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024**2) / time_delta
                    disk_write_mb = (current_disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024**2) / time_delta
                except (AttributeError, TypeError):
                    logger.warning("PERCEPTION_COLLECT", "磁盘IO计数器属性异常，使用默认值", module="perception")
                    disk_read_mb = 0.0
                    disk_write_mb = 0.0
            
            self._last_disk_io = current_disk_io
            
            # 网络 IO
            current_net_io = psutil.net_io_counters()
            net_sent_mb = 0.0
            net_recv_mb = 0.0
            if self._last_net_io and current_net_io and time_delta > 0:
                try:
                    net_sent_mb = (current_net_io.bytes_sent - self._last_net_io.bytes_sent) / (1024**2) / time_delta
                    net_recv_mb = (current_net_io.bytes_recv - self._last_net_io.bytes_recv) / (1024**2) / time_delta
                except (AttributeError, TypeError):
                    logger.warning("PERCEPTION_COLLECT", "网络IO计数器属性异常，使用默认值", module="perception")
                    net_sent_mb = 0.0
                    net_recv_mb = 0.0
            
            self._last_net_io = current_net_io
            
            # 进程数
            process_count = len(psutil.pids())
            
            # 负载
            load_avg = (0.0, 0.0, 0.0)
            if hasattr(os, 'getloadavg'):
                try:
                    load_avg = os.getloadavg()
                except OSError:
                    logger.warning("PERCEPTION_COLLECT", "获取系统负载失败，使用默认值", module="perception")
            
            self._last_check_time = current_time
            
            metrics = SystemMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                cpu_freq_mhz=cpu_freq_mhz,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_available_gb=memory_available_gb,
                disk_percent=disk_percent,
                disk_read_mb=disk_read_mb,
                disk_write_mb=disk_write_mb,
                net_sent_mb=net_sent_mb,
                net_recv_mb=net_recv_mb,
                process_count=process_count,
                load_avg_1m=load_avg[0],
                load_avg_5m=load_avg[1],
                load_avg_15m=load_avg[2],
            )
            
            self.system_history.append(metrics)
            
            logger.debug("PERCEPTION_COLLECT", "系统层指标采集完成", module="perception",
                        extra={"cpu": cpu_percent, "mem": memory_percent, "disk": disk_percent})
            
            return metrics
            
        except Exception as e:
            logger.error("PERCEPTION_COLLECT", f"系统层指标采集失败: {str(e)}", module="perception",
                        extra={"error": str(e)})
            # 返回空指标
            return SystemMetrics(
                timestamp=time.time(), cpu_percent=0.0, cpu_count=0, cpu_freq_mhz=0.0,
                memory_percent=0.0, memory_used_gb=0.0, memory_available_gb=0.0,
                disk_percent=0.0, disk_read_mb=0.0, disk_write_mb=0.0,
                net_sent_mb=0.0, net_recv_mb=0.0, process_count=0,
                load_avg_1m=0.0, load_avg_5m=0.0, load_avg_15m=0.0
            )
    
    @log_execution
    def collect_log_metrics(self) -> LogMetrics:
        """采集日志层指标"""
        logger.debug("PERCEPTION_COLLECT", "开始采集日志层指标", module="perception")
        
        # 计算错误率
        total_logs = len(self._error_buffer) + len(self._warning_buffer) + 10  # +10 避免除零
        error_rate = len(self._error_buffer) / total_logs
        
        # 计算日志熵（简化版）
        if len(self._error_buffer) > 0:
            log_entropy = -sum([p * (p > 0 and p or 1e-10) for p in [
                len(self._error_buffer) / total_logs,
                len(self._warning_buffer) / total_logs,
                10 / total_logs
            ]])
        else:
            log_entropy = 0.0
        
        metrics = LogMetrics(
            timestamp=time.time(),
            error_count=len(self._error_buffer),
            warning_count=len(self._warning_buffer),
            error_rate=error_rate,
            recent_errors=list(self._error_buffer)[-10:],
            log_entropy=log_entropy,
        )
        
        self.log_history.append(metrics)
        
        logger.debug("PERCEPTION_COLLECT", "日志层指标采集完成", module="perception",
                    extra={"errors": metrics.error_count, "warnings": metrics.warning_count})
        
        return metrics
    
    @log_execution
    def collect_file_metrics(self, watch_dir: str = ".") -> FileMetrics:
        """采集文件层指标"""
        logger.debug("PERCEPTION_COLLECT", f"开始采集文件层指标: {watch_dir}", module="perception")
        
        try:
            current_snapshot = {}
            new_files = 0
            deleted_files = 0
            modified_files = 0
            total_files = 0
            
            for root, dirs, files in os.walk(watch_dir):
                # 限制扫描深度和文件数
                if root.count(os.sep) - watch_dir.count(os.sep) >= 3:
                    break
                
                for filename in files:
                    if total_files >= 1000:  # 限制扫描数量
                        break
                    
                    filepath = os.path.join(root, filename)
                    try:
                        stat = os.stat(filepath)
                        current_snapshot[filepath] = stat.st_mtime
                        total_files += 1
                        
                        if filepath not in self._file_snapshot:
                            new_files += 1
                        elif self._file_snapshot[filepath] != stat.st_mtime:
                            modified_files += 1
                    except (OSError, PermissionError):
                        continue
            
            # 检测删除的文件
            for filepath in self._file_snapshot:
                if filepath not in current_snapshot:
                    deleted_files += 1
            
            self._file_snapshot = current_snapshot
            
            metrics = FileMetrics(
                timestamp=time.time(),
                file_change_count=new_files + deleted_files + modified_files,
                new_files=new_files,
                deleted_files=deleted_files,
                modified_files=modified_files,
                total_files=total_files,
            )
            
            self.file_history.append(metrics)
            
            logger.debug("PERCEPTION_COLLECT", "文件层指标采集完成", module="perception",
                        extra={"new": new_files, "deleted": deleted_files, "modified": modified_files})
            
            return metrics
            
        except Exception as e:
            logger.error("PERCEPTION_COLLECT", f"文件层指标采集失败: {str(e)}", module="perception",
                        extra={"error": str(e)})
            return FileMetrics(
                timestamp=time.time(), file_change_count=0, new_files=0,
                deleted_files=0, modified_files=0, total_files=0
            )
    
    @log_execution
    def collect_self_metrics(self, agent_process: Optional[psutil.Process] = None) -> SelfMetrics:
        """采集自身层指标"""
        logger.debug("PERCEPTION_COLLECT", "开始采集自身层指标", module="perception")
        
        try:
            if agent_process is None:
                agent_process = psutil.Process()
            
            agent_cpu = agent_process.cpu_percent(interval=0.1)
            agent_mem = agent_process.memory_info().rss / (1024**2)  # MB
            
            metrics = SelfMetrics(
                timestamp=time.time(),
                model_accuracy=0.0,  # 由模型模块更新
                prediction_error=0.0,  # 由误差模块更新
                model_entropy=0.0,  # 由模型模块更新
                agent_cpu_percent=agent_cpu,
                agent_memory_mb=agent_mem,
                recent_tasks_completed=0,  # 由决策模块更新
                recent_tasks_failed=0,  # 由决策模块更新
            )
            
            self.self_history.append(metrics)
            
            logger.debug("PERCEPTION_COLLECT", "自身层指标采集完成", module="perception",
                        extra={"agent_cpu": agent_cpu, "agent_mem_mb": agent_mem})
            
            return metrics
            
        except Exception as e:
            logger.error("PERCEPTION_COLLECT", f"自身层指标采集失败: {str(e)}", module="perception",
                        extra={"error": str(e)})
            return SelfMetrics(
                timestamp=time.time(), model_accuracy=0.0, prediction_error=0.0,
                model_entropy=0.0, agent_cpu_percent=0.0, agent_memory_mb=0.0,
                recent_tasks_completed=0, recent_tasks_failed=0
            )
    
    def _safe_get(self, metric, attr: str, default: float = 0.0) -> float:
        """安全获取指标值，支持对象属性和字典键访问 —— 高频调用，不使用log_execution装饰器"""
        try:
            if isinstance(metric, dict):
                val = metric.get(attr, default)
            else:
                val = getattr(metric, attr, default)
            # 处理None和非法数值
            if val is None:
                return default
            val = float(val)
            if math.isnan(val) or math.isinf(val):
                return default
            return val
        except (TypeError, ValueError, AttributeError):
            return default

    def get_state_vector(self) -> Dict[str, float]:
        """获取标准化环境状态向量 —— 带短期缓存优化"""
        now = time.time()
        # 缓存命中且未过期则直接返回
        if self._state_cache is not None and (now - self._state_cache_time) < self._state_cache_ttl:
            return self._state_cache
        
        logger.debug("PERCEPTION_STATE", "开始构建环境状态向量", module="perception")
        
        # 获取最新指标
        sys_metric = self.system_history[-1] if self.system_history else self.collect_system_metrics()
        log_metric = self.log_history[-1] if self.log_history else self.collect_log_metrics()
        file_metric = self.file_history[-1] if self.file_history else self.collect_file_metrics()
        self_metric = self.self_history[-1] if self.self_history else self.collect_self_metrics()
        
        # 构建状态向量（标准化到 [0, 1]）
        state_vector = {
            # 系统层
            StateKeys.CPU_PERCENT: min(self._safe_get(sys_metric, "cpu_percent") / 100.0, 1.0),
            StateKeys.MEMORY_PERCENT: min(self._safe_get(sys_metric, "memory_percent") / 100.0, 1.0),
            StateKeys.DISK_PERCENT: min(self._safe_get(sys_metric, "disk_percent") / 100.0, 1.0),
            StateKeys.DISK_IO: min((self._safe_get(sys_metric, "disk_read_mb") + self._safe_get(sys_metric, "disk_write_mb")) / 100.0, 1.0),
            StateKeys.NET_IO: min((self._safe_get(sys_metric, "net_sent_mb") + self._safe_get(sys_metric, "net_recv_mb")) / 100.0, 1.0),
            "process_count": min(self._safe_get(sys_metric, "process_count") / 500.0, 1.0),
            StateKeys.LOAD_AVG: min(self._safe_get(sys_metric, "load_avg_1m") / 10.0, 1.0),
            
            # 日志层
            StateKeys.ERROR_RATE: min(self._safe_get(log_metric, "error_rate"), 1.0),
            "log_entropy": min(self._safe_get(log_metric, "log_entropy") / 10.0, 1.0),
            
            # 文件层
            "file_change_rate": min(self._safe_get(file_metric, "file_change_count") / 100.0, 1.0),
            
            # 自身层
            StateKeys.AGENT_CPU: min(self._safe_get(self_metric, "agent_cpu_percent") / 100.0, 1.0),
            StateKeys.AGENT_MEMORY: min(self._safe_get(self_metric, "agent_memory_mb") / 2048.0, 1.0),  # 假设 2GB 上限
            StateKeys.MODEL_ACCURACY: self._safe_get(self_metric, "model_accuracy"),
            "prediction_error": min(self._safe_get(self_metric, "prediction_error"), 1.0),
        }
        
        logger.debug("PERCEPTION_STATE", "环境状态向量构建完成", module="perception",
                    extra={"vector_dim": len(state_vector), "keys": list(state_vector.keys())})
        
        # 更新缓存
        self._state_cache = state_vector
        self._state_cache_time = now
        
        return state_vector
    
    def start(self) -> None:
        """启动感知模块"""
        logger.info("PERCEPTION_START", "启动环境感知模块", module="perception")
        self.running = True
        self._thread = threading.Thread(target=self._perception_loop, daemon=True)
        self._thread.start()
        logger.info("PERCEPTION_START", "环境感知模块已启动", module="perception")
    
    def stop(self) -> None:
        """停止感知模块"""
        logger.info("PERCEPTION_STOP", "停止环境感知模块", module="perception")
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("PERCEPTION_STOP", "环境感知模块已停止", module="perception")
    
    def _perception_loop(self) -> None:
        """感知循环线程"""
        logger.info("PERCEPTION_LOOP", "感知循环线程启动", module="perception")
        
        while self.running:
            try:
                # 采集所有维度指标
                self.collect_system_metrics()
                self.collect_log_metrics()
                self.collect_file_metrics()
                self.collect_self_metrics()
                
                logger.debug("PERCEPTION_LOOP", "感知循环完成一轮采集", module="perception")
                
                # 按配置间隔休眠
                time.sleep(self.config.perception_interval)
                
            except Exception as e:
                logger.error("PERCEPTION_LOOP", f"感知循环异常: {str(e)}", module="perception",
                            extra={"error": str(e)})
                time.sleep(1.0)  # 异常后短暂休眠
        
        logger.info("PERCEPTION_LOOP", "感知循环线程结束", module="perception")
