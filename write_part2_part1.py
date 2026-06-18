#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write Phases 21-25 of Part 2."""

content = r'''

## 阶段 21：环境感知模块总体架构

### 21.1 引言

在完成了第一部分（阶段 1-20）的理论基础构建之后，我们正式进入第二部分——工程实现篇。环境感知模块（Environment Perception Module）是本系统七大模块中的第一个模块，也是整个系统的"感官层"。它承担着将本机多维度的原始物理状态转化为标准化数字向量的核心职责。没有准确、全面、及时的环境感知，后续的驱力引擎、预测模型、决策模块都将成为无源之水、无本之木。

环境感知模块的设计必须回答以下几个核心问题：第一，我们需要感知什么？本机是一个复杂的数字生态系统，CPU的指令流水线、内存的页表映射、磁盘的I/O队列、网络的TCP连接状态、日志文件的追加写入、进程的fork和exit——这些信息维度极其丰富，但并非所有信息都同等重要。第二，感知的频率应该是多少？不同的信息维度有不同的变化速率——CPU使用率可能在毫秒级波动，而磁盘使用率可能以小时为单位缓慢变化。第三，如何将异构的多模态数据统一为下游模块可以直接消费的标准化格式？第四，如何在资源约束下（CPU≤20%，内存≤20%）高效地完成感知任务？

本章将建立环境感知模块的总体架构，回答上述全部问题，并为后续13个阶段的详细设计奠定框架基础。

### 21.2 模块总体设计

#### 21.2.1 四层感知体系

环境感知模块采用四层感知体系，每一层负责一个维度的数据采集：

第一层：系统层（System Layer）——负责采集操作系统级别的核心指标，包括CPU使用率、CPU频率、CPU温度、内存使用量、内存使用率、Swap使用量、磁盘I/O读写速率、磁盘使用率、网络收发包速率、网络连接数、进程列表、进程资源占用等。系统层是感知体系中最基础、最核心的一层，因为所有其他层的数据都依赖于系统资源的存在。

第二层：日志层（Log Layer）——负责采集和解析系统和应用的日志数据。包括系统日志（/var/log/syslog、/var/log/messages）、应用日志（用户自定义路径）、内核日志（dmesg）、安全日志（auth.log）等。日志层的数据是半结构化的文本，需要经过解析、模式提取、异常检测等处理才能转化为结构化数据。

第三层：文件层（File Layer）——负责监控文件系统的变化。包括文件的创建、修改、删除、移动、权限变更等事件。文件层使用watchdog库实现事件驱动的文件系统监控，可以感知到哪些文件正在被修改、哪些新文件被创建、哪些目录结构发生了变化。

第四层：自身层（Self Layer）——负责对系统自身进行内省。包括模型预测精度的变化趋势、模型推理的延迟、模型参数的规模、内存/CPU使用量、错误率、学习率变化等。自身层实现了系统的"自我意识"——系统不仅知道环境在发生什么，还知道自己在环境中的表现如何。

#### 21.2.2 数据流架构

环境感知模块的数据流遵循采集→缓冲→处理→融合→输出的五阶段流水线。四个采集器并行运行，各自以适当的频率采集原始数据。系统层使用psutil进行高频轮询（每秒1次），日志层使用文件tail和inotify进行增量读取（每5秒1次），文件层使用watchdog事件驱动（实时），自身层使用内省API（每秒1次）。

采集到的原始数据首先进入时序缓冲层。每个采集器有自己的独立缓冲区（使用collections.deque实现），缓冲区长度根据维度的时间特性设定：系统层60秒，日志层300秒，文件层60秒，自身层60秒。从缓冲区中取出原始数据后，经过降噪处理（中值滤波、卡尔曼滤波）、异常值过滤（IQR方法）、缺失值填充（前向填充或均值填充）。然后将四个维度的处理后的数据拼接为一个统一的特征向量，并进行标准化（Z-score归一化）。最终输出标准化的环境状态向量x_t∈ℝ^d（d≈128-256），供驱力引擎模块和预测模型使用。

#### 21.2.3 线程模型

环境感知模块使用异步I/O+线程池的混合并发模型。主线程是一个asyncio Event Loop，上面运行六个异步任务：系统层采集协程（1s间隔）、日志层采集协程（5s间隔）、文件层事件监听协程（watchdog回调）、自身层内省协程（1s间隔）、数据处理协程（1s间隔）、健康检查与监控协程（10s间隔）。CPU密集型操作（如异常检测、数据融合）提交到线程池执行，避免阻塞事件循环。所有采集器之间完全独立，一个采集器故障不影响其他采集器。使用asyncio.Queue进行采集器到处理器的数据传递。

### 21.3 状态向量的维度设计

环境状态向量x_t是环境感知模块的最终输出，也是下游所有模块的输入。其维度设计必须满足以下约束：维度不能太高（否则预测模型的计算量过大）、维度不能太低（否则丢失关键信息）、每个维度必须有明确的物理含义（便于调试和理解）。

状态向量x_t按来源分为四个子向量。系统层子向量x_t^sys∈ℝ^48：包括CPU维度（8维：cpu_percent, per_core[4], freq_current, freq_max, temp）、内存维度（8维：total, available, used, percent, swap_total, swap_used, swap_percent, active）、磁盘维度（8维：read_bytes, write_bytes, read_count, write_count, usage_percent, free, total, io_time）、网络维度（8维：bytes_sent, bytes_recv, packets_sent, packets_recv, errin, errout, connections, conn_rate）、进程维度（16维：total, running, sleeping, zombie, top5_cpu_pids/percent, top5_mem_pids/percent）。

日志层子向量x_t^log∈ℝ^32：日志速率（8维：total_lines_per_min, error/warning/info/debug/critical_lines_per_min, avg_line_length, unique_patterns）、错误分布（8维：error_type_distribution）、异常检测（8维：anomaly_score, anomaly_confidence, recent_anomaly_count, anomaly_trend, isolation_forest_score, cluster_distance, novelty_score, pattern_shift）、日志源（8维：source_activity）。

文件层子向量x_t^file∈ℝ^16：文件事件（8维：created/modified/deleted/moved_count, total_events, event_rate, unique_paths, avg_event_interval）、文件类型（4维：config/code/data/log_changes）、目录统计（4维：monitored/active_dirs, dir_depth_max, avg_path_length）。

自身层子向量x_t^self∈ℝ^32：预测精度（8维：mse/mae_recent, mse/mae_trend, accuracy_ema, error_std, prediction_confidence, calibration_error）、模型资源（8维：model_ram_mb, model_cpu_percent, inference_time_ms, param_count, gradient_norm, last_update_time, update_frequency）、模型状态（8维：model_entropy, knowledge_coverage, feature_importance_std, overfitting_signal, learning_rate, exploration_rate, drive_weights[3]）、错误统计（8维：error_autocorr, error_skewness, error_kurtosis, top3_error_features/magnitudes, error_burst_count/duration）。总计128维。

### 21.4 模块间的接口定义

环境感知模块对外暴露以下核心接口：async start()启动所有采集器和处理器；async stop()优雅停止所有采集器；async get_state_vector()获取当前环境状态向量x_t∈ℝ^128；async get_raw_data(layer)获取指定层的原始数据（用于调试）；get_health_status()获取模块健康状态；subscribe(callback)订阅状态更新事件。

```python
import asyncio
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor

@dataclass
class PerceptionConfig:
    system_poll_interval: float = 1.0
    log_poll_interval: float = 5.0
    self_poll_interval: float = 1.0
    system_buffer_size: int = 60
    log_buffer_size: int = 60
    file_buffer_size: int = 60
    self_buffer_size: int = 60
    state_vector_dim: int = 128
    max_threadpool_workers: int = 4
    health_check_interval: float = 10.0

class EnvironmentPerceptionModule:
    def __init__(self, config=None):
        self.config = config or PerceptionConfig()
        self.logger = logging.getLogger(__name__)
        self.system_buffer = deque(maxlen=self.config.system_buffer_size)
        self.log_buffer = deque(maxlen=self.config.log_buffer_size)
        self.file_buffer = deque(maxlen=self.config.file_buffer_size)
        self.self_buffer = deque(maxlen=self.config.self_buffer_size)
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_threadpool_workers)
        self._current_state_vector = None
        self._subscribers = []
        self._running = False
        self._tasks = []

    async def start(self):
        self._running = True
        self._tasks = [
            asyncio.create_task(self._system_loop(), name="system_loop"),
            asyncio.create_task(self._log_loop(), name="log_loop"),
            asyncio.create_task(self._file_loop(), name="file_loop"),
            asyncio.create_task(self._self_loop(), name="self_loop"),
            asyncio.create_task(self._process_loop(), name="process_loop"),
            asyncio.create_task(self._health_check_loop(), name="health_check"),
        ]
        self.logger.info("环境感知模块已启动")

    async def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.executor.shutdown(wait=True)

    async def get_state_vector(self) -> np.ndarray:
        if self._current_state_vector is None:
            return np.zeros(self.config.state_vector_dim, dtype=np.float32)
        return self._current_state_vector.copy()
```

### 21.5 工程考虑

资源约束下的运行：环境感知模块必须在CPU≤20%和内存≤20%的约束下运行。为此，所有采集器使用异步I/O而非阻塞I/O，CPU密集型操作提交到线程池而非主线程，缓冲区大小严格控制（系统层60秒对应约60条数据，每条约512字节，总计约30KB）。内存占用主要来自采集器对象和缓冲区，总计不超过50MB。

故障隔离：每个采集器运行在独立的asyncio Task中，一个采集器故障不会影响其他采集器。数据处理循环持续运行，即使某个采集器暂时没有数据，也会使用默认值填充。

可观测性：通过健康检查协程，系统持续监控各采集器的运行状态、缓冲区大小、处理延迟等指标，确保模块运行在健康状态。

### 21.6 阶段小结

本阶段建立了环境感知模块的总体架构：四层感知体系（系统层、日志层、文件层、自身层）覆盖本机全维度状态；五阶段数据流（采集→缓冲→处理→融合→输出）将原始数据转化为标准化状态向量；异步I/O+线程池的混合并发模型确保在资源约束下高效运行；128维状态向量按来源分为四个子向量，便于下游模块按需使用；完善的健康检查和故障隔离机制确保模块的鲁棒性。

---

## 阶段 22：系统层数据采集——CPU监控

### 22.1 引言

CPU（中央处理器）是计算机系统的心脏。对本系统而言，CPU监控是第一优先级——不仅因为CPU是系统运行的核心资源，更因为CPU使用率直接关系到本系统的刚性边界约束（CPU≤20%）。如果我们不能准确感知CPU的状态，就无法保证系统不在资源约束上越界。本阶段将深入设计CPU监控子模块，从数据采集、指标计算、异常检测到工程实现，全面覆盖CPU监控的每一个细节。

### 22.2 CPU监控的核心指标体系

总体CPU使用率是最基本的CPU指标。psutil提供了psutil.cpu_percent(interval)方法，可以在指定时间间隔内测量CPU使用率。重要工程细节：psutil.cpu_percent()的第一次调用总是返回0.0，因为它需要至少两个时间点来计算差值。因此在采集循环中，第一次调用用于初始化，后续调用才返回有效值。

每核心CPU使用率也是关键指标。现代CPU通常是多核的，仅监控总体使用率是不够的——如果是一个单线程程序占满了某个核心，总体使用率可能只有25%（4核CPU），但那个核心已经100%满载。因此必须监控每个核心的使用率（psutil.cpu_percent(percpu=True)）。

CPU频率决定了CPU当前的运行速度。现代CPU支持动态频率调整（Intel SpeedStep、AMD Cool'n'Quiet），频率会根据负载动态变化。通过psutil.cpu_freq()获取当前频率、最低频率和最高频率，并计算频率利用率（current/max）。

CPU温度是判断系统是否存在散热问题的关键指标。psutil提供了sensors_temperatures()方法获取温度信息。注意CPU温度传感器在不同操作系统上的可用性不同，需要做跨平台兼容处理。

CPU时间统计（psutil.cpu_times()）提供了CPU在不同状态下的累计时间：user（用户态）、system（内核态）、idle（空闲）、iowait（等待I/O）。此外，CPU负载（Load Average）是运行队列中等待CPU的进程数的指数移动平均，可以通过os.getloadavg()获取1分钟、5分钟、15分钟平均负载。负载<CPU核心数意味着CPU未饱和，负载>CPU核心数意味着CPU过载。

### 22.3 CPU监控的数学指标

CPU使用率的变化率（一阶导数）反映负载变化的剧烈程度：ΔCPU(t)=CPU(t)-CPU(t-1)。变化率为正且很大意味着CPU负载急剧增加，变化率为负且很大意味着CPU负载急剧下降。

二阶导数（加速度）反映负载变化的加速趋势：Δ²CPU(t)=CPU(t)-2·CPU(t-1)+CPU(t-2)。Δ²CPU(t)>0表示CPU负载正在加速增长（需要警惕），Δ²CPU(t)<0表示CPU负载减速增长或加速下降（趋于稳定）。

我们定义CPU压力指数为：CPU_Stress(t)=α·CPU_percent(t)+β·|ΔCPU(t)|+γ·max(0,Load_Avg(t)-Core_Count)，其中α=0.5,β=0.3,γ=0.2。压力指数>0.7为黄色预警，>0.85为橙色预警，>0.95为红色预警。

### 22.4 CPU监控的完整实现

```python
import asyncio
import psutil
import os
import time
import numpy as np
from dataclasses import dataclass, field
from collections import deque

@dataclass
class CPUMetrics:
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    per_cpu_percent: list = field(default_factory=list)
    cpu_freq_current: float = 0.0
    cpu_freq_max: float = 0.0
    cpu_freq_min: float = 0.0
    cpu_temp: float = 0.0
    cpu_temp_high: float = 85.0
    cpu_temp_critical: float = 100.0
    cpu_times_user: float = 0.0
    cpu_times_system: float = 0.0
    cpu_times_idle: float = 0.0
    cpu_times_iowait: float = 0.0
    load_avg_1min: float = 0.0
    load_avg_5min: float = 0.0
    load_avg_15min: float = 0.0
    ctx_switches: float = 0.0
    cpu_stress_index: float = 0.0
    core_count: int = 0

class CPUCollector:
    def __init__(self):
        self.core_count = psutil.cpu_count(logical=True)
        self.physical_core_count = psutil.cpu_count(logical=False)
        self._prev_cpu_percent = 0.0
        self._initialized = False
        self._history = deque(maxlen=10)

    async def initialize(self):
        psutil.cpu_percent(interval=None)
        await asyncio.sleep(0.2)
        self._initialized = True

    async def collect(self) -> CPUMetrics:
        if not self._initialized:
            await self.initialize()
        metrics = CPUMetrics(core_count=self.core_count)
        metrics.cpu_percent = psutil.cpu_percent(interval=None)
        metrics.per_cpu_percent = psutil.cpu_percent(interval=None, percpu=True)

        try:
            freq = psutil.cpu_freq()
            metrics.cpu_freq_current = freq.current
            metrics.cpu_freq_max = freq.max
            metrics.cpu_freq_min = freq.min
        except Exception:
            pass

        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if 'core' in entry.label.lower() or 'package' in entry.label.lower():
                            metrics.cpu_temp = entry.current
                            metrics.cpu_temp_high = entry.high or 85.0
                            metrics.cpu_temp_critical = entry.critical or 100.0
                            break
        except Exception:
            pass

        cpu_times = psutil.cpu_times()
        metrics.cpu_times_user = cpu_times.user
        metrics.cpu_times_system = cpu_times.system
        metrics.cpu_times_idle = getattr(cpu_times, 'idle', 0.0)
        metrics.cpu_times_iowait = getattr(cpu_times, 'iowait', 0.0)

        try:
            load_avg = os.getloadavg()
            metrics.load_avg_1min = load_avg[0]
            metrics.load_avg_5min = load_avg[1]
            metrics.load_avg_15min = load_avg[2]
        except OSError:
            pass

        try:
            stats = psutil.cpu_stats()
            metrics.ctx_switches = stats.ctx_switches
        except Exception:
            pass

        # 计算CPU压力指数
        usage_contrib = metrics.cpu_percent / 100.0
        delta = abs(metrics.cpu_percent - self._prev_cpu_percent) / 100.0
        load_excess = max(0, metrics.load_avg_1min - self.core_count)
        load_contrib = min(1.0, load_excess / self.core_count)
        temp_contrib = 0.0
        if metrics.cpu_temp > 0 and metrics.cpu_temp > metrics.cpu_temp_high:
            temp_contrib = (metrics.cpu_temp - metrics.cpu_temp_high) / (metrics.cpu_temp_critical - metrics.cpu_temp_high)
        metrics.cpu_stress_index = min(1.0, 0.35*usage_contrib + 0.25*delta + 0.25*load_contrib + 0.15*temp_contrib)

        self._history.append(metrics.cpu_percent)
        self._prev_cpu_percent = metrics.cpu_percent
        return metrics
```

### 22.5 多级预警与边界约束集成

多级预警阈值体系：绿色（CPU<50%, 温度<60°C, 负载/核心<0.7, 压力<0.5）→正常运行；黄色（50-70%, 60-75°C, 0.7-1.0, 0.5-0.7）→增加监控频率；橙色（70-85%, 75-90°C, 1.0-1.5, 0.7-0.85）→降低Agent活跃度；红色（>85%, >90°C, >1.5, >0.85）→暂停非必要任务。

CPU监控直接服务于刚性边界约束（CPU≤20%）。当CPU使用率超过20%时，系统触发三级响应：第一级（>20%）降低采集频率至2s；第二级（>25%）跳过非必要任务；第三级（>30%）暂停所有采集器。每次约束检查记录审计日志，确保可追溯。

### 22.6 阶段小结

本阶段详细设计了CPU监控子模块：建立了涵盖总体使用率、每核心使用率、频率、温度、时间统计、负载和上下文切换的完整指标体系；定义了CPU压力指数作为综合评估指标，结合使用率、变化率、负载和温度四个维度；实现了完整的CPU采集器，包括初始化、采集、趋势分析；与刚性边界约束深度集成，通过三级响应机制确保系统不越界。

---

## 阶段 23：系统层数据采集——内存与磁盘

### 23.1 引言

内存和磁盘是计算机系统中仅次于CPU的核心资源。内存（RAM）是系统的"工作台"——所有正在运行的程序和数据都存储在内存中。磁盘是系统的"仓库"——所有持久化数据都存储在磁盘中。对本系统而言，内存监控直接关系到刚性边界约束（内存≤20%），而磁盘监控则关系到数据持久化的可靠性。本阶段将深入设计内存和磁盘监控子模块。

### 23.2 内存监控

虚拟内存的核心指标通过psutil.virtual_memory()获取：total（总物理内存）、available（可用内存，包括可回收缓存）、used（已使用）、free（完全空闲）、percent（使用率）、active（活跃使用中）、inactive（非活跃可回收）、buffers（缓冲区内存）、cached（缓存内存）、shared（共享内存）。关键理解：free和available是不同的。Linux内核会使用空闲内存作为文件缓存（cached），这些缓存可以在需要时立即释放。因此available是比free更准确的"可用内存"指标。

交换内存（Swap Memory）通过psutil.swap_memory()获取：total（交换空间总大小）、used（已使用）、free（空闲）、percent（使用率）、sin（从磁盘交换入的字节数）、sout（交换出到磁盘的字节数）。Swap使用率>0是内存压力的重要信号——这意味着物理内存不足，系统正在将内存页写入磁盘（速度极慢，比RAM慢约1000倍）。

内存压力指数定义为：Mem_Pressure(t)=α·Mem_Percent(t)+β·Swap_Percent(t)+γ·Swap_Rate(t)，其中Swap_Rate(t)=(sout(t)-sout(t-1))/Δt，α=0.5,β=0.3,γ=0.2。压力指数<0.3为绿色（内存充足），0.3-0.6为黄色（内存偏紧），0.6-0.8为橙色（内存紧张），≥0.8为红色（内存严重不足）。

内存碎片化指数定义为：Mem_Fragmentation=1-(largest_free_block/total_free)。碎片化越接近1，内存越碎片化，越难分配大块连续内存。

```python
import psutil
from dataclasses import dataclass, field
import time

@dataclass
class MemoryMetrics:
    timestamp: float = field(default_factory=time.time)
    mem_total: int = 0
    mem_available: int = 0
    mem_used: int = 0
    mem_free: int = 0
    mem_percent: float = 0.0
    mem_active: int = 0
    mem_inactive: int = 0
    mem_buffers: int = 0
    mem_cached: int = 0
    swap_total: int = 0
    swap_used: int = 0
    swap_free: int = 0
    swap_percent: float = 0.0
    swap_sin: int = 0
    swap_sout: int = 0
    swap_rate: float = 0.0
    mem_pressure: float = 0.0

class MemoryCollector:
    def __init__(self):
        self._prev_swap_sout = 0
        self._prev_timestamp = time.time()

    def collect(self) -> MemoryMetrics:
        metrics = MemoryMetrics()
        mem = psutil.virtual_memory()
        metrics.mem_total = mem.total
        metrics.mem_available = mem.available
        metrics.mem_used = mem.used
        metrics.mem_free = mem.free
        metrics.mem_percent = mem.percent
        metrics.mem_active = getattr(mem, 'active', 0)
        metrics.mem_inactive = getattr(mem, 'inactive', 0)
        metrics.mem_buffers = getattr(mem, 'buffers', 0)
        metrics.mem_cached = getattr(mem, 'cached', 0)

        swap = psutil.swap_memory()
        metrics.swap_total = swap.total
        metrics.swap_used = swap.used
        metrics.swap_free = swap.free
        metrics.swap_percent = swap.percent
        metrics.swap_sin = swap.sin
        metrics.swap_sout = swap.sout

        now = time.time()
        delta_t = now - self._prev_timestamp
        if delta_t > 0:
            metrics.swap_rate = (swap.sout - self._prev_swap_sout) / delta_t
        self._prev_swap_sout = swap.sout
        self._prev_timestamp = now

        metrics.mem_pressure = 0.5*(metrics.mem_percent/100.0) + 0.3*(metrics.swap_percent/100.0) + 0.2*min(1.0, metrics.swap_rate/(1024*1024))
        return metrics
```

### 23.3 磁盘监控

磁盘I/O统计通过psutil.disk_io_counters()获取：read_bytes/write_bytes（读写字节数累计）、read_count/write_count（读写操作次数累计）、read_time/write_time（读写耗时毫秒累计）。从这些累计值可以计算读写速率（字节/秒）、平均读延迟（read_time/read_count ms/次）、平均写延迟（write_time/write_count ms/次）。

I/O压力指数定义：IO_Pressure(t)=α·(read_rate/max_read_rate)+β·(write_rate/max_write_rate)+γ·(avg_latency/max_latency)。其中max_read_rate假设为100MB/s，max_latency为100ms。

磁盘分区使用率通过psutil.disk_partitions()遍历所有分区，再通过psutil.disk_usage()获取每个分区的使用情况。系统监控最满的分区（使用率最高的分区）。磁盘使用率预警：>80%黄色（提醒清理）、>90%橙色（加速清理、淘汰旧数据）、>95%红色（紧急清理、暂停新数据写入）。

```python
@dataclass
class DiskMetrics:
    timestamp: float = field(default_factory=time.time)
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    disk_read_count: int = 0
    disk_write_count: int = 0
    disk_read_time: int = 0
    disk_write_time: int = 0
    disk_read_rate: float = 0.0
    disk_write_rate: float = 0.0
    disk_avg_read_latency: float = 0.0
    disk_avg_write_latency: float = 0.0
    disk_io_pressure: float = 0.0
    disk_usage_percent: float = 0.0
    disk_total: int = 0
    disk_used: int = 0
    disk_free: int = 0
    disk_mountpoint: str = ""

class DiskCollector:
    def __init__(self):
        self._prev_io = None
        self._prev_time = time.time()

    def collect(self) -> DiskMetrics:
        metrics = DiskMetrics()
        try:
            io = psutil.disk_io_counters()
            metrics.disk_read_bytes = io.read_bytes
            metrics.disk_write_bytes = io.write_bytes
            metrics.disk_read_count = io.read_count
            metrics.disk_write_count = io.write_count
            metrics.disk_read_time = io.read_time
            metrics.disk_write_time = io.write_time
            now = time.time()
            if self._prev_io is not None:
                delta_t = now - self._prev_time
                if delta_t > 0:
                    metrics.disk_read_rate = (io.read_bytes - self._prev_io.read_bytes) / delta_t
                    metrics.disk_write_rate = (io.write_bytes - self._prev_io.write_bytes) / delta_t
                    read_delta = io.read_count - self._prev_io.read_count
                    write_delta = io.write_count - self._prev_io.write_count
                    if read_delta > 0:
                        metrics.disk_avg_read_latency = (io.read_time - self._prev_io.read_time) / read_delta
                    if write_delta > 0:
                        metrics.disk_avg_write_latency = (io.write_time - self._prev_io.write_time) / write_delta
            self._prev_io = io
            self._prev_time = now
        except Exception:
            pass

        try:
            max_usage = 0.0
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    if usage.percent > max_usage:
                        max_usage = usage.percent
                        metrics.disk_usage_percent = usage.percent
                        metrics.disk_total = usage.total
                        metrics.disk_used = usage.used
                        metrics.disk_free = usage.free
                        metrics.disk_mountpoint = partition.mountpoint
                except PermissionError:
                    continue
        except Exception:
            pass

        max_rate = 100 * 1024 * 1024
        rate_contrib = (metrics.disk_read_rate + metrics.disk_write_rate) / (2 * max_rate)
        latency_contrib = max(metrics.disk_avg_read_latency, metrics.disk_avg_write_latency) / 100.0
        metrics.disk_io_pressure = min(1.0, 0.6 * rate_contrib + 0.4 * latency_contrib)
        return metrics
```

### 23.4 内存与磁盘的联动监控

内存和磁盘不是独立的——当内存不足时，系统会使用Swap（磁盘），当磁盘I/O过高时，可能是Swap导致的。因此需要联动监控。定义内存-磁盘联动压力指数：MD_Combined_Pressure=max(Mem_Pressure,IO_Pressure)+0.3·min(Mem_Pressure,IO_Pressure)。如果内存压力和磁盘I/O压力同时高，可能存在Swap风暴（系统在内存和磁盘之间疯狂交换数据），需要紧急处理。

### 23.5 阶段小结

本阶段详细设计了内存和磁盘监控子模块：内存监控覆盖虚拟内存和交换内存的完整指标，包括内存压力指数和碎片化指数；磁盘监控覆盖I/O统计和分区使用率，包括I/O压力指数；实现了内存和磁盘的联动监控，通过联动压力指数检测Swap风暴等异常情况；建立了多级预警阈值体系，与刚性边界约束深度集成。

---

## 阶段 24：系统层数据采集——网络与进程

### 24.1 引言

网络和进程是系统层监控的最后两个维度。网络监控让我们感知系统与外界的通信模式——是否有人在大量传输数据？是否出现了异常的网络连接？进程监控让我们感知系统内部的活动——哪些程序在运行？哪些程序消耗了大量资源？是否有僵尸进程？本阶段将完成系统层监控的全部设计。

### 24.2 网络监控

网络I/O统计通过psutil.net_io_counters()获取：bytes_sent/bytes_recv（发送/接收字节数累计）、packets_sent/packets_recv（数据包数累计）、errin/errout（错误数累计）、dropin/dropout（丢弃数累计）。网络错误率error_rate=errin/packets_recv，错误率>1%表示网络质量下降。网络流量速率：Net_Send_Rate(t)=(bytes_sent(t)-bytes_sent(t-1))/Δt，Net_Recv_Rate(t)=(bytes_recv(t)-bytes_recv(t-1))/Δt。

网络连接监控通过psutil.net_connections(kind='inet')获取所有网络连接的列表。关键指标包括总连接数、各状态连接数（ESTABLISHED, LISTEN, TIME_WAIT, CLOSE_WAIT）、监听端口数、外部连接数和连接速率（新连接数/秒）。连接状态分布的意义：大量TIME_WAIT连接→可能有短连接风暴；大量CLOSE_WAIT连接→可能有应用层连接泄漏；大量ESTABLISHED连接→正常的活跃连接。

网络连接异常检测：Net_Anomaly_Score=α·(connection_rate/max_normal_rate)+β·(error_rate/0.01)+γ·(unusual_ports/total_ports)。其中max_normal_rate是历史正常连接速率。

```python
from dataclasses import dataclass, field
import time
import psutil
import os

@dataclass
class NetworkMetrics:
    timestamp: float = field(default_factory=time.time)
    net_bytes_sent: int = 0
    net_bytes_recv: int = 0
    net_packets_sent: int = 0
    net_packets_recv: int = 0
    net_errin: int = 0
    net_errout: int = 0
    net_dropin: int = 0
    net_dropout: int = 0
    net_send_rate: float = 0.0
    net_recv_rate: float = 0.0
    net_error_rate: float = 0.0
    net_total_rate: float = 0.0
    net_connections: int = 0
    net_established: int = 0
    net_listen: int = 0
    net_time_wait: int = 0
    net_close_wait: int = 0
    net_conn_rate: float = 0.0
    net_anomaly_score: float = 0.0

class NetworkCollector:
    def __init__(self):
        self._prev_io = None
        self._prev_conn_count = 0
        self._prev_time = time.time()

    def collect(self) -> NetworkMetrics:
        metrics = NetworkMetrics()
        now = time.time()
        try:
            net_io = psutil.net_io_counters()
            metrics.net_bytes_sent = net_io.bytes_sent
            metrics.net_bytes_recv = net_io.bytes_recv
            metrics.net_packets_sent = net_io.packets_sent
            metrics.net_packets_recv = net_io.packets_recv
            metrics.net_errin = net_io.errin
            metrics.net_errout = net_io.errout
            metrics.net_dropin = net_io.dropin
            metrics.net_dropout = net_io.dropout
            if self._prev_io is not None:
                delta_t = now - self._prev_time
                if delta_t > 0:
                    metrics.net_send_rate = (net_io.bytes_sent - self._prev_io.bytes_sent) / delta_t
                    metrics.net_recv_rate = (net_io.bytes_recv - self._prev_io.bytes_recv) / delta_t
                    metrics.net_total_rate = metrics.net_send_rate + metrics.net_recv_rate
                    packets_delta = net_io.packets_recv - self._prev_io.packets_recv
                    if packets_delta > 0:
                        err_delta = net_io.errin - self._prev_io.errin
                        metrics.net_error_rate = err_delta / packets_delta
            self._prev_io = net_io
        except Exception:
            pass

        try:
            connections = psutil.net_connections(kind='inet')
            metrics.net_connections = len(connections)
            status_counts = {'ESTABLISHED': 0, 'LISTEN': 0, 'TIME_WAIT': 0, 'CLOSE_WAIT': 0}
            for conn in connections:
                status = conn.status
                if status in status_counts:
                    status_counts[status] += 1
            metrics.net_established = status_counts['ESTABLISHED']
            metrics.net_listen = status_counts['LISTEN']
            metrics.net_time_wait = status_counts['TIME_WAIT']
            metrics.net_close_wait = status_counts['CLOSE_WAIT']
            delta_t = now - self._prev_time
            if delta_t > 0:
                metrics.net_conn_rate = (len(connections) - self._prev_conn_count) / delta_t
            self._prev_conn_count = len(connections)
        except Exception:
            pass

        self._prev_time = now
        score = 0.0
        if metrics.net_error_rate > 0.01:
            score += 0.3
        if metrics.net_time_wait > 1000:
            score += 0.2
        if metrics.net_close_wait > 100:
            score += 0.3
        if metrics.net_total_rate > 50 * 1024 * 1024:
            score += 0.2
        metrics.net_anomaly_score = min(1.0, score)
        return metrics
```

### 24.3 进程监控

进程列表通过psutil.process_iter()获取。关键指标包括进程总数、各状态进程数（running, sleeping, zombie, stopped, disk-sleep）、Top5 CPU消耗进程、Top5内存消耗进程、僵尸进程数和孤儿进程检测。

僵尸进程（Zombie）虽然本身不消耗资源，但占用进程表条目，大量僵尸进程是系统异常的信号。系统设定僵尸进程数>10触发预警。

此外，进程监控还需要追踪本Agent自身的进程信息：Agent的CPU使用率、内存使用率和内存占用（MB）。这直接服务于刚性边界约束的监控。

```python
@dataclass
class ProcessInfo:
    pid: int = 0
    name: str = ""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    status: str = ""

@dataclass
class ProcessMetrics:
    timestamp: float = field(default_factory=time.time)
    proc_total: int = 0
    proc_running: int = 0
    proc_sleeping: int = 0
    proc_zombie: int = 0
    proc_stopped: int = 0
    top_cpu_procs: list = field(default_factory=list)
    top_mem_procs: list = field(default_factory=list)
    agent_cpu_percent: float = 0.0
    agent_memory_percent: float = 0.0
    agent_memory_mb: float = 0.0
    zombie_alert: bool = False
    high_cpu_alert: bool = False

class ProcessCollector:
    def __init__(self):
        self._agent_pid = os.getpid()

    def collect(self) -> ProcessMetrics:
        metrics = ProcessMetrics()
        status_counts = {'running': 0, 'sleeping': 0, 'zombie': 0, 'stopped': 0}
        proc_list = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                try:
                    info = proc.info
                    pid = info['pid']
                    proc_list.append(ProcessInfo(
                        pid=pid, name=info['name'] or '',
                        cpu_percent=info['cpu_percent'] or 0.0,
                        memory_percent=info['memory_percent'] or 0.0,
                        status=info['status']
                    ))
                    if info['status'] in status_counts:
                        status_counts[info['status']] += 1
                    if pid == self._agent_pid:
                        metrics.agent_cpu_percent = info['cpu_percent'] or 0.0
                        metrics.agent_memory_percent = info['memory_percent'] or 0.0
                        try:
                            mem_info = proc.memory_info()
                            metrics.agent_memory_mb = mem_info.rss / (1024 * 1024)
                        except Exception:
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass

        metrics.proc_total = len(proc_list)
        metrics.proc_running = status_counts['running']
        metrics.proc_sleeping = status_counts['sleeping']
        metrics.proc_zombie = status_counts['zombie']
        metrics.proc_stopped = status_counts['stopped']
        proc_list.sort(key=lambda p: p.cpu_percent, reverse=True)
        metrics.top_cpu_procs = proc_list[:5]
        proc_list.sort(key=lambda p: p.memory_percent, reverse=True)
        metrics.top_mem_procs = proc_list[:5]
        metrics.zombie_alert = metrics.proc_zombie > 10
        metrics.high_cpu_alert = any(p.cpu_percent > 80 for p in proc_list[:5])
        return metrics
```

### 24.4 阶段小结

本阶段完成了系统层监控的最后两个维度：网络监控覆盖I/O统计（发送/接收速率、错误率）、连接统计（各状态连接数、连接速率）和异常检测（错误率、TIME_WAIT堆积、流量突变）；进程监控覆盖进程总数、状态分布、Top5 CPU/内存消耗进程、僵尸进程检测和本Agent进程自监控。至此，系统层的四个维度全部完成，为环境感知提供了48维的系统层特征。

---

## 阶段 25：日志层数据采集——结构化日志解析

### 25.1 引言

日志是系统的"黑匣子"。系统日志、应用日志、安全日志中包含了丰富的运行时信息——错误、警告、异常、性能指标、安全事件等。然而，日志数据是半结构化的文本，计算机无法直接理解。本阶段将设计日志层的结构化解析子模块，将非结构化的日志文本转化为结构化的数据特征。

### 25.2 日志采集架构

日志采集器需要监控多个日志源，系统配置了system（syslog/messages）、kernel（kern.log）、auth（auth.log）、application（用户自定义）四类日志源，每类日志源配置了对应的关键词模式（如error、fail、critical、panic、segfault、authentication failure等）和优先级。

增量读取策略是日志采集的核心工程挑战。日志文件是持续追加写入的，采集器需要记住上次读取的位置，只读取新增的内容。同时需要处理日志轮转（log rotation）——当日志文件被轮转时，inode会改变，采集器需要检测到这一变化并从头开始读取新文件；如果文件被截断（truncate），采集器也需要从0重新开始。

```python
import os
import re
from collections import Counter, deque
from dataclasses import dataclass, field
import time

class LogFileTracker:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.inode = None
        self.position = 0
        self._init_tracking()

    def _init_tracking(self):
        try:
            stat = os.stat(self.filepath)
            self.inode = stat.st_ino
            self.position = stat.st_size
        except FileNotFoundError:
            self.inode = None
            self.position = 0

    def read_new_lines(self) -> list:
        lines = []
        try:
            current_stat = os.stat(self.filepath)
            if current_stat.st_ino != self.inode:
                self.position = 0
                self.inode = current_stat.st_ino
            if current_stat.st_size < self.position:
                self.position = 0
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.position)
                new_content = f.read()
                if new_content:
                    lines = new_content.strip().split('\n')
                    self.position = f.tell()
        except Exception:
            pass
        return lines
```

### 25.3 日志解析引擎

使用正则表达式解析多种日志格式。syslog格式使用模式：(?P<timestamp>\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(?P<hostname>\S+)\s+(?P<process>\S+?)(?:\[(?P<pid>\d+)\])?:\s+(?P<message>.*)。ISO时间戳格式使用模式：(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})\s+(?P<level>\w+)\s+(?P<message>.*)。同时使用关键词模式匹配错误类型（error/exception/fail/critical/fatal/panic/traceback）和IP地址提取。

```python
LOG_PATTERNS = {
    'syslog': re.compile(
        r'(?P<timestamp>\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+'
        r'(?P<hostname>\S+)\s+'
        r'(?P<process>\S+?)(?:\[(?P<pid>\d+)\])?:\s+'
        r'(?P<message>.*)'
    ),
    'iso_timestamp': re.compile(
        r'(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+'
        r'(?P<level>\w+)\s+'
        r'(?P<message>.*)'
    ),
    'error_pattern': re.compile(r'(?i)(error|exception|fail|critical|fatal|panic|traceback)'),
    'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
}

def parse_log_line(line: str) -> dict:
    result = {'raw': line, 'timestamp': None, 'level': 'UNKNOWN', 'process': '',
              'pid': None, 'message': line, 'has_error': False, 'error_type': None, 'ip_addresses': []}
    m = LOG_PATTERNS['syslog'].match(line)
    if m:
        result['process'] = m.group('process')
        result['pid'] = int(m.group('pid')) if m.group('pid') else None
        result['message'] = m.group('message')
    m = LOG_PATTERNS['iso_timestamp'].match(line)
    if m:
        result['level'] = m.group('level').upper()
        result['message'] = m.group('message')
    m = LOG_PATTERNS['error_pattern'].search(line)
    if m:
        result['has_error'] = True
        result['error_type'] = m.group(1).upper()
    for level in ['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'WARN', 'INFO', 'DEBUG']:
        if level in line.upper():
            result['level'] = level
            break
    result['ip_addresses'] = LOG_PATTERNS['ip_address'].findall(line)
    return result
```

### 25.4 日志特征提取

从解析后的日志中提取数值特征，形成结构化的LogMetrics：日志速率指标（total_lines, error/warning/info/debug/critical_lines）、速率指标（lines/errors/warnings_per_minute）、错误分布（error_counts字典）、模式统计（unique_patterns, avg_line_length）和突发检测（lines_per_minute>1000触发burst_detected）。

完整的LogCollector类管理所有日志文件追踪器，周期性地读取新增行、解析每条日志并更新统计计数器，计算速率指标，检测日志突发。

### 25.5 阶段小结

本阶段设计了日志层的结构化解析子模块：使用正则表达式模式匹配解析syslog、ISO时间戳等多种日志格式；使用文件追踪器实现日志文件的增量读取和轮转检测；从解析后的日志中提取日志速率、错误率、级别分布、错误类型分布等特征；为后续的异常序列检测提供了结构化的数据基础。

'''

with open('/workspace/自驱永续进化预测修正Agent_技术白皮书.md', 'a', encoding='utf-8') as f:
    f.write(content)

print("Phases 21-25 written successfully.")