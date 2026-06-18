#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write Phases 26-30 of Part 2."""

content = r'''

## 阶段 26：日志层数据采集——异常序列检测

### 26.1 引言

阶段25解决了日志的结构化解析问题。但仅知道"有多少条错误日志"是不够的——我们需要知道这些错误是否构成了一个异常序列。例如，一条错误日志可能是偶然的，但如果在10秒内连续出现5条相同类型的错误日志，这就构成了一个异常序列。本阶段将使用Isolation Forest（隔离森林）算法来检测日志中的异常序列。

### 26.2 Isolation Forest算法原理

Isolation Forest是一种基于"隔离"概念的异常检测算法。它的核心直觉是：异常点比正常点更容易被隔离。想象你在一个二维平面上随机选择一条分割线，将一个点与其他点分开。异常点（远离群体的点）通常只需要很少的几次分割就能被隔离，而正常点（在群体中心）需要很多次分割。

隔离森林的构建过程：给定数据集X={x_1,x_2,...,x_n}，构建t棵隔离树（iTree）。每棵iTree随机选择一个特征，随机选择该特征的一个分割值（在最小值和最大值之间），根据分割值将数据分为左右两个子集，递归执行直到每个叶子节点只有一个数据点或达到最大深度。

异常分数定义为：s(x,n)=2^(-E[h(x)]/c(n))。其中h(x)是样本x在iTree中的路径长度（需要多少次分割才能隔离），E[h(x)]是所有iTree中路径长度的平均值，c(n)=2H(n-1)-2(n-1)/n是二叉搜索树的平均路径长度（归一化因子），H(i)是调和数约等于ln(i)+0.5772156649（欧拉常数）。异常分数的解读：s(x,n)≈1表示x是异常点（路径很短，很容易被隔离），s(x,n)≈0.5表示x没有明显的异常特征，s(x,n)<<0.5表示x是非常正常的点。

选择Isolation Forest的理由：线性时间复杂度O(n)，比基于距离的方法（O(n²)）或基于密度的方法（O(n log n)）更快；低内存占用，只需要存储树结构；无需数据分布假设；支持增量学习；处理高维数据有良好的可扩展性。

### 26.3 日志序列特征提取

首先将连续的日志条目转化为固定长度的特征向量序列。使用LogSequenceFeatureExtractor维护一个窗口大小为10的日志缓冲区，每次添加一条解析后的日志，然后从当前窗口中提取20维特征向量。特征包括：窗口大小、错误数、警告数、严重数、平均行长度、行长度标准差、最大行长度、不同错误类型数、最频繁错误类型计数、不同进程数、IP地址出现次数、最大连续错误数等。

### 26.4 Isolation Forest异常检测器实现

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque, Counter

class LogSequenceFeatureExtractor:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.log_buffer = deque(maxlen=window_size)

    def add_log(self, parsed_log: dict):
        self.log_buffer.append(parsed_log)

    def extract_features(self) -> np.ndarray:
        if len(self.log_buffer) == 0:
            return np.zeros(20, dtype=np.float32)
        features = []
        features.append(len(self.log_buffer))
        features.append(sum(1 for l in self.log_buffer if l['has_error']))
        features.append(sum(1 for l in self.log_buffer if l['level'] == 'WARNING'))
        features.append(sum(1 for l in self.log_buffer if l['level'] == 'CRITICAL'))
        line_lengths = [len(l['raw']) for l in self.log_buffer]
        if line_lengths:
            features.append(np.mean(line_lengths))
            features.append(np.std(line_lengths))
            features.append(np.max(line_lengths))
        else:
            features.extend([0, 0, 0])
        error_types = Counter()
        for l in self.log_buffer:
            if l['error_type']:
                error_types[l['error_type']] += 1
        features.append(len(error_types))
        features.append(max(error_types.values()) if error_types else 0)
        processes = Counter()
        for l in self.log_buffer:
            if l['process']:
                processes[l['process']] += 1
        features.append(len(processes))
        ip_count = sum(len(l['ip_addresses']) for l in self.log_buffer)
        features.append(ip_count)
        max_consecutive_errors = 0
        current_streak = 0
        for l in self.log_buffer:
            if l['has_error']:
                current_streak += 1
                max_consecutive_errors = max(max_consecutive_errors, current_streak)
            else:
                current_streak = 0
        features.append(max_consecutive_errors)
        while len(features) < 20:
            features.append(0.0)
        return np.array(features[:20], dtype=np.float32)

class LogAnomalyDetector:
    def __init__(self, n_estimators: int = 100, contamination: float = 0.1):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination,
            random_state=42, n_jobs=1
        )
        self._fitted = False
        self._training_buffer = deque(maxlen=5000)
        self.feature_extractor = LogSequenceFeatureExtractor(window_size=10)
        self.anomaly_scores = deque(maxlen=100)

    def add_log(self, parsed_log: dict):
        self.feature_extractor.add_log(parsed_log)

    def fit_if_needed(self):
        if len(self._training_buffer) >= 1000 and not self._fitted:
            X = np.array(list(self._training_buffer))
            self.model.fit(X)
            self._fitted = True
            self._training_buffer.clear()

    def detect(self) -> float:
        features = self.feature_extractor.extract_features()
        if not self._fitted:
            self._training_buffer.append(features)
            self.fit_if_needed()
            return 0.5
        features_2d = features.reshape(1, -1)
        score = self.model.decision_function(features_2d)[0]
        normalized_score = 1.0 - (score + 0.5)
        normalized_score = max(0.0, min(1.0, normalized_score))
        self.anomaly_scores.append(normalized_score)
        if normalized_score > 0.7:
            self._training_buffer.append(features)
        return normalized_score

    def get_anomaly_trend(self) -> float:
        if len(self.anomaly_scores) < 10:
            return 0.0
        recent = list(self.anomaly_scores)[-10:]
        x = np.arange(len(recent))
        y = np.array(recent)
        slope = np.polyfit(x, y, 1)[0]
        return slope

    def incremental_update(self):
        if len(self._training_buffer) >= 500:
            X_new = np.array(list(self._training_buffer))
            new_model = IsolationForest(
                n_estimators=self.n_estimators + 20,
                contamination=self.contamination,
                random_state=42, n_jobs=1
            )
            new_model.fit(X_new)
            self.model = new_model
            self._training_buffer.clear()
            self._fitted = True
```

### 26.5 异常序列的分类与驱力集成

除了Isolation Forest的异常分数，系统还将异常序列分为五种类型：SPIKE（突发错误潮，error_count>5）、PATTERN（周期性异常模式）、NOVEL（前所未见的日志模式）、CASCADE（级联错误，一个错误触发多个，error_types>3）、SILENCE（异常安静，日志突然停止，buffer<3）。

当检测到异常序列时，系统会提高安全驱力的权重。高异常（anomaly_score>0.8）时安全驱力权重提升至0.7；中异常（>0.6）时提升至0.5。这确保了系统在感知到风险时优先保障安全。

### 26.6 阶段小结

本阶段设计了日志异常序列检测子模块：使用Isolation Forest算法检测日志序列中的异常模式，基于"异常点更容易被隔离"的核心直觉；构建了20维的日志序列特征向量，包括统计特征、时间特征、错误类型特征、进程特征和序列模式特征；实现了增量学习机制，模型可以持续适应新的日志模式；定义了五种异常类型，为安全驱力提供精确的异常信号。

---

## 阶段 27：文件层数据采集——文件系统监控

### 27.1 引言

文件系统是计算机存储的基础。文件的变化——创建、修改、删除、权限变更——反映了系统状态的变化。例如，配置文件被修改可能意味着系统行为即将改变，新文件被创建可能意味着新应用正在安装，日志文件快速增长可能意味着系统出现了问题。本阶段将使用watchdog库实现事件驱动的文件系统监控。

### 27.2 watchdog文件系统监控原理

watchdog是一个跨平台的Python库，用于监控文件系统事件。其核心架构包括Observer（观察者，运行在后台线程中）、Event Handler（事件处理器，用户自定义）和Event（事件抽象）。watchdog在Linux上使用inotify（内核文件系统事件监控接口），在macOS上使用FSEvents，在Windows上使用ReadDirectoryChangesW。这些底层API都是事件驱动的，效率极高——不需要轮询，有事件发生时才触发回调。

watchdog支持以下文件系统事件类型：FileCreatedEvent（文件创建）、FileModifiedEvent（文件修改）、FileDeletedEvent（文件删除）、FileMovedEvent（文件移动/重命名）、DirCreatedEvent（目录创建）、DirModifiedEvent（目录修改）、DirDeletedEvent（目录删除）、DirMovedEvent（目录移动/重命名）。

### 27.3 文件系统监控的完整实现

```python
import asyncio
import time
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

@dataclass
class FileEventRecord:
    timestamp: float
    event_type: str
    path: str
    is_directory: bool
    file_extension: str = ""

@dataclass
class FileMetrics:
    timestamp: float = field(default_factory=time.time)
    created_count: int = 0
    modified_count: int = 0
    deleted_count: int = 0
    moved_count: int = 0
    total_events: int = 0
    event_rate: float = 0.0
    avg_event_interval: float = 0.0
    config_changes: int = 0
    code_changes: int = 0
    data_changes: int = 0
    log_changes: int = 0
    unique_paths: int = 0
    monitored_dirs: int = 0
    active_dirs: int = 0
    burst_detected: bool = False
    anomaly_score: float = 0.0

EXTENSION_CATEGORIES = {
    'config': {'.conf', '.cfg', '.ini', '.yaml', '.yml', '.json', '.xml', '.toml', '.properties', '.env'},
    'code': {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.rs', '.go', '.rb', '.php', '.sh', '.bash'},
    'data': {'.csv', '.tsv', '.parquet', '.jsonl', '.db', '.sqlite', '.sqlite3', '.bin', '.dat', '.pkl', '.pickle', '.npy', '.npz'},
    'log': {'.log', '.out', '.err', '.trace'},
}

class FileEventHandler(FileSystemEventHandler):
    def __init__(self, event_queue: asyncio.Queue):
        super().__init__()
        self.event_queue = event_queue

    def on_created(self, event):
        self._push_event('created', event)
    def on_modified(self, event):
        self._push_event('modified', event)
    def on_deleted(self, event):
        self._push_event('deleted', event)
    def on_moved(self, event):
        self._push_event('moved', event)

    def _push_event(self, event_type: str, event: FileSystemEvent):
        path = event.src_path
        ext = Path(path).suffix.lower()
        record = FileEventRecord(
            timestamp=time.time(), event_type=event_type,
            path=path, is_directory=event.is_directory, file_extension=ext
        )
        try:
            self.event_queue.put_nowait(record)
        except asyncio.QueueFull:
            pass

class FileCollector:
    def __init__(self, watch_paths=None):
        self.watch_paths = watch_paths or ['/var/log', '/etc', str(Path.home())]
        self.observer = None
        self.event_queue = asyncio.Queue(maxsize=10000)
        self.event_handler = FileEventHandler(self.event_queue)
        self._event_buffer = deque(maxlen=1000)
        self._prev_time = time.time()
        self._total_events = 0

    def start(self):
        self.observer = Observer()
        for path in self.watch_paths:
            if os.path.exists(path):
                self.observer.schedule(self.event_handler, path, recursive=True)
        self.observer.start()

    def stop(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()

    async def collect(self) -> FileMetrics:
        metrics = FileMetrics()
        new_events = []
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                new_events.append(event)
                self._event_buffer.append(event)
            except asyncio.QueueEmpty:
                break

        for event in new_events:
            self._total_events += 1
            if event.event_type == 'created':
                metrics.created_count += 1
            elif event.event_type == 'modified':
                metrics.modified_count += 1
            elif event.event_type == 'deleted':
                metrics.deleted_count += 1
            elif event.event_type == 'moved':
                metrics.moved_count += 1

        metrics.total_events = len(new_events)
        now = time.time()
        delta_t = now - self._prev_time
        if delta_t > 0:
            metrics.event_rate = metrics.total_events / delta_t
            if metrics.total_events > 0:
                metrics.avg_event_interval = delta_t / metrics.total_events
        self._prev_time = now

        for event in new_events:
            ext = event.file_extension
            if ext in EXTENSION_CATEGORIES['config']:
                metrics.config_changes += 1
            elif ext in EXTENSION_CATEGORIES['code']:
                metrics.code_changes += 1
            elif ext in EXTENSION_CATEGORIES['data']:
                metrics.data_changes += 1
            elif ext in EXTENSION_CATEGORIES['log']:
                metrics.log_changes += 1

        unique_paths = set()
        active_dirs = set()
        for event in new_events:
            unique_paths.add(event.path)
            active_dirs.add(str(Path(event.path).parent))
        metrics.unique_paths = len(unique_paths)
        metrics.active_dirs = len(active_dirs)
        metrics.monitored_dirs = len(self.watch_paths)

        if metrics.event_rate > 100:
            metrics.burst_detected = True
            metrics.anomaly_score = 0.8
        elif metrics.event_rate > 50:
            metrics.anomaly_score = 0.5

        return metrics
```

### 27.4 文件变更的智能过滤

并非所有文件变更都同等重要。临时文件、缓存文件、编译中间文件的变更往往是噪声。系统维护了一个IGNORE_PATTERNS列表（包括*.pyc, __pycache__/*, *.swp, *.tmp, .git/*, node_modules/*, .venv/*, *.cache, *.lock, *.class, *.log.[0-9]*等），使用fnmatch进行路径匹配过滤。

此外，系统实现了文件重要性评分函数score_file_importance()：基础分0.5，配置文件（/etc/、.conf、.yaml）加0.3，可执行文件（/bin/、/sbin/）加0.3，系统关键文件（/etc/passwd、/etc/shadow、/boot/）直接设为1.0，临时文件（/tmp/、cache、.tmp）减0.3，删除操作加0.1。重要性评分用于后续的事件优先级排序。

### 27.5 阶段小结

本阶段设计了文件系统监控子模块：使用watchdog库实现事件驱动的文件系统监控，利用操作系统的inotify/FSEvents等底层API实现高效监控；支持所有文件系统事件类型，按文件类型（配置、代码、数据、日志）进行分类统计；实现了智能过滤机制，忽略临时文件和缓存文件的噪声，优先关注系统关键文件和配置文件的变更；为环境感知提供了16维的文件层特征。

---

## 阶段 28：自身层数据采集——模型内省

### 28.1 引言

环境感知不仅要感知外部环境（系统、日志、文件），还要感知系统自身——这就是"自身层"（Self Layer）。自身层实现了系统的自我意识：系统知道自己的预测精度如何、消耗了多少资源、模型熵是多少、是否存在过拟合信号。没有自身层，系统就无法知道"我做得好不好"，也就无法进行自我优化。

### 28.2 模型内省的核心维度

预测精度指标是自身层最重要的监控指标。均方误差MSE_t=(1/n)·Σ(x_i-x̂_i)²对大的误差特别敏感（误差被平方放大）。均方绝对误差MAE_t=(1/n)·Σ|x_i-x̂_i|对所有误差一视同仁。MSE/MAE比值>1且持续增大意味着存在少量大幅偏差（可能是异常事件）。误差趋势使用指数移动平均（EMA）平滑误差序列。预测置信度衡量模型输出的概率估计的校准程度。

模型资源指标包括：推理延迟（模型从接收输入到输出预测的时间，单位毫秒）、模型参数规模（总参数数量，sum(p.numel() for p in model.parameters())）、模型内存占用（参数和缓冲区大小之和）、梯度范数（最近一次梯度更新的范数，||∇L||=sqrt(Σ(∂L/∂θ_i)²)）。梯度范数过大意味着模型参数可能在剧烈变化（不稳定），过小意味着模型可能陷入局部最优。

模型状态指标包括：模型熵（H_model=-ΣP(ŷ_i)·log P(ŷ_i)，预测不确定性）、认知覆盖度（已知状态空间占观测到的状态空间的比例）、过拟合信号（当训练误差持续下降但验证误差上升时）、学习率和探索率的当前值。

### 28.3 自身层监控的完整实现

```python
import torch
import numpy as np
import time
from dataclasses import dataclass, field
from collections import deque

@dataclass
class SelfMetrics:
    timestamp: float = field(default_factory=time.time)
    mse_recent: float = 0.0
    mae_recent: float = 0.0
    mse_trend: float = 0.0
    mae_trend: float = 0.0
    accuracy_ema: float = 0.0
    error_std: float = 0.0
    prediction_confidence: float = 0.0
    calibration_error: float = 0.0
    model_ram_mb: float = 0.0
    model_cpu_percent: float = 0.0
    inference_time_ms: float = 0.0
    model_param_count: int = 0
    gradient_norm: float = 0.0
    last_update_time: float = 0.0
    update_frequency: float = 0.0
    model_entropy: float = 0.0
    knowledge_coverage: float = 0.0
    overfitting_signal: float = 0.0
    learning_rate_current: float = 0.0
    exploration_rate_current: float = 0.0
    error_autocorr: float = 0.0
    error_skewness: float = 0.0
    error_kurtosis: float = 0.0

class SelfCollector:
    def __init__(self):
        self._error_history = deque(maxlen=1000)
        self._inference_times = deque(maxlen=100)
        self._mse_ema = None
        self._prev_mse_ema = 0.0
        self.model = None

    def set_model(self, model):
        self.model = model

    def record_prediction(self, predicted: np.ndarray, actual: np.ndarray, confidence: float = 0.0):
        error = actual - predicted
        mse = np.mean(error ** 2)
        mae = np.mean(np.abs(error))
        self._error_history.append({'mse': mse, 'mae': mae, 'error': error,
                                     'confidence': confidence, 'timestamp': time.time()})

    def record_inference_time(self, duration_ms: float):
        self._inference_times.append(duration_ms)

    def record_gradient(self, gradient_norm: float):
        self._gradient_norm = gradient_norm

    def collect(self) -> SelfMetrics:
        metrics = SelfMetrics()
        if len(self._error_history) > 0:
            recent_errors = list(self._error_history)[-100:]
            mse_values = [e['mse'] for e in recent_errors]
            mae_values = [e['mae'] for e in recent_errors]
            metrics.mse_recent = np.mean(mse_values)
            metrics.mae_recent = np.mean(mae_values)
            metrics.error_std = np.std(mse_values)
            if self._mse_ema is None:
                self._mse_ema = metrics.mse_recent
            else:
                self._mse_ema = 0.1 * metrics.mse_recent + 0.9 * self._mse_ema
            metrics.accuracy_ema = 1.0 / (1.0 + self._mse_ema)
            metrics.mse_trend = self._mse_ema - self._prev_mse_ema
            self._prev_mse_ema = self._mse_ema
            confidences = [e['confidence'] for e in recent_errors if e['confidence'] > 0]
            if confidences:
                metrics.prediction_confidence = np.mean(confidences)
            if metrics.prediction_confidence > 0:
                metrics.calibration_error = abs(metrics.mse_recent - (1.0 - metrics.prediction_confidence))
            if len(mse_values) >= 2:
                metrics.error_autocorr = np.corrcoef(mse_values[:-1], mse_values[1:])[0, 1]
            all_errors = np.concatenate([e['error'].flatten() for e in recent_errors])
            if len(all_errors) > 0:
                mean_e = np.mean(all_errors)
                std_e = np.std(all_errors)
                metrics.error_skewness = float(np.mean((all_errors - mean_e)**3) / (std_e**3 + 1e-8))
                metrics.error_kurtosis = float(np.mean((all_errors - mean_e)**4) / (std_e**4 + 1e-8))

        if len(self._inference_times) > 0:
            metrics.inference_time_ms = np.mean(list(self._inference_times))

        if self.model is not None:
            metrics.model_param_count = sum(p.numel() for p in self.model.parameters())
            try:
                param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
                metrics.model_ram_mb = (param_size + buffer_size) / (1024 * 1024)
            except Exception:
                pass

        if hasattr(self, '_gradient_norm'):
            metrics.gradient_norm = self._gradient_norm

        if len(self._error_history) > 0:
            mse_vals = np.array([e['mse'] for e in self._error_history])
            if len(mse_vals) > 1:
                hist, _ = np.histogram(mse_vals, bins=10, density=True)
                hist = hist[hist > 0]
                metrics.model_entropy = -np.sum(hist * np.log(hist + 1e-8))

        if len(self._error_history) >= 100:
            first_half = np.mean([e['mse'] for e in list(self._error_history)[:50]])
            second_half = np.mean([e['mse'] for e in list(self._error_history)[50:100]])
            if first_half > 0:
                metrics.overfitting_signal = max(0.0, (first_half - second_half) / first_half)

        return metrics

    def to_vector(self, metrics: SelfMetrics) -> list:
        return [
            metrics.mse_recent, metrics.mae_recent, metrics.mse_trend, metrics.mae_trend,
            metrics.accuracy_ema, metrics.error_std, metrics.prediction_confidence,
            metrics.calibration_error, metrics.model_ram_mb, metrics.model_cpu_percent,
            metrics.inference_time_ms, float(metrics.model_param_count), metrics.gradient_norm,
            metrics.last_update_time, metrics.update_frequency, metrics.model_entropy,
            metrics.knowledge_coverage, metrics.overfitting_signal, metrics.learning_rate_current,
            metrics.exploration_rate_current, metrics.error_autocorr, metrics.error_skewness,
            metrics.error_kurtosis, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
```

### 28.4 阶段小结

本阶段完成了自身层（模型内省）子模块的设计：监控预测精度（MSE、MAE、趋势、置信度、校准误差）、模型资源（推理延迟、参数规模、内存占用、梯度范数）、模型状态（模型熵、认知覆盖度、过拟合信号）和错误统计（自相关、偏度、峰度）四大维度；实现了完整的SelfCollector类，支持记录预测结果、推理时间和梯度信息；为环境感知提供了32维的自身层特征，使系统具备"自我意识"。

---

## 阶段 29：多模态数据融合、标准化与状态向量编码

### 29.1 引言

前八个阶段（阶段21-28）已经完成了四层感知的数据采集。现在，我们拥有来自四个不同来源、四个不同时间尺度、四种不同数据类型的异构数据流。本阶段将解决一个核心问题：如何将这些异构数据融合为一个统一的、标准化的、可供下游模块直接消费的状态向量x_t∈ℝ^128？本阶段将系统性地介绍数据融合、标准化和最终状态向量编码的完整流程。

### 29.2 多模态数据融合

四个数据源具有不同的特性：系统层是连续数值、1Hz更新、极少缺失、量纲0-100（百分比）；日志层是离散计数、0.2Hz更新、可能较多缺失、量纲0-10000+（计数）；文件层是离散计数、事件驱动、缺失不确定、量纲0-1000+（计数）；自身层是连续数值、1Hz更新、极少缺失、量纲0-1（比率）。

核心挑战包括时间不对齐（不同采集频率）、量纲不统一（CPU使用率0-100 vs日志计数0-10000+）、缺失值（日志层和文件层可能在某些时刻没有数据）和数据类型混合（连续值和离散值混合）。

时间对齐策略使用最近邻插值：每个primary时间点取最近的secondary数据：nearest_idx=argmin(|secondary_timestamps-primary_t|)。

缺失值处理策略因数据源而异：系统层和自身层使用前向填充（用上一个有效值填充缺失值，适用于连续变化的指标）；日志层和文件层使用零填充（用0填充缺失值，适用于计数类指标——"没有日志"意味着计数为0）。

### 29.3 数据融合引擎实现

```python
import numpy as np

class DataFusionEngine:
    def __init__(self):
        self.last_system_vector = None
        self.last_log_vector = None
        self.last_file_vector = None
        self.last_self_vector = None
        self.dims = {'system': 48, 'log': 32, 'file': 16, 'self': 32}
        self.total_dim = sum(self.dims.values())

    def fuse(self, system_data=None, log_data=None, file_data=None, self_data=None) -> np.ndarray:
        vectors = []
        if system_data is not None:
            vec = system_data.to_vector()
            self.last_system_vector = vec
        elif self.last_system_vector is not None:
            vec = self.last_system_vector
        else:
            vec = np.zeros(self.dims['system'])
        vectors.append(vec)

        if log_data is not None:
            vec = log_data.to_vector()
            self.last_log_vector = vec
        elif self.last_log_vector is not None:
            vec = self.last_log_vector
        else:
            vec = np.zeros(self.dims['log'])
        vectors.append(vec)

        if file_data is not None:
            vec = file_data.to_vector()
        else:
            vec = np.zeros(self.dims['file'])
        vectors.append(vec)

        if self_data is not None:
            vec = self_data.to_vector()
            self.last_self_vector = vec
        elif self.last_self_vector is not None:
            vec = self.last_self_vector
        else:
            vec = np.zeros(self.dims['self'])
        vectors.append(vec)

        return np.concatenate(vectors).astype(np.float32)
```

### 29.4 数据标准化

在线Z-score标准化使用Welford算法，可以在线计算均值和方差而无需存储所有历史数据。x_norm=(x-μ)/σ，标准化后数据均值为0，标准差为1。

```python
class OnlineStandardizer:
    def __init__(self, dim: int):
        self.dim = dim
        self.mean = np.zeros(dim, dtype=np.float32)
        self.M2 = np.zeros(dim, dtype=np.float32)
        self.count = 0
        self.eps = 1e-8

    def update(self, x: np.ndarray):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_std(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.dim, dtype=np.float32)
        variance = self.M2 / (self.count - 1)
        return np.sqrt(variance + self.eps)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.count < 2:
            return x
        return (x - self.mean) / self.get_std()

    def inverse_normalize(self, x_norm: np.ndarray) -> np.ndarray:
        if self.count < 2:
            return x_norm
        return x_norm * self.get_std() + self.mean
```

对于已知范围的指标（如CPU使用率0-100%），使用Min-Max标准化：x_norm=(x-x_min)/(x_max-x_min)。对于存在异常值的数据，使用鲁棒标准化：x_norm=(x-median)/IQR，其中IQR=Q3-Q1。

### 29.5 状态向量编码器

完整的状态向量编码器StateVectorEncoder整合了融合和标准化两个步骤：

```python
class StateVectorEncoder:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.fusion_engine = DataFusionEngine()
        self.standardizer = OnlineStandardizer(dim=dim)
        self._current_vector = np.zeros(dim, dtype=np.float32)

    async def encode(self, system_data=None, log_data=None,
                     file_data=None, self_data=None) -> np.ndarray:
        fused = self.fusion_engine.fuse(system_data, log_data, file_data, self_data)
        self.standardizer.update(fused)
        normalized = self.standardizer.normalize(fused)
        normalized = np.clip(normalized, -5.0, 5.0)
        self._current_vector = normalized
        return normalized

    def get_current_state(self) -> np.ndarray:
        return self._current_vector.copy()
```

最终输出128维标准化环境状态向量：x_t=[x_t^sys(48维)|x_t^log(32维)|x_t^file(16维)|x_t^self(32维)]，每个维度经过Z-score标准化后被裁剪到[-5,5]范围以防止极端值。

### 29.6 阶段小结

本阶段完整实现了多模态数据融合、标准化和状态向量编码：使用最近邻插值实现时间对齐，使用前向填充和零填充处理不同来源的缺失值；使用Welford算法的在线Z-score标准化，配合Min-Max和鲁棒标准化应对不同数据特征；最终输出128维的标准化环境状态向量，可直接供下游驱力引擎和预测模型消费。

---

## 阶段 30：数据降噪与异常值处理

### 30.1 引言

采集到的原始数据不可避免地包含噪声和异常值。噪声来自传感器的精度限制、操作系统的调度抖动、网络传输的随机波动等。异常值可能来自系统瞬间的峰值、采集错误或真正需要关注的异常事件。本阶段将设计数据降噪和异常值处理子模块，使用中值滤波、卡尔曼滤波和IQR方法三种互补的技术。

### 30.2 中值滤波

中值滤波（Median Filter）是一种非线性滤波技术，将每个数据点替换为其邻域窗口内的中位数：y[t]=median(x[t-k],...,x[t],...,x[t+k])。它对脉冲噪声（椒盐噪声）特别有效——可以去除孤立的尖峰，同时保持边缘（突变）信息。中值滤波的特性：对孤立的异常值非常有效、不改变数据的阶跃变化（边缘保持）、计算复杂度O(k log k)、窗口大小k越大平滑效果越强但延迟也越大。

```python
import numpy as np
from collections import deque

class MedianFilter:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def filter(self, value: float) -> float:
        self.buffer.append(value)
        if len(self.buffer) < 3:
            return value
        sorted_values = sorted(self.buffer)
        mid = len(sorted_values) // 2
        return sorted_values[mid]
```

### 30.3 卡尔曼滤波

卡尔曼滤波（Kalman Filter）是一种递归的、最优的状态估计算法。它假设系统状态服从线性动态模型，观测带有高斯噪声，然后通过"预测-更新"循环来估计系统的最优状态。

预测步骤：x̂_k|k-1=F·x̂_k-1|k-1（状态预测），P_k|k-1=F·P_k-1|k-1·F^T+Q（协方差预测）。

更新步骤：y_k=z_k-H·x̂_k|k-1（测量残差），S_k=H·P_k|k-1·H^T+R（残差协方差），K_k=P_k|k-1·H^T·S_k^{-1}（卡尔曼增益），x̂_k|k=x̂_k|k-1+K_k·y_k（更新状态估计），P_k|k=(I-K_k·H)·P_k|k-1（更新协方差）。

卡尔曼增益K的直观理解：K决定了"更相信预测"还是"更相信观测"。如果观测噪声R很大（观测不可靠），K变小，更相信预测；如果过程噪声Q很大（系统变化剧烈），K变大，更相信观测。

```python
class KalmanFilter1D:
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1, initial_value: float = 0.0):
        self.x = initial_value
        self.P = 1.0
        self.F = 1.0
        self.H = 1.0
        self.Q = process_noise
        self.R = measurement_noise
        self._initialized = False

    def update(self, measurement: float) -> float:
        if not self._initialized:
            self.x = measurement
            self._initialized = True
            return self.x
        x_pred = self.F * self.x
        P_pred = self.F * self.P * self.F + self.Q
        y = measurement - self.H * x_pred
        S = self.H * P_pred * self.H + self.R
        K = P_pred * self.H / S
        self.x = x_pred + K * y
        self.P = (1 - K * self.H) * P_pred
        return self.x

class MultiKalmanFilter:
    def __init__(self, dim: int, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.dim = dim
        self.filters = [KalmanFilter1D(process_noise, measurement_noise) for _ in range(dim)]

    def filter(self, vector: np.ndarray) -> np.ndarray:
        filtered = np.zeros_like(vector)
        for i in range(self.dim):
            filtered[i] = self.filters[i].update(float(vector[i]))
        return filtered
```

### 30.4 IQR异常值检测

IQR（Interquartile Range，四分位距）方法是一种基于分布的异常值检测方法。它不假设数据服从正态分布，对偏态分布和有异常值的数据都具有鲁棒性。IQR=Q3-Q1，下界=Q1-1.5×IQR，上界=Q3+1.5×IQR。落在[下界,上界]之外的数据点被视为异常值。调整因子1.5是标准因子，更严格的检测使用2.0或3.0，更宽松的检测使用1.0。

```python
class IQRFilter:
    def __init__(self, factor: float = 1.5, window_size: int = 100):
        self.factor = factor
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def is_outlier(self, value: float) -> bool:
        if len(self.buffer) < 4:
            self.buffer.append(value)
            return False
        data = np.array(list(self.buffer))
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.factor * iqr
        upper_bound = q3 + self.factor * iqr
        self.buffer.append(value)
        return value < lower_bound or value > upper_bound

    def filter(self, value: float) -> tuple:
        is_outlier = self.is_outlier(value)
        if is_outlier and len(self.buffer) >= 2:
            replacement = np.median(list(self.buffer)[:-1])
            return replacement, True
        return value, False
```

### 30.5 降噪流水线的组合使用

三种方法各有优势，在实际系统中组合使用形成降噪流水线DenoisingPipeline。首先使用中值滤波去除尖峰脉冲，然后使用卡尔曼滤波平滑连续特征，最后使用IQR过滤异常值。对于不同类型的特征，使用不同的降噪策略：系统层和自身层的连续特征使用卡尔曼滤波，日志层和文件层的计数特征使用IQR过滤。

### 30.6 阶段小结

本阶段设计了数据降噪和异常值处理子模块：中值滤波用于去除孤立的脉冲噪声（尖峰），保持数据边缘信息；卡尔曼滤波通过"预测-更新"循环实现最优状态估计，平滑连续变化指标；IQR方法基于四分位距检测异常值，不依赖数据分布假设，对异常值本身具有鲁棒性；三种方法组合使用，形成完整的降噪流水线，确保下游模块接收到干净、稳定的数据。

'''

with open('/workspace/自驱永续进化预测修正Agent_技术白皮书.md', 'a', encoding='utf-8') as f:
    f.write(content)

print("Phases 26-30 written successfully.")