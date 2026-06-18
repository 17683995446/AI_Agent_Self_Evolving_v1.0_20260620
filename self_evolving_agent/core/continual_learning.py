"""
阶段44：持续学习与灾难性遗忘防护

基于第一性原理：
- 神经网络学习新任务时会覆盖旧知识（灾难性遗忘）
- 弹性权重巩固（EWC）：保护重要参数不被大幅修改
- 经验回放：混合旧任务样本与新任务样本训练

核心设计：
1. EWCRegularizer: 弹性权重巩固正则化器
2. ExperienceReplay: 经验回放缓冲区
3. ProgressiveNetwork: 渐进式神经网络
4. ForgettingMonitor: 遗忘监控器
5. ContinualLearningController: 高层封装

接口契约：
- 任务: Dict，包含task_id、data、labels
- 重要性: Dict[str, np.ndarray]，参数重要性矩阵
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque, OrderedDict
import copy

logger = logging.getLogger(__name__)


@dataclass
class TaskExperience:
    """任务经验"""
    task_id: str
    data: np.ndarray
    labels: np.ndarray
    performance: float = 0.0


class EWCRegularizer:
    """
    弹性权重巩固（Elastic Weight Consolidation）
    核心思想：对重要参数施加正则化，防止大幅修改
    """

    def __init__(self, lambda_ewc: float = 1.0):
        self.lambda_ewc = lambda_ewc
        self.fisher_info: Dict[str, np.ndarray] = {}
        self.optimal_params: Dict[str, np.ndarray] = {}
        self.task_count = 0
        logger.info("EWCRegularizer initialized: lambda=%.3f", lambda_ewc)

    def compute_fisher_information(self, model_forward: Callable,
                                   model_params: Dict[str, np.ndarray],
                                   data: np.ndarray,
                                   labels: np.ndarray,
                                   n_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        计算Fisher信息矩阵（近似）
        衡量每个参数对任务输出的敏感度
        """
        fisher = {}
        for key in model_params:
            fisher[key] = np.zeros_like(model_params[key])

        n = min(n_samples, len(data))
        indices = np.random.choice(len(data), n, replace=False)

        for idx in indices:
            x = data[idx:idx + 1]
            y = int(labels[idx])
            # 数值梯度近似Fisher信息
            eps = 1e-4
            for key in model_params:
                orig = model_params[key].copy()
                it = np.nditer(orig, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    i = it.multi_index
                    model_params[key][i] = orig[i] + eps
                    logits1 = model_forward(x)
                    model_params[key][i] = orig[i] - eps
                    logits2 = model_forward(x)
                    model_params[key][i] = orig[i]
                    # Fisher信息 ≈ (dL/dw)^2
                    grad = (logits1[0, y] - logits2[0, y]) / (2 * eps)
                    fisher[key][i] += grad ** 2
                    it.iternext()

        for key in fisher:
            fisher[key] /= n

        return fisher

    def register_task(self, model_forward: Callable,
                      model_params: Dict[str, np.ndarray],
                      data: np.ndarray,
                      labels: np.ndarray):
        """注册新任务，累积Fisher信息"""
        new_fisher = self.compute_fisher_information(model_forward, model_params, data, labels)

        if self.task_count == 0:
            self.fisher_info = new_fisher
        else:
            # 在线累积：Fisher信息取加权平均
            for key in self.fisher_info:
                self.fisher_info[key] = (self.fisher_info[key] * self.task_count + new_fisher[key]) / (self.task_count + 1)

        # 保存当前最优参数
        self.optimal_params = {k: v.copy() for k, v in model_params.items()}
        self.task_count += 1
        logger.info("EWC task %d registered, fisher keys: %s", self.task_count, list(self.fisher_info.keys()))

    def compute_penalty(self, current_params: Dict[str, np.ndarray]) -> float:
        """计算EWC正则化惩罚项"""
        if self.task_count == 0:
            return 0.0
        penalty = 0.0
        for key in current_params:
            if key in self.fisher_info:
                diff = current_params[key] - self.optimal_params[key]
                penalty += np.sum(self.fisher_info[key] * diff ** 2)
        return float(self.lambda_ewc * penalty)

    def get_param_importance(self) -> Dict[str, float]:
        """获取参数重要性摘要"""
        return {k: float(np.mean(v)) for k, v in self.fisher_info.items()}


class ExperienceReplay:
    """
    经验回放缓冲区
    存储旧任务样本，与新任务混合训练
    """

    def __init__(self, capacity_per_task: int = 100):
        self.capacity_per_task = capacity_per_task
        self.experiences: OrderedDict[str, TaskExperience] = OrderedDict()
        logger.info("ExperienceReplay initialized: capacity=%d/task", capacity_per_task)

    def add_task(self, task_id: str, data: np.ndarray, labels: np.ndarray,
                 performance: float = 0.0):
        """添加任务经验"""
        # 采样存储
        n = min(len(data), self.capacity_per_task)
        indices = np.random.choice(len(data), n, replace=False)
        self.experiences[task_id] = TaskExperience(
            task_id=task_id,
            data=data[indices],
            labels=labels[indices],
            performance=performance,
        )
        logger.info("Experience added: task=%s, samples=%d", task_id, n)

    def sample_replay_batch(self, batch_size: int = 32,
                            exclude_task: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """采样回放批次"""
        available_tasks = [t for t in self.experiences if t != exclude_task]
        if not available_tasks:
            return np.array([]), np.array([])

        all_data = []
        all_labels = []
        for task_id in available_tasks:
            exp = self.experiences[task_id]
            all_data.append(exp.data)
            all_labels.append(exp.labels)

        combined_data = np.vstack(all_data)
        combined_labels = np.hstack(all_labels)

        n = min(batch_size, len(combined_data))
        indices = np.random.choice(len(combined_data), n, replace=False)
        return combined_data[indices], combined_labels[indices]

    def get_all_experiences(self) -> List[TaskExperience]:
        return list(self.experiences.values())

    def __len__(self):
        return sum(len(exp.data) for exp in self.experiences.values())


class ProgressiveNetwork:
    """
    渐进式神经网络
    每个新任务添加新列（新网络），旧列冻结
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.columns: List[Dict[str, np.ndarray]] = []
        self.lateral_weights: List[List[np.ndarray]] = []  # 列间横向连接
        self._add_column()
        logger.info("ProgressiveNetwork initialized: dim=%d->%d->%d", input_dim, hidden_dim, output_dim)

    def _add_column(self):
        """添加新列"""
        col = {
            'W1': np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim),
            'b1': np.zeros(self.hidden_dim),
            'W2': np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(2.0 / self.hidden_dim),
            'b2': np.zeros(self.output_dim),
        }
        self.columns.append(col)
        # 添加横向连接
        if len(self.columns) > 1:
            lateral = []
            for _ in range(len(self.columns) - 1):
                lateral.append(np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1)
            self.lateral_weights.append(lateral)
        logger.info("Column %d added", len(self.columns))

    def forward(self, x: np.ndarray, use_column: int = None) -> np.ndarray:
        """前向传播"""
        if use_column is None:
            use_column = len(self.columns) - 1  # 默认使用最新列

        # 计算所有列的隐藏层输出
        hidden_outputs = []
        for i, col in enumerate(self.columns):
            h = np.maximum(0, np.dot(x, col['W1']) + col['b1'])
            hidden_outputs.append(h)

        # 当前列的隐藏层 + 横向连接
        current_hidden = hidden_outputs[use_column]
        if use_column > 0 and use_column - 1 < len(self.lateral_weights):
            for prev_col_idx, lateral_w in enumerate(self.lateral_weights[use_column - 1]):
                if prev_col_idx < len(hidden_outputs):
                    current_hidden += np.dot(hidden_outputs[prev_col_idx], lateral_w)

        # 输出层
        output = np.dot(current_hidden, self.columns[use_column]['W2']) + self.columns[use_column]['b2']
        return output

    def add_task(self):
        """为新任务添加列"""
        self._add_column()

    def freeze_column(self, col_idx: int):
        """冻结指定列（标记为不可训练）"""
        # 在实际实现中通过mask实现，此处简化
        logger.info("Column %d frozen", col_idx)

    def get_active_column_count(self) -> int:
        return len(self.columns)


class ForgettingMonitor:
    """
    遗忘监控器
    跟踪模型在旧任务上的性能变化
    """

    def __init__(self):
        self.baselines: Dict[str, float] = {}  # task_id -> baseline_performance
        self.history: Dict[str, deque] = {}  # task_id -> performance history
        logger.info("ForgettingMonitor initialized")

    def register_baseline(self, task_id: str, performance: float):
        """注册任务基线性能"""
        self.baselines[task_id] = performance
        self.history[task_id] = deque(maxlen=100)
        self.history[task_id].append(performance)
        logger.info("Baseline registered: task=%s, perf=%.4f", task_id, performance)

    def record_performance(self, task_id: str, performance: float):
        """记录任务当前性能"""
        if task_id not in self.history:
            self.history[task_id] = deque(maxlen=100)
            self.baselines[task_id] = performance
        self.history[task_id].append(performance)

    def compute_forgetting(self, task_id: str) -> Dict[str, float]:
        """计算遗忘程度"""
        if task_id not in self.baselines:
            return {"forgetting": 0.0, "baseline": 0.0, "current": 0.0}

        baseline = self.baselines[task_id]
        current = self.history[task_id][-1] if self.history[task_id] else baseline
        forgetting = max(0.0, baseline - current)
        return {
            "forgetting": float(forgetting),
            "baseline": float(baseline),
            "current": float(current),
            "forgetting_ratio": float(forgetting / (baseline + 1e-10)),
        }

    def get_all_forgetting(self) -> Dict[str, Dict[str, float]]:
        return {tid: self.compute_forgetting(tid) for tid in self.baselines}

    def get_summary(self) -> Dict[str, float]:
        all_forgetting = self.get_all_forgetting()
        if not all_forgetting:
            return {"mean_forgetting": 0.0, "max_forgetting": 0.0, "n_tasks": 0}
        forgettings = [f["forgetting"] for f in all_forgetting.values()]
        return {
            "mean_forgetting": float(np.mean(forgettings)),
            "max_forgetting": float(np.max(forgettings)),
            "n_tasks": len(all_forgetting),
        }


class ContinualLearningController:
    """
    持续学习控制器高层封装
    整合EWC、经验回放、渐进网络、遗忘监控
    """

    def __init__(self, input_dim: int, output_dim: int = 2,
                 hidden_dim: int = 32,
                 lambda_ewc: float = 1.0,
                 replay_capacity: int = 100,
                 strategy: str = "ewc_replay"):
        """
        strategy: "ewc", "replay", "ewc_replay", "progressive"
        """
        self.input_dim = max(input_dim, 1)  # 防止input_dim=0导致除零
        self.output_dim = output_dim
        self.strategy = strategy

        self.ewc = EWCRegularizer(lambda_ewc=lambda_ewc) if "ewc" in strategy else None
        self.replay = ExperienceReplay(capacity_per_task=replay_capacity) if "replay" in strategy else None
        self.progressive = ProgressiveNetwork(input_dim, hidden_dim, output_dim) if strategy == "progressive" else None
        self.monitor = ForgettingMonitor()

        # 简单模型参数（用于非progressive策略）
        self.model_params = {
            'W1': np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim),
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim),
            'b2': np.zeros(output_dim),
        }

        self.task_count = 0
        self.current_task_id: Optional[str] = None
        logger.info("ContinualLearningController initialized: strategy=%s", strategy)

    def _model_forward(self, x: np.ndarray) -> np.ndarray:
        """模型前向传播"""
        h = np.maximum(0, np.dot(x, self.model_params['W1']) + self.model_params['b1'])
        return np.dot(h, self.model_params['W2']) + self.model_params['b2']

    def _compute_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        logits = self._model_forward(x)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        n = len(y)
        correct_probs = probs[np.arange(n), y.astype(int)]
        return float(-np.mean(np.log(correct_probs + 1e-10)))

    def learn_task(self, task_id: str, data: np.ndarray, labels: np.ndarray,
                   n_steps: int = 50, lr: float = 0.01) -> Dict[str, Any]:
        """学习新任务"""
        self.current_task_id = task_id
        self.task_count += 1

        if self.strategy == "progressive":
            if self.task_count > 1:
                self.progressive.add_task()

        losses = []
        for step in range(n_steps):
            # 混合回放数据
            if self.replay and self.task_count > 1:
                replay_data, replay_labels = self.replay.sample_replay_batch(
                    batch_size=min(16, len(data) // 2))
                if len(replay_data) > 0:
                    combined_data = np.vstack([data, replay_data])
                    combined_labels = np.hstack([labels, replay_labels])
                else:
                    combined_data, combined_labels = data, labels
            else:
                combined_data, combined_labels = data, labels

            # 计算损失
            loss = self._compute_loss(combined_data, combined_labels)

            # EWC惩罚
            ewc_penalty = 0.0
            if self.ewc:
                ewc_penalty = self.ewc.compute_penalty(self.model_params)
            total_loss = loss + ewc_penalty

            # 数值梯度更新
            eps = 1e-4
            for key in self.model_params:
                orig = self.model_params[key].copy()
                grad = np.zeros_like(orig)
                it = np.nditer(orig, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    idx = it.multi_index
                    self.model_params[key][idx] = orig[idx] + eps
                    loss1 = self._compute_loss(combined_data, combined_labels)
                    if self.ewc:
                        loss1 += self.ewc.compute_penalty(self.model_params)
                    self.model_params[key][idx] = orig[idx] - eps
                    loss2 = self._compute_loss(combined_data, combined_labels)
                    if self.ewc:
                        loss2 += self.ewc.compute_penalty(self.model_params)
                    grad[idx] = (loss1 - loss2) / (2 * eps)
                    self.model_params[key][idx] = orig[idx]
                    it.iternext()
                self.model_params[key] -= lr * grad

            losses.append(total_loss)

        # 注册EWC
        if self.ewc:
            self.ewc.register_task(self._model_forward, self.model_params, data, labels)

        # 存储经验
        if self.replay:
            final_perf = 1.0 - self._compute_loss(data, labels)
            self.replay.add_task(task_id, data, labels, performance=max(0.0, final_perf))

        # 记录基线性能
        final_acc = self._evaluate(data, labels)
        self.monitor.register_baseline(task_id, final_acc)

        logger.info("Task %s learned: final_loss=%.4f, acc=%.4f", task_id, losses[-1], final_acc)
        return {
            "task_id": task_id,
            "final_loss": losses[-1],
            "accuracy": final_acc,
            "ewc_penalty": ewc_penalty,
        }

    def _evaluate(self, data: np.ndarray, labels: np.ndarray) -> float:
        """评估准确率"""
        if self.strategy == "progressive":
            logits = self.progressive.forward(data)
        else:
            logits = self._model_forward(data)
        preds = np.argmax(logits, axis=1)
        return float(np.mean(preds == labels))

    def evaluate_task(self, task_id: str, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """评估指定任务性能"""
        acc = self._evaluate(data, labels)
        self.monitor.record_performance(task_id, acc)
        forgetting = self.monitor.compute_forgetting(task_id)
        return {
            "accuracy": acc,
            "forgetting": forgetting["forgetting"],
            "forgetting_ratio": forgetting["forgetting_ratio"],
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "task_count": self.task_count,
            "current_task": self.current_task_id,
            "forgetting_summary": self.monitor.get_summary(),
            "replay_size": len(self.replay) if self.replay else 0,
            "ewc_tasks": self.ewc.task_count if self.ewc else 0,
        }

    def save(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "task_count": self.task_count,
            "model_params": {k: v.tolist() for k, v in self.model_params.items()},
            "forgetting_summary": self.monitor.get_summary(),
        }

    def load(self, data: Dict[str, Any]):
        self.strategy = data.get("strategy", self.strategy)
        self.task_count = data.get("task_count", 0)
        params = data.get("model_params", {})
        for k, v in params.items():
            self.model_params[k] = np.array(v)
