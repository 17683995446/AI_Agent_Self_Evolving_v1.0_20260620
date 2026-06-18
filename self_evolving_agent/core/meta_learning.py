"""
阶段39：元学习与快速适应

基于第一性原理：
- 学习如何学习：元知识是跨任务的通用规律
- 快速适应：基于少量样本快速调整策略
- 元梯度：优化初始化参数使得少量梯度步即可适应新任务

核心设计：
1. TaskDistribution: 任务分布采样器
2. MAMLModel: 模型无关元学习(MAML)模型
3. MetaLearner: 元学习训练器（外循环优化）
4. FastAdapter: 快速适应器（内循环适应）
5. MetaReplayBuffer: 元经验回放缓冲区
6. MetaLearningController: 高层封装

接口契约：
- 任务: Dict，包含support_set和query_set
- 适应步数: int，内循环梯度步数
- 元学习率: float，外循环学习率
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import copy

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """元学习任务"""
    task_id: str
    support_x: np.ndarray  # 支持集输入 (K, input_dim)
    support_y: np.ndarray  # 支持集标签 (K,)
    query_x: np.ndarray    # 查询集输入 (Q, input_dim)
    query_y: np.ndarray    # 查询集标签 (Q,)
    task_type: str = "classification"


class TaskDistribution:
    """
    任务分布采样器
    生成多样化的元学习任务
    """

    def __init__(self, input_dim: int, n_classes: int = 5,
                 k_shot: int = 5, q_query: int = 15):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.k_shot = k_shot
        self.q_query = q_query
        self.task_counter = 0
        logger.info("TaskDistribution initialized: input_dim=%d, k_shot=%d", input_dim, k_shot)

    def sample_task(self) -> Task:
        """采样一个随机分类任务"""
        self.task_counter += 1
        # 随机选择n_classes个类别（池大小为n_classes*2，至少n_classes）
        pool_size = max(self.n_classes * 2, self.n_classes)
        selected_classes = np.random.choice(pool_size, self.n_classes, replace=False)
        # 为每个类别生成原型权重
        prototypes = np.random.randn(self.n_classes, self.input_dim)

        support_x = []
        support_y = []
        query_x = []
        query_y = []

        for class_idx, cls in enumerate(selected_classes):
            # 生成该类别的数据（围绕原型的高斯分布）
            mean = prototypes[class_idx]
            cov = np.eye(self.input_dim) * 0.5
            samples = np.random.multivariate_normal(mean, cov, self.k_shot + self.q_query)
            support_x.append(samples[:self.k_shot])
            support_y.append(np.full(self.k_shot, class_idx))
            query_x.append(samples[self.k_shot:])
            query_y.append(np.full(self.q_query, class_idx))

        return Task(
            task_id=f"task_{self.task_counter}",
            support_x=np.vstack(support_x),
            support_y=np.hstack(support_y),
            query_x=np.vstack(query_x),
            query_y=np.hstack(query_y),
        )

    def sample_batch(self, batch_size: int) -> List[Task]:
        """采样一批任务"""
        return [self.sample_task() for _ in range(batch_size)]


class SimpleModel:
    """
    简单神经网络，用于MAML
    不依赖外部深度学习框架
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 40):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        # Xavier初始化
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
        logger.info("SimpleModel initialized: %d -> %d -> %d", input_dim, hidden_dim, output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播，返回logits"""
        h = np.maximum(0, np.dot(x, self.W1) + self.b1)  # ReLU
        logits = np.dot(h, self.W2) + self.b2
        return logits

    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测类别"""
        logits = self.forward(x)
        return np.argmax(logits, axis=-1)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """交叉熵损失"""
        logits = self.forward(x)
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        n = len(y)
        correct_probs = probs[np.arange(n), y.astype(int)]
        return float(-np.mean(np.log(correct_probs + 1e-10)))

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        preds = self.predict(x)
        return float(np.mean(preds == y))

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_params(self, params: Dict[str, np.ndarray]):
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()
        self.W2 = params["W2"].copy()
        self.b2 = params["b2"].copy()

    def copy(self) -> 'SimpleModel':
        m = SimpleModel(self.input_dim, self.output_dim, self.hidden_dim)
        m.set_params(self.get_params())
        return m


class MAMLModel:
    """
    MAML (Model-Agnostic Meta-Learning)
    学习一个好的初始化，使得少量梯度步即可适应新任务
    """

    def __init__(self, model: SimpleModel, inner_lr: float = 0.01,
                 inner_steps: int = 5, first_order: bool = False):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order  # 一阶近似（FOMAML）
        logger.info("MAMLModel initialized: inner_lr=%.4f, steps=%d", inner_lr, inner_steps)

    def _compute_gradients(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """数值梯度计算"""
        eps = 1e-4
        grads = {}
        params = self.model.get_params()
        for key in params:
            orig = params[key].copy()
            grad = np.zeros_like(orig)
            it = np.nditer(orig, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                params[key][idx] = orig[idx] + eps
                self.model.set_params(params)
                loss1 = self.model.loss(x, y)

                params[key][idx] = orig[idx] - eps
                self.model.set_params(params)
                loss2 = self.model.loss(x, y)

                grad[idx] = (loss1 - loss2) / (2 * eps)
                params[key][idx] = orig[idx]
                it.iternext()
            self.model.set_params(params)
            grads[key] = grad
        return grads

    def adapt(self, task: Task) -> SimpleModel:
        """
        内循环：在支持集上适应任务
        Returns: 适应后的模型
        """
        # 保存元参数，adapt结束后恢复
        meta_params = self.model.get_params()
        adapted = self.model.copy()
        for step in range(self.inner_steps):
            # 在支持集上计算梯度
            self.model.set_params(adapted.get_params())
            grads = self._compute_gradients(task.support_x, task.support_y)
            # 更新参数
            params = adapted.get_params()
            for key in params:
                params[key] -= self.inner_lr * grads[key]
            adapted.set_params(params)
        # 恢复元参数，确保后续任务的adapt基于正确的初始参数
        self.model.set_params(meta_params)
        return adapted

    def meta_loss(self, task: Task) -> Tuple[float, float]:
        """
        计算元损失：适应后在查询集上的损失
        Returns: (loss, accuracy)
        """
        adapted = self.adapt(task)
        loss = adapted.loss(task.query_x, task.query_y)
        acc = adapted.accuracy(task.query_x, task.query_y)
        return loss, acc


class MetaLearner:
    """
    元学习训练器
    外循环：优化元参数使得所有任务的查询损失最小
    """

    def __init__(self, maml: MAMLModel, meta_lr: float = 0.001,
                 task_distribution: TaskDistribution = None):
        self.maml = maml
        self.meta_lr = meta_lr
        self.task_distribution = task_distribution or TaskDistribution(
            maml.model.input_dim, n_classes=5)
        self.meta_loss_history: deque = deque(maxlen=1000)
        self.meta_acc_history: deque = deque(maxlen=1000)
        self.step_count = 0
        logger.info("MetaLearner initialized: meta_lr=%.4f", meta_lr)

    def meta_train_step(self, batch_size: int = 4) -> Dict[str, float]:
        """执行一步元训练"""
        tasks = self.task_distribution.sample_batch(batch_size)
        total_loss = 0.0
        total_acc = 0.0

        # 保存当前元参数
        meta_params = {k: v.copy() for k, v in self.maml.model.get_params().items()}

        for task in tasks:
            loss, acc = self.maml.meta_loss(task)
            total_loss += loss
            total_acc += acc

        avg_loss = total_loss / batch_size
        avg_acc = total_acc / batch_size

        # 元梯度：使用一阶近似（FOMAML）
        # 对每个参数计算 meta_loss 对元参数的数值梯度
        eps = 1e-3
        for key in meta_params:
            orig = meta_params[key].copy()
            grad = np.zeros_like(orig)
            # 采样少量维度计算梯度（降低计算量）
            n_dims = min(orig.size, 5)  # 最多采样5个维度
            flat_indices = np.random.choice(orig.size, n_dims, replace=False)
            for flat_idx in flat_indices:
                idx = np.unravel_index(flat_idx, orig.shape)
                meta_params[key][idx] = orig[idx] + eps
                self.maml.model.set_params(meta_params)
                # 仅用第1个任务评估
                loss_plus, _ = self.maml.meta_loss(tasks[0])

                meta_params[key][idx] = orig[idx] - eps
                self.maml.model.set_params(meta_params)
                loss_minus, _ = self.maml.meta_loss(tasks[0])

                grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                meta_params[key][idx] = orig[idx]

            # 梯度裁剪
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1.0:
                grad = grad / grad_norm
            meta_params[key] = orig - self.meta_lr * grad

        self.maml.model.set_params(meta_params)
        self.step_count += 1
        self.meta_loss_history.append(avg_loss)
        self.meta_acc_history.append(avg_acc)

        logger.info("Meta step %d: loss=%.4f, acc=%.4f", self.step_count, avg_loss, avg_acc)
        return {"meta_loss": avg_loss, "meta_accuracy": avg_acc}

    def train(self, n_iterations: int = 100, batch_size: int = 4) -> Dict[str, Any]:
        """执行完整元训练"""
        logger.info("Starting meta-training for %d iterations", n_iterations)
        for i in range(n_iterations):
            metrics = self.meta_train_step(batch_size)
            if (i + 1) % 20 == 0:
                logger.info("Meta-training progress: %d/%d, loss=%.4f, acc=%.4f",
                            i + 1, n_iterations, metrics["meta_loss"], metrics["meta_accuracy"])
        return {
            "final_loss": np.mean(list(self.meta_loss_history)[-10:]),
            "final_accuracy": np.mean(list(self.meta_acc_history)[-10:]),
            "steps": self.step_count,
        }

    def evaluate_on_new_task(self, n_eval_tasks: int = 10) -> Dict[str, float]:
        """在全新任务上评估快速适应能力"""
        losses = []
        accs_before = []
        accs_after = []

        for _ in range(n_eval_tasks):
            task = self.task_distribution.sample_task()
            # 适应前
            acc_before = self.maml.model.accuracy(task.query_x, task.query_y)
            accs_before.append(acc_before)

            # 适应后
            adapted = self.maml.adapt(task)
            loss_after = adapted.loss(task.query_x, task.query_y)
            acc_after = adapted.accuracy(task.query_x, task.query_y)
            losses.append(loss_after)
            accs_after.append(acc_after)

        return {
            "acc_before_adapt": float(np.mean(accs_before)),
            "acc_after_adapt": float(np.mean(accs_after)),
            "loss_after_adapt": float(np.mean(losses)),
            "improvement": float(np.mean(accs_after) - np.mean(accs_before)),
        }


class FastAdapter:
    """
    快速适应器
    将元学习模型部署到具体任务时的快速微调
    """

    def __init__(self, base_model: SimpleModel, lr: float = 0.01, max_steps: int = 10):
        self.base_model = base_model
        self.lr = lr
        self.max_steps = max_steps
        self.adapted_model: Optional[SimpleModel] = None
        logger.info("FastAdapter initialized: lr=%.4f, max_steps=%d", lr, max_steps)

    def adapt(self, support_x: np.ndarray, support_y: np.ndarray,
              progress_fn: Optional[Callable] = None) -> SimpleModel:
        """在支持集上快速微调"""
        self.adapted_model = self.base_model.copy()
        maml = MAMLModel(self.adapted_model, inner_lr=self.lr, inner_steps=self.max_steps)
        task = Task("adapt_task", support_x, support_y, support_x, support_y)
        self.adapted_model = maml.adapt(task)
        if progress_fn:
            progress_fn(1.0)
        return self.adapted_model

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.adapted_model is None:
            return self.base_model.predict(x)
        return self.adapted_model.predict(x)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "max_steps": self.max_steps,
            "adapted": self.adapted_model is not None,
        }


class MetaReplayBuffer:
    """
    元经验回放缓冲区
    存储历史任务用于元训练复用
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        logger.info("MetaReplayBuffer initialized: capacity=%d", capacity)

    def add(self, task: Task):
        self.buffer.append(task)

    def sample(self, batch_size: int) -> List[Task]:
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

    def get_stats(self) -> Dict[str, Any]:
        return {"size": len(self.buffer), "capacity": self.capacity}


class MetaLearningController:
    """
    元学习控制器高层封装
    整合任务分布、MAML训练、快速适应
    """

    def __init__(self, input_dim: int, output_dim: int,
                 n_classes: int = 5, k_shot: int = 5,
                 inner_lr: float = 0.01, meta_lr: float = 0.001,
                 inner_steps: int = 5):
        self.input_dim = input_dim
        self.output_dim = output_dim

        model = SimpleModel(input_dim, output_dim)
        self.maml = MAMLModel(model, inner_lr=inner_lr, inner_steps=inner_steps)
        self.task_dist = TaskDistribution(input_dim, n_classes=n_classes, k_shot=k_shot)
        self.meta_learner = MetaLearner(self.maml, meta_lr=meta_lr,
                                        task_distribution=self.task_dist)
        self.replay_buffer = MetaReplayBuffer()
        self.fast_adapter = FastAdapter(model)

        logger.info("MetaLearningController initialized")

    def meta_train(self, n_iterations: int = 100, batch_size: int = 4) -> Dict[str, Any]:
        """执行元训练"""
        return self.meta_learner.train(n_iterations, batch_size)

    def adapt_to_task(self, support_x: np.ndarray, support_y: np.ndarray) -> SimpleModel:
        """快速适应到新任务"""
        return self.fast_adapter.adapt(support_x, support_y)

    def evaluate_adaptation(self, n_tasks: int = 10) -> Dict[str, float]:
        """评估适应能力"""
        return self.meta_learner.evaluate_on_new_task(n_tasks)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "meta_steps": self.meta_learner.step_count,
            "replay_size": len(self.replay_buffer),
            "model_params": sum(p.size for p in self.maml.model.get_params().values()),
        }

    def save(self) -> Dict[str, Any]:
        return {
            "model_params": {k: v.tolist() for k, v in self.maml.model.get_params().items()},
            "meta_steps": self.meta_learner.step_count,
            "inner_lr": self.maml.inner_lr,
            "meta_lr": self.meta_learner.meta_lr,
        }

    def load(self, data: Dict[str, Any]):
        params = {k: np.array(v) for k, v in data.get("model_params", {}).items()}
        self.maml.model.set_params(params)
        self.fast_adapter.base_model.set_params(params)
        self.meta_learner.step_count = data.get("meta_steps", 0)
