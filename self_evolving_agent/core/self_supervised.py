"""
阶段42：自监督学习与表征学习

基于第一性原理：
- 数据本身包含结构信息，无需外部标签即可学习有用表征
- 对比学习：拉近正样本对、推远负样本对
- 掩码预测：遮蔽部分输入，预测被遮蔽内容

核心设计：
1. ContrastiveLearner: 对比学习引擎（SimCLR风格）
2. MaskedPredictor: 掩码预测引擎
3. RepresentationEncoder: 表征编码器
4. SelfSupervisedPretrainer: 自监督预训练器
5. RepresentationEvaluator: 表征质量评估器
6. SelfSupervisedController: 高层封装

接口契约：
- 输入: np.ndarray, 原始未标注数据
- 表征: np.ndarray, 编码后的特征向量
- 损失: float, 对比/掩码损失
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import copy

logger = logging.getLogger(__name__)


class RepresentationEncoder:
    """
    表征编码器
    将原始输入映射到低维特征空间
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = None,
                 latent_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.latent_dim = latent_dim
        self.params = self._init_params()
        logger.info("RepresentationEncoder initialized: %d -> %d", input_dim, latent_dim)

    def _init_params(self) -> Dict[str, np.ndarray]:
        dims = [self.input_dim] + self.hidden_dims + [self.latent_dim]
        params = {}
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i + 1])
            params[f'W{i}'] = w
            params[f'b{i}'] = b
        return params

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """前向编码"""
        h = np.clip(x, -1e6, 1e6)
        for i in range(len(self.hidden_dims)):
            h = self._relu(np.dot(h, self.params[f'W{i}']) + self.params[f'b{i}'])
            h = np.clip(h, -1e6, 1e6)
        z = np.dot(h, self.params[f'W{len(self.hidden_dims)}']) + self.params[f'b{len(self.hidden_dims)}']
        return z

    def encode_batch(self, x: np.ndarray) -> np.ndarray:
        """批量编码"""
        return self.encode(x)

    def get_params(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params: Dict[str, np.ndarray]):
        self.params = {k: v.copy() for k, v in params.items()}

    def copy(self) -> 'RepresentationEncoder':
        enc = RepresentationEncoder(self.input_dim, self.hidden_dims.copy(), self.latent_dim)
        enc.set_params(self.params)
        return enc


class DataAugmentor:
    """
    数据增强器
    生成同一输入的不同视图（正样本对）
    """

    def __init__(self, noise_std: float = 0.1, mask_ratio: float = 0.15,
                 jitter_scale: float = 0.1):
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio
        self.jitter_scale = jitter_scale
        logger.info("DataAugmentor initialized: noise=%.3f, mask=%.3f", noise_std, mask_ratio)

    def augment(self, x: np.ndarray) -> np.ndarray:
        """对输入进行随机增强"""
        augmented = x.copy()
        # 高斯噪声
        augmented += np.random.randn(*augmented.shape) * self.noise_std
        # 随机遮蔽
        mask = np.random.random(augmented.shape) < self.mask_ratio
        augmented[mask] = 0.0
        # 尺度抖动
        scale = 1.0 + np.random.uniform(-self.jitter_scale, self.jitter_scale)
        augmented *= scale
        return augmented

    def augment_batch(self, x: np.ndarray) -> np.ndarray:
        """批量增强"""
        return np.array([self.augment(sample) for sample in x])


class ContrastiveLearner:
    """
    对比学习引擎（SimCLR风格）
    核心思想：同一输入的两个增强视图应在表征空间中接近
    """

    def __init__(self, encoder: RepresentationEncoder,
                 temperature: float = 0.5,
                 augmentor: DataAugmentor = None):
        self.encoder = encoder
        self.temperature = temperature
        self.augmentor = augmentor or DataAugmentor()
        self.loss_history: deque = deque(maxlen=1000)
        logger.info("ContrastiveLearner initialized: temperature=%.3f", temperature)

    def compute_loss(self, x: np.ndarray) -> float:
        """
        计算NT-Xent损失（Normalized Temperature-scaled Cross Entropy）
        """
        batch_size = len(x)
        if batch_size < 2:
            return 0.0

        # 生成两个增强视图
        x1 = self.augmentor.augment_batch(x)
        x2 = self.augmentor.augment_batch(x)

        # 编码
        z1 = self.encoder.encode_batch(x1)
        z2 = self.encoder.encode_batch(x2)

        # L2归一化
        z1 = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-10)
        z2 = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-10)

        # 计算相似度矩阵
        z = np.vstack([z1, z2])  # (2N, dim)
        sim = np.dot(z, z.T) / self.temperature  # (2N, 2N)

        # 构造标签：第i个样本的正样本是i+N（或i-N）
        labels = np.arange(2 * batch_size)
        labels = (labels + batch_size) % (2 * batch_size)

        # 计算损失
        sim = sim - np.max(sim, axis=1, keepdims=True)  # 数值稳定
        exp_sim = np.exp(sim)
        # 排除自身
        mask = np.eye(2 * batch_size, dtype=bool)
        exp_sim[mask] = 0
        log_prob = sim - np.log(np.sum(exp_sim, axis=1, keepdims=True) + 1e-10)

        # 正样本对的损失
        pos_loss = -log_prob[np.arange(2 * batch_size), labels]
        loss = float(np.mean(pos_loss))

        self.loss_history.append(loss)
        return loss

    def train_step(self, x: np.ndarray, lr: float = 0.001) -> Dict[str, float]:
        """执行一步对比学习训练"""
        loss = self.compute_loss(x)

        # 数值梯度更新
        eps = 1e-4
        for key in self.encoder.params:
            orig = self.encoder.params[key].copy()
            grad = np.zeros_like(orig)
            it = np.nditer(orig, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                self.encoder.params[key][idx] = orig[idx] + eps
                loss1 = self.compute_loss(x)
                self.encoder.params[key][idx] = orig[idx] - eps
                loss2 = self.compute_loss(x)
                grad[idx] = (loss1 - loss2) / (2 * eps)
                self.encoder.params[key][idx] = orig[idx]
                it.iternext()
            # 梯度裁剪防止参数爆炸
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1.0:
                grad = grad / grad_norm
            self.encoder.params[key] -= lr * grad

        return {"contrastive_loss": loss}

    def get_embeddings(self, x: np.ndarray) -> np.ndarray:
        """获取输入的表征向量"""
        return self.encoder.encode_batch(x)


class MaskedPredictor:
    """
    掩码预测引擎
    遮蔽部分输入维度，预测被遮蔽内容
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = None,
                 mask_ratio: float = 0.15):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.mask_ratio = mask_ratio

        # 编码器：输入（含mask标记）-> 表征
        self.encoder = RepresentationEncoder(input_dim, hidden_dims, latent_dim=input_dim)
        # 解码器：表征 -> 重建
        self.decoder = RepresentationEncoder(input_dim, [input_dim], input_dim)
        self.loss_history: deque = deque(maxlen=1000)
        logger.info("MaskedPredictor initialized: mask_ratio=%.3f", mask_ratio)

    def _apply_mask(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用随机掩码"""
        mask = np.random.random(x.shape) < self.mask_ratio
        masked_x = x.copy()
        masked_x[mask] = 0.0  # 遮蔽位置设为0
        return masked_x, mask

    def compute_loss(self, x: np.ndarray) -> float:
        """计算掩码重建损失"""
        masked_x, mask = self._apply_mask(x)
        # 编码 + 解码
        z = self.encoder.encode_batch(masked_x)
        reconstructed = self.decoder.encode_batch(z)

        # 仅计算被遮蔽位置的重建损失
        if mask.any():
            rec_clipped = np.clip(reconstructed, -1e6, 1e6)
            x_clipped = np.clip(x, -1e6, 1e6)
            diff = (rec_clipped - x_clipped) ** 2
            masked_loss = np.clip(diff[mask], 0, 1e12)
            loss = float(np.mean(masked_loss))
        else:
            loss = 0.0

        self.loss_history.append(loss)
        return loss

    def train_step(self, x: np.ndarray, lr: float = 0.001) -> Dict[str, float]:
        """执行一步掩码预测训练"""
        loss = self.compute_loss(x)

        # 数值梯度更新（编码器+解码器）
        eps = 1e-4
        for network in [self.encoder, self.decoder]:
            for key in network.params:
                orig = network.params[key].copy()
                grad = np.zeros_like(orig)
                it = np.nditer(orig, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    idx = it.multi_index
                    network.params[key][idx] = orig[idx] + eps
                    loss1 = self.compute_loss(x)
                    network.params[key][idx] = orig[idx] - eps
                    loss2 = self.compute_loss(x)
                    grad[idx] = (loss1 - loss2) / (2 * eps)
                    network.params[key][idx] = orig[idx]
                    it.iternext()
                network.params[key] -= lr * grad

        return {"masked_loss": loss}

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """重建输入"""
        masked_x, _ = self._apply_mask(x)
        z = self.encoder.encode_batch(masked_x)
        return self.decoder.encode_batch(z)


class RepresentationEvaluator:
    """
    表征质量评估器
    评估学到的表征的质量
    """

    def __init__(self):
        logger.info("RepresentationEvaluator initialized")

    def evaluate_linear_probe(self, embeddings: np.ndarray,
                              labels: np.ndarray,
                              n_folds: int = 5) -> Dict[str, float]:
        """线性探测评估：在表征上训练线性分类器"""
        from numpy.linalg import lstsq
        n_classes = len(np.unique(labels))
        # One-hot编码
        y_onehot = np.eye(n_classes)[labels.astype(int)]

        # 简化的线性回归分类
        W = lstsq(embeddings, y_onehot, rcond=None)[0]
        preds = np.argmax(embeddings @ W, axis=1)
        accuracy = float(np.mean(preds == labels))

        return {
            "linear_probe_accuracy": accuracy,
            "n_classes": n_classes,
            "embedding_dim": embeddings.shape[1],
        }

    def evaluate_clustering(self, embeddings: np.ndarray,
                            labels: np.ndarray) -> Dict[str, float]:
        """聚类评估：表征空间的聚类质量"""
        n_classes = len(np.unique(labels))
        # 简化K-means
        centroids = embeddings[np.random.choice(len(embeddings), n_classes, replace=False)]
        for _ in range(10):
            dists = np.linalg.norm(embeddings[:, None] - centroids[None, :], axis=2)
            assignments = np.argmin(dists, axis=1)
            for k in range(n_classes):
                mask = assignments == k
                if mask.any():
                    centroids[k] = embeddings[mask].mean(axis=0)

        # 计算聚类纯度
        cluster_purities = []
        for k in range(n_classes):
            mask = assignments == k
            if mask.any():
                cluster_labels = labels[mask]
                most_common = np.bincount(cluster_labels.astype(int)).max()
                purity = most_common / len(cluster_labels)
                cluster_purities.append(purity)

        purity = float(np.mean(cluster_purities)) if cluster_purities else 0.0
        return {
            "clustering_purity": purity,
            "n_clusters": n_classes,
        }

    def evaluate_uniformity(self, embeddings: np.ndarray) -> float:
        """评估表征空间的均匀性"""
        normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        # 计算所有点对距离的平方
        dists_sq = 2 - 2 * np.dot(normalized, normalized.T)
        # 修正浮点误差导致的负值
        dists_sq = np.maximum(dists_sq, 0)
        np.fill_diagonal(dists_sq, 0)
        # 均匀性损失：-log(E[dist^2])
        mask = dists_sq > 0
        if mask.any():
            uniformity = -float(np.log(np.mean(dists_sq[mask])))
        else:
            uniformity = 0.0
        return uniformity


class SelfSupervisedController:
    """
    自监督学习控制器高层封装
    整合对比学习、掩码预测、表征评估
    """

    def __init__(self, input_dim: int, latent_dim: int = 64,
                 method: str = "contrastive",
                 temperature: float = 0.5,
                 mask_ratio: float = 0.15):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.method = method

        self.encoder = RepresentationEncoder(input_dim, latent_dim=latent_dim)
        self.augmentor = DataAugmentor()
        self.evaluator = RepresentationEvaluator()

        if method == "contrastive":
            self.learner = ContrastiveLearner(self.encoder, temperature, self.augmentor)
        elif method == "masked":
            self.learner = MaskedPredictor(input_dim, mask_ratio=mask_ratio)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.step_count = 0
        logger.info("SelfSupervisedController initialized: method=%s", method)

    def train(self, data: np.ndarray, n_steps: int = 100,
              lr: float = 0.001, batch_size: int = 32) -> Dict[str, Any]:
        """执行自监督训练"""
        logger.info("Starting self-supervised training: %d steps", n_steps)
        for step in range(n_steps):
            idx = np.random.choice(len(data), min(batch_size, len(data)), replace=False)
            batch = data[idx]
            metrics = self.learner.train_step(batch, lr=lr)
            self.step_count += 1
            if (step + 1) % 20 == 0:
                logger.info("SSL step %d/%d: %s", step + 1, n_steps, metrics)

        return {
            "final_loss": list(self.learner.loss_history)[-1] if self.learner.loss_history else 0.0,
            "mean_loss": float(np.mean(list(self.learner.loss_history)[-10:])) if len(self.learner.loss_history) >= 10 else 0.0,
            "steps": self.step_count,
        }

    def get_embeddings(self, data: np.ndarray) -> np.ndarray:
        """获取表征向量"""
        if self.method == "contrastive":
            return self.learner.get_embeddings(data)
        else:
            return self.encoder.encode_batch(data)

    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """评估表征质量"""
        embeddings = self.get_embeddings(data)
        probe_result = self.evaluator.evaluate_linear_probe(embeddings, labels)
        cluster_result = self.evaluator.evaluate_clustering(embeddings, labels)
        uniformity = self.evaluator.evaluate_uniformity(embeddings)
        return {
            "linear_probe": probe_result,
            "clustering": cluster_result,
            "uniformity": uniformity,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "step_count": self.step_count,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
        }

    def save(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "encoder_params": {k: v.tolist() for k, v in self.encoder.params.items()},
            "step_count": self.step_count,
        }

    def load(self, data: Dict[str, Any]):
        self.method = data.get("method", self.method)
        params = {k: np.array(v) for k, v in data.get("encoder_params", {}).items()}
        if params:
            self.encoder.set_params(params)
        self.step_count = data.get("step_count", 0)
