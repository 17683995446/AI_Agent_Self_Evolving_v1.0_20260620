"""
阶段43：多模态融合引擎

基于第一性原理：
- 不同模态携带互补信息，融合后信息量大于单一模态
- 注意力机制：让模型自动学习各模态的贡献权重
- 跨模态对齐：不同模态的表征映射到统一语义空间

核心设计：
1. ModalityEncoder: 单模态编码器
2. CrossModalAligner: 跨模态对齐器
3. AttentionFusion: 注意力融合器
4. MultimodalPredictor: 多模态预测器
5. MultimodalController: 高层封装

接口契约：
- 模态数据: Dict[str, np.ndarray]，键为模态名，值为该模态的特征矩阵
- 融合表征: np.ndarray，统一维度的融合特征
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import copy

logger = logging.getLogger(__name__)


@dataclass
class ModalityConfig:
    """模态配置"""
    name: str
    input_dim: int
    latent_dim: int = 64
    modality_type: str = "numeric"  # numeric, text, image, timeseries


class ModalityEncoder:
    """
    单模态编码器
    将特定模态的原始特征映射到统一维度的潜空间
    """

    def __init__(self, config: ModalityConfig):
        self.config = config
        dims = [config.input_dim, config.latent_dim * 2, config.latent_dim]
        self.params = {}
        for i in range(len(dims) - 1):
            self.params[f'W{i}'] = np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i])
            self.params[f'b{i}'] = np.zeros(dims[i + 1])
        logger.info("ModalityEncoder initialized: %s (%d -> %d)",
                    config.name, config.input_dim, config.latent_dim)

    def _relu(self, x):
        return np.maximum(0, x)

    def encode(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i in range(len(self.params) // 2):
            h = self._relu(np.dot(h, self.params[f'W{i}']) + self.params[f'b{i}'])
        return h

    def get_params(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params: Dict[str, np.ndarray]):
        self.params = {k: v.copy() for k, v in params.items()}


class CrossModalAligner:
    """
    跨模态对齐器
    将不同模态的表征映射到统一语义空间
    基于对比学习：同一时刻的不同模态表征应接近
    """

    def __init__(self, latent_dim: int = 64, temperature: float = 0.1):
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.alignment_history: deque = deque(maxlen=500)
        logger.info("CrossModalAligner initialized: dim=%d", latent_dim)

    def compute_alignment_loss(self, modality_embeddings: Dict[str, np.ndarray]) -> float:
        """计算跨模态对齐损失"""
        modality_names = list(modality_embeddings.keys())
        if len(modality_names) < 2:
            return 0.0

        total_loss = 0.0
        n_pairs = 0
        for i in range(len(modality_names)):
            for j in range(i + 1, len(modality_names)):
                z1 = modality_embeddings[modality_names[i]]
                z2 = modality_embeddings[modality_names[j]]
                # L2归一化
                z1 = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-10)
                z2 = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-10)
                # 对比损失（带数值稳定化）
                sim = np.dot(z1, z2.T) / self.temperature
                sim = sim - np.max(sim, axis=1, keepdims=True)  # 数值稳定
                n = len(z1)
                labels = np.arange(n)
                loss = -np.mean(sim[np.arange(n), labels] -
                               np.log(np.sum(np.exp(sim), axis=1) + 1e-10))
                total_loss += loss
                n_pairs += 1

        avg_loss = total_loss / max(n_pairs, 1)
        self.alignment_history.append(avg_loss)
        return float(avg_loss)

    def align(self, embeddings: np.ndarray, target_dim: int = None) -> np.ndarray:
        """通过线性投影对齐维度"""
        if target_dim is None:
            target_dim = self.latent_dim
        if embeddings.shape[1] == target_dim:
            return embeddings
        # 简单PCA式降维/升维
        if embeddings.shape[1] > target_dim:
            U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)
            n_components = min(target_dim, U.shape[1], len(S))
            result = np.zeros((embeddings.shape[0], target_dim))
            result[:, :n_components] = U[:, :n_components] * S[:n_components]
            return result
        else:
            padded = np.zeros((embeddings.shape[0], target_dim))
            padded[:, :embeddings.shape[1]] = embeddings
            return padded


class AttentionFusion:
    """
    注意力融合器
    使用注意力机制自动加权不同模态
    """

    def __init__(self, n_modalities: int, latent_dim: int = 64,
                 n_heads: int = 4):
        self.n_modalities = n_modalities
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        # 注意力权重
        self.query_weights = np.random.randn(latent_dim, latent_dim) * 0.1
        self.key_weights = np.random.randn(latent_dim, latent_dim) * 0.1
        self.value_weights = np.random.randn(latent_dim, latent_dim) * 0.1
        # 模态权重
        self.modality_weights = np.ones(n_modalities) / n_modalities
        self.fusion_history: deque = deque(maxlen=500)
        logger.info("AttentionFusion initialized: n_modalities=%d, heads=%d", n_modalities, n_heads)

    def compute_attention(self, modality_embeddings: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算跨模态注意力
        Returns: (fused_embedding, attention_weights)
        """
        # 堆叠所有模态: (n_modalities, batch, dim)
        stacked = np.stack(modality_embeddings)  # (M, B, D)
        M, B, D = stacked.shape

        # 简化注意力：基于模态间余弦相似度计算权重
        # 计算每个模态的均值表征
        modality_means = np.mean(stacked, axis=1)  # (M, D)

        # 模态间相似度矩阵
        modality_means_norm = modality_means / (np.linalg.norm(modality_means, axis=1, keepdims=True) + 1e-10)
        sim_matrix = np.dot(modality_means_norm, modality_means_norm.T)  # (M, M)
        sim_matrix = sim_matrix / np.sqrt(D)

        # 注意力权重：每行softmax
        sim_matrix = sim_matrix - np.max(sim_matrix, axis=1, keepdims=True)
        exp_sim = np.exp(sim_matrix)
        attention_weights_matrix = exp_sim / (np.sum(exp_sim, axis=1, keepdims=True) + 1e-10)  # (M, M)

        # 加权融合各模态
        attended = np.zeros((B, D))
        for m in range(M):
            # 该模态的注意力权重（来自所有模态对该模态的关注）
            weights_m = attention_weights_matrix[m]  # (M,)
            # 加权求和所有模态的表征
            context = np.sum(stacked * weights_m[:, None, None], axis=0)  # (B, D)
            attended += context * self.modality_weights[m]

        attention_weights = self.modality_weights.copy()
        return attended, attention_weights

    def fuse(self, modality_embeddings: List[np.ndarray]) -> np.ndarray:
        """融合多个模态的表征"""
        fused, weights = self.compute_attention(modality_embeddings)
        self.fusion_history.append({"fused_shape": fused.shape, "weights": weights.tolist()})
        return fused

    def update_modality_weights(self, modality_losses: List[float], lr: float = 0.01):
        """根据各模态损失更新模态权重（损失低的模态权重高）"""
        losses = np.array(modality_losses)
        # 损失越低，权重越高
        weights = 1.0 / (losses + 1e-10)
        weights = weights / weights.sum()
        self.modality_weights = self.modality_weights * (1 - lr) + weights * lr
        logger.info("Modality weights updated: %s", self.modality_weights)


class MultimodalPredictor:
    """
    多模态预测器
    基于融合表征进行预测
    """

    def __init__(self, latent_dim: int = 64, output_dim: int = 1,
                 hidden_dim: int = 32):
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.W1 = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / latent_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
        logger.info("MultimodalPredictor initialized: %d -> %d -> %d",
                    latent_dim, hidden_dim, output_dim)

    def predict(self, fused_embedding: np.ndarray) -> np.ndarray:
        h = np.maximum(0, np.dot(fused_embedding, self.W1) + self.b1)
        return np.dot(h, self.W2) + self.b2

    def compute_loss(self, fused_embedding: np.ndarray, targets: np.ndarray) -> float:
        preds = self.predict(fused_embedding)
        return float(np.mean((preds - targets) ** 2))


class MultimodalController:
    """
    多模态融合控制器高层封装
    """

    def __init__(self, modality_configs: List[ModalityConfig],
                 output_dim: int = 1, latent_dim: int = 64):
        self.modality_configs = modality_configs
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        self.encoders: Dict[str, ModalityEncoder] = {
            cfg.name: ModalityEncoder(cfg) for cfg in modality_configs
        }
        self.aligner = CrossModalAligner(latent_dim)
        self.fusion = AttentionFusion(len(modality_configs), latent_dim)
        self.predictor = MultimodalPredictor(latent_dim, output_dim)

        self.step_count = 0
        logger.info("MultimodalController initialized: %d modalities", len(modality_configs))

    def encode_modalities(self, modality_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """编码各模态数据"""
        embeddings = {}
        for name, data in modality_data.items():
            if name in self.encoders:
                embeddings[name] = self.encoders[name].encode(data)
        return embeddings

    def fuse(self, modality_data: Dict[str, np.ndarray]) -> np.ndarray:
        """编码+融合"""
        embeddings = self.encode_modalities(modality_data)
        # 对齐到统一维度
        aligned = []
        for name in self.encoders:
            if name in embeddings:
                emb = self.aligner.align(embeddings[name], self.latent_dim)
                aligned.append(emb)
        if not aligned:
            return np.zeros((0, self.latent_dim))
        fused = self.fusion.fuse(aligned)
        return fused

    def predict(self, modality_data: Dict[str, np.ndarray]) -> np.ndarray:
        """端到端预测"""
        fused = self.fuse(modality_data)
        return self.predictor.predict(fused)

    def train_step(self, modality_data: Dict[str, np.ndarray],
                   targets: np.ndarray, lr: float = 0.001) -> Dict[str, float]:
        """训练一步"""
        fused = self.fuse(modality_data)
        loss = self.predictor.compute_loss(fused, targets)

        # 简化梯度更新
        eps = 1e-4
        for param_dict in [self.predictor.__dict__]:
            for key in ['W1', 'W2']:
                if not isinstance(param_dict.get(key), np.ndarray):
                    continue
                orig = param_dict[key].copy()
                grad = np.zeros_like(orig)
                it = np.nditer(orig, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    idx = it.multi_index
                    param_dict[key][idx] = orig[idx] + eps
                    loss1 = self.predictor.compute_loss(fused, targets)
                    param_dict[key][idx] = orig[idx] - eps
                    loss2 = self.predictor.compute_loss(fused, targets)
                    grad[idx] = (loss1 - loss2) / (2 * eps)
                    param_dict[key][idx] = orig[idx]
                    it.iternext()
                param_dict[key] -= lr * grad

        self.step_count += 1
        return {"fusion_loss": loss, "step": self.step_count}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "n_modalities": len(self.encoders),
            "modality_names": list(self.encoders.keys()),
            "latent_dim": self.latent_dim,
            "step_count": self.step_count,
            "modality_weights": self.fusion.modality_weights.tolist(),
        }

    def save(self) -> Dict[str, Any]:
        return {
            "modality_configs": [
                {"name": c.name, "input_dim": c.input_dim, "latent_dim": c.latent_dim}
                for c in self.modality_configs
            ],
            "step_count": self.step_count,
            "modality_weights": self.fusion.modality_weights.tolist(),
        }

    def load(self, data: Dict[str, Any]):
        self.step_count = data.get("step_count", 0)
        weights = data.get("modality_weights")
        if weights:
            self.fusion.modality_weights = np.array(weights)
