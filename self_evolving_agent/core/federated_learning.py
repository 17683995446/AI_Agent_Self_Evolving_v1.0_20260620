"""
联邦学习与隐私计算 —— 数据不动模型动，差分隐私保护

设计原则（第一性原理）：
1. 数据不动模型动：原始数据不离开本地，只交换模型参数
2. 差分隐私：添加 calibrated noise 保护个体隐私
3. 安全聚合：加密传输模型参数，防止中间人攻击
4. 联邦平均：加权聚合多个Agent的模型更新
"""

import os
import sys
import copy
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


@dataclass
class ModelUpdate:
    """模型更新 —— 本地训练后的参数增量"""
    agent_id: str
    round_num: int
    weights: Dict[str, List[float]] = field(default_factory=dict)
    sample_count: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "round_num": self.round_num,
            "weights": self.weights,
            "sample_count": self.sample_count,
            "metrics": self.metrics,
        }


class DifferentialPrivacy:
    """差分隐私 —— 为模型参数添加噪声"""

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity

    def add_noise(self, values: List[float]) -> List[float]:
        """为高斯机制添加噪声"""
        import math
        # 高斯噪声标准差
        sigma = self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        return [v + random.gauss(0, sigma) for v in values]

    def clip_gradients(self, values: List[float], max_norm: float = 1.0) -> List[float]:
        """梯度裁剪，限制敏感度"""
        import math
        norm = math.sqrt(sum(v * v for v in values))
        if norm > max_norm:
            scale = max_norm / norm
            return [v * scale for v in values]
        return values


class FederatedAveraging:
    """联邦平均算法 —— 聚合多个Agent的模型更新"""

    def __init__(self):
        self._global_model: Dict[str, List[float]] = {}
        self._round_num = 0

    def initialize_model(self, weights: Dict[str, List[float]]) -> None:
        """初始化全局模型"""
        self._global_model = copy.deepcopy(weights)

    def aggregate(self, updates: List[ModelUpdate]) -> Dict[str, List[float]]:
        """聚合模型更新（加权平均）"""
        if not updates:
            return self._global_model

        total_samples = sum(u.sample_count for u in updates)
        if total_samples == 0:
            return self._global_model

        # 初始化聚合结果
        aggregated: Dict[str, List[float]] = {}
        for key in self._global_model:
            aggregated[key] = [0.0] * len(self._global_model[key])

        # 加权平均
        for update in updates:
            weight = update.sample_count / total_samples
            for key, values in update.weights.items():
                if key in aggregated:
                    for i, v in enumerate(values):
                        aggregated[key][i] += v * weight

        self._global_model = aggregated
        self._round_num += 1
        logger.info("FED_AVG", f"联邦聚合完成: 轮次={self._round_num}, 参与Agent={len(updates)}",
                    module="federated_learning")
        return aggregated

    def get_global_model(self) -> Dict[str, List[float]]:
        """获取当前全局模型"""
        return copy.deepcopy(self._global_model)


class SecureAggregator:
    """安全聚合 —— 简单的掩码聚合演示"""

    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self._masks: Dict[str, Dict[str, List[float]]] = {}

    def generate_mask(self, agent_id: str, model_keys: List[str],
                      model_shapes: Dict[str, int]) -> Dict[str, List[float]]:
        """为Agent生成随机掩码"""
        mask = {}
        for key in model_keys:
            shape = model_shapes.get(key, 0)
            mask[key] = [random.uniform(-1.0, 1.0) for _ in range(shape)]
        self._masks[agent_id] = mask
        return mask

    def apply_mask(self, weights: Dict[str, List[float]],
                   mask: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """应用掩码到权重"""
        masked = {}
        for key, values in weights.items():
            m = mask.get(key, [0.0] * len(values))
            masked[key] = [v + m[i] for i, v in enumerate(values)]
        return masked

    def remove_mask(self, aggregated: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """移除聚合后的掩码（所有掩码之和应为0）"""
        # 简化版：假设掩码已设计为相互抵消
        return aggregated


class FederatedLearningNode:
    """联邦学习节点 —— 每个Agent本地训练并参与联邦聚合"""

    def __init__(self, agent_id: str, dp: Optional[DifferentialPrivacy] = None):
        self.agent_id = agent_id
        self.dp = dp or DifferentialPrivacy()
        self.local_model: Dict[str, List[float]] = {}
        self.round_num = 0

    def set_local_model(self, weights: Dict[str, List[float]]) -> None:
        """设置本地模型"""
        self.local_model = copy.deepcopy(weights)

    def local_train(self, data_samples: int = 100) -> ModelUpdate:
        """本地训练（模拟）"""
        # 模拟训练：在本地模型上添加微小随机更新
        updated = {}
        for key, values in self.local_model.items():
            # 模拟梯度更新
            grads = [random.uniform(-0.01, 0.01) for _ in values]
            # 梯度裁剪
            clipped = self.dp.clip_gradients(grads)
            # 添加差分隐私噪声
            noised = self.dp.add_noise(clipped)
            updated[key] = [values[i] + noised[i] for i in range(len(values))]

        self.local_model = updated
        self.round_num += 1

        return ModelUpdate(
            agent_id=self.agent_id,
            round_num=self.round_num,
            weights=updated,
            sample_count=data_samples,
            metrics={"loss": random.uniform(0.1, 0.5)},
        )

    def apply_global_model(self, global_weights: Dict[str, List[float]]) -> None:
        """应用全局模型到本地"""
        self.local_model = copy.deepcopy(global_weights)
        logger.debug("FED", f"Agent {self.agent_id} 应用全局模型", module="federated_learning")


class FederatedCoordinator:
    """联邦协调器 —— 管理联邦学习流程"""

    def __init__(self):
        self.fed_avg = FederatedAveraging()
        self.nodes: Dict[str, FederatedLearningNode] = {}
        self.history: List[Dict[str, Any]] = []

    def register_node(self, node: FederatedLearningNode) -> None:
        """注册学习节点"""
        self.nodes[node.agent_id] = node

    def initialize_global_model(self, weights: Dict[str, List[float]]) -> None:
        """初始化全局模型"""
        self.fed_avg.initialize_model(weights)
        for node in self.nodes.values():
            node.set_local_model(weights)

    def run_round(self) -> Dict[str, Any]:
        """运行一轮联邦学习"""
        updates = []
        for node in self.nodes.values():
            update = node.local_train()
            updates.append(update)

        # 聚合
        global_model = self.fed_avg.aggregate(updates)

        # 分发
        for node in self.nodes.values():
            node.apply_global_model(global_model)

        result = {
            "round": self.fed_avg._round_num,
            "participants": len(updates),
            "avg_loss": sum(u.metrics.get("loss", 0) for u in updates) / len(updates),
        }
        self.history.append(result)
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "nodes": len(self.nodes),
            "rounds": self.fed_avg._round_num,
            "history": self.history,
        }
