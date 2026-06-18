"""
阶段41：对抗鲁棒性训练

基于第一性原理：
- 对抗样本：输入空间中微小但恶意的扰动可导致模型错误预测
- 鲁棒性边界：模型在扰动邻域内保持预测一致的能力
- 对抗训练：在训练过程中加入对抗样本，提升模型鲁棒性

核心设计：
1. AdversarialExampleGenerator: 对抗样本生成器（FGSM/PGD）
2. RobustnessEvaluator: 鲁棒性评估器
3. AdversarialTrainer: 对抗训练器
4. RobustnessCertifier: 鲁棒性认证器（简化版）
5. PerturbationBudget: 扰动预算管理
6. AdversarialRobustnessController: 高层封装

接口契约：
- 输入: np.ndarray, 原始样本
- 扰动: np.ndarray, 对抗扰动
- 预算: float (epsilon), 最大扰动幅度
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import copy

logger = logging.getLogger(__name__)


@dataclass
class PerturbationBudget:
    """扰动预算"""
    epsilon: float = 0.1
    norm: str = "linf"  # "linf", "l2", "l1"

    def is_valid(self, original: np.ndarray, perturbed: np.ndarray) -> bool:
        """检查扰动是否在预算内"""
        delta = perturbed - original
        if self.norm == "linf":
            return float(np.max(np.abs(delta))) <= self.epsilon
        elif self.norm == "l2":
            return float(np.linalg.norm(delta)) <= self.epsilon
        elif self.norm == "l1":
            return float(np.sum(np.abs(delta))) <= self.epsilon
        return True

    def clip(self, original: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
        """将扰动裁剪到预算内"""
        delta = perturbed - original
        if self.norm == "linf":
            delta = np.clip(delta, -self.epsilon, self.epsilon)
        elif self.norm == "l2":
            norm = np.linalg.norm(delta)
            if norm > self.epsilon:
                delta = delta / norm * self.epsilon
        return original + delta


class AdversarialExampleGenerator:
    """
    对抗样本生成器
    实现FGSM和PGD攻击
    """

    def __init__(self, model_forward_fn: Callable[[np.ndarray], np.ndarray],
                 model_loss_fn: Optional[Callable] = None):
        """
        model_forward_fn: 输入x，返回logits
        model_loss_fn: 输入(x, y)，返回损失值（可选，用于数值梯度）
        """
        self.model_forward = model_forward_fn
        self.model_loss = model_loss_fn
        logger.info("AdversarialExampleGenerator initialized")

    def _compute_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算损失对输入的梯度（数值近似）"""
        eps = 1e-4
        grad = np.zeros_like(x)
        orig = x.copy()
        for idx in np.ndindex(x.shape):
            x[idx] = orig[idx] + eps
            loss1 = self._get_loss(x, y)
            x[idx] = orig[idx] - eps
            loss2 = self._get_loss(x, y)
            grad[idx] = (loss1 - loss2) / (2 * eps)
            x[idx] = orig[idx]
        return grad

    def _get_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.model_loss:
            return self.model_loss(x, y)
        # 默认交叉熵损失
        logits = self.model_forward(x)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
            y = np.array([y])
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        n = len(y)
        correct_probs = probs[np.arange(n), y.astype(int)]
        return float(-np.mean(np.log(correct_probs + 1e-10)))

    def fgsm(self, x: np.ndarray, y: np.ndarray,
             budget: PerturbationBudget) -> np.ndarray:
        """
        FGSM (Fast Gradient Sign Method)
        x_adv = x + epsilon * sign(grad_x loss)
        """
        grad = self._compute_gradient(x, y)
        if budget.norm == "linf":
            perturbation = budget.epsilon * np.sign(grad)
        else:
            # 对于L2/L1，使用归一化梯度
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                perturbation = budget.epsilon * grad / grad_norm
            else:
                perturbation = np.zeros_like(grad)
        adv = x + perturbation
        return budget.clip(x, adv)

    def pgd(self, x: np.ndarray, y: np.ndarray,
            budget: PerturbationBudget,
            steps: int = 10,
            step_size: float = None) -> np.ndarray:
        """
        PGD (Projected Gradient Descent)
        多步迭代FGSM，每步投影回预算球内
        """
        if step_size is None:
            step_size = budget.epsilon / steps * 2

        adv = x.copy()
        for _ in range(steps):
            grad = self._compute_gradient(adv, y)
            # 梯度上升（最大化损失）
            adv = adv + step_size * np.sign(grad)
            # 投影回原始点的epsilon邻域
            adv = budget.clip(x, adv)
            # 确保数值范围有效（如图像[0,1]）
            adv = np.clip(adv, -10.0, 10.0)
        return adv

    def generate(self, x: np.ndarray, y: np.ndarray,
                 method: str = "pgd",
                 budget: PerturbationBudget = None,
                 **kwargs) -> np.ndarray:
        """通用对抗样本生成接口"""
        if budget is None:
            budget = PerturbationBudget()
        if method == "fgsm":
            return self.fgsm(x, y, budget)
        elif method == "pgd":
            return self.pgd(x, y, budget, **kwargs)
        else:
            raise ValueError(f"Unknown attack method: {method}")


class RobustnessEvaluator:
    """
    鲁棒性评估器
    评估模型在对抗攻击下的表现
    """

    def __init__(self, model_forward_fn: Callable[[np.ndarray], np.ndarray],
                 model_predict_fn: Callable[[np.ndarray], np.ndarray]):
        self.model_forward = model_forward_fn
        self.model_predict = model_predict_fn
        self.attack_generator = AdversarialExampleGenerator(model_forward_fn)
        logger.info("RobustnessEvaluator initialized")

    def evaluate(self, x: np.ndarray, y: np.ndarray,
                 budget: PerturbationBudget = None,
                 attack_method: str = "pgd") -> Dict[str, float]:
        """评估单批次数据的鲁棒性"""
        if budget is None:
            budget = PerturbationBudget(epsilon=0.1)

        clean_preds = self.model_predict(x)
        clean_acc = float(np.mean(clean_preds == y))

        # 生成对抗样本
        adv_x = np.array([self.attack_generator.generate(
            x[i], y[i], method=attack_method, budget=budget)
            for i in range(len(x))])

        adv_preds = self.model_predict(adv_x)
        adv_acc = float(np.mean(adv_preds == y))

        # 计算扰动大小
        perturbations = adv_x - x
        max_perturbation = float(np.max(np.abs(perturbations)))
        mean_perturbation = float(np.mean(np.abs(perturbations)))

        return {
            "clean_accuracy": clean_acc,
            "adversarial_accuracy": adv_acc,
            "accuracy_drop": clean_acc - adv_acc,
            "max_perturbation": max_perturbation,
            "mean_perturbation": mean_perturbation,
            "budget_epsilon": budget.epsilon,
        }

    def evaluate_robustness_curve(self, x: np.ndarray, y: np.ndarray,
                                  epsilons: List[float] = None) -> List[Dict[str, float]]:
        """评估不同扰动预算下的鲁棒性曲线"""
        if epsilons is None:
            epsilons = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

        results = []
        for eps in epsilons:
            budget = PerturbationBudget(epsilon=eps)
            result = self.evaluate(x, y, budget=budget)
            result["epsilon"] = eps
            results.append(result)
        return results


class AdversarialTrainer:
    """
    对抗训练器
    在训练过程中混合原始样本和对抗样本
    """

    def __init__(self, model: Any,
                 model_forward_fn: Callable,
                 model_update_fn: Callable,
                 budget: PerturbationBudget = None,
                 adversarial_ratio: float = 0.5):
        self.model = model
        self.model_forward = model_forward_fn
        self.model_update = model_update_fn
        self.budget = budget or PerturbationBudget(epsilon=0.1)
        self.adversarial_ratio = adversarial_ratio
        self.attack_generator = AdversarialExampleGenerator(model_forward_fn)
        self.training_history: deque = deque(maxlen=1000)
        logger.info("AdversarialTrainer initialized: ratio=%.2f", adversarial_ratio)

    def train_step(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """执行一步对抗训练"""
        n = len(x)
        n_adv = int(n * self.adversarial_ratio)

        # 生成对抗样本
        adv_x = x.copy()
        for i in range(n_adv):
            adv_x[i] = self.attack_generator.generate(
                x[i], y[i], method="pgd", budget=self.budget, steps=5)

        # 混合训练
        combined_x = np.vstack([x, adv_x[:n_adv]])
        combined_y = np.hstack([y, y[:n_adv]])

        # 更新模型（调用外部更新函数）
        loss = self.model_update(combined_x, combined_y)

        # 评估
        clean_preds = np.argmax(self.model_forward(x), axis=-1)
        adv_preds = np.argmax(self.model_forward(adv_x), axis=-1)
        clean_acc = float(np.mean(clean_preds == y))
        adv_acc = float(np.mean(adv_preds == y))

        metrics = {
            "loss": float(loss),
            "clean_acc": clean_acc,
            "adversarial_acc": adv_acc,
        }
        self.training_history.append(metrics)
        return metrics

    def train(self, data_generator: Callable, n_steps: int = 100) -> Dict[str, Any]:
        """执行对抗训练"""
        logger.info("Starting adversarial training for %d steps", n_steps)
        for step in range(n_steps):
            x, y = data_generator()
            metrics = self.train_step(x, y)
            if (step + 1) % 20 == 0:
                logger.info("Adv training step %d/%d: %s", step + 1, n_steps, metrics)
        return {
            "final_clean_acc": np.mean([m["clean_acc"] for m in list(self.training_history)[-10:]]),
            "final_adv_acc": np.mean([m["adversarial_acc"] for m in list(self.training_history)[-10:]]),
        }


class RobustnessCertifier:
    """
    鲁棒性认证器（简化版）
    基于局部Lipschitz常数估计认证鲁棒性半径
    """

    def __init__(self, model_forward_fn: Callable):
        self.model_forward = model_forward_fn
        logger.info("RobustnessCertifier initialized")

    def estimate_local_lipschitz(self, x: np.ndarray, n_samples: int = 50,
                                  radius: float = 0.01) -> float:
        """估计局部Lipschitz常数"""
        y0 = self.model_forward(x.reshape(1, -1))
        max_ratio = 0.0
        for _ in range(n_samples):
            noise = np.random.randn(*x.shape) * radius
            x_perturbed = x + noise
            y1 = self.model_forward(x_perturbed.reshape(1, -1))
            output_diff = np.linalg.norm(y1 - y0)
            input_diff = np.linalg.norm(noise)
            if input_diff > 1e-10:
                ratio = output_diff / input_diff
                max_ratio = max(max_ratio, ratio)
        return float(max_ratio)

    def certify_radius(self, x: np.ndarray, y_true: int,
                       lipschitz: float = None) -> float:
        """
        认证鲁棒性半径
        如果扰动半径 < margin / (2 * L)，则保证分类不变
        """
        if lipschitz is None:
            lipschitz = self.estimate_local_lipschitz(x)
        logits = self.model_forward(x.reshape(1, -1)).flatten()
        sorted_logits = np.sort(logits)[::-1]
        margin = sorted_logits[0] - sorted_logits[1]
        if lipschitz > 1e-10:
            certified_radius = margin / (2 * lipschitz)
        else:
            certified_radius = float('inf')
        return float(certified_radius)


class AdversarialRobustnessController:
    """
    对抗鲁棒性控制器高层封装
    整合攻击生成、鲁棒性评估、对抗训练、认证
    """

    def __init__(self, model: Any,
                 model_forward_fn: Callable[[np.ndarray], np.ndarray],
                 model_predict_fn: Callable[[np.ndarray], np.ndarray],
                 model_update_fn: Callable = None):
        self.model = model
        self.model_forward = model_forward_fn
        self.model_predict = model_predict_fn
        self.model_update = model_update_fn

        self.attack_generator = AdversarialExampleGenerator(model_forward_fn)
        self.evaluator = RobustnessEvaluator(model_forward_fn, model_predict_fn)
        self.certifier = RobustnessCertifier(model_forward_fn)
        self.trainer: Optional[AdversarialTrainer] = None

        if model_update_fn:
            self.trainer = AdversarialTrainer(
                model, model_forward_fn, model_update_fn)

        logger.info("AdversarialRobustnessController initialized")

    def generate_attack(self, x: np.ndarray, y: np.ndarray,
                        method: str = "pgd", epsilon: float = 0.1) -> np.ndarray:
        """生成对抗样本"""
        budget = PerturbationBudget(epsilon=epsilon)
        return self.attack_generator.generate(x, y, method=method, budget=budget)

    def evaluate_robustness(self, x: np.ndarray, y: np.ndarray,
                            epsilon: float = 0.1) -> Dict[str, float]:
        """评估鲁棒性"""
        budget = PerturbationBudget(epsilon=epsilon)
        return self.evaluator.evaluate(x, y, budget=budget)

    def certify(self, x: np.ndarray, y_true: int) -> Dict[str, float]:
        """认证鲁棒性"""
        lipschitz = self.certifier.estimate_local_lipschitz(x)
        radius = self.certifier.certify_radius(x, y_true, lipschitz)
        return {
            "lipschitz_estimate": lipschitz,
            "certified_radius": radius,
        }

    def adversarial_train(self, data_generator: Callable, n_steps: int = 100) -> Dict[str, Any]:
        """执行对抗训练"""
        if self.trainer is None:
            raise ValueError("Model update function not provided. Cannot perform adversarial training.")
        return self.trainer.train(data_generator, n_steps)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "has_trainer": self.trainer is not None,
            "attack_methods": ["fgsm", "pgd"],
        }

    def save(self) -> Dict[str, Any]:
        return {
            "has_trainer": self.trainer is not None,
        }

    def load(self, data: Dict[str, Any]):
        pass
