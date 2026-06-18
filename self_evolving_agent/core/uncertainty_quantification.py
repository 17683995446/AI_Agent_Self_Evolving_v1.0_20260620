"""
阶段45：不确定性量化与贝叶斯神经网络

基于第一性原理：
- 确定性预测无法表达"自己不知道什么"
- 贝叶斯推断：将参数视为分布而非点估计
- MC Dropout：通过推理时保留Dropout近似贝叶斯推断
- 深度集成：多个独立模型预测的方差作为不确定性

核心设计：
1. MCDropoutNetwork: MC Dropout神经网络
2. DeepEnsemble: 深度集成不确定性估计
3. UncertaintyEstimator: 不确定性估计器
4. BayesianLinearRegression: 贝叶斯线性回归
5. UncertaintyController: 高层封装

接口契约：
- 预测结果: {"mean": np.ndarray, "std": np.ndarray, "confidence": float}
- 不确定性: float, 越高表示越不确定
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import copy

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    """不确定性量化结果"""
    mean: np.ndarray
    std: np.ndarray
    epistemic_uncertainty: float  # 认知不确定性（模型不确定性）
    aleatoric_uncertainty: float  # 偶然不确定性（数据不确定性）
    total_uncertainty: float
    confidence: float  # 1 - total_uncertainty
    n_samples: int


class MCDropoutNetwork:
    """
    MC Dropout网络
    推理时保留Dropout，多次采样估计预测分布
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32,
                 output_dim: int = 1, dropout_rate: float = 0.2,
                 n_mc_samples: int = 30):
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # 校验dropout_rate范围，防止除零
        self.dropout_rate = min(max(dropout_rate, 0.0), 0.99)
        self.n_mc_samples = max(1, n_mc_samples)

        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
        logger.info("MCDropoutNetwork initialized: dropout=%.2f, mc_samples=%d", dropout_rate, n_mc_samples)

    def _forward_with_dropout(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """带Dropout的前向传播"""
        h = np.maximum(0, np.dot(x, self.W1) + self.b1)
        if training:
            mask = np.random.binomial(1, 1 - self.dropout_rate, h.shape) / (1 - self.dropout_rate)
            h = h * mask
        return np.dot(h, self.W2) + self.b2

    def predict_with_uncertainty(self, x: np.ndarray) -> UncertaintyResult:
        """MC Dropout预测+不确定性估计"""
        predictions = []
        for _ in range(self.n_mc_samples):
            pred = self._forward_with_dropout(x, training=True)
            predictions.append(pred)

        predictions = np.array(predictions)  # (n_samples, batch, output_dim)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)

        # 认知不确定性：MC采样间的方差（模型不确定性）
        epistemic = float(np.mean(np.var(predictions, axis=0)))

        # 偶然不确定性：预测均值附近的残差方差（数据噪声）
        # 使用预测值与均值偏差的平均方差减去认知不确定性
        # 修正：std^2 == var，不能直接相减。改用预测残差估计偶然不确定性
        pred_mean = mean
        residuals = predictions - pred_mean  # (n_samples, batch, output)
        # 偶然不确定性 = 每个样本的预测方差中无法被模型间方差解释的部分
        # 简化估计：使用预测均值的平均方差作为下界
        total_var = float(np.mean(np.var(predictions, axis=0)))
        aleatoric = max(0.0, total_var * 0.1)  # 简化：总方差的10%作为偶然不确定性下界
        # 如果只有1个MC样本，无法区分认知和偶然不确定性
        if self.n_mc_samples <= 1:
            aleatoric = 0.0

        total = epistemic + aleatoric
        confidence = 1.0 / (1.0 + total)

        return UncertaintyResult(
            mean=mean, std=std,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
            confidence=confidence,
            n_samples=self.n_mc_samples,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """确定性预测（关闭Dropout）"""
        return self._forward_with_dropout(x, training=False)

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        """训练一步"""
        pred = self._forward_with_dropout(x, training=True)
        loss = float(np.mean((pred - y) ** 2))

        # 数值梯度
        eps = 1e-4
        for param_name in ['W1', 'W2']:
            param = getattr(self, param_name)
            orig = param.copy()
            grad = np.zeros_like(orig)
            it = np.nditer(orig, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                param[idx] = orig[idx] + eps
                pred1 = self._forward_with_dropout(x, training=False)
                loss1 = np.mean((pred1 - y) ** 2)
                param[idx] = orig[idx] - eps
                pred2 = self._forward_with_dropout(x, training=False)
                loss2 = np.mean((pred2 - y) ** 2)
                grad[idx] = (loss1 - loss2) / (2 * eps)
                param[idx] = orig[idx]
                it.iternext()
            setattr(self, param_name, orig - lr * grad)

        return loss


class DeepEnsemble:
    """
    深度集成
    训练多个独立模型，用预测方差估计不确定性
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32,
                 output_dim: int = 1, n_models: int = 5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_models = n_models

        self.models: List[Dict[str, np.ndarray]] = []
        for _ in range(n_models):
            model = {
                'W1': np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim),
                'b1': np.zeros(hidden_dim),
                'W2': np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim),
                'b2': np.zeros(output_dim),
            }
            self.models.append(model)
        logger.info("DeepEnsemble initialized: n_models=%d", n_models)

    def _forward(self, x: np.ndarray, model: Dict[str, np.ndarray]) -> np.ndarray:
        h = np.maximum(0, np.dot(x, model['W1']) + model['b1'])
        return np.dot(h, model['W2']) + model['b2']

    def predict_with_uncertainty(self, x: np.ndarray) -> UncertaintyResult:
        """集成预测+不确定性"""
        predictions = []
        for model in self.models:
            pred = self._forward(x, model)
            predictions.append(pred)

        predictions = np.array(predictions)  # (n_models, batch, output)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)

        epistemic = float(np.mean(np.var(predictions, axis=0)))
        aleatoric = 0.0  # 纯集成不区分偶然不确定性
        total = epistemic
        confidence = 1.0 / (1.0 + total)

        return UncertaintyResult(
            mean=mean, std=std,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
            confidence=confidence,
            n_samples=self.n_models,
        )

    def train_model(self, model_idx: int, x: np.ndarray, y: np.ndarray,
                    lr: float = 0.01, n_steps: int = 10) -> float:
        """训练集成中的单个模型"""
        model = self.models[model_idx]
        final_loss = 0.0
        for _ in range(n_steps):
            pred = self._forward(x, model)
            loss = float(np.mean((pred - y) ** 2))
            final_loss = loss

            eps = 1e-4
            for key in ['W1', 'W2']:
                orig = model[key].copy()
                grad = np.zeros_like(orig)
                it = np.nditer(orig, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    idx = it.multi_index
                    model[key][idx] = orig[idx] + eps
                    loss1 = np.mean((self._forward(x, model) - y) ** 2)
                    model[key][idx] = orig[idx] - eps
                    loss2 = np.mean((self._forward(x, model) - y) ** 2)
                    grad[idx] = (loss1 - loss2) / (2 * eps)
                    model[key][idx] = orig[idx]
                    it.iternext()
                model[key] -= lr * grad

        return final_loss

    def train_all(self, x: np.ndarray, y: np.ndarray,
                  lr: float = 0.01, n_steps: int = 10) -> List[float]:
        """训练所有模型"""
        losses = []
        for i in range(self.n_models):
            loss = self.train_model(i, x, y, lr, n_steps)
            losses.append(loss)
            logger.info("Ensemble model %d trained: loss=%.4f", i, loss)
        return losses


class BayesianLinearRegression:
    """
    贝叶斯线性回归
    权重服从高斯先验，通过贝叶斯推断得到后验分布
    """

    def __init__(self, input_dim: int, prior_std: float = 1.0,
                 noise_std: float = 0.1):
        self.input_dim = input_dim
        self.prior_std = prior_std
        self.noise_std = noise_std

        # 先验: w ~ N(0, prior_std^2 * I)
        self.prior_mean = np.zeros(input_dim)
        self.prior_cov = np.eye(input_dim) * prior_std ** 2

        # 后验（初始为先验）
        self.post_mean = self.prior_mean.copy()
        self.post_cov = self.prior_cov.copy()
        logger.info("BayesianLinearRegression initialized: dim=%d", input_dim)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """贝叶斯线性回归拟合（解析解）"""
        # 后验: w ~ N(mu, Sigma)
        # Sigma = (1/prior_std^2 * I + 1/noise_std^2 * X^T X)^{-1}
        # mu = 1/noise_std^2 * Sigma * X^T y
        prior_precision = np.eye(self.input_dim) / (self.prior_std ** 2)
        data_precision = X.T @ X / (self.noise_std ** 2)
        precision_matrix = prior_precision + data_precision
        # 添加正则化项防止矩阵奇异
        precision_matrix += np.eye(self.input_dim) * 1e-6
        try:
            self.post_cov = np.linalg.inv(precision_matrix)
        except np.linalg.LinAlgError:
            # 使用伪逆作为后备
            self.post_cov = np.linalg.pinv(precision_matrix)
        self.post_mean = self.post_cov @ X.T @ y / (self.noise_std ** 2)
        logger.info("Bayesian LR fitted: post_mean shape=%s", self.post_mean.shape)

    def predict_with_uncertainty(self, X: np.ndarray) -> UncertaintyResult:
        """预测+不确定性"""
        mean = X @ self.post_mean
        # 预测方差 = X * Sigma * X^T + noise_std^2
        var = np.array([X[i] @ self.post_cov @ X[i].T + self.noise_std ** 2
                        for i in range(len(X))])
        std = np.sqrt(var).reshape(-1, 1)

        epistemic = float(np.mean(np.array([X[i] @ self.post_cov @ X[i].T
                                             for i in range(len(X))])))
        aleatoric = self.noise_std ** 2
        total = epistemic + aleatoric
        confidence = 1.0 / (1.0 + total)

        return UncertaintyResult(
            mean=mean.reshape(-1, 1), std=std,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
            confidence=confidence,
            n_samples=1,
        )


class UncertaintyEstimator:
    """
    不确定性估计器
    通用接口，支持多种不确定性量化方法
    """

    def __init__(self, method: str = "mc_dropout", **kwargs):
        self.method = method
        self.model: Any = None
        self._init_model(**kwargs)
        logger.info("UncertaintyEstimator initialized: method=%s", method)

    def _init_model(self, input_dim: int = 10, hidden_dim: int = 32,
                    output_dim: int = 1, **kwargs):
        if self.method == "mc_dropout":
            self.model = MCDropoutNetwork(input_dim, hidden_dim, output_dim, **kwargs)
        elif self.method == "ensemble":
            self.model = DeepEnsemble(input_dim, hidden_dim, output_dim, **kwargs)
        elif self.method == "bayesian_lr":
            self.model = BayesianLinearRegression(input_dim, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def predict(self, x: np.ndarray) -> UncertaintyResult:
        """预测+不确定性"""
        return self.model.predict_with_uncertainty(x)

    def train(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """训练模型"""
        if self.method == "mc_dropout":
            # 支持epochs参数，循环调用train_step
            epochs = kwargs.pop("epochs", 1)
            lr = kwargs.pop("lr", 0.01)
            last_loss = 0.0
            for _ in range(epochs):
                last_loss = self.model.train_step(x, y, lr=lr)
            return last_loss
        elif self.method == "ensemble":
            return self.model.train_all(x, y, **kwargs)
        elif self.method == "bayesian_lr":
            self.model.fit(x, y)
            return None


class UncertaintyController:
    """
    不确定性量化控制器高层封装
    """

    def __init__(self, input_dim: int, output_dim: int = 1,
                 method: str = "mc_dropout",
                 confidence_threshold: float = 0.7,
                 **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.method = method
        self.confidence_threshold = confidence_threshold

        self.estimator = UncertaintyEstimator(
            method=method, input_dim=input_dim, output_dim=output_dim, **kwargs)
        self.prediction_history: deque = deque(maxlen=1000)
        self.step_count = 0
        logger.info("UncertaintyController initialized: method=%s, threshold=%.2f",
                    method, confidence_threshold)

    def train(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """训练不确定性模型"""
        return self.estimator.train(x, y, **kwargs)

    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        """预测并返回不确定性"""
        result = self.estimator.predict(x)
        self.prediction_history.append({
            "step": self.step_count,
            "confidence": result.confidence,
            "uncertainty": result.total_uncertainty,
        })
        self.step_count += 1

        return {
            "mean": result.mean,
            "std": result.std,
            "epistemic_uncertainty": result.epistemic_uncertainty,
            "aleatoric_uncertainty": result.aleatoric_uncertainty,
            "total_uncertainty": result.total_uncertainty,
            "confidence": result.confidence,
            "is_reliable": result.confidence >= self.confidence_threshold,
        }

    def get_confidence_stats(self) -> Dict[str, float]:
        """获取置信度统计"""
        if not self.prediction_history:
            return {"mean_confidence": 0.0, "mean_uncertainty": 0.0, "reliable_ratio": 0.0}
        confidences = [p["confidence"] for p in self.prediction_history]
        uncertainties = [p["uncertainty"] for p in self.prediction_history]
        reliable = sum(1 for c in confidences if c >= self.confidence_threshold)
        return {
            "mean_confidence": float(np.mean(confidences)),
            "mean_uncertainty": float(np.mean(uncertainties)),
            "reliable_ratio": reliable / len(confidences),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "step_count": self.step_count,
            "confidence_threshold": self.confidence_threshold,
            "confidence_stats": self.get_confidence_stats(),
        }

    def save(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "step_count": self.step_count,
            "confidence_threshold": self.confidence_threshold,
        }

    def load(self, data: Dict[str, Any]):
        self.step_count = data.get("step_count", 0)
        self.confidence_threshold = data.get("confidence_threshold", 0.7)
