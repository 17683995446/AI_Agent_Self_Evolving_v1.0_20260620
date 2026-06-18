"""
阶段37：强化学习策略优化（PPO/TRPO）

基于第一性原理：
- 策略梯度定理：智能体的目标是最大化期望累积回报
- 信任区域（Trust Region）：策略更新不能偏离旧策略太远，保证学习稳定性
- 裁剪目标（Clipped Objective）：PPO通过裁剪比率防止策略突变

核心设计：
1. PolicyNetwork: 简单神经网络策略（状态->动作概率分布）
2. ValueNetwork: 价值函数估计（状态->期望回报）
3. PPOTrainer: 近端策略优化训练器
4. TRPOTrainer: 信任区域策略优化训练器（简化版）
5. RolloutBuffer: 轨迹数据收集与GAE计算
6. PolicyOptimizer: 高层封装，自动选择PPO/TRPO

接口契约：
- 状态向量: np.ndarray, shape=(state_dim,)
- 动作: int (离散) 或 np.ndarray (连续)
- 策略输出: {"action": ..., "log_prob": ..., "value": ...}
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from collections import deque
import copy

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """单步转移数据"""
    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float = 0.0
    value: float = 0.0


@dataclass
class Rollout:
    """完整轨迹数据"""
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[Union[int, np.ndarray]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)


class PolicyNetwork:
    """
    策略网络：状态 -> 动作概率分布
    使用简单全连接网络实现，不依赖外部深度学习框架
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None,
                 discrete: bool = True, learning_rate: float = 3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.lr = learning_rate
        self.hidden_dims = hidden_dims or [64, 64]

        # 初始化网络参数 (Xavier初始化)
        self.params = self._init_params()
        self._grads = None
        logger.info("PolicyNetwork initialized: state_dim=%d, action_dim=%d, discrete=%s",
                    state_dim, action_dim, discrete)

    def _init_params(self) -> Dict[str, np.ndarray]:
        dims = [self.state_dim] + self.hidden_dims
        params = {}
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i + 1])
            params[f'W{i}'] = w
            params[f'b{i}'] = b
        # 输出层
        w_out = np.random.randn(dims[-1], self.action_dim) * np.sqrt(2.0 / dims[-1])
        b_out = np.zeros(self.action_dim)
        params['W_out'] = w_out
        params['b_out'] = b_out
        return params

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """前向传播，返回logits和中间激活值"""
        x = state
        activations = [x]
        for i in range(len(self.hidden_dims)):
            x = np.dot(x, self.params[f'W{i}']) + self.params[f'b{i}']
            x = self._relu(x)
            activations.append(x)
        logits = np.dot(x, self.params['W_out']) + self.params['b_out']
        activations.append(logits)
        return logits, activations

    def get_action(self, state: np.ndarray) -> Dict[str, Any]:
        """根据状态采样动作"""
        logits, _ = self.forward(state)
        if self.discrete:
            probs = self._softmax(logits)
            action = np.random.choice(self.action_dim, p=probs)
            log_prob = np.log(probs[action] + 1e-10)
            return {"action": int(action), "log_prob": float(log_prob), "probs": probs}
        else:
            # 连续动作：假设高斯分布，logits为均值
            std = 1.0
            action = logits + std * np.random.randn(self.action_dim)
            log_prob = -0.5 * np.sum((action - logits) ** 2) / (std ** 2) - 0.5 * self.action_dim * np.log(2 * np.pi * std ** 2)
            return {"action": action, "log_prob": float(log_prob), "mean": logits}

    def evaluate(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """评估状态-动作对的log_prob和熵"""
        logits, _ = self.forward(states)
        if self.discrete:
            probs = self._softmax(logits)
            log_probs = np.log(probs + 1e-10)
            action_log_probs = log_probs[np.arange(len(actions)), actions.astype(int)]
            entropy = -np.sum(probs * log_probs, axis=-1)
            return action_log_probs, entropy
        else:
            std = 1.0
            log_probs = -0.5 * np.sum((actions - logits) ** 2, axis=-1) / (std ** 2)
            entropy = 0.5 * self.action_dim * np.log(2 * np.pi * np.e * std ** 2)
            return log_probs, np.full(len(states), entropy)

    def _backward(self, states: np.ndarray, actions: np.ndarray, advantages: np.ndarray,
                  old_log_probs: Optional[np.ndarray] = None,
                  clip_epsilon: Optional[float] = None) -> Dict[str, np.ndarray]:
        """计算策略梯度 (简化版，使用数值梯度近似)
        若提供old_log_probs和clip_epsilon，使用PPO clipped surrogate梯度"""
        eps = 1e-5
        grads = {}
        for key in self.params:
            orig = self.params[key].copy()
            grad = np.zeros_like(orig)
            it = np.nditer(orig, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                self.params[key][idx] = orig[idx] + eps
                lp1, _ = self.evaluate(states, actions)
                if old_log_probs is not None and clip_epsilon is not None:
                    # PPO clipped surrogate loss
                    ratio1 = np.exp(np.clip(lp1 - old_log_probs, -20, 20))
                    surr1 = ratio1 * advantages
                    surr2 = np.clip(ratio1, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
                    loss1 = -np.mean(np.minimum(surr1, surr2))
                else:
                    loss1 = -np.mean(lp1 * advantages)

                self.params[key][idx] = orig[idx] - eps
                lp2, _ = self.evaluate(states, actions)
                if old_log_probs is not None and clip_epsilon is not None:
                    ratio2 = np.exp(np.clip(lp2 - old_log_probs, -20, 20))
                    surr1 = ratio2 * advantages
                    surr2 = np.clip(ratio2, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
                    loss2 = -np.mean(np.minimum(surr1, surr2))
                else:
                    loss2 = -np.mean(lp2 * advantages)

                grad[idx] = (loss1 - loss2) / (2 * eps)
                self.params[key][idx] = orig[idx]
                it.iternext()
            grads[key] = grad
        return grads

    def apply_gradients(self, grads: Dict[str, np.ndarray]):
        """应用梯度更新"""
        for key in self.params:
            if key in grads:
                self.params[key] -= self.lr * grads[key]

    def get_flat_params(self) -> np.ndarray:
        """将所有参数展平为一维向量"""
        return np.concatenate([self.params[k].flatten() for k in sorted(self.params.keys())])

    def set_flat_params(self, flat_params: np.ndarray):
        """从展平向量恢复参数"""
        offset = 0
        for key in sorted(self.params.keys()):
            size = self.params[key].size
            self.params[key] = flat_params[offset:offset + size].reshape(self.params[key].shape)
            offset += size

    def copy(self) -> 'PolicyNetwork':
        """深拷贝网络"""
        new_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dims.copy(),
                                self.discrete, self.lr)
        new_net.params = {k: v.copy() for k, v in self.params.items()}
        return new_net


class ValueNetwork:
    """价值网络：状态 -> 状态价值估计"""

    def __init__(self, state_dim: int, hidden_dims: List[int] = None, learning_rate: float = 1e-3):
        self.state_dim = state_dim
        self.lr = learning_rate
        self.hidden_dims = hidden_dims or [64, 64]
        self.params = self._init_params()
        logger.info("ValueNetwork initialized: state_dim=%d", state_dim)

    def _init_params(self) -> Dict[str, np.ndarray]:
        dims = [self.state_dim] + self.hidden_dims + [1]
        params = {}
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i + 1])
            params[f'W{i}'] = w
            params[f'b{i}'] = b
        return params

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def predict(self, state: np.ndarray):
        """预测状态价值（支持单状态和批量）"""
        x = state
        for i in range(len(self.hidden_dims)):
            x = np.dot(x, self.params[f'W{i}']) + self.params[f'b{i}']
            x = self._relu(x)
        v = np.dot(x, self.params[f'W{len(self.hidden_dims)}']) + self.params[f'b{len(self.hidden_dims)}']
        result = v.squeeze()
        # 单状态返回float，批量返回数组
        if result.ndim == 0:
            return float(result)
        return result

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """批量预测"""
        x = states
        for i in range(len(self.hidden_dims)):
            x = np.dot(x, self.params[f'W{i}']) + self.params[f'b{i}']
            x = self._relu(x)
        v = np.dot(x, self.params[f'W{len(self.hidden_dims)}']) + self.params[f'b{len(self.hidden_dims)}']
        return v.squeeze()

    def update(self, states: np.ndarray, targets: np.ndarray, epochs: int = 5) -> float:
        """使用梯度下降更新价值网络"""
        for _ in range(epochs):
            # 前向
            x = states
            activations = [x]
            for i in range(len(self.hidden_dims)):
                x = np.dot(x, self.params[f'W{i}']) + self.params[f'b{i}']
                x = self._relu(x)
                activations.append(x)
            values = np.dot(x, self.params[f'W{len(self.hidden_dims)}']) + self.params[f'b{len(self.hidden_dims)}']
            values = values.squeeze()

            # 反向传播 (MSE损失)
            loss = np.mean((values - targets) ** 2)
            dvalues = 2 * (values - targets) / len(states)

            # 输出层梯度
            dW = np.dot(activations[-1].T, dvalues.reshape(-1, 1)) / len(states)
            db = np.mean(dvalues)
            self.params[f'W{len(self.hidden_dims)}'] -= self.lr * dW
            self.params[f'b{len(self.hidden_dims)}'] -= self.lr * db

            # 隐藏层梯度 (简化，仅更新最后一层以保持稳定性)
        return float(loss)


class RolloutBuffer:
    """
    轨迹数据缓冲区
    收集环境交互数据，计算GAE优势函数
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollout = Rollout()
        logger.info("RolloutBuffer initialized: gamma=%.3f, gae_lambda=%.3f", gamma, gae_lambda)

    def add(self, state, action, reward, log_prob, value, done):
        self.rollout.add(state, action, reward, log_prob, value, done)

    def compute_advantages(self, last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用GAE计算优势函数和回报
        Returns: (advantages, returns)
        """
        rewards = np.array(self.rollout.rewards)
        values = np.array(self.rollout.values + [last_value])
        dones = np.array(self.rollout.dones)
        T = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            if dones[t]:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1.0 - float(dones[t]))
            advantages[t] = gae

        returns = advantages + values[:T]
        return advantages, returns

    def get_batch(self) -> Dict[str, np.ndarray]:
        """获取完整批次数据"""
        return {
            "states": np.array(self.rollout.states),
            "actions": np.array(self.rollout.actions),
            "rewards": np.array(self.rollout.rewards),
            "log_probs": np.array(self.rollout.log_probs),
            "values": np.array(self.rollout.values),
            "dones": np.array(self.rollout.dones),
        }

    def clear(self):
        self.rollout.clear()

    def __len__(self):
        return len(self.rollout)


class PPOTrainer:
    """
    近端策略优化训练器
    核心思想：clip(ratio, 1-eps, 1+eps) 限制策略更新幅度
    """

    def __init__(self, policy: PolicyNetwork, value_net: ValueNetwork,
                 clip_epsilon: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5,
                 ppo_epochs: int = 4, batch_size: int = 64):
        self.policy = policy
        self.value_net = value_net
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.old_policy = policy.copy()
        logger.info("PPOTrainer initialized: clip=%.3f, entropy_coef=%.3f", clip_epsilon, entropy_coef)

    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> Dict[str, float]:
        """使用PPO算法更新策略和价值网络"""
        advantages, returns = buffer.compute_advantages(last_value)
        batch = buffer.get_batch()
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]

        # 标准化优势
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(self.ppo_epochs):
            # 简化版：全量更新（小批量逻辑在完整实现中扩展）
            new_log_probs, entropy = self.policy.evaluate(states, actions)
            ratio = np.exp(np.clip(new_log_probs - old_log_probs, -20, 20))

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -np.mean(np.minimum(surr1, surr2))

            # 熵奖励
            entropy_bonus = -self.entropy_coef * np.mean(entropy)
            policy_loss += entropy_bonus

            # 策略梯度更新 (使用PPO clipped surrogate梯度)
            grads = self.policy._backward(states, actions, advantages,
                                          old_log_probs=old_log_probs,
                                          clip_epsilon=self.clip_epsilon)
            self.policy.apply_gradients(grads)

            # 价值网络更新
            value_loss = self.value_net.update(states, returns)

            total_policy_loss += float(policy_loss)
            total_value_loss += float(value_loss)
            total_entropy += float(np.mean(entropy))

        # 更新旧策略
        self.old_policy = self.policy.copy()

        n = self.ppo_epochs
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": float(np.mean(np.exp(np.clip(new_log_probs - old_log_probs, -20, 20)) - 1)),
        }


class TRPOTrainer:
    """
    信任区域策略优化训练器（简化版）
    核心思想：限制KL散度，在自然梯度方向更新
    由于TRPO需要二阶优化，此处提供概念实现
    """

    def __init__(self, policy: PolicyNetwork, value_net: ValueNetwork,
                 max_kl: float = 0.01, damping: float = 0.1, cg_iters: int = 10,
                 line_search_iters: int = 10):
        self.policy = policy
        self.value_net = value_net
        self.max_kl = max_kl
        self.damping = damping
        self.cg_iters = cg_iters
        self.line_search_iters = line_search_iters
        self.old_policy = policy.copy()  # 初始化old_policy避免AttributeError
        logger.info("TRPOTrainer initialized: max_kl=%.4f", max_kl)

    def _compute_kl(self, states: np.ndarray) -> float:
        """计算新旧策略的KL散度（简化）"""
        # 使用旧策略的均值作为参考（连续动作）或概率分布（离散动作）
        logits_old, _ = self.old_policy.forward(states)
        logits_new, _ = self.policy.forward(states)
        if self.policy.discrete:
            p_old = self.old_policy._softmax(logits_old)
            p_new = self.policy._softmax(logits_new)
            kl = np.mean(np.sum(p_old * (np.log(p_old + 1e-10) - np.log(p_new + 1e-10)), axis=-1))
        else:
            # 连续高斯，比较均值
            kl = np.mean(np.sum((logits_old - logits_new) ** 2, axis=-1))
        return float(kl)

    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> Dict[str, float]:
        """使用TRPO算法更新（简化实现，使用PPO作为实际回退）"""
        # 由于TRPO需要Hessian矩阵和共轭梯度，纯numpy实现较复杂
        # 此处采用PPO作为实际更新机制，但增加KL约束检查
        advantages, returns = buffer.compute_advantages(last_value)
        batch = buffer.get_batch()
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # 保存旧策略
        self.old_policy = self.policy.copy()
        old_params = self.policy.get_flat_params()

        # 计算策略梯度
        grads = self.policy._backward(states, actions, advantages)
        self.policy.apply_gradients(grads)

        # KL约束检查
        kl = self._compute_kl(states)
        if kl > self.max_kl * 2:
            # 回退到旧参数
            self.policy.set_flat_params(old_params)
            logger.warning("TRPO: KL divergence %.4f exceeds limit, rolling back", kl)

        # 更新价值网络
        value_loss = self.value_net.update(states, returns)

        return {
            "policy_loss": float(np.mean(-advantages)),
            "value_loss": float(value_loss),
            "kl_divergence": kl,
            "max_kl": self.max_kl,
        }


class PolicyOptimizer:
    """
    策略优化器高层封装
    自动选择PPO/TRPO，管理训练循环
    """

    def __init__(self, state_dim: int, action_dim: int, algorithm: str = "ppo",
                 discrete: bool = True, hidden_dims: List[int] = None,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm.lower()
        self.discrete = discrete

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims, discrete)
        self.value_net = ValueNetwork(state_dim, hidden_dims)
        self.buffer = RolloutBuffer(gamma, gae_lambda)

        if self.algorithm == "ppo":
            self.trainer = PPOTrainer(self.policy, self.value_net, **kwargs)
        elif self.algorithm == "trpo":
            self.trainer = TRPOTrainer(self.policy, self.value_net, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.step_count = 0
        self.update_count = 0
        self.episode_rewards: deque = deque(maxlen=100)
        logger.info("PolicyOptimizer initialized: algorithm=%s", algorithm)

    def select_action(self, state: np.ndarray) -> Dict[str, Any]:
        """选择动作并记录价值估计"""
        result = self.policy.get_action(state)
        value = self.value_net.predict(state)
        result["value"] = value
        return result

    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """存储转移数据"""
        self.buffer.add(state, action, reward, log_prob, value, done)
        self.step_count += 1

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """执行策略更新"""
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0}
        metrics = self.trainer.update(self.buffer, last_value)
        self.buffer.clear()
        self.update_count += 1
        logger.info("Policy update #%d: %s", self.update_count, metrics)
        return metrics

    def train_episode(self, env_step_fn: Callable, max_steps: int = 1000,
                      update_interval: int = 2048) -> Dict[str, Any]:
        """
        运行单轮训练
        env_step_fn: 接收action，返回(next_state, reward, done, info)
        """
        state = env_step_fn(None)[0]  # 假设传入None重置环境
        episode_reward = 0.0
        step = 0

        while step < max_steps:
            action_result = self.select_action(state)
            action = action_result["action"]
            log_prob = action_result["log_prob"]
            value = action_result["value"]

            next_state, reward, done, _ = env_step_fn(action)
            self.store_transition(state, action, reward, next_state, done, log_prob, value)
            episode_reward += reward
            state = next_state
            step += 1

            if len(self.buffer) >= update_interval or done:
                last_v = self.value_net.predict(next_state) if not done else 0.0
                metrics = self.update(last_v)

            if done:
                break

        self.episode_rewards.append(episode_reward)
        return {
            "episode_reward": episode_reward,
            "episode_length": step,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "step_count": self.step_count,
            "update_count": self.update_count,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "buffer_size": len(self.buffer),
        }

    def save(self) -> Dict[str, Any]:
        return {
            "policy_params": {k: v.tolist() for k, v in self.policy.params.items()},
            "value_params": {k: v.tolist() for k, v in self.value_net.params.items()},
            "step_count": self.step_count,
            "update_count": self.update_count,
            "episode_rewards": list(self.episode_rewards),
        }

    def load(self, data: Dict[str, Any]):
        for k, v in data.get("policy_params", {}).items():
            self.policy.params[k] = np.array(v)
        for k, v in data.get("value_params", {}).items():
            self.value_net.params[k] = np.array(v)
        self.step_count = data.get("step_count", 0)
        self.update_count = data.get("update_count", 0)
        self.episode_rewards = deque(data.get("episode_rewards", []), maxlen=100)
