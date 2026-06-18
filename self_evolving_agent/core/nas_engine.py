"""
阶段38：神经网络架构搜索(NAS)

基于第一性原理：
- 架构即假设空间：不同的网络结构对应不同的函数假设
- 搜索即优化：在离散架构空间中寻找最优结构
- 权重共享：不同架构共享权重，加速评估

核心设计：
1. ArchitectureSpec: 架构描述（层类型、维度、连接方式）
2. SearchSpace: 定义可搜索的架构空间
3. WeightSharingNAS: 权重共享的NAS引擎
4. ArchitectureEvaluator: 架构评估器（基于代理任务）
5. EvolutionarySearcher: 进化搜索策略
6. NASController: 高层封装，管理搜索-评估-进化全流程

接口契约：
- 架构编码: Dict[str, Any]，包含layers/connections/skips
- 评估分数: float，越高越好
- 搜索输出: 最优架构 + 评估历史
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import copy
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class LayerSpec:
    """单层规格"""
    layer_id: str
    layer_type: str  # "fc", "conv", "relu", "dropout", "skip"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchitectureSpec:
    """神经网络架构描述"""
    layers: List[LayerSpec] = field(default_factory=list)
    connections: List[Tuple[str, str]] = field(default_factory=list)
    input_dim: int = 0
    output_dim: int = 0

    def add_layer(self, layer_type: str, layer_id: str = None, **params) -> str:
        if layer_id is None:
            layer_id = f"{layer_type}_{len(self.layers)}"
        self.layers.append(LayerSpec(layer_id, layer_type, params))
        return layer_id

    def add_connection(self, from_id: str, to_id: str):
        self.connections.append((from_id, to_id))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layers": [{"id": l.layer_id, "type": l.layer_type, "params": l.params} for l in self.layers],
            "connections": self.connections,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }

    def hash(self) -> str:
        """生成架构唯一标识"""
        s = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()[:12]

    def copy(self) -> 'ArchitectureSpec':
        return ArchitectureSpec(
            layers=[LayerSpec(l.layer_id, l.layer_type, copy.deepcopy(l.params)) for l in self.layers],
            connections=self.connections.copy(),
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )


class SearchSpace:
    """
    定义架构搜索空间
    基于第一性原理：搜索空间 = 层类型 × 层维度 × 连接方式
    """

    def __init__(self, input_dim: int, output_dim: int,
                 layer_types: List[str] = None,
                 hidden_dims: List[int] = None,
                 max_layers: int = 5,
                 allow_skip: bool = True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_types = layer_types or ["fc", "relu", "dropout"]
        self.hidden_dims = hidden_dims or [16, 32, 64]
        self.max_layers = max_layers
        self.allow_skip = allow_skip
        logger.info("SearchSpace initialized: types=%s, dims=%s, max_layers=%d",
                    self.layer_types, self.hidden_dims, max_layers)

    def sample_random(self) -> ArchitectureSpec:
        """随机采样一个架构"""
        arch = ArchitectureSpec(input_dim=self.input_dim, output_dim=self.output_dim)
        prev_id = "input"
        arch.add_layer("input", "input", dim=self.input_dim)

        n_layers = np.random.randint(2, self.max_layers + 1)
        for i in range(n_layers):
            layer_type = np.random.choice(self.layer_types)
            if layer_type == "fc":
                dim = int(np.random.choice(self.hidden_dims))
                layer_id = arch.add_layer("fc", params={"dim": dim})
            elif layer_type == "dropout":
                rate = float(np.random.uniform(0.1, 0.5))
                layer_id = arch.add_layer("dropout", params={"rate": rate})
            else:
                layer_id = arch.add_layer(layer_type)

            arch.add_connection(prev_id, layer_id)
            prev_id = layer_id

            # 随机添加skip connection
            if self.allow_skip and i > 1 and np.random.random() < 0.3:
                skip_from = arch.layers[np.random.randint(0, i)].layer_id
                if skip_from != layer_id:
                    arch.add_connection(skip_from, layer_id)

        # 输出层
        arch.add_layer("output", "output", dim=self.output_dim)
        arch.add_connection(prev_id, "output")
        return arch

    def mutate(self, arch: ArchitectureSpec) -> ArchitectureSpec:
        """对架构进行变异"""
        new_arch = arch.copy()
        if len(new_arch.layers) <= 2:
            return new_arch

        mutation_type = np.random.choice(["add_layer", "remove_layer", "change_dim", "add_skip", "remove_skip"])

        if mutation_type == "add_layer" and len(new_arch.layers) < self.max_layers + 2:
            insert_pos = np.random.randint(1, len(new_arch.layers) - 1)
            layer_type = np.random.choice(self.layer_types)
            if layer_type == "fc":
                dim = int(np.random.choice(self.hidden_dims))
                new_id = new_arch.add_layer("fc", params={"dim": dim})
            elif layer_type == "dropout":
                rate = float(np.random.uniform(0.1, 0.5))
                new_id = new_arch.add_layer("dropout", params={"rate": rate})
            else:
                new_id = new_arch.add_layer(layer_type)
            # 简单插入连接
            prev = new_arch.layers[insert_pos - 1].layer_id
            nxt = new_arch.layers[insert_pos + 1].layer_id if insert_pos + 1 < len(new_arch.layers) else "output"
            new_arch.add_connection(prev, new_id)
            new_arch.add_connection(new_id, nxt)

        elif mutation_type == "remove_layer" and len(new_arch.layers) > 3:
            remove_idx = np.random.randint(1, len(new_arch.layers) - 1)
            removed_id = new_arch.layers[remove_idx].layer_id
            new_arch.layers.pop(remove_idx)
            new_arch.connections = [(a, b) for a, b in new_arch.connections
                                    if a != removed_id and b != removed_id]

        elif mutation_type == "change_dim":
            fc_layers = [l for l in new_arch.layers if l.layer_type == "fc"]
            if fc_layers:
                target = np.random.choice(fc_layers)
                target.params["dim"] = int(np.random.choice(self.hidden_dims))

        elif mutation_type == "add_skip" and self.allow_skip:
            candidates = [(a.layer_id, b.layer_id) for i, a in enumerate(new_arch.layers)
                          for j, b in enumerate(new_arch.layers) if i < j - 1]
            if candidates:
                a, b = candidates[np.random.randint(len(candidates))]
                if (a, b) not in new_arch.connections:
                    new_arch.add_connection(a, b)

        elif mutation_type == "remove_skip":
            skips = [c for c in new_arch.connections
                     if abs(next(i for i, l in enumerate(new_arch.layers) if l.layer_id == c[0])
                            - next(i for i, l in enumerate(new_arch.layers) if l.layer_id == c[1])) > 1]
            if skips:
                c = skips[np.random.randint(len(skips))]
                new_arch.connections.remove(c)

        return new_arch

    def crossover(self, arch1: ArchitectureSpec, arch2: ArchitectureSpec) -> ArchitectureSpec:
        """两个架构的交叉"""
        child = ArchitectureSpec(input_dim=self.input_dim, output_dim=self.output_dim)
        child.add_layer("input", "input", dim=self.input_dim)

        # 取前半部分来自arch1，后半部分来自arch2
        mid1 = len(arch1.layers) // 2
        mid2 = len(arch2.layers) // 2

        prev = "input"
        for i in range(1, mid1):
            if i < len(arch1.layers) - 1:
                l = arch1.layers[i]
                lid = child.add_layer(l.layer_type, l.layer_id, **copy.deepcopy(l.params))
                child.add_connection(prev, lid)
                prev = lid

        for i in range(mid2, len(arch2.layers) - 1):
            l = arch2.layers[i]
            lid = child.add_layer(l.layer_type, f"{l.layer_id}_c", **copy.deepcopy(l.params))
            child.add_connection(prev, lid)
            prev = lid

        child.add_layer("output", "output", dim=self.output_dim)
        child.add_connection(prev, "output")
        return child


class WeightSharingNAS:
    """
    权重共享NAS引擎
    核心思想：所有候选架构共享一个超网络的权重
    """

    def __init__(self, search_space: SearchSpace, state_dim: int, action_dim: int):
        self.search_space = search_space
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.shared_weights: Dict[str, np.ndarray] = {}
        self.architecture_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("WeightSharingNAS initialized")

    def _get_weight_key(self, layer_type: str, in_dim: int, out_dim: int) -> str:
        return f"{layer_type}_{in_dim}_{out_dim}"

    def _init_shared_weight(self, key: str, shape: Tuple[int, ...]):
        if key not in self.shared_weights:
            self.shared_weights[key] = np.random.randn(*shape) * np.sqrt(2.0 / shape[0])

    def get_weight(self, layer_type: str, in_dim: int, out_dim: int) -> np.ndarray:
        key = self._get_weight_key(layer_type, in_dim, out_dim)
        self._init_shared_weight(key, (in_dim, out_dim))
        return self.shared_weights[key]

    def evaluate_architecture(self, arch: ArchitectureSpec,
                              eval_fn: Callable[[ArchitectureSpec], float]) -> float:
        """评估架构，使用缓存避免重复评估"""
        h = arch.hash()
        if h in self.architecture_cache:
            logger.info("Architecture %s cache hit", h)
            return self.architecture_cache[h]["score"]
        score = eval_fn(arch)
        self.architecture_cache[h] = {"score": score, "arch": arch.to_dict()}
        logger.info("Architecture %s evaluated: score=%.4f", h, score)
        return score

    def get_top_architectures(self, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """获取评分最高的k个架构"""
        sorted_archs = sorted(self.architecture_cache.items(),
                              key=lambda x: x[1]["score"], reverse=True)
        return [(h, info["score"], info["arch"]) for h, info in sorted_archs[:k]]


class ArchitectureEvaluator:
    """
    架构评估器
    使用代理任务快速评估架构质量
    """

    def __init__(self, proxy_task_fn: Optional[Callable] = None,
                 eval_steps: int = 100):
        self.proxy_task_fn = proxy_task_fn
        self.eval_steps = eval_steps
        self.eval_history: deque = deque(maxlen=1000)
        logger.info("ArchitectureEvaluator initialized")

    def default_evaluate(self, arch: ArchitectureSpec) -> float:
        """
        默认评估函数：基于架构复杂度和连接性的启发式评分
        实际部署时替换为真实任务评估
        """
        score = 0.0
        n_layers = len(arch.layers)
        n_connections = len(arch.connections)

        # 复杂度惩罚（避免过深）
        score -= n_layers * 0.5

        # 连接奖励（适度连接性）
        score += min(n_connections, n_layers * 1.5) * 0.3

        # 隐藏层维度奖励（适中维度）
        for l in arch.layers:
            if l.layer_type == "fc":
                dim = l.params.get("dim", 32)
                score -= abs(dim - 48) * 0.02  # 偏好48维附近

        # Skip connection奖励
        skips = [c for c in arch.connections
                 if abs(next((i for i, la in enumerate(arch.layers) if la.layer_id == c[0]), 0)
                        - next((i for i, la in enumerate(arch.layers) if la.layer_id == c[1]), 0)) > 1]
        score += len(skips) * 0.5

        # 加少量随机噪声模拟真实评估波动
        score += np.random.randn() * 0.1

        return float(score)

    def evaluate(self, arch: ArchitectureSpec) -> float:
        if self.proxy_task_fn:
            score = self.proxy_task_fn(arch)
        else:
            score = self.default_evaluate(arch)
        self.eval_history.append({"arch_hash": arch.hash(), "score": score})
        return score


class EvolutionarySearcher:
    """
    进化搜索策略
    基于遗传算法在架构空间搜索
    """

    def __init__(self, search_space: SearchSpace,
                 population_size: int = 20,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.3,
                 elitism: int = 2):
        self.search_space = search_space
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.population: List[Tuple[ArchitectureSpec, float]] = []
        self.generation = 0
        logger.info("EvolutionarySearcher initialized: pop=%d", population_size)

    def initialize_population(self, evaluator: ArchitectureEvaluator):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            arch = self.search_space.sample_random()
            score = evaluator.evaluate(arch)
            self.population.append((arch, score))
        self.population.sort(key=lambda x: x[1], reverse=True)
        logger.info("Population initialized, best_score=%.4f", self.population[0][1])

    def evolve(self, evaluator: ArchitectureEvaluator) -> ArchitectureSpec:
        """进化一代"""
        new_population = self.population[:self.elitism]  # 保留精英

        while len(new_population) < self.population_size:
            if np.random.random() < self.crossover_rate and len(self.population) >= 2:
                p1 = self._select_parent()
                p2 = self._select_parent()
                child = self.search_space.crossover(p1, p2)
            else:
                parent = self._select_parent()
                child = self.search_space.mutate(parent) if np.random.random() < self.mutation_rate else parent.copy()

            score = evaluator.evaluate(child)
            new_population.append((child, score))

        new_population.sort(key=lambda x: x[1], reverse=True)
        self.population = new_population
        self.generation += 1
        logger.info("Generation %d: best=%.4f, mean=%.4f",
                    self.generation, self.population[0][1],
                    np.mean([s for _, s in self.population]))
        return self.population[0][0]

    def _select_parent(self) -> ArchitectureSpec:
        """锦标赛选择"""
        tournament_size = 3
        candidates = [self.population[np.random.randint(len(self.population))]
                      for _ in range(tournament_size)]
        return max(candidates, key=lambda x: x[1])[0]

    def get_best(self) -> Tuple[ArchitectureSpec, float]:
        return self.population[0]


class NASController:
    """
    NAS控制器高层封装
    管理搜索-评估-进化全流程
    """

    def __init__(self, state_dim: int, action_dim: int,
                 algorithm: str = "evolution",
                 population_size: int = 20,
                 generations: int = 10,
                 **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm
        self.generations = generations

        self.search_space = SearchSpace(state_dim, action_dim, **kwargs)
        self.nas_engine = WeightSharingNAS(self.search_space, state_dim, action_dim)
        self.evaluator = ArchitectureEvaluator()
        self.searcher = EvolutionarySearcher(self.search_space, population_size=population_size)

        self.best_architecture: Optional[ArchitectureSpec] = None
        self.best_score = -float('inf')
        self.search_history: List[Dict[str, Any]] = []
        logger.info("NASController initialized: algorithm=%s, generations=%d", algorithm, generations)

    def search(self) -> Dict[str, Any]:
        """执行架构搜索"""
        logger.info("Starting NAS search for %d generations", self.generations)
        self.searcher.initialize_population(self.evaluator)

        for gen in range(self.generations):
            best_arch = self.searcher.evolve(self.evaluator)
            best_score = self.searcher.get_best()[1]

            self.search_history.append({
                "generation": gen + 1,
                "best_score": best_score,
                "mean_score": np.mean([s for _, s in self.searcher.population]),
                "best_arch_hash": best_arch.hash(),
            })

            if best_score > self.best_score:
                self.best_score = best_score
                self.best_architecture = best_arch.copy()
                logger.info("New best architecture found: score=%.4f, hash=%s", best_score, best_arch.hash())

        logger.info("NAS search completed. Best score=%.4f", self.best_score)
        return {
            "best_architecture": self.best_architecture.to_dict() if self.best_architecture else None,
            "best_score": self.best_score,
            "history": self.search_history,
        }

    def evaluate_architecture(self, arch: ArchitectureSpec) -> float:
        """使用NAS引擎评估单个架构"""
        return self.nas_engine.evaluate_architecture(arch, self.evaluator.evaluate)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "generations_completed": len(self.search_history),
            "best_score": self.best_score,
            "population_size": self.searcher.population_size,
            "evaluated_architectures": len(self.nas_engine.architecture_cache),
        }

    def save(self) -> Dict[str, Any]:
        return {
            "best_architecture": self.best_architecture.to_dict() if self.best_architecture else None,
            "best_score": self.best_score,
            "history": self.search_history,
            "stats": self.get_stats(),
        }

    def load(self, data: Dict[str, Any]):
        if data.get("best_architecture"):
            d = data["best_architecture"]
            self.best_architecture = ArchitectureSpec(
                layers=[LayerSpec(l["id"], l["type"], l["params"]) for l in d.get("layers", [])],
                connections=[tuple(c) for c in d.get("connections", [])],
                input_dim=d.get("input_dim", 0),
                output_dim=d.get("output_dim", 0),
            )
        self.best_score = data.get("best_score", -float('inf'))
        self.search_history = data.get("history", [])
