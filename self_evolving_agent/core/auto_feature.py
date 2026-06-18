"""
阶段46：自动特征工程与选择

基于第一性原理：
- 原始特征不一定是最优表征，组合变换可揭示隐藏模式
- 冗余特征增加噪声和计算成本，需筛选最有效子集
- 信息论度量（互信息）可量化特征与目标的关联强度

核心设计：
1. FeatureTransformer: 特征变换器（数学运算组合）
2. GeneticFeatureEngineer: 遗传编程特征构造
3. MutualInfoSelector: 互信息特征选择器
4. FeaturePipeline: 自动特征工程管道
5. AutoFeatureController: 高层封装

接口契约：
- 特征矩阵: np.ndarray, shape=(n_samples, n_features)
- 特征名: List[str]
- 重要性: Dict[str, float]
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from collections import deque
import itertools

logger = logging.getLogger(__name__)


@dataclass
class FeatureDef:
    """特征定义"""
    name: str
    source_features: List[str]  # 源特征名
    operation: str  # 运算类型
    formula: str  # 公式描述


class FeatureTransformer:
    """
    特征变换器
    对原始特征进行数学运算生成新特征
    """

    # 支持的二元运算
    BINARY_OPS = {
        'add': lambda a, b: a + b,
        'sub': lambda a, b: a - b,
        'mul': lambda a, b: a * b,
        'div': lambda a, b: a / (np.where(np.abs(b) < 1e-10, 1e-10, b)),
        'max': lambda a, b: np.maximum(a, b),
        'min': lambda a, b: np.minimum(a, b),
    }

    # 支持的一元运算
    UNARY_OPS = {
        'log': lambda a: np.log(np.abs(a) + 1e-10),
        'sqrt': lambda a: np.sqrt(np.abs(a)),
        'square': lambda a: np.clip(a, -1e154, 1e154) ** 2,
        'abs': lambda a: np.abs(a),
        'neg': lambda a: -a,
        'sigmoid': lambda a: 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500))),
        'tanh': lambda a: np.tanh(np.clip(a, -500, 500)),
    }

    def __init__(self):
        self.generated_features: List[FeatureDef] = []
        logger.info("FeatureTransformer initialized")

    def apply_binary(self, X: np.ndarray, feature_names: List[str],
                     i: int, j: int, op_name: str) -> Tuple[np.ndarray, FeatureDef]:
        """应用二元运算"""
        op = self.BINARY_OPS[op_name]
        result = op(X[:, i], X[:, j])
        name = f"{feature_names[i]}_{op_name}_{feature_names[j]}"
        feat_def = FeatureDef(
            name=name,
            source_features=[feature_names[i], feature_names[j]],
            operation=op_name,
            formula=f"{feature_names[i]} {op_name} {feature_names[j]}",
        )
        return result, feat_def

    def apply_unary(self, X: np.ndarray, feature_names: List[str],
                    i: int, op_name: str) -> Tuple[np.ndarray, FeatureDef]:
        """应用一元运算"""
        op = self.UNARY_OPS[op_name]
        result = op(X[:, i])
        name = f"{op_name}_{feature_names[i]}"
        feat_def = FeatureDef(
            name=name,
            source_features=[feature_names[i]],
            operation=op_name,
            formula=f"{op_name}({feature_names[i]})",
        )
        return result, feat_def

    def generate_all(self, X: np.ndarray, feature_names: List[str],
                     max_features: int = 50) -> Tuple[np.ndarray, List[str], List[FeatureDef]]:
        """生成所有可能的特征组合"""
        n_samples, n_features = X.shape
        new_features = []
        new_names = []
        defs = []

        # 一元变换
        for i in range(n_features):
            for op_name in self.UNARY_OPS:
                if len(new_features) >= max_features:
                    break
                result, feat_def = self.apply_unary(X, feature_names, i, op_name)
                if np.all(np.isfinite(result)):
                    new_features.append(result)
                    new_names.append(feat_def.name)
                    defs.append(feat_def)

        # 二元变换（仅前几个特征避免组合爆炸）
        n_pairs = min(n_features, 5)
        for i in range(n_pairs):
            for j in range(i + 1, n_pairs):
                for op_name in self.BINARY_OPS:
                    if len(new_features) >= max_features:
                        break
                    result, feat_def = self.apply_binary(X, feature_names, i, j, op_name)
                    if np.all(np.isfinite(result)):
                        new_features.append(result)
                        new_names.append(feat_def.name)
                        defs.append(feat_def)

        if new_features:
            new_X = np.column_stack(new_features)
        else:
            new_X = np.zeros((n_samples, 0))

        self.generated_features = defs
        logger.info("Generated %d new features", len(defs))
        return new_X, new_names, defs


class MutualInfoSelector:
    """
    互信息特征选择器
    基于互信息量选择与目标最相关的特征
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.feature_scores: Dict[str, float] = {}
        self.selected_features: List[str] = []
        logger.info("MutualInfoSelector initialized: n_bins=%d", n_bins)

    def _discretize(self, x: np.ndarray) -> np.ndarray:
        """离散化连续变量"""
        bins = np.linspace(x.min(), x.max() + 1e-10, self.n_bins + 1)
        return np.digitize(x, bins[:-1])

    def _entropy(self, x: np.ndarray) -> float:
        """计算熵"""
        _, counts = np.unique(x, return_counts=True)
        probs = counts / len(x)
        return float(-np.sum(probs * np.log2(probs + 1e-10)))

    def _conditional_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算条件熵 H(x|y)"""
        result = 0.0
        n = len(x)
        for y_val in np.unique(y):
            mask = y == y_val
            p_y = mask.sum() / n
            h_x_given_y = self._entropy(x[mask])
            result += p_y * h_x_given_y
        return result

    def compute_mutual_info(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算互信息 I(x;y) = H(x) - H(x|y)"""
        x_disc = self._discretize(x)
        y_disc = y if y.dtype == int else self._discretize(y)
        h_x = self._entropy(x_disc)
        h_x_given_y = self._conditional_entropy(x_disc, y_disc)
        return h_x - h_x_given_y

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
            n_select: int = 10) -> List[str]:
        """选择top-n特征"""
        scores = []
        for i in range(X.shape[1]):
            mi = self.compute_mutual_info(X[:, i], y)
            scores.append((feature_names[i], mi, i))
            self.feature_scores[feature_names[i]] = mi

        # 按互信息降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        self.selected_features = [s[0] for s in scores[:n_select]]
        logger.info("Selected %d features: %s", len(self.selected_features), self.selected_features[:5])
        return self.selected_features

    def transform(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """保留选中的特征"""
        indices = [feature_names.index(f) for f in self.selected_features if f in feature_names]
        return X[:, indices]

    def get_scores(self) -> Dict[str, float]:
        return self.feature_scores.copy()


class GeneticFeatureEngineer:
    """
    遗传编程特征构造
    通过进化搜索发现有用的特征组合
    """

    def __init__(self, population_size: int = 20, n_generations: int = 10,
                 mutation_rate: float = 0.3, max_depth: int = 3):
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.max_depth = max_depth
        self.transformer = FeatureTransformer()
        self.best_features: List[FeatureDef] = []
        self.fitness_history: deque = deque(maxlen=100)
        logger.info("GeneticFeatureEngineer initialized: pop=%d, gen=%d", population_size, n_generations)

    def _generate_individual(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, FeatureDef]:
        """随机生成一个特征（个体）"""
        n_features = X.shape[1]
        depth = np.random.randint(1, self.max_depth + 1)

        # 随机选择运算
        if depth == 1 or n_features == 1:
            # 一元运算
            i = np.random.randint(n_features)
            op_name = np.random.choice(list(self.transformer.UNARY_OPS.keys()))
            return self.transformer.apply_unary(X, feature_names, i, op_name)
        else:
            # 二元运算
            i, j = np.random.choice(n_features, 2, replace=False)
            op_name = np.random.choice(list(self.transformer.BINARY_OPS.keys()))
            return self.transformer.apply_binary(X, feature_names, i, j, op_name)

    def _evaluate_fitness(self, feature: np.ndarray, y: np.ndarray) -> float:
        """评估特征适应度（与目标的绝对相关）"""
        if np.std(feature) < 1e-10:
            return 0.0
        corr = abs(np.corrcoef(feature, y)[0, 1])
        return float(corr) if not np.isnan(corr) else 0.0

    def evolve(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[FeatureDef]:
        """执行进化搜索"""
        population = []
        for _ in range(self.population_size):
            try:
                feat, defn = self._generate_individual(X, feature_names)
                fitness = self._evaluate_fitness(feat, y)
                population.append((feat, defn, fitness))
            except Exception:
                continue

        for gen in range(self.n_generations):
            # 排序选择
            population.sort(key=lambda x: x[2], reverse=True)

            # 保留前50%
            survivors = population[:max(2, len(population) // 2)]

            # 生成后代
            new_population = list(survivors)
            while len(new_population) < self.population_size:
                try:
                    feat, defn = self._generate_individual(X, feature_names)
                    fitness = self._evaluate_fitness(feat, y)
                    new_population.append((feat, defn, fitness))
                except Exception:
                    continue

            population = new_population
            best_fitness = population[0][2] if population else 0.0
            self.fitness_history.append(best_fitness)
            logger.info("Gen %d: best_fitness=%.4f", gen + 1, best_fitness)

        # 最终排序
        population.sort(key=lambda x: x[2], reverse=True)
        self.best_features = [p[1] for p in population[:5]]
        return self.best_features


class FeaturePipeline:
    """
    自动特征工程管道
    串联变换 -> 选择 -> 评估
    """

    def __init__(self, max_generated: int = 50, n_select: int = 10):
        self.transformer = FeatureTransformer()
        self.selector = MutualInfoSelector()
        self.max_generated = max_generated
        self.n_select = n_select
        self.feature_defs: List[FeatureDef] = []
        self.selected_names: List[str] = []
        logger.info("FeaturePipeline initialized: max_gen=%d, n_select=%d", max_generated, n_select)

    def fit_transform(self, X: np.ndarray, y: np.ndarray,
                      feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """拟合管道并变换数据"""
        # 1. 生成新特征
        new_X, new_names, defs = self.transformer.generate_all(
            X, feature_names, max_features=self.max_generated)
        self.feature_defs = defs

        # 2. 合并原始+新特征
        if new_X.shape[1] > 0:
            combined_X = np.hstack([X, new_X])
            combined_names = feature_names + new_names
        else:
            combined_X = X
            combined_names = feature_names

        # 3. 特征选择
        self.selected_names = self.selector.fit(
            combined_X, y, combined_names, n_select=self.n_select)

        # 4. 变换
        result_X = self.selector.transform(combined_X, combined_names)
        logger.info("Pipeline: %d -> %d -> %d features",
                    X.shape[1], combined_X.shape[1], result_X.shape[1])
        return result_X, self.selected_names

    def transform(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """使用已拟合的管道变换新数据"""
        # 重新生成特征（与fit时相同的生成逻辑）
        new_X, new_names, _ = self.transformer.generate_all(
            X, feature_names, max_features=self.max_generated)
        if new_X.shape[1] > 0:
            combined_X = np.hstack([X, new_X])
            combined_names = feature_names + new_names
        else:
            combined_X = X
            combined_names = feature_names
        return self.selector.transform(combined_X, combined_names)

    def get_feature_importance(self) -> Dict[str, float]:
        return self.selector.get_scores()


class AutoFeatureController:
    """
    自动特征工程控制器高层封装
    """

    def __init__(self, max_generated: int = 50, n_select: int = 10,
                 use_genetic: bool = False,
                 genetic_generations: int = 5):
        self.max_generated = max_generated
        self.n_select = n_select
        self.use_genetic = use_genetic

        self.pipeline = FeaturePipeline(max_generated, n_select)
        self.genetic_engineer: Optional[GeneticFeatureEngineer] = None
        if use_genetic:
            self.genetic_engineer = GeneticFeatureEngineer(n_generations=genetic_generations)

        self.is_fitted = False
        self.original_feature_names: List[str] = []
        self.selected_feature_names: List[str] = []
        logger.info("AutoFeatureController initialized: genetic=%s", use_genetic)

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str] = None) -> Dict[str, Any]:
        """拟合自动特征工程"""
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        self.original_feature_names = feature_names

        # 管道特征工程
        transformed_X, selected_names = self.pipeline.fit_transform(X, y, feature_names)
        self.selected_feature_names = selected_names

        result = {
            "original_features": len(feature_names),
            "generated_features": len(self.pipeline.feature_defs),
            "selected_features": len(selected_names),
            "selected_names": selected_names,
            "feature_importance": self.pipeline.get_feature_importance(),
        }

        # 遗传特征工程（可选）
        if self.use_genetic and self.genetic_engineer:
            genetic_features = self.genetic_engineer.evolve(X, y, feature_names)
            result["genetic_features"] = [f.formula for f in genetic_features]
            result["genetic_best_fitness"] = list(self.genetic_engineer.fitness_history)[-1] if self.genetic_engineer.fitness_history else 0.0

        self.is_fitted = True
        logger.info("AutoFeature fit complete: %s", {k: v for k, v in result.items() if isinstance(v, (int, float))})
        return result

    def transform(self, X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """变换新数据"""
        if not self.is_fitted:
            raise ValueError("AutoFeatureController not fitted. Call fit() first.")
        if feature_names is None:
            feature_names = self.original_feature_names
        return self.pipeline.transform(X, feature_names)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_fitted": self.is_fitted,
            "original_features": len(self.original_feature_names),
            "selected_features": len(self.selected_feature_names),
            "use_genetic": self.use_genetic,
            "max_generated": self.max_generated,
        }

    def save(self) -> Dict[str, Any]:
        return {
            "is_fitted": self.is_fitted,
            "original_feature_names": self.original_feature_names,
            "selected_feature_names": self.selected_feature_names,
            "feature_importance": self.pipeline.get_feature_importance(),
            "max_generated": self.max_generated,
            "n_select": self.n_select,
            "use_genetic": self.use_genetic,
        }

    def load(self, data: Dict[str, Any]):
        self.is_fitted = data.get("is_fitted", False)
        self.original_feature_names = data.get("original_feature_names", [])
        self.selected_feature_names = data.get("selected_feature_names", [])
        self.max_generated = data.get("max_generated", 50)
        self.n_select = data.get("n_select", 10)
        self.use_genetic = data.get("use_genetic", False)
