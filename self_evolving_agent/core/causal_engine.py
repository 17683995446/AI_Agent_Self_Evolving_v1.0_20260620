"""
阶段40：因果推断引擎

基于第一性原理：
- 相关不等于因果：统计关联可能由混杂因素导致
- do-演算：干预（intervention）与观察（observation）的区别
- 因果图：用有向无环图（DAG）表示变量间的因果关系

核心设计：
1. CausalVariable: 因果变量定义
2. CausalGraph: 因果图（DAG）构建与操作
3. CausalDiscovery: 因果发现算法（基于约束/分数）
4. DoCalculus: do-演算引擎（P(y|do(x))）
5. BackdoorCriterion: 后门准则调整
6. CausalInferenceController: 高层封装

接口契约：
- 变量名: str
- 因果效应: float (ATE: Average Treatment Effect)
- 因果图: Dict[str, List[str]] 邻接表
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import itertools

logger = logging.getLogger(__name__)


@dataclass
class CausalVariable:
    """因果变量"""
    name: str
    domain: Optional[List] = None
    is_treatment: bool = False
    is_outcome: bool = False
    is_confounder: bool = False


class CausalGraph:
    """
    因果图（有向无环图DAG）
    表示变量间的因果关系
    """

    def __init__(self):
        self.nodes: Dict[str, CausalVariable] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)  # parent -> children
        self._parents: Dict[str, Set[str]] = defaultdict(set)
        logger.info("CausalGraph initialized")

    def add_variable(self, name: str, **kwargs):
        self.nodes[name] = CausalVariable(name=name, **kwargs)

    def add_edge(self, from_var: str, to_var: str):
        if from_var not in self.nodes:
            self.add_variable(from_var)
        if to_var not in self.nodes:
            self.add_variable(to_var)
        self.edges[from_var].append(to_var)
        self._parents[to_var].add(from_var)
        logger.info("Added edge: %s -> %s", from_var, to_var)

    def get_parents(self, var: str) -> Set[str]:
        return self._parents[var].copy()

    def get_children(self, var: str) -> List[str]:
        return self.edges[var].copy()

    def get_ancestors(self, var: str) -> Set[str]:
        """获取变量的所有祖先"""
        ancestors = set()
        queue = deque([var])
        visited = {var}
        while queue:
            current = queue.popleft()
            for parent in self.get_parents(current):
                if parent not in visited:
                    visited.add(parent)
                    ancestors.add(parent)
                    queue.append(parent)
        return ancestors

    def get_descendants(self, var: str) -> Set[str]:
        """获取变量的所有后代"""
        descendants = set()
        queue = deque([var])
        visited = {var}
        while queue:
            current = queue.popleft()
            for child in self.get_children(current):
                if child not in visited:
                    visited.add(child)
                    descendants.add(child)
                    queue.append(child)
        return descendants

    def is_dag(self) -> bool:
        """检查是否为有向无环图"""
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for child in self.edges.get(node, []):
                if child not in visited:
                    if dfs(child):
                        return True
                elif child in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    return False
        return True

    def get_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """
        获取从treatment到outcome的所有后门路径
        后门路径：以指向treatment的箭头开始的路径
        """
        paths = []
        # 找到所有以指向treatment的节点为起点的路径
        for parent in self.get_parents(treatment):
            # 从parent到outcome的所有简单路径
            self._find_paths(parent, outcome, [treatment], paths)
        return paths

    def _find_paths(self, current: str, target: str, visited: List[str], paths: List[List[str]]):
        if current == target:
            paths.append(visited.copy())
            return
        if current in visited:
            return
        visited.append(current)
        for child in self.edges.get(current, []):
            self._find_paths(child, target, visited, paths)
        for parent in self.get_parents(current):
            self._find_paths(parent, target, visited, paths)
        visited.pop()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.nodes.keys()),
            "edges": {k: v for k, v in self.edges.items()},
        }


class CausalDiscovery:
    """
    因果发现算法
    从观测数据中推断因果结构
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        logger.info("CausalDiscovery initialized: alpha=%.3f", alpha)

    def partial_correlation(self, x: np.ndarray, y: np.ndarray,
                           z: Optional[np.ndarray] = None) -> float:
        """计算偏相关系数"""
        # 零方差检查
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        if z is None or len(z) == 0:
            corr = np.corrcoef(x, y)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0

        # 多元回归残差
        z = np.atleast_2d(z)
        if z.shape[0] != len(x):
            z = z.T

        # x对z回归
        beta_x = np.linalg.lstsq(z, x, rcond=None)[0]
        resid_x = x - z @ beta_x

        # y对z回归
        beta_y = np.linalg.lstsq(z, y, rcond=None)[0]
        resid_y = y - z @ beta_y

        return float(np.corrcoef(resid_x, resid_y)[0, 1])

    def pc_skeleton(self, data: Dict[str, np.ndarray],
                    var_names: List[str]) -> Set[Tuple[str, str]]:
        """
        PC算法第一阶段：构建骨架（无向图）
        基于条件独立性检验
        """
        n_vars = len(var_names)
        # 初始化完全图
        adj = {v: set(var_names) - {v} for v in var_names}

        depth = 0
        while True:
            removed = False
            for x in var_names:
                for y in list(adj[x]):
                    if y not in adj[x]:
                        continue
                    # 寻找条件集
                    neighbors = list(adj[x] - {y})
                    if len(neighbors) < depth:
                        continue
                    for cond in itertools.combinations(neighbors, depth):
                        cond_vars = [data[v] for v in cond]
                        z = np.vstack(cond_vars).T if cond_vars else None
                        pc = self.partial_correlation(data[x], data[y], z)
                        # 简化独立性检验：|偏相关| < threshold 认为独立
                        if abs(pc) < 0.1:  # 简化阈值
                            adj[x].discard(y)
                            adj[y].discard(x)
                            removed = True
                            break
                    if removed:
                        break
                if removed:
                    break
            if not removed:
                break
            depth += 1

        edges = set()
        for x in var_names:
            for y in adj[x]:
                if (y, x) not in edges:
                    edges.add((x, y))
        return edges

    def discover(self, data: Dict[str, np.ndarray],
                 var_names: List[str]) -> CausalGraph:
        """
        简化版因果发现
        返回发现的因果图（实际PC算法更复杂，此处为概念实现）
        """
        graph = CausalGraph()
        for v in var_names:
            graph.add_variable(v)

        # 基于偏相关构建骨架
        skeleton = self.pc_skeleton(data, var_names)

        # 简化定向：基于时间顺序或启发式规则
        # 实际应使用PC算法的定向规则（v-structure识别等）
        for x, y in skeleton:
            # 简化：基于相关性强弱定向（相关性强的作为原因）
            corr_x = np.mean([abs(np.corrcoef(data[x], data[z])[0, 1])
                              for z in var_names if z != x])
            corr_y = np.mean([abs(np.corrcoef(data[y], data[z])[0, 1])
                              for z in var_names if z != y])
            if corr_x >= corr_y:
                graph.add_edge(x, y)
            else:
                graph.add_edge(y, x)

        logger.info("Causal discovery completed: %d nodes, %d edges",
                    len(graph.nodes), sum(len(v) for v in graph.edges.values()))
        return graph


class BackdoorCriterion:
    """
    后门准则
    识别有效的调整集以估计因果效应
    """

    def __init__(self, graph: CausalGraph):
        self.graph = graph
        logger.info("BackdoorCriterion initialized")

    def find_valid_adjustment_sets(self, treatment: str, outcome: str,
                                    max_size: int = 3) -> List[Set[str]]:
        """
        寻找满足后门准则的调整集
        条件：
        1. 调整集中的变量不能是treatment的后代
        2. 调整集能阻断所有后门路径
        """
        treatment_descendants = self.graph.get_descendants(treatment)
        treatment_descendants.add(treatment)

        candidates = [v for v in self.graph.nodes
                      if v not in treatment_descendants and v != outcome]

        valid_sets = []
        for size in range(min(max_size, len(candidates)) + 1):
            for subset in itertools.combinations(candidates, size):
                adjustment_set = set(subset)
                if self._satisfies_backdoor(treatment, outcome, adjustment_set):
                    valid_sets.append(adjustment_set)

        return valid_sets

    def _satisfies_backdoor(self, treatment: str, outcome: str,
                            adjustment_set: Set[str]) -> bool:
        """检查调整集是否满足后门准则
        后门准则：
        1. 调整集中没有treatment的后代
        2. 调整集阻断了所有从treatment到outcome的后门路径
        """
        # 条件1已在上层find_valid_adjustment_sets中保证
        # 条件2：检查调整集是否包含所有混杂因素（同时是treatment和outcome祖先的变量）
        treatment_ancestors = self.graph.get_ancestors(treatment)
        outcome_ancestors = self.graph.get_ancestors(outcome)
        confounders = treatment_ancestors & outcome_ancestors

        # 调整集必须包含所有混杂因素
        if not confounders.issubset(adjustment_set):
            return False

        return True

    def estimate_ate(self, data: Dict[str, np.ndarray],
                     treatment: str, outcome: str,
                     adjustment_set: Optional[Set[str]] = None) -> float:
        """
        使用后门调整估计平均处理效应（ATE）
        ATE = E[Y|do(X=1)] - E[Y|do(X=0)]
        简化实现：使用回归调整
        """
        if adjustment_set is None:
            valid_sets = self.find_valid_adjustment_sets(treatment, outcome)
            if not valid_sets:
                # 无混杂因素，直接计算差异
                treated = data[outcome][data[treatment] == 1]
                control = data[outcome][data[treatment] == 0]
                if len(treated) == 0 or len(control) == 0:
                    return 0.0  # 无数据，返回0
                return float(np.mean(treated) - np.mean(control))
            adjustment_set = valid_sets[0]

        # 构建设计矩阵
        X_treat = data[treatment]
        Y = data[outcome]

        if adjustment_set:
            Z = np.column_stack([data[v] for v in adjustment_set])
            # 包含治疗和协变量的回归
            design = np.column_stack([np.ones(len(X_treat)), X_treat, Z])
            beta = np.linalg.lstsq(design, Y, rcond=None)[0]
            ate = beta[1]
        else:
            design = np.column_stack([np.ones(len(X_treat)), X_treat])
            beta = np.linalg.lstsq(design, Y, rcond=None)[0]
            ate = beta[1]

        return float(ate)


class DoCalculus:
    """
    do-演算引擎
    计算干预分布 P(y|do(x))
    """

    def __init__(self, graph: CausalGraph):
        self.graph = graph
        logger.info("DoCalculus initialized")

    def identify(self, treatment: str, outcome: str) -> Optional[str]:
        """
        识别因果效应是否可识别
        简化版：如果存在后门路径且可调整，则可识别
        """
        backdoor = BackdoorCriterion(self.graph)
        valid_sets = backdoor.find_valid_adjustment_sets(treatment, outcome)
        if valid_sets:
            return "backdoor"
        # 实际还应检查front-door准则等
        return None

    def estimate_intervention(self, data: Dict[str, np.ndarray],
                              treatment: str, outcome: str,
                              treatment_value: Any = 1,
                              control_value: Any = 0) -> float:
        """
        估计干预效应 P(Y|do(X=x))
        使用后门调整
        """
        backdoor = BackdoorCriterion(self.graph)
        ate = backdoor.estimate_ate(data, treatment, outcome)
        # P(Y|do(X=1)) - P(Y|do(X=0)) = ATE
        baseline = float(np.mean(data[outcome][data[treatment] == control_value]))
        return baseline + ate * (treatment_value - control_value)


class CausalInferenceController:
    """
    因果推断控制器高层封装
    整合因果发现、后门调整、do-演算
    """

    def __init__(self):
        self.graph: Optional[CausalGraph] = None
        self.discovery = CausalDiscovery()
        self.backdoor: Optional[BackdoorCriterion] = None
        self.do_calculus: Optional[DoCalculus] = None
        self.inference_history: deque = deque(maxlen=1000)
        logger.info("CausalInferenceController initialized")

    def discover_graph(self, data: Dict[str, np.ndarray],
                       var_names: List[str]) -> CausalGraph:
        """从数据中发现因果图"""
        self.graph = self.discovery.discover(data, var_names)
        self.backdoor = BackdoorCriterion(self.graph)
        self.do_calculus = DoCalculus(self.graph)
        return self.graph

    def set_graph(self, graph: CausalGraph):
        """直接设置因果图"""
        self.graph = graph
        self.backdoor = BackdoorCriterion(graph)
        self.do_calculus = DoCalculus(graph)

    def estimate_causal_effect(self, data: Dict[str, np.ndarray],
                               treatment: str, outcome: str) -> Dict[str, Any]:
        """估计因果效应"""
        if self.graph is None:
            raise ValueError("Causal graph not set. Call discover_graph() or set_graph() first.")

        identification = self.do_calculus.identify(treatment, outcome)
        if identification is None:
            return {"identifiable": False, "reason": "No valid adjustment set found"}

        valid_sets = self.backdoor.find_valid_adjustment_sets(treatment, outcome)
        ate = self.backdoor.estimate_ate(data, treatment, outcome,
                                          valid_sets[0] if valid_sets else None)

        result = {
            "identifiable": True,
            "method": identification,
            "adjustment_set": list(valid_sets[0]) if valid_sets else [],
            "ate": ate,
        }
        self.inference_history.append(result)
        logger.info("Causal effect estimated: %s -> %s, ATE=%.4f", treatment, outcome, ate)
        return result

    def get_mediators(self, treatment: str, outcome: str) -> List[str]:
        """识别中介变量（在treatment到outcome路径上的变量）"""
        if self.graph is None:
            return []
        descendants = self.graph.get_descendants(treatment)
        ancestors = self.graph.get_ancestors(outcome)
        return [v for v in descendants if v in ancestors and v != outcome]

    def get_confounders(self, treatment: str, outcome: str) -> List[str]:
        """识别混杂因素（同时影响treatment和outcome的变量）"""
        if self.graph is None:
            return []
        treatment_ancestors = self.graph.get_ancestors(treatment)
        outcome_ancestors = self.graph.get_ancestors(outcome)
        return [v for v in treatment_ancestors if v in outcome_ancestors and v != treatment]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "n_nodes": len(self.graph.nodes) if self.graph else 0,
            "n_edges": sum(len(v) for v in self.graph.edges.values()) if self.graph else 0,
            "is_dag": self.graph.is_dag() if self.graph else False,
            "inference_count": len(self.inference_history),
        }

    def save(self) -> Dict[str, Any]:
        return {
            "graph": self.graph.to_dict() if self.graph else None,
            "inference_count": len(self.inference_history),
        }

    def load(self, data: Dict[str, Any]):
        if data.get("graph"):
            g = CausalGraph()
            for node in data["graph"].get("nodes", []):
                g.add_variable(node)
            for src, targets in data["graph"].get("edges", {}).items():
                for tgt in targets:
                    g.add_edge(src, tgt)
            self.set_graph(g)
