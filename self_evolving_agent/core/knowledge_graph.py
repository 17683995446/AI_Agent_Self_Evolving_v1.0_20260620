"""
知识图谱与语义推理 —— 实体关系建模、路径查询、因果推理

设计原则（第一性原理）：
1. 连接即智能：知识的价值在于实体间的关联
2. 图即记忆：知识图谱是系统的外化记忆网络
3. 推理即导航：从已知节点沿边到达未知结论
4. 持续生长：新实体和新关系不断加入
"""

import os
import sys
import time
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


@dataclass
class Entity:
    """实体节点"""
    entity_id: str
    entity_type: str
    name: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "name": self.name,
            "properties": self.properties,
        }


@dataclass
class Relation:
    """关系边"""
    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "properties": self.properties,
        }


class KnowledgeGraph:
    """知识图谱 —— 管理实体和关系"""

    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}
        self._adj_out: Dict[str, List[str]] = defaultdict(list)  # source -> [relation_id]
        self._adj_in: Dict[str, List[str]] = defaultdict(list)   # target -> [relation_id]
        self._type_index: Dict[str, Set[str]] = defaultdict(set) # entity_type -> {entity_id}

    def add_entity(self, entity: Entity) -> bool:
        """添加实体"""
        if entity.entity_id in self._entities:
            return False
        self._entities[entity.entity_id] = entity
        self._type_index[entity.entity_type].add(entity.entity_id)
        logger.debug("KG", f"实体添加: {entity.entity_id} [{entity.entity_type}]", module="knowledge_graph")
        return True

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        return self._entities.get(entity_id)

    def add_relation(self, relation: Relation) -> bool:
        """添加关系"""
        if relation.relation_id in self._relations:
            return False
        if relation.source_id not in self._entities or relation.target_id not in self._entities:
            return False
        self._relations[relation.relation_id] = relation
        self._adj_out[relation.source_id].append(relation.relation_id)
        self._adj_in[relation.target_id].append(relation.relation_id)
        logger.debug("KG", f"关系添加: {relation.source_id} -[{relation.relation_type}]-> {relation.target_id}",
                     module="knowledge_graph")
        return True

    def get_neighbors(self, entity_id: str, relation_type: Optional[str] = None,
                      direction: str = "out") -> List[Tuple[Entity, Relation]]:
        """获取邻居实体"""
        result = []
        rel_ids = self._adj_out.get(entity_id, []) if direction == "out" else self._adj_in.get(entity_id, [])
        for rid in rel_ids:
            rel = self._relations.get(rid)
            if not rel:
                continue
            if relation_type and rel.relation_type != relation_type:
                continue
            neighbor_id = rel.target_id if direction == "out" else rel.source_id
            neighbor = self._entities.get(neighbor_id)
            if neighbor:
                result.append((neighbor, rel))
        return result

    def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> Optional[List[Relation]]:
        """查找两实体间的最短路径（BFS）"""
        if start_id not in self._entities or end_id not in self._entities:
            return None

        visited = {start_id}
        queue = deque([(start_id, [])])

        while queue:
            current_id, path = queue.popleft()
            if len(path) >= max_depth:
                continue

            for rel_id in self._adj_out.get(current_id, []):
                rel = self._relations.get(rel_id)
                if not rel:
                    continue
                next_id = rel.target_id
                if next_id == end_id:
                    return path + [rel]
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [rel]))

        return None

    def find_causes(self, entity_id: str, relation_type: str = "causes",
                    max_depth: int = 3) -> List[List[Relation]]:
        """查找导致某实体的所有原因路径"""
        causes = []
        visited = set()

        def dfs(current_id: str, path: List[Relation], depth: int):
            if depth > max_depth:
                return
            for rel_id in self._adj_in.get(current_id, []):
                rel = self._relations.get(rel_id)
                if not rel or rel.relation_type != relation_type:
                    continue
                if rel.source_id in visited:
                    continue
                visited.add(rel.source_id)
                new_path = [rel] + path
                causes.append(new_path)
                dfs(rel.source_id, new_path, depth + 1)
                visited.discard(rel.source_id)

        dfs(entity_id, [], 0)
        return causes

    def infer_relation(self, entity_a: str, entity_b: str) -> Optional[str]:
        """推断两实体间可能的关系（通过共同邻居）"""
        neighbors_a = set()
        for rel_id in self._adj_out.get(entity_a, []):
            rel = self._relations.get(rel_id)
            if rel:
                neighbors_a.add(rel.target_id)

        neighbors_b = set()
        for rel_id in self._adj_in.get(entity_b, []):
            rel = self._relations.get(rel_id)
            if rel:
                neighbors_b.add(rel.source_id)

        common = neighbors_a & neighbors_b
        if common:
            return f"indirect_via_{len(common)}"
        return None

    def query_by_type(self, entity_type: str) -> List[Entity]:
        """按类型查询实体"""
        return [self._entities[eid] for eid in self._type_index.get(entity_type, [])]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "entities": len(self._entities),
            "relations": len(self._relations),
            "entity_types": len(self._type_index),
        }


class SemanticReasoner:
    """语义推理器 —— 基于知识图谱进行推理"""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def transitive_inference(self, entity_a: str, relation_type: str) -> Set[str]:
        """传递闭包推理：A->B, B->C 则 A->C"""
        result = set()
        visited = set()
        queue = deque([entity_a])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for neighbor, rel in self.kg.get_neighbors(current, relation_type, "out"):
                if neighbor.entity_id not in visited:
                    result.add(neighbor.entity_id)
                    queue.append(neighbor.entity_id)

        return result

    def pattern_match(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """模式匹配查询"""
        results = []
        entity_type = pattern.get("entity_type")
        prop_key = pattern.get("property_key")
        prop_value = pattern.get("property_value")

        candidates = self.kg.query_by_type(entity_type) if entity_type else list(self.kg._entities.values())

        for entity in candidates:
            if prop_key is not None:
                if entity.properties.get(prop_key) != prop_value:
                    continue
            results.append(entity.to_dict())

        return results

    def explain(self, entity_id: str) -> Dict[str, Any]:
        """解释某实体的来源和关联"""
        entity = self.kg.get_entity(entity_id)
        if not entity:
            return {}

        incoming = self.kg.get_neighbors(entity_id, direction="in")
        outgoing = self.kg.get_neighbors(entity_id, direction="out")

        return {
            "entity": entity.to_dict(),
            "caused_by": [{"from": e.name, "relation": r.relation_type} for e, r in incoming],
            "causes": [{"to": e.name, "relation": r.relation_type} for e, r in outgoing],
        }
