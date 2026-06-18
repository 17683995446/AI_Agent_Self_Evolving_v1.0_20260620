"""
知识图谱与语义推理测试 —— 验证实体管理、关系查询、路径查找、因果推理

测试前置：
1. 拆解KnowledgeGraph接口：add_entity/add_relation/get_neighbors/find_path/find_causes/infer_relation
2. 拆解SemanticReasoner接口：transitive_inference/pattern_match/explain
3. 边界条件：重复实体、不存在的实体、环路、无路径
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge_graph import KnowledgeGraph, SemanticReasoner, Entity, Relation


class TestKnowledgeGraph:
    """测试知识图谱"""

    def setup_method(self):
        self.kg = KnowledgeGraph()

    def test_add_entity(self):
        """添加实体"""
        e = Entity("e1", "server", "Server-A", {"cpu": 80})
        assert self.kg.add_entity(e) is True
        assert self.kg.get_entity("e1") is not None

    def test_add_duplicate_entity(self):
        """重复添加失败"""
        e = Entity("e1", "server", "Server-A")
        self.kg.add_entity(e)
        assert self.kg.add_entity(e) is False

    def test_add_relation(self):
        """添加关系"""
        self.kg.add_entity(Entity("e1", "server"))
        self.kg.add_entity(Entity("e2", "service"))
        r = Relation("r1", "e1", "e2", "hosts")
        assert self.kg.add_relation(r) is True

    def test_add_relation_missing_entity(self):
        """实体不存在时添加关系失败"""
        r = Relation("r1", "e1", "e2", "hosts")
        assert self.kg.add_relation(r) is False

    def test_get_neighbors(self):
        """获取邻居"""
        self.kg.add_entity(Entity("e1", "server"))
        self.kg.add_entity(Entity("e2", "service"))
        self.kg.add_entity(Entity("e3", "service"))
        self.kg.add_relation(Relation("r1", "e1", "e2", "hosts"))
        self.kg.add_relation(Relation("r2", "e1", "e3", "hosts"))

        neighbors = self.kg.get_neighbors("e1")
        assert len(neighbors) == 2

    def test_get_neighbors_filtered(self):
        """按关系类型过滤邻居"""
        self.kg.add_entity(Entity("e1", "server"))
        self.kg.add_entity(Entity("e2", "service"))
        self.kg.add_entity(Entity("e3", "service"))
        self.kg.add_relation(Relation("r1", "e1", "e2", "hosts"))
        self.kg.add_relation(Relation("r2", "e1", "e3", "depends"))

        neighbors = self.kg.get_neighbors("e1", relation_type="hosts")
        assert len(neighbors) == 1
        assert neighbors[0][0].entity_id == "e2"

    def test_find_path(self):
        """查找路径"""
        self.kg.add_entity(Entity("a", "node"))
        self.kg.add_entity(Entity("b", "node"))
        self.kg.add_entity(Entity("c", "node"))
        self.kg.add_relation(Relation("r1", "a", "b", "connects"))
        self.kg.add_relation(Relation("r2", "b", "c", "connects"))

        path = self.kg.find_path("a", "c")
        assert path is not None
        assert len(path) == 2

    def test_find_path_no_path(self):
        """无路径时返回None"""
        self.kg.add_entity(Entity("a", "node"))
        self.kg.add_entity(Entity("b", "node"))
        assert self.kg.find_path("a", "b") is None

    def test_find_path_max_depth(self):
        """超过最大深度返回None"""
        self.kg.add_entity(Entity("a", "node"))
        self.kg.add_entity(Entity("b", "node"))
        self.kg.add_entity(Entity("c", "node"))
        self.kg.add_relation(Relation("r1", "a", "b", "connects"))
        self.kg.add_relation(Relation("r2", "b", "c", "connects"))

        assert self.kg.find_path("a", "c", max_depth=1) is None

    def test_find_causes(self):
        """查找原因"""
        self.kg.add_entity(Entity("cpu_high", "symptom"))
        self.kg.add_entity(Entity("traffic_spike", "cause"))
        self.kg.add_entity(Entity("ddos", "cause"))
        self.kg.add_relation(Relation("r1", "traffic_spike", "cpu_high", "causes"))
        self.kg.add_relation(Relation("r2", "ddos", "traffic_spike", "causes"))

        causes = self.kg.find_causes("cpu_high", "causes")
        assert len(causes) >= 1

    def test_infer_relation(self):
        """推断间接关系"""
        self.kg.add_entity(Entity("a", "node"))
        self.kg.add_entity(Entity("b", "node"))
        self.kg.add_entity(Entity("c", "node"))
        # a->c->b 形成间接关系
        self.kg.add_relation(Relation("r1", "a", "c", "connects"))
        self.kg.add_relation(Relation("r2", "c", "b", "connects"))

        result = self.kg.infer_relation("a", "b")
        assert result is not None
        assert "indirect" in result

    def test_query_by_type(self):
        """按类型查询"""
        self.kg.add_entity(Entity("s1", "server"))
        self.kg.add_entity(Entity("s2", "server"))
        self.kg.add_entity(Entity("d1", "database"))

        servers = self.kg.query_by_type("server")
        assert len(servers) == 2

    def test_stats(self):
        """统计信息"""
        self.kg.add_entity(Entity("e1", "server"))
        self.kg.add_entity(Entity("e2", "service"))
        self.kg.add_relation(Relation("r1", "e1", "e2", "hosts"))
        stats = self.kg.get_stats()
        assert stats["entities"] == 2
        assert stats["relations"] == 1


class TestSemanticReasoner:
    """测试语义推理器"""

    def setup_method(self):
        self.kg = KnowledgeGraph()
        self.reasoner = SemanticReasoner(self.kg)

    def test_transitive_inference(self):
        """传递闭包推理"""
        self.kg.add_entity(Entity("a", "node"))
        self.kg.add_entity(Entity("b", "node"))
        self.kg.add_entity(Entity("c", "node"))
        self.kg.add_entity(Entity("d", "node"))
        self.kg.add_relation(Relation("r1", "a", "b", "depends"))
        self.kg.add_relation(Relation("r2", "b", "c", "depends"))
        self.kg.add_relation(Relation("r3", "c", "d", "depends"))

        result = self.reasoner.transitive_inference("a", "depends")
        assert "b" in result
        assert "c" in result
        assert "d" in result

    def test_pattern_match(self):
        """模式匹配"""
        self.kg.add_entity(Entity("s1", "server", properties={"region": "us-east"}))
        self.kg.add_entity(Entity("s2", "server", properties={"region": "us-west"}))
        self.kg.add_entity(Entity("s3", "database", properties={"region": "us-east"}))

        results = self.reasoner.pattern_match({"entity_type": "server", "property_key": "region", "property_value": "us-east"})
        assert len(results) == 1
        assert results[0]["entity_id"] == "s1"

    def test_explain(self):
        """解释实体"""
        self.kg.add_entity(Entity("cpu_high", "symptom", "CPU高"))
        self.kg.add_entity(Entity("traffic", "cause", "流量大"))
        self.kg.add_relation(Relation("r1", "traffic", "cpu_high", "causes"))

        explanation = self.reasoner.explain("cpu_high")
        assert explanation["entity"]["name"] == "CPU高"
        assert len(explanation["caused_by"]) == 1

    def test_explain_missing(self):
        """解释不存在的实体"""
        assert self.reasoner.explain("none") == {}
