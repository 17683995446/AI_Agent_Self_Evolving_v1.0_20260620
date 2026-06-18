"""
联邦学习与隐私计算测试 —— 验证差分隐私、联邦平均、安全聚合

测试前置：
1. 拆解DifferentialPrivacy接口：add_noise/clip_gradients
2. 拆解FederatedAveraging接口：initialize_model/aggregate/get_global_model
3. 拆解FederatedCoordinator接口：register_node/initialize_global_model/run_round
4. 边界条件：空更新、单节点、多节点、噪声影响
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.federated_learning import (
    DifferentialPrivacy, FederatedAveraging, FederatedLearningNode,
    FederatedCoordinator, SecureAggregator, ModelUpdate,
)


class TestDifferentialPrivacy:
    """测试差分隐私"""

    def setup_method(self):
        self.dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, sensitivity=1.0)

    def test_add_noise_changes_values(self):
        """噪声改变值"""
        values = [1.0, 2.0, 3.0]
        noised = self.dp.add_noise(values)
        assert noised != values

    def test_clip_gradients(self):
        """梯度裁剪"""
        grads = [3.0, 4.0]  # norm = 5
        clipped = self.dp.clip_gradients(grads, max_norm=1.0)
        import math
        norm = math.sqrt(sum(g * g for g in clipped))
        assert abs(norm - 1.0) < 0.01

    def test_clip_small_gradients_unchanged(self):
        """小梯度不裁剪"""
        grads = [0.3, 0.4]
        clipped = self.dp.clip_gradients(grads, max_norm=1.0)
        assert clipped == grads


class TestFederatedAveraging:
    """测试联邦平均"""

    def setup_method(self):
        self.fed_avg = FederatedAveraging()

    def test_initialize_model(self):
        """初始化模型"""
        weights = {"w1": [1.0, 2.0], "b1": [0.5]}
        self.fed_avg.initialize_model(weights)
        assert self.fed_avg.get_global_model() == weights

    def test_aggregate_single_update(self):
        """单更新聚合"""
        self.fed_avg.initialize_model({"w1": [0.0, 0.0]})
        update = ModelUpdate("a1", 1, {"w1": [1.0, 2.0]}, sample_count=100)
        result = self.fed_avg.aggregate([update])
        assert result["w1"] == [1.0, 2.0]

    def test_aggregate_multiple_updates(self):
        """多更新加权平均"""
        self.fed_avg.initialize_model({"w1": [0.0]})
        u1 = ModelUpdate("a1", 1, {"w1": [10.0]}, sample_count=100)
        u2 = ModelUpdate("a2", 1, {"w1": [20.0]}, sample_count=100)
        result = self.fed_avg.aggregate([u1, u2])
        assert result["w1"] == [15.0]  # 加权平均

    def test_aggregate_weighted(self):
        """不同样本数加权"""
        self.fed_avg.initialize_model({"w1": [0.0]})
        u1 = ModelUpdate("a1", 1, {"w1": [10.0]}, sample_count=100)
        u2 = ModelUpdate("a2", 1, {"w1": [20.0]}, sample_count=300)
        result = self.fed_avg.aggregate([u1, u2])
        expected = (10.0 * 100 + 20.0 * 300) / 400
        assert abs(result["w1"][0] - expected) < 0.01

    def test_aggregate_empty(self):
        """空更新返回原模型"""
        self.fed_avg.initialize_model({"w1": [1.0]})
        result = self.fed_avg.aggregate([])
        assert result["w1"] == [1.0]


class TestSecureAggregator:
    """测试安全聚合"""

    def test_generate_mask(self):
        """生成掩码"""
        sa = SecureAggregator(3)
        mask = sa.generate_mask("a1", ["w1"], {"w1": 5})
        assert "w1" in mask
        assert len(mask["w1"]) == 5

    def test_apply_mask(self):
        """应用掩码"""
        sa = SecureAggregator(2)
        weights = {"w1": [1.0, 2.0, 3.0]}
        mask = {"w1": [0.1, 0.2, 0.3]}
        masked = sa.apply_mask(weights, mask)
        assert masked["w1"] == [1.1, 2.2, 3.3]


class TestFederatedLearningNode:
    """测试联邦学习节点"""

    def test_local_train(self):
        """本地训练产生更新"""
        node = FederatedLearningNode("agent1")
        node.set_local_model({"w1": [0.0, 0.0]})
        update = node.local_train(data_samples=50)
        assert update.agent_id == "agent1"
        assert update.sample_count == 50
        assert "w1" in update.weights

    def test_apply_global_model(self):
        """应用全局模型"""
        node = FederatedLearningNode("agent1")
        node.set_local_model({"w1": [0.0]})
        node.apply_global_model({"w1": [5.0]})
        assert node.local_model["w1"] == [5.0]


class TestFederatedCoordinator:
    """测试联邦协调器"""

    def setup_method(self):
        self.coordinator = FederatedCoordinator()

    def test_register_and_initialize(self):
        """注册节点并初始化"""
        node1 = FederatedLearningNode("n1")
        node2 = FederatedLearningNode("n2")
        self.coordinator.register_node(node1)
        self.coordinator.register_node(node2)
        self.coordinator.initialize_global_model({"w1": [0.0, 0.0]})
        assert node1.local_model["w1"] == [0.0, 0.0]
        assert node2.local_model["w1"] == [0.0, 0.0]

    def test_run_round(self):
        """运行一轮联邦学习"""
        node1 = FederatedLearningNode("n1")
        node2 = FederatedLearningNode("n2")
        self.coordinator.register_node(node1)
        self.coordinator.register_node(node2)
        self.coordinator.initialize_global_model({"w1": [1.0]})

        result = self.coordinator.run_round()
        assert result["participants"] == 2
        assert result["round"] == 1
        assert "avg_loss" in result

    def test_multiple_rounds(self):
        """运行多轮联邦学习"""
        node = FederatedLearningNode("n1")
        self.coordinator.register_node(node)
        self.coordinator.initialize_global_model({"w1": [0.0]})

        for _ in range(3):
            self.coordinator.run_round()

        stats = self.coordinator.get_stats()
        assert stats["rounds"] == 3
        assert len(stats["history"]) == 3

    def test_stats(self):
        """统计信息"""
        self.coordinator.register_node(FederatedLearningNode("n1"))
        stats = self.coordinator.get_stats()
        assert stats["nodes"] == 1
        assert stats["rounds"] == 0
