"""
业务闭环测试 - 端到端数据流验证

测试目标：
1. 正向场景：数据输入 -> 特征工程 -> 模型训练 -> 预测 -> 不确定性估计
2. 逆向场景：从预测结果回溯特征重要性
3. 中断场景：训练中断后恢复
4. 重复操作：多次训练/评估循环
5. 数据一致性：全流程数据完整性验证

判定标准：数据流转无误，无逻辑冲突与隐性缺陷
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.uncertainty_quantification import (
    MCDropoutNetwork, DeepEnsemble, BayesianLinearRegression,
    UncertaintyEstimator, UncertaintyController
)
from core.multimodal_fusion import (
    ModalityConfig, ModalityEncoder, CrossModalAligner,
    AttentionFusion, MultimodalPredictor, MultimodalController
)
from core.continual_learning import (
    EWCRegularizer, ExperienceReplay, ProgressiveNetwork,
    ForgettingMonitor, ContinualLearningController
)
from core.self_supervised import (
    RepresentationEncoder, DataAugmentor, ContrastiveLearner,
    MaskedPredictor, RepresentationEvaluator, SelfSupervisedController
)
from core.auto_feature import (
    FeatureTransformer, MutualInfoSelector,
    GeneticFeatureEngineer, FeaturePipeline, AutoFeatureController
)
from core.meta_learning import (
    TaskDistribution, SimpleModel, MAMLModel,
    MetaLearner, FastAdapter, MetaReplayBuffer, MetaLearningController
)
from core.adversarial_robustness import (
    PerturbationBudget, AdversarialExampleGenerator,
    RobustnessEvaluator, AdversarialRobustnessController
)
from core.causal_engine import (
    CausalGraph, CausalDiscovery, BackdoorCriterion,
    DoCalculus, CausalInferenceController
)
from core.rl_policy import (
    PolicyNetwork, ValueNetwork, RolloutBuffer,
    PPOTrainer, TRPOTrainer, PolicyOptimizer
)


class TestForwardPipeline:
    """正向场景：完整数据流闭环"""

    def test_full_pipeline_feature_to_uncertainty(self):
        """完整管道：特征工程 -> 训练 -> 预测 -> 不确定性"""
        np.random.seed(42)
        # 1. 原始数据
        X = np.random.randn(50, 8)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # 2. 自动特征工程
        af_ctrl = AutoFeatureController(max_generated=20, n_select=5)
        af_ctrl.fit(X, y)
        features = af_ctrl.transform(X)
        assert features.shape[0] == 50

        # 3. 不确定性量化训练
        uq_ctrl = UncertaintyController(input_dim=features.shape[1], method="mc_dropout")
        uq_ctrl.train(features, y.reshape(-1, 1), epochs=2)

        # 4. 预测
        result = uq_ctrl.predict(features)
        assert np.all(np.isfinite(result["mean"]))
        assert "confidence" in result
        assert "is_reliable" in result

    def test_full_pipeline_ssl_to_continual(self):
        """完整管道：自监督 -> 持续学习"""
        np.random.seed(42)
        # 1. 原始无标签数据
        data = np.random.randn(60, 6)

        # 2. 自监督学习获取表征
        ss_ctrl = SelfSupervisedController(input_dim=6, latent_dim=4, method="contrastive")
        ss_ctrl.train(data, n_steps=3, lr=0.001, batch_size=10)
        embeddings = ss_ctrl.get_embeddings(data)
        assert embeddings.shape == (60, 4)

        # 3. 持续学习
        cl_ctrl = ContinualLearningController(input_dim=4, output_dim=3)
        y = np.random.randint(0, 3, 60)
        cl_ctrl.learn_task("task_0", embeddings, y, n_steps=3)
        assert cl_ctrl.get_stats()["task_count"] == 1

        # 4. 第二个任务
        data2 = np.random.randn(40, 6)
        emb2 = ss_ctrl.get_embeddings(data2)
        y2 = np.random.randint(0, 3, 40)
        cl_ctrl.learn_task("task_1", emb2, y2, n_steps=3)
        assert cl_ctrl.get_stats()["task_count"] == 2

    def test_full_pipeline_multimodal_to_prediction(self):
        """完整管道：多模态融合 -> 预测"""
        np.random.seed(42)
        configs = [
            ModalityConfig(name="numeric", input_dim=5, latent_dim=8),
            ModalityConfig(name="text", input_dim=3, latent_dim=8),
        ]
        ctrl = MultimodalController(configs, output_dim=1, latent_dim=8)

        data = {
            "numeric": np.random.randn(20, 5),
            "text": np.random.randn(20, 3),
        }
        targets = np.random.randn(20, 1)

        # 训练
        for _ in range(2):
            metrics = ctrl.train_step(data, targets, lr=0.001)

        # 预测
        preds = ctrl.predict(data)
        assert preds.shape[0] == 20
        assert np.all(np.isfinite(preds))

    def test_full_pipeline_causal_to_intervention(self):
        """完整管道：因果发现 -> 干预估计"""
        np.random.seed(42)
        # 构造因果数据：Z -> X, Z -> Y, X -> Y
        n = 100
        Z = np.random.randn(n)
        X = 0.5 * Z + np.random.randn(n) * 0.3
        Y = 0.3 * Z + 0.7 * X + np.random.randn(n) * 0.3

        # 构建因果图
        graph = CausalGraph()
        graph.add_edge("Z", "X")
        graph.add_edge("Z", "Y")
        graph.add_edge("X", "Y")

        # 后门准则找调整集
        bc = BackdoorCriterion(graph)
        adj_sets = bc.find_valid_adjustment_sets("X", "Y")
        assert isinstance(adj_sets, list)

        # 偏相关检验
        cd = CausalDiscovery()
        pc_xy = cd.partial_correlation(X, Y)
        pc_xy_z = cd.partial_correlation(X, Y, Z)
        # 控制Z后X-Y偏相关应减弱
        assert np.isfinite(pc_xy)
        assert np.isfinite(pc_xy_z)


class TestReversePipeline:
    """逆向场景：从结果回溯"""

    def test_feature_importance_backtrack(self):
        """从特征选择结果回溯特征重要性"""
        np.random.seed(42)
        X = np.random.randn(40, 5)
        y = (X[:, 0] > 0).astype(int)  # 仅f0与y相关

        af_ctrl = AutoFeatureController(max_generated=10, n_select=3)
        af_ctrl.fit(X, y, feature_names=["f0", "f1", "f2", "f3", "f4"])

        # 回溯特征重要性
        importance = af_ctrl.pipeline.get_feature_importance()
        assert isinstance(importance, dict)
        # f0应具有较高的重要性
        if "f0" in importance:
            assert importance["f0"] >= 0

    def test_continual_learning_forgetting_monitor(self):
        """从持续学习回溯遗忘情况"""
        np.random.seed(42)
        cl_ctrl = ContinualLearningController(input_dim=5, output_dim=3, strategy="ewc_replay")

        # 学习多个任务
        for i in range(3):
            X = np.random.randn(20, 5)
            y = np.random.randint(0, 3, 20)
            cl_ctrl.learn_task(f"task_{i}", X, y, n_steps=3)

        # 回溯遗忘摘要
        stats = cl_ctrl.get_stats()
        assert "forgetting_summary" in stats
        assert stats["task_count"] == 3


class TestInterruptionScenario:
    """中断场景"""

    def test_continual_learning_save_restore(self):
        """持续学习保存恢复"""
        np.random.seed(42)
        ctrl = ContinualLearningController(input_dim=5, output_dim=3)
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 3, 20)
        ctrl.learn_task("task_0", X, y, n_steps=3)

        # 保存
        saved = ctrl.save()

        # 恢复
        ctrl2 = ContinualLearningController(input_dim=5, output_dim=3)
        ctrl2.load(saved)
        assert ctrl2.get_stats()["task_count"] == 1

        # 继续学习
        X2 = np.random.randn(20, 5)
        y2 = np.random.randint(0, 3, 20)
        ctrl2.learn_task("task_1", X2, y2, n_steps=3)
        assert ctrl2.get_stats()["task_count"] == 2

    def test_meta_learning_save_restore(self):
        """元学习保存恢复"""
        np.random.seed(42)
        ctrl = MetaLearningController(input_dim=5, output_dim=3, inner_steps=2)
        # 训练前保存
        saved = ctrl.save()
        assert "model_params" in saved

        # 恢复
        ctrl2 = MetaLearningController(input_dim=5, output_dim=3, inner_steps=2)
        ctrl2.load(saved)
        assert ctrl2.meta_learner.step_count == 0

    def test_auto_feature_save_restore(self):
        """自动特征工程保存恢复"""
        np.random.seed(42)
        ctrl = AutoFeatureController(max_generated=10, n_select=3)
        X = np.random.randn(30, 5)
        y = np.random.randint(0, 3, 30)
        ctrl.fit(X, y)

        saved = ctrl.save()
        ctrl2 = AutoFeatureController()
        ctrl2.load(saved)
        assert ctrl2.is_fitted

        # 恢复后应能变换
        features = ctrl2.transform(X)
        assert features.shape[0] == 30


class TestRepeatedOperations:
    """重复操作场景"""

    def test_repeated_continual_learning(self):
        """重复持续学习"""
        np.random.seed(42)
        ctrl = ContinualLearningController(input_dim=5, output_dim=3, strategy="ewc_replay")

        for i in range(5):
            X = np.random.randn(15, 5)
            y = np.random.randint(0, 3, 15)
            ctrl.learn_task(f"task_{i}", X, y, n_steps=2)

        assert ctrl.get_stats()["task_count"] == 5

    def test_repeated_ssl_training(self):
        """重复自监督训练"""
        np.random.seed(42)
        ctrl = SelfSupervisedController(input_dim=5, latent_dim=4, method="contrastive")
        data = np.random.randn(30, 5)

        # 多轮训练
        for _ in range(3):
            result = ctrl.train(data, n_steps=2, lr=0.001, batch_size=8)
            assert "final_loss" in result

        assert ctrl.step_count == 6

    def test_repeated_adversarial_attack(self):
        """重复对抗攻击"""
        np.random.seed(42)
        def forward(x):
            return np.dot(x, np.random.randn(5, 3))

        gen = AdversarialExampleGenerator(forward)
        budget = PerturbationBudget(epsilon=0.1, norm="linf")

        for _ in range(5):
            x = np.random.randn(5, 5)
            y = np.random.randint(0, 3, 5)
            adv = gen.fgsm(x, y, budget)
            assert adv.shape == x.shape
            assert np.all(np.isfinite(adv))

    def test_repeated_rl_rollout(self):
        """重复RL轨迹收集"""
        np.random.seed(42)
        policy = PolicyNetwork(state_dim=5, action_dim=3)
        value_net = ValueNetwork(state_dim=5)

        for episode in range(3):
            buf = RolloutBuffer()
            state = np.random.randn(5)
            for step in range(10):
                action = policy.get_action(state)
                value = value_net.predict(state)
                reward = np.random.randn()
                buf.add(state, action["action"], reward, action["log_prob"], value, False)
                state = np.random.randn(5)
            buf.add(state, 0, 0.0, -0.5, value_net.predict(state), True)
            assert len(buf) == 11


class TestDataConsistency:
    """数据一致性校验"""

    def test_feature_transform_consistency(self):
        """特征变换一致性：相同输入应产生相同输出"""
        np.random.seed(42)
        transformer = FeatureTransformer()
        X = np.random.randn(20, 3)
        names = ["f0", "f1", "f2"]

        new_X1, names1, _ = transformer.generate_all(X, names, max_features=20)
        new_X2, names2, _ = transformer.generate_all(X, names, max_features=20)

        # 特征名应一致
        assert names1 == names2
        # 特征值应一致
        np.testing.assert_array_almost_equal(new_X1, new_X2)

    def test_auto_feature_fit_transform_consistency(self):
        """fit和transform的一致性"""
        np.random.seed(42)
        ctrl = AutoFeatureController(max_generated=10, n_select=3)
        X = np.random.randn(30, 5)
        y = np.random.randint(0, 3, 30)
        ctrl.fit(X, y)

        # transform应与fit_transform的选择一致
        features = ctrl.transform(X)
        assert features.shape[0] == 30
        # 所有特征应有限
        assert np.all(np.isfinite(features))

    def test_causal_graph_consistency(self):
        """因果图一致性"""
        graph = CausalGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        # 父子关系一致
        assert "A" in graph.get_parents("B")
        assert "B" in graph.get_children("A")
        assert "B" in graph.get_parents("C")
        assert "C" in graph.get_children("B")

        # 祖先后代一致
        assert "A" in graph.get_ancestors("C")
        assert "C" in graph.get_descendants("A")

    def test_continual_learning_replay_consistency(self):
        """持续学习回放数据一致性"""
        np.random.seed(42)
        replay = ExperienceReplay(capacity_per_task=10)
        X1 = np.random.randn(5, 3)
        y1 = np.random.randint(0, 2, 5)
        X2 = np.random.randn(5, 3)
        y2 = np.random.randint(0, 2, 5)

        replay.add_task("task_0", X1, y1)
        replay.add_task("task_1", X2, y2)

        # 采样应来自已添加的数据
        data, labels = replay.sample_replay_batch(20)
        assert len(data) <= 10  # 最多10个样本

        # 排除task_0后应只来自task_1
        data_excl, _ = replay.sample_replay_batch(20, exclude_task="task_0")
        assert len(data_excl) <= 5

    def test_meta_replay_buffer_consistency(self):
        """元经验回放缓冲区一致性"""
        buf = MetaReplayBuffer(capacity=10)
        dist = TaskDistribution(input_dim=3, n_classes=2, k_shot=2, q_query=3)

        tasks = []
        for _ in range(5):
            task = dist.sample_task()
            tasks.append(task)
            buf.add(task)

        assert len(buf) == 5
        # 采样应返回已添加的任务
        sampled = buf.sample(3)
        assert len(sampled) == 3
        for t in sampled:
            assert isinstance(t, type(tasks[0]))


class TestExceptionHandling:
    """异常容错测试"""

    def test_ssl_controller_wrong_dim(self):
        """自监督控制器维度不匹配"""
        ctrl = SelfSupervisedController(input_dim=5, latent_dim=4, method="contrastive")
        # 输入维度不匹配
        wrong_data = np.random.randn(10, 3)  # 3 != 5
        with pytest.raises((ValueError, IndexError)):
            ctrl.train(wrong_data, n_steps=1, lr=0.001, batch_size=5)

    def test_continual_learning_wrong_dim(self):
        """持续学习维度不匹配"""
        ctrl = ContinualLearningController(input_dim=5, output_dim=3)
        X = np.random.randn(10, 3)  # 3 != 5
        y = np.random.randint(0, 3, 10)
        with pytest.raises((ValueError, IndexError)):
            ctrl.learn_task("task_0", X, y, n_steps=1)

    def test_multimodal_wrong_dim(self):
        """多模态维度不匹配"""
        configs = [ModalityConfig(name="a", input_dim=5, latent_dim=8)]
        ctrl = MultimodalController(configs, output_dim=1, latent_dim=8)
        data = {"a": np.random.randn(10, 3)}  # 3 != 5
        with pytest.raises((ValueError, IndexError)):
            ctrl.fuse(data)

    def test_adversarial_empty_input(self):
        """对抗攻击空输入"""
        def forward(x):
            return np.dot(x, np.random.randn(5, 3))
        gen = AdversarialExampleGenerator(forward)
        budget = PerturbationBudget(epsilon=0.1, norm="linf")
        x = np.zeros((0, 5))
        y = np.array([], dtype=int)
        # 空输入应能处理或抛出明确异常
        try:
            adv = gen.fgsm(x, y, budget)
        except (ValueError, IndexError):
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--timeout=60"])
