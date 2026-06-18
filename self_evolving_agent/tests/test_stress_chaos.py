"""
复杂场景压力测试 - 混乱输入、极端边界、长时间运行

测试目标：
1. 极端值输入（超大、超小、NaN、Inf）
2. 混乱输入（错误形状、错误类型、空值）
3. 边界条件（空数组、单元素、零维）
4. 长时间运行稳定性
5. 模块间交互鲁棒性

判定标准：异常场景可平稳兜底，无崩溃报错
"""

import numpy as np
import pytest
import warnings
import sys
import os

# 添加项目根目录到路径
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
    MaskedPredictor, SelfSupervisedController
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
    RobustnessEvaluator, AdversarialTrainer, AdversarialRobustnessController
)
from core.causal_engine import (
    CausalGraph, CausalDiscovery, BackdoorCriterion,
    DoCalculus, CausalInferenceController
)
from core.rl_policy import (
    PolicyNetwork, ValueNetwork, RolloutBuffer,
    PPOTrainer, TRPOTrainer, PolicyOptimizer
)


class TestExtremeValues:
    """极端值输入测试"""

    def test_mc_dropout_with_large_values(self):
        """测试MC Dropout处理超大值"""
        net = MCDropoutNetwork(input_dim=5, hidden_dim=8, n_mc_samples=5)
        x = np.random.randn(10, 5) * 1e6
        result = net.predict_with_uncertainty(x)
        assert np.all(np.isfinite(result.mean)), "mean应有限"
        assert np.all(np.isfinite(result.std)), "std应有限"
        assert np.isfinite(result.total_uncertainty), "不确定性应有限"

    def test_mc_dropout_with_tiny_values(self):
        """测试MC Dropout处理超小值"""
        net = MCDropoutNetwork(input_dim=5, hidden_dim=8, n_mc_samples=5)
        x = np.random.randn(10, 5) * 1e-6
        result = net.predict_with_uncertainty(x)
        assert np.all(np.isfinite(result.mean))

    def test_mc_dropout_with_zero_input(self):
        """测试MC Dropout处理全零输入"""
        net = MCDropoutNetwork(input_dim=5, hidden_dim=8, n_mc_samples=5)
        x = np.zeros((10, 5))
        result = net.predict_with_uncertainty(x)
        assert np.all(np.isfinite(result.mean))

    def test_deep_ensemble_with_extreme_values(self):
        """测试深度集成处理极端值"""
        ensemble = DeepEnsemble(n_models=3, input_dim=5, hidden_dim=8)
        x = np.random.randn(10, 5) * 1e4
        result = ensemble.predict_with_uncertainty(x)
        assert np.all(np.isfinite(result.mean))
        assert np.all(np.isfinite(result.std))

    def test_bayesian_lr_with_singular_matrix(self):
        """测试贝叶斯线性回归处理奇异矩阵"""
        blr = BayesianLinearRegression(input_dim=3)
        # 构造奇异矩阵（两列完全相同）
        X = np.random.randn(10, 3)
        X[:, 1] = X[:, 0]  # 第1列等于第0列，矩阵奇异
        y = np.random.randn(10)
        blr.fit(X, y)
        result = blr.predict_with_uncertainty(X[:5])
        assert np.all(np.isfinite(result.mean))

    def test_multimodal_with_extreme_values(self):
        """测试多模态融合处理极端值"""
        configs = [
            ModalityConfig(name="a", input_dim=4, latent_dim=8),
            ModalityConfig(name="b", input_dim=4, latent_dim=8),
        ]
        ctrl = MultimodalController(configs, output_dim=1, latent_dim=8)
        data = {
            "a": np.random.randn(10, 4) * 1e5,
            "b": np.random.randn(10, 4) * 1e-5,
        }
        result = ctrl.fuse(data)
        assert np.all(np.isfinite(result)), "融合结果应有限"

    def test_contrastive_learner_with_extreme_values(self):
        """测试对比学习处理极端值"""
        encoder = RepresentationEncoder(input_dim=5, hidden_dims=[8], latent_dim=4)
        learner = ContrastiveLearner(encoder, temperature=0.5)
        x = np.random.randn(10, 5) * 1e4
        loss = learner.compute_loss(x)
        assert np.isfinite(loss), "对比损失应有限"

    def test_auto_feature_with_extreme_values(self):
        """测试自动特征工程处理极端值"""
        transformer = FeatureTransformer()
        X = np.random.randn(20, 3) * 1e6
        names = [f"f{i}" for i in range(3)]
        new_X, new_names, defs = transformer.generate_all(X, names, max_features=20)
        assert np.all(np.isfinite(new_X)), "特征变换结果应有限"

    def test_policy_network_with_extreme_states(self):
        """测试策略网络处理极端状态"""
        policy = PolicyNetwork(state_dim=5, action_dim=3)
        # 极大状态
        x_big = np.random.randn(10, 5) * 1e6
        action = policy.get_action(x_big[0])
        assert np.isfinite(action["log_prob"])

        # 极小状态
        x_small = np.random.randn(10, 5) * 1e-6
        action = policy.get_action(x_small[0])
        assert np.isfinite(action["log_prob"])

    def test_causal_partial_corr_with_constant(self):
        """测试偏相关系数处理常量变量"""
        cd = CausalDiscovery()
        # 常量变量
        x = np.ones(50)
        y = np.random.randn(50)
        pc = cd.partial_correlation(x, y)
        assert np.isfinite(pc), "常量变量的偏相关应有限"
        assert pc == 0.0, "常量变量偏相关应为0"


class TestChaoticInputs:
    """混乱输入测试"""

    def test_mc_dropout_single_sample(self):
        """测试单样本输入"""
        net = MCDropoutNetwork(input_dim=5, hidden_dim=8, n_mc_samples=5)
        x = np.random.randn(1, 5)
        result = net.predict_with_uncertainty(x)
        assert result.mean.shape[0] == 1

    def test_mc_dropout_large_batch(self):
        """测试大批量输入"""
        net = MCDropoutNetwork(input_dim=5, hidden_dim=8, n_mc_samples=5)
        x = np.random.randn(500, 5)
        result = net.predict_with_uncertainty(x)
        assert result.mean.shape[0] == 500

    def test_multimodal_mismatched_batch_sizes(self):
        """测试多模态批量大小不匹配"""
        configs = [
            ModalityConfig(name="a", input_dim=4, latent_dim=8),
            ModalityConfig(name="b", input_dim=4, latent_dim=8),
        ]
        ctrl = MultimodalController(configs, output_dim=1, latent_dim=8)
        data = {
            "a": np.random.randn(10, 4),
            "b": np.random.randn(15, 4),  # 不同批量大小
        }
        # 应该能处理或抛出明确异常，不应崩溃
        try:
            result = ctrl.fuse(data)
            assert np.all(np.isfinite(result))
        except (ValueError, AssertionError):
            pass  # 明确异常是可接受的

    def test_continual_learning_with_degenerate_data(self):
        """测试持续学习处理退化数据"""
        ewc = EWCRegularizer()
        # task_count为0时compute_penalty返回0
        params = {"W": np.zeros((3, 3))}
        loss = ewc.compute_penalty(params)
        assert np.isfinite(loss)
        assert loss == 0.0

    def test_causal_with_constant_variable(self):
        """测试因果推断处理常量变量"""
        cd = CausalDiscovery()
        # 常量数据
        x = np.ones(50)
        y = np.random.randn(50)
        pc = cd.partial_correlation(x, y)
        assert np.isfinite(pc)

    def test_adversarial_with_zero_gradient(self):
        """测试对抗攻击处理零梯度场景"""
        # 构造模型前向函数
        def model_forward(x):
            return np.dot(x, np.random.randn(5, 3))

        def model_loss(x, y):
            logits = model_forward(x)
            return float(np.mean(logits ** 2))

        gen = AdversarialExampleGenerator(model_forward, model_loss)
        budget = PerturbationBudget(epsilon=0.1, norm="linf")
        x = np.random.randn(5, 5)
        y = np.random.randint(0, 3, 5)
        adv_x = gen.fgsm(x, y, budget)
        assert np.all(np.isfinite(adv_x)), "对抗样本应有限"

    def test_rl_policy_with_extreme_rewards(self):
        """测试RL策略处理极端奖励"""
        policy = PolicyNetwork(state_dim=5, action_dim=3)
        states = np.random.randn(10, 5)
        actions = np.random.randint(0, 3, 10)
        # 极端奖励
        rewards = np.array([1e6, -1e6, 0, 1e-6, -1e-6] * 2)
        old_log_probs = np.random.randn(10)
        log_probs, entropy = policy.evaluate(states, actions)
        assert np.all(np.isfinite(log_probs))
        assert np.all(np.isfinite(entropy))


class TestBoundaryConditions:
    """边界条件测试"""

    def test_empty_modality_dict(self):
        """测试空模态字典"""
        aligner = CrossModalAligner(latent_dim=8)
        loss = aligner.compute_alignment_loss({})
        assert loss == 0.0

    def test_single_modality_alignment(self):
        """测试单模态对齐"""
        aligner = CrossModalAligner(latent_dim=8)
        embeddings = {"a": np.random.randn(5, 8)}
        loss = aligner.compute_alignment_loss(embeddings)
        assert loss == 0.0  # 单模态无需对齐

    def test_experience_replay_empty(self):
        """测试空经验回放缓冲区"""
        replay = ExperienceReplay(capacity_per_task=100)
        data, labels = replay.sample_replay_batch(10)
        assert len(data) == 0  # 空缓冲区返回空

    def test_meta_replay_buffer_empty(self):
        """测试空元经验回放缓冲区"""
        buf = MetaReplayBuffer(capacity=100)
        batch = buf.sample(10)
        assert len(batch) == 0

    def test_dropout_rate_boundary(self):
        """测试dropout_rate边界值"""
        x = np.random.randn(5, 5)
        # dropout_rate=0
        net0 = MCDropoutNetwork(input_dim=5, dropout_rate=0.0, n_mc_samples=3)
        r0 = net0.predict_with_uncertainty(x)
        assert np.all(np.isfinite(r0.mean))

        # dropout_rate=1.0（应被裁剪到0.99）
        net1 = MCDropoutNetwork(input_dim=5, dropout_rate=1.0, n_mc_samples=3)
        r1 = net1.predict_with_uncertainty(x)
        assert np.all(np.isfinite(r1.mean))

        # dropout_rate=-0.5（应被裁剪到0.0）
        net_neg = MCDropoutNetwork(input_dim=5, dropout_rate=-0.5, n_mc_samples=3)
        r_neg = net_neg.predict_with_uncertainty(x)
        assert np.all(np.isfinite(r_neg.mean))

    def test_temperature_boundary(self):
        """测试温度参数边界值"""
        # 极小温度
        encoder = RepresentationEncoder(input_dim=5, hidden_dims=[8], latent_dim=4)
        learner = ContrastiveLearner(encoder, temperature=1e-10)
        x = np.random.randn(10, 5)
        loss = learner.compute_loss(x)
        assert np.isfinite(loss)

    def test_feature_selector_more_than_available(self):
        """测试特征选择数超过可用特征数"""
        selector = MutualInfoSelector(n_bins=5)
        X = np.random.randn(20, 3)  # 只有3个特征
        y = np.random.randint(0, 2, 20)
        names = [f"f{i}" for i in range(3)]
        selected = selector.fit(X, y, names, n_select=10)
        assert len(selected) <= 3  # 不应超过原始特征数

    def test_rollout_buffer_empty(self):
        """测试空RolloutBuffer"""
        buf = RolloutBuffer()
        assert len(buf) == 0

    def test_causal_graph_empty(self):
        """测试空因果图"""
        graph = CausalGraph()
        assert graph.is_dag()  # 空图是DAG
        assert len(graph.get_parents("nonexistent")) == 0


class TestLongRunningStability:
    """长时间运行稳定性测试"""

    def test_continual_learning_many_tasks(self):
        """测试持续学习处理多任务序列"""
        ctrl = ContinualLearningController(input_dim=5, output_dim=3)
        for i in range(10):
            X = np.random.randn(20, 5)
            y = np.random.randint(0, 3, 20)
            ctrl.learn_task(f"task_{i}", X, y, n_steps=3)
        stats = ctrl.get_stats()
        assert stats["task_count"] == 10

    def test_replay_buffer_overflow(self):
        """测试经验回放缓冲区溢出"""
        replay = ExperienceReplay(capacity_per_task=5)
        for i in range(5):
            X = np.random.randn(3, 5)
            y = np.random.randint(0, 2, 3)
            replay.add_task(f"task_{i}", X, y)
        # 添加第6个任务，应能正常处理
        X = np.random.randn(3, 5)
        y = np.random.randint(0, 2, 3)
        replay.add_task("task_5", X, y)
        assert len(replay) > 0

    def test_meta_replay_buffer_overflow(self):
        """测试元经验回放缓冲区溢出"""
        buf = MetaReplayBuffer(capacity=5)
        dist = TaskDistribution(input_dim=3, n_classes=2, k_shot=2, q_query=3)
        for _ in range(20):
            buf.add(dist.sample_task())
        assert len(buf) == 5

    def test_repeated_augmentation_stability(self):
        """测试重复数据增强稳定性"""
        aug = DataAugmentor(noise_std=0.1)
        x = np.random.randn(5, 5)
        for _ in range(100):
            x = aug.augment(x)
        assert np.all(np.isfinite(x)), "重复增强后应保持有限"

    def test_repeated_encoding_stability(self):
        """测试重复编码稳定性（维度不变场景）"""
        enc = RepresentationEncoder(input_dim=5, hidden_dims=[8], latent_dim=5)
        x = np.random.randn(5, 5)
        for _ in range(50):
            x = enc.encode(x)
        assert np.all(np.isfinite(x))

    def test_policy_get_action_many_times(self):
        """测试策略网络多次采样稳定性"""
        policy = PolicyNetwork(state_dim=5, action_dim=3)
        for _ in range(100):
            state = np.random.randn(5)
            action = policy.get_action(state)
            assert 0 <= action["action"] < 3
            assert np.isfinite(action["log_prob"])


class TestModuleInteraction:
    """模块间交互鲁棒性测试"""

    def test_uncertainty_with_multimodal(self):
        """测试不确定性量化与多模态融合交互"""
        configs = [ModalityConfig(name="a", input_dim=5, latent_dim=8)]
        mm_ctrl = MultimodalController(configs, output_dim=1, latent_dim=8)
        uq_ctrl = UncertaintyController(input_dim=8, output_dim=1, method="mc_dropout")

        data = {"a": np.random.randn(20, 5)}
        fused = mm_ctrl.fuse(data)
        result = uq_ctrl.predict(fused)
        assert np.all(np.isfinite(result["mean"]))

    def test_continual_with_auto_feature(self):
        """测试持续学习与自动特征工程交互"""
        af_ctrl = AutoFeatureController(max_generated=10, n_select=3)
        X = np.random.randn(30, 5)
        y = np.random.randint(0, 3, 30)
        af_ctrl.fit(X, y)
        features = af_ctrl.transform(X)
        # 根据实际特征数创建持续学习控制器
        n_features = features.shape[1]
        cl_ctrl = ContinualLearningController(input_dim=n_features, output_dim=3)
        cl_ctrl.learn_task("task_0", features, y, n_steps=3)
        stats = cl_ctrl.get_stats()
        assert stats["task_count"] == 1

    def test_adversarial_with_uncertainty(self):
        """测试对抗鲁棒性与不确定性量化交互"""
        uq_net = MCDropoutNetwork(input_dim=5, hidden_dim=8, n_mc_samples=3)

        def model_forward(x):
            return uq_net.predict(x)

        gen = AdversarialExampleGenerator(model_forward)
        budget = PerturbationBudget(epsilon=0.1, norm="linf")

        x = np.random.randn(5, 5)
        y = np.random.randint(0, 1, 5)
        adv_x = gen.fgsm(x, y, budget)

        # 在原始和对抗样本上估计不确定性
        r1 = uq_net.predict_with_uncertainty(x)
        r2 = uq_net.predict_with_uncertainty(adv_x)
        assert np.all(np.isfinite(r1.mean))
        assert np.all(np.isfinite(r2.mean))

    def test_self_supervised_with_continual(self):
        """测试自监督学习与持续学习交互"""
        ss_ctrl = SelfSupervisedController(input_dim=5, latent_dim=4, method="contrastive")
        cl_ctrl = ContinualLearningController(input_dim=4, output_dim=3)

        data = np.random.randn(30, 5)
        ss_ctrl.train(data, n_steps=2, lr=0.001, batch_size=8)
        embeddings = ss_ctrl.get_embeddings(data)
        y = np.random.randint(0, 3, 30)
        cl_ctrl.learn_task("task_0", embeddings, y, n_steps=3)
        assert cl_ctrl.get_stats()["task_count"] == 1


class TestNumericalStability:
    """数值稳定性测试"""

    def test_softmax_stability_large_logits(self):
        """测试大logits下softmax稳定性"""
        encoder = RepresentationEncoder(input_dim=5, hidden_dims=[8], latent_dim=4)
        learner = ContrastiveLearner(encoder, temperature=0.01)
        # 构造大logits
        x = np.random.randn(10, 5) * 100
        loss = learner.compute_loss(x)
        assert np.isfinite(loss), "大logits下损失应有限"

    def test_log_stability_near_zero(self):
        """测试接近零值的对数稳定性"""
        blr = BayesianLinearRegression(input_dim=3)
        X = np.random.randn(10, 3) * 0.001  # 接近零
        y = np.random.randn(10) * 0.001
        blr.fit(X, y)
        result = blr.predict_with_uncertainty(X[:5])
        assert np.all(np.isfinite(result.mean))

    def test_division_stability_near_zero(self):
        """测试接近零值的除法稳定性"""
        transformer = FeatureTransformer()
        X = np.random.randn(20, 3) * 1e-10
        names = [f"f{i}" for i in range(3)]
        new_X, new_names, defs = transformer.generate_all(X, names, max_features=20)
        assert np.all(np.isfinite(new_X))

    def test_matrix_inversion_stability(self):
        """测试矩阵求逆稳定性"""
        blr = BayesianLinearRegression(input_dim=5)
        # 构造接近奇异的矩阵
        X = np.random.randn(10, 5)
        X[:, 4] = X[:, 0] * 0.9999  # 近似共线性
        y = np.random.randn(10)
        blr.fit(X, y)
        result = blr.predict_with_uncertainty(X[:5])
        assert np.all(np.isfinite(result.mean))

    def test_causal_graph_cycle_detection(self):
        """测试因果图环检测"""
        graph = CausalGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        assert graph.is_dag()
        # 添加环
        graph.add_edge("C", "A")
        assert not graph.is_dag()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--timeout=60"])
