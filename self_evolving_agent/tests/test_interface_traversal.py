"""
接口全量遍历测试 - 正常请求、异常入参、超限参数、空值、非法格式

测试目标：
1. 全覆盖所有接口的公共方法
2. 校验正常请求、异常入参、超限参数、空值、非法格式
3. 确保异常场景有明确异常或平稳兜底

判定标准：接口行为与代码定义完全一致
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


class TestUncertaintyInterfaces:
    """不确定性量化模块接口遍历"""

    def test_mc_dropout_normal(self):
        """正常请求"""
        net = MCDropoutNetwork(input_dim=5, hidden_dim=8, n_mc_samples=5)
        x = np.random.randn(10, 5)
        result = net.predict_with_uncertainty(x)
        assert result.mean.shape == (10, 1)
        assert result.std.shape == (10, 1)

    def test_mc_dropout_negative_hidden_dim(self):
        """异常入参：负隐藏层维度"""
        with pytest.raises(ValueError):
            MCDropoutNetwork(input_dim=5, hidden_dim=-1)

    def test_mc_dropout_zero_input_dim(self):
        """异常入参：零输入维度"""
        with pytest.raises(ValueError):
            MCDropoutNetwork(input_dim=0)

    def test_mc_dropout_negative_n_samples(self):
        """异常入参：负采样数"""
        net = MCDropoutNetwork(input_dim=5, n_mc_samples=-1)
        # 应被裁剪到至少1
        assert net.n_mc_samples >= 1

    def test_deep_ensemble_normal(self):
        """正常请求"""
        ensemble = DeepEnsemble(n_models=3, input_dim=5, hidden_dim=8)
        x = np.random.randn(10, 5)
        result = ensemble.predict_with_uncertainty(x)
        assert result.mean.shape == (10, 1)

    def test_bayesian_lr_normal(self):
        """正常请求"""
        blr = BayesianLinearRegression(input_dim=3)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        blr.fit(X, y)
        result = blr.predict_with_uncertainty(X[:5])
        assert result.mean.shape[0] == 5

    def test_bayesian_lr_wrong_dim(self):
        """异常入参：维度不匹配"""
        blr = BayesianLinearRegression(input_dim=3)
        X = np.random.randn(20, 5)  # 5 != 3
        y = np.random.randn(20)
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            blr.fit(X, y)

    def test_uncertainty_controller_normal(self):
        """正常请求"""
        ctrl = UncertaintyController(input_dim=5, method="mc_dropout")
        x = np.random.randn(10, 5)
        y = np.random.randn(10, 1)
        ctrl.train(x, y, epochs=1)
        result = ctrl.predict(x)
        assert "mean" in result
        assert "confidence" in result

    def test_uncertainty_controller_unknown_method(self):
        """异常入参：未知方法"""
        with pytest.raises((ValueError, KeyError)):
            UncertaintyController(input_dim=5, method="unknown_method")


class TestMultimodalInterfaces:
    """多模态融合模块接口遍历"""

    def test_modality_encoder_normal(self):
        """正常请求"""
        config = ModalityConfig(name="test", input_dim=5, latent_dim=8)
        enc = ModalityEncoder(config)
        x = np.random.randn(10, 5)
        result = enc.encode(x)
        assert result.shape == (10, 8)

    def test_cross_modal_aligner_normal(self):
        """正常请求"""
        aligner = CrossModalAligner(latent_dim=8)
        embeddings = {
            "a": np.random.randn(5, 8),
            "b": np.random.randn(5, 8),
        }
        loss = aligner.compute_alignment_loss(embeddings)
        assert np.isfinite(loss)

    def test_cross_modal_aligner_align_dim_reduction(self):
        """维度降维对齐"""
        aligner = CrossModalAligner(latent_dim=4)
        emb = np.random.randn(5, 8)
        result = aligner.align(emb, target_dim=4)
        assert result.shape == (5, 4)

    def test_cross_modal_aligner_align_dim_increase(self):
        """维度升维对齐"""
        aligner = CrossModalAligner(latent_dim=8)
        emb = np.random.randn(5, 4)
        result = aligner.align(emb, target_dim=8)
        assert result.shape == (5, 8)

    def test_attention_fusion_normal(self):
        """正常请求"""
        fusion = AttentionFusion(n_modalities=2, latent_dim=8)
        embeddings = [np.random.randn(10, 8), np.random.randn(10, 8)]
        result = fusion.fuse(embeddings)
        assert result.shape == (10, 8)

    def test_multimodal_controller_normal(self):
        """正常请求"""
        configs = [
            ModalityConfig(name="a", input_dim=5, latent_dim=8),
            ModalityConfig(name="b", input_dim=3, latent_dim=8),
        ]
        ctrl = MultimodalController(configs, output_dim=1, latent_dim=8)
        data = {"a": np.random.randn(10, 5), "b": np.random.randn(10, 3)}
        result = ctrl.fuse(data)
        assert result.shape[0] == 10

    def test_multimodal_controller_missing_modality(self):
        """异常入参：缺失模态"""
        configs = [
            ModalityConfig(name="a", input_dim=5, latent_dim=8),
            ModalityConfig(name="b", input_dim=3, latent_dim=8),
        ]
        ctrl = MultimodalController(configs, output_dim=1, latent_dim=8)
        data = {"a": np.random.randn(10, 5)}  # 缺失b
        result = ctrl.fuse(data)
        # 应能处理部分模态缺失
        assert np.all(np.isfinite(result))

    def test_multimodal_controller_empty_data(self):
        """异常入参：空数据"""
        configs = [ModalityConfig(name="a", input_dim=5, latent_dim=8)]
        ctrl = MultimodalController(configs, output_dim=1, latent_dim=8)
        result = ctrl.fuse({})
        assert result.shape[0] == 0  # 空输入返回空


class TestContinualLearningInterfaces:
    """持续学习模块接口遍历"""

    def test_ewc_normal(self):
        """正常请求"""
        ewc = EWCRegularizer(lambda_ewc=1.0)
        params = {"W": np.zeros((3, 3))}
        # task_count=0时返回0
        assert ewc.compute_penalty(params) == 0.0

    def test_experience_replay_normal(self):
        """正常请求"""
        replay = ExperienceReplay(capacity_per_task=10)
        X = np.random.randn(5, 3)
        y = np.random.randint(0, 2, 5)
        replay.add_task("task_0", X, y)
        data, labels = replay.sample_replay_batch(3)
        assert len(data) <= 3

    def test_progressive_network_normal(self):
        """正常请求"""
        net = ProgressiveNetwork(input_dim=5, hidden_dim=8, output_dim=3)
        x = np.random.randn(10, 5)
        result = net.forward(x)
        assert result.shape[0] == 10

    def test_continual_controller_ewc_strategy(self):
        """EWC策略"""
        ctrl = ContinualLearningController(input_dim=5, output_dim=3, strategy="ewc")
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 3, 20)
        ctrl.learn_task("task_0", X, y, n_steps=3)
        assert ctrl.get_stats()["task_count"] == 1

    def test_continual_controller_replay_strategy(self):
        """回放策略"""
        ctrl = ContinualLearningController(input_dim=5, output_dim=3, strategy="replay")
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 3, 20)
        ctrl.learn_task("task_0", X, y, n_steps=3)
        assert ctrl.get_stats()["task_count"] == 1

    def test_continual_controller_progressive_strategy(self):
        """渐进网络策略"""
        ctrl = ContinualLearningController(input_dim=5, output_dim=3, strategy="progressive")
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 3, 20)
        ctrl.learn_task("task_0", X, y, n_steps=3)
        assert ctrl.get_stats()["task_count"] == 1

    def test_continual_controller_unknown_strategy(self):
        """异常入参：未知策略"""
        ctrl = ContinualLearningController(input_dim=5, output_dim=3, strategy="unknown")
        # 应不崩溃，只是没有ewc/replay
        assert ctrl.ewc is None
        assert ctrl.replay is None

    def test_continual_controller_save_load(self):
        """保存加载"""
        ctrl = ContinualLearningController(input_dim=5, output_dim=3)
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 3, 20)
        ctrl.learn_task("task_0", X, y, n_steps=3)
        saved = ctrl.save()
        ctrl2 = ContinualLearningController(input_dim=5, output_dim=3)
        ctrl2.load(saved)
        assert ctrl2.get_stats()["task_count"] == 1


class TestSelfSupervisedInterfaces:
    """自监督学习模块接口遍历"""

    def test_representation_encoder_normal(self):
        """正常请求"""
        enc = RepresentationEncoder(input_dim=5, hidden_dims=[8], latent_dim=4)
        x = np.random.randn(10, 5)
        result = enc.encode(x)
        assert result.shape == (10, 4)

    def test_data_augmentor_normal(self):
        """正常请求"""
        aug = DataAugmentor(noise_std=0.1)
        x = np.random.randn(10, 5)
        result = aug.augment(x)
        assert result.shape == x.shape

    def test_contrastive_learner_normal(self):
        """正常请求"""
        enc = RepresentationEncoder(input_dim=5, hidden_dims=[8], latent_dim=4)
        learner = ContrastiveLearner(enc, temperature=0.5)
        x = np.random.randn(10, 5)
        loss = learner.compute_loss(x)
        assert np.isfinite(loss)

    def test_masked_predictor_normal(self):
        """正常请求"""
        predictor = MaskedPredictor(input_dim=5, hidden_dims=[8])
        x = np.random.randn(10, 5)
        loss = predictor.compute_loss(x)
        assert np.isfinite(loss)

    def test_ssl_controller_contrastive(self):
        """对比学习方法"""
        ctrl = SelfSupervisedController(input_dim=5, latent_dim=4, method="contrastive")
        data = np.random.randn(30, 5)
        result = ctrl.train(data, n_steps=3, lr=0.001, batch_size=8)
        assert "final_loss" in result

    def test_ssl_controller_masked(self):
        """掩码预测方法"""
        ctrl = SelfSupervisedController(input_dim=5, latent_dim=4, method="masked")
        data = np.random.randn(30, 5)
        result = ctrl.train(data, n_steps=3, lr=0.001, batch_size=8)
        assert "final_loss" in result

    def test_ssl_controller_unknown_method(self):
        """异常入参：未知方法"""
        with pytest.raises(ValueError):
            SelfSupervisedController(input_dim=5, method="unknown")


class TestAutoFeatureInterfaces:
    """自动特征工程模块接口遍历"""

    def test_feature_transformer_normal(self):
        """正常请求"""
        transformer = FeatureTransformer()
        X = np.random.randn(20, 3)
        names = ["f0", "f1", "f2"]
        new_X, new_names, defs = transformer.generate_all(X, names, max_features=20)
        assert new_X.shape[0] == 20

    def test_mutual_info_selector_normal(self):
        """正常请求"""
        selector = MutualInfoSelector(n_bins=5)
        X = np.random.randn(20, 3)
        y = np.random.randint(0, 2, 20)
        names = ["f0", "f1", "f2"]
        selected = selector.fit(X, y, names, n_select=2)
        assert len(selected) <= 2

    def test_auto_feature_controller_normal(self):
        """正常请求"""
        ctrl = AutoFeatureController(max_generated=10, n_select=3)
        X = np.random.randn(30, 5)
        y = np.random.randint(0, 3, 30)
        result = ctrl.fit(X, y)
        assert "selected_features" in result
        features = ctrl.transform(X)
        assert features.shape[0] == 30

    def test_auto_feature_controller_not_fitted(self):
        """异常入参：未拟合就变换"""
        ctrl = AutoFeatureController()
        X = np.random.randn(10, 3)
        with pytest.raises(ValueError):
            ctrl.transform(X)

    def test_auto_feature_controller_save_load(self):
        """保存加载"""
        ctrl = AutoFeatureController(max_generated=10, n_select=3)
        X = np.random.randn(30, 5)
        y = np.random.randint(0, 3, 30)
        ctrl.fit(X, y)
        saved = ctrl.save()
        ctrl2 = AutoFeatureController()
        ctrl2.load(saved)
        assert ctrl2.is_fitted


class TestMetaLearningInterfaces:
    """元学习模块接口遍历"""

    def test_task_distribution_normal(self):
        """正常请求"""
        dist = TaskDistribution(input_dim=5, n_classes=3, k_shot=2, q_query=5)
        task = dist.sample_task()
        assert task.support_x.shape == (6, 5)  # 3类 * 2shot
        assert task.query_x.shape == (15, 5)  # 3类 * 5query

    def test_simple_model_normal(self):
        """正常请求"""
        model = SimpleModel(input_dim=5, output_dim=3)
        x = np.random.randn(10, 5)
        logits = model.forward(x)
        assert logits.shape == (10, 3)
        preds = model.predict(x)
        assert preds.shape == (10,)

    def test_maml_model_normal(self):
        """正常请求"""
        model = SimpleModel(input_dim=5, output_dim=3)
        maml = MAMLModel(model, inner_lr=0.01, inner_steps=2)
        dist = TaskDistribution(input_dim=5, n_classes=3, k_shot=2, q_query=3)
        task = dist.sample_task()
        adapted = maml.adapt(task)
        assert adapted is not None

    def test_meta_replay_buffer_normal(self):
        """正常请求"""
        buf = MetaReplayBuffer(capacity=10)
        dist = TaskDistribution(input_dim=3, n_classes=2, k_shot=2, q_query=3)
        buf.add(dist.sample_task())
        assert len(buf) == 1
        batch = buf.sample(1)
        assert len(batch) == 1

    def test_fast_adapter_normal(self):
        """正常请求"""
        model = SimpleModel(input_dim=5, output_dim=3)
        adapter = FastAdapter(model, lr=0.01, max_steps=2)
        x = np.random.randn(6, 5)
        y = np.random.randint(0, 3, 6)
        adapted = adapter.adapt(x, y)
        assert adapted is not None


class TestAdversarialInterfaces:
    """对抗鲁棒性模块接口遍历"""

    def test_perturbation_budget_normal(self):
        """正常请求"""
        budget = PerturbationBudget(epsilon=0.1, norm="linf")
        assert budget.epsilon == 0.1

    def test_adversarial_generator_fgsm(self):
        """FGSM攻击"""
        def forward(x):
            return np.dot(x, np.random.randn(5, 3))
        gen = AdversarialExampleGenerator(forward)
        budget = PerturbationBudget(epsilon=0.1, norm="linf")
        x = np.random.randn(5, 5)
        y = np.random.randint(0, 3, 5)
        adv = gen.fgsm(x, y, budget)
        assert adv.shape == x.shape

    def test_adversarial_generator_pgd(self):
        """PGD攻击"""
        def forward(x):
            return np.dot(x, np.random.randn(5, 3))
        gen = AdversarialExampleGenerator(forward)
        budget = PerturbationBudget(epsilon=0.1, norm="linf")
        x = np.random.randn(5, 5)
        y = np.random.randint(0, 3, 5)
        adv = gen.pgd(x, y, budget, steps=3)
        assert adv.shape == x.shape

    def test_adversarial_generator_generate(self):
        """generate接口"""
        def forward(x):
            return np.dot(x, np.random.randn(5, 3))
        gen = AdversarialExampleGenerator(forward)
        budget = PerturbationBudget(epsilon=0.1, norm="linf")
        x = np.random.randn(5, 5)
        y = np.random.randint(0, 3, 5)
        adv = gen.generate(x, y, method="fgsm", budget=budget)
        assert adv.shape == x.shape


class TestCausalInterfaces:
    """因果推断模块接口遍历"""

    def test_causal_graph_normal(self):
        """正常请求"""
        graph = CausalGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        assert graph.is_dag()
        assert "A" in graph.get_parents("B")

    def test_causal_discovery_partial_corr(self):
        """偏相关系数"""
        cd = CausalDiscovery()
        x = np.random.randn(50)
        y = np.random.randn(50)
        pc = cd.partial_correlation(x, y)
        assert -1 <= pc <= 1

    def test_causal_discovery_partial_corr_with_cond(self):
        """带条件集的偏相关系数"""
        cd = CausalDiscovery()
        x = np.random.randn(50)
        y = np.random.randn(50)
        z = np.random.randn(50)
        pc = cd.partial_correlation(x, y, z)
        assert np.isfinite(pc)

    def test_backdoor_criterion_normal(self):
        """后门准则"""
        graph = CausalGraph()
        graph.add_edge("Z", "X")
        graph.add_edge("Z", "Y")
        graph.add_edge("X", "Y")
        bc = BackdoorCriterion(graph)
        # 查找X->Y的调整集
        result = bc.find_valid_adjustment_sets("X", "Y")
        assert isinstance(result, list)


class TestRLPolicyInterfaces:
    """强化学习策略模块接口遍历"""

    def test_policy_network_normal(self):
        """正常请求"""
        policy = PolicyNetwork(state_dim=5, action_dim=3)
        state = np.random.randn(5)
        action = policy.get_action(state)
        assert 0 <= action["action"] < 3

    def test_policy_network_evaluate(self):
        """评估接口"""
        policy = PolicyNetwork(state_dim=5, action_dim=3)
        states = np.random.randn(10, 5)
        actions = np.random.randint(0, 3, 10)
        log_probs, entropy = policy.evaluate(states, actions)
        assert log_probs.shape == (10,)
        assert entropy.shape == (10,)

    def test_value_network_normal(self):
        """价值网络"""
        vn = ValueNetwork(state_dim=5)
        x = np.random.randn(10, 5)
        values = vn.predict(x)
        assert values.shape[0] == 10

    def test_rollout_buffer_normal(self):
        """轨迹缓冲区"""
        buf = RolloutBuffer()
        for _ in range(5):
            buf.add(
                state=np.random.randn(5),
                action=np.random.randint(0, 3),
                reward=0.1,
                log_prob=-0.5,
                value=0.0,
                done=False,
            )
        assert len(buf) == 5
        advantages, returns = buf.compute_advantages()
        assert advantages.shape == (5,)

    def test_ppo_trainer_normal(self):
        """PPO训练器"""
        policy = PolicyNetwork(state_dim=5, action_dim=3)
        value_net = ValueNetwork(state_dim=5)
        trainer = PPOTrainer(policy, value_net)
        buf = RolloutBuffer()
        for _ in range(5):
            buf.add(
                state=np.random.randn(5),
                action=np.random.randint(0, 3),
                reward=0.1,
                log_prob=-0.5,
                value=0.0,
                done=False,
            )
        metrics = trainer.update(buf)
        assert "policy_loss" in metrics or "loss" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--timeout=60"])
