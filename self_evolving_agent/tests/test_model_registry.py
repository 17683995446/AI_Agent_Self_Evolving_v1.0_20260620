"""
模型版本管理与A/B测试 —— 验证模型注册、版本切换、A/B对比

测试前置：
1. 拆解ModelRegistry接口：register/get_version/list_versions/get_active_version/activate/delete_version
2. 拆解ABTestFramework接口：start_test/record_prediction/evaluate/auto_promote/get_stats
3. 边界条件：空注册表、重复激活、删除不存在版本、A/B类型不匹配、样本不足、零误差对照组
"""
import sys
import os
import tempfile
import shutil
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_registry import ModelRegistry, ABTestFramework, ModelVersion


class TestModelRegistry:
    """测试模型注册表"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(registry_dir=self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_and_get_version(self):
        """注册模型并获取"""
        version = self.registry.register("base", {"lr": 0.01}, {"mse": 0.1})
        assert version.version_id.startswith("base_")
        assert version.model_type == "base"
        assert version.metrics["mse"] == 0.1

        fetched = self.registry.get_version(version.version_id)
        assert fetched is not None
        assert fetched.checksum == version.checksum

    def test_list_versions_all(self):
        """列出所有版本"""
        v1 = self.registry.register("base", {"lr": 0.01})
        v2 = self.registry.register("mid", {"units": 64})
        v3 = self.registry.register("base", {"lr": 0.02})

        all_versions = self.registry.list_versions()
        assert len(all_versions) == 3
        # 按时间倒序
        assert all_versions[0].version_id == v3.version_id

    def test_list_versions_filter_by_type(self):
        """按类型过滤版本"""
        self.registry.register("base", {"lr": 0.01})
        self.registry.register("mid", {"units": 64})
        self.registry.register("base", {"lr": 0.02})

        base_versions = self.registry.list_versions(model_type="base")
        assert len(base_versions) == 2
        assert all(v.model_type == "base" for v in base_versions)

    def test_activate_and_get_active(self):
        """激活版本并获取当前激活版本"""
        v1 = self.registry.register("base", {"lr": 0.01})
        v2 = self.registry.register("base", {"lr": 0.02})

        # 初始无激活
        assert self.registry.get_active_version("base") is None

        # 激活v1
        assert self.registry.activate(v1.version_id) is True
        active = self.registry.get_active_version("base")
        assert active is not None
        assert active.version_id == v1.version_id
        assert active.is_active is True

        # 激活v2，v1自动失效
        assert self.registry.activate(v2.version_id) is True
        active = self.registry.get_active_version("base")
        assert active.version_id == v2.version_id
        assert self.registry.get_version(v1.version_id).is_active is False

    def test_activate_nonexistent(self):
        """激活不存在的版本返回False"""
        assert self.registry.activate("not_exist") is False

    def test_delete_version(self):
        """删除版本"""
        v1 = self.registry.register("base", {"lr": 0.01})
        assert self.registry.get_version(v1.version_id) is not None

        assert self.registry.delete_version(v1.version_id) is True
        assert self.registry.get_version(v1.version_id) is None
        # 文件也应被删除
        assert not os.path.exists(os.path.join(self.temp_dir, f"model_{v1.version_id}.pkl"))

    def test_delete_nonexistent(self):
        """删除不存在的版本返回False"""
        assert self.registry.delete_version("not_exist") is False

    def test_persistence_load_on_init(self):
        """持久化：新注册表应加载已有版本"""
        v1 = self.registry.register("base", {"lr": 0.01})

        # 创建新注册表实例，指向同一目录
        registry2 = ModelRegistry(registry_dir=self.temp_dir)
        fetched = registry2.get_version(v1.version_id)
        assert fetched is not None
        assert fetched.checksum == v1.checksum

    def test_checksum_consistency(self):
        """相同参数产生相同checksum"""
        v1 = self.registry.register("base", {"lr": 0.01, "batch": 32})
        v2 = self.registry.register("base", {"batch": 32, "lr": 0.01})
        assert v1.checksum == v2.checksum

    def test_checksum_difference(self):
        """不同参数产生不同checksum"""
        v1 = self.registry.register("base", {"lr": 0.01})
        v2 = self.registry.register("base", {"lr": 0.02})
        assert v1.checksum != v2.checksum

    def test_version_to_dict(self):
        """版本序列化为字典"""
        v = self.registry.register("high", {"depth": 3}, {"accuracy": 0.95})
        d = v.to_dict()
        assert d["model_type"] == "high"
        assert d["metrics"]["accuracy"] == 0.95
        assert "params" not in d  # to_dict不包含params

    def test_empty_registry_list(self):
        """空注册表列出返回空列表"""
        assert self.registry.list_versions() == []
        assert self.registry.get_active_version("base") is None

    def test_register_with_none_metrics(self):
        """注册时metrics为None应默认为空字典"""
        v = self.registry.register("base", {"lr": 0.01}, metrics=None)
        assert v.metrics == {}


class TestABTestFramework:
    """测试A/B测试框架"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(registry_dir=self.temp_dir)
        self.ab = ABTestFramework(self.registry)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_start_test_success(self):
        """正常启动A/B测试"""
        ctrl = self.registry.register("base", {"lr": 0.01})
        cand = self.registry.register("base", {"lr": 0.02})

        assert self.ab.start_test(ctrl.version_id, cand.version_id) is True
        assert cand.is_candidate is True
        assert cand.version_id in self.ab._candidate_metrics
        assert ctrl.version_id in self.ab._control_metrics

    def test_start_test_nonexistent_version(self):
        """版本不存在时启动失败"""
        assert self.ab.start_test("ctrl", "cand") is False

    def test_start_test_type_mismatch(self):
        """模型类型不匹配时启动失败"""
        ctrl = self.registry.register("base", {"lr": 0.01})
        cand = self.registry.register("mid", {"units": 64})

        assert self.ab.start_test(ctrl.version_id, cand.version_id) is False

    def test_record_prediction(self):
        """记录预测误差"""
        ctrl = self.registry.register("base", {"lr": 0.01})
        cand = self.registry.register("base", {"lr": 0.02})
        self.ab.start_test(ctrl.version_id, cand.version_id)

        self.ab.record_prediction(cand.version_id, 0.1, is_candidate=True)
        self.ab.record_prediction(cand.version_id, 0.2, is_candidate=True)
        self.ab.record_prediction(ctrl.version_id, 0.15, is_candidate=False)

        assert len(self.ab._candidate_metrics[cand.version_id]) == 2
        assert len(self.ab._control_metrics[ctrl.version_id]) == 1

    def test_record_prediction_ignore_unknown(self):
        """记录未知版本应被忽略不抛异常"""
        self.ab.record_prediction("unknown", 0.1, is_candidate=True)
        self.ab.record_prediction("unknown", 0.1, is_candidate=False)
        # 不应抛异常

    def test_evaluate_insufficient_samples(self):
        """样本不足时返回 inconclusive"""
        ctrl = self.registry.register("base", {"lr": 0.01})
        cand = self.registry.register("base", {"lr": 0.02})
        self.ab.start_test(ctrl.version_id, cand.version_id)

        # 只记录少量样本
        self.ab.record_prediction(cand.version_id, 0.1, is_candidate=True)
        self.ab.record_prediction(ctrl.version_id, 0.15, is_candidate=False)

        result = self.ab.evaluate(cand.version_id, ctrl.version_id)
        assert result["conclusive"] is False
        assert "样本不足" in result["reason"]

    def test_evaluate_candidate_wins(self):
        """候选模型显著优于对照组"""
        ctrl = self.registry.register("base", {"lr": 0.01})
        cand = self.registry.register("base", {"lr": 0.02})
        self.ab.start_test(ctrl.version_id, cand.version_id)

        # 候选误差明显更小
        for _ in range(15):
            self.ab.record_prediction(cand.version_id, 0.05, is_candidate=True)
            self.ab.record_prediction(ctrl.version_id, 0.20, is_candidate=False)

        result = self.ab.evaluate(cand.version_id, ctrl.version_id)
        assert result["conclusive"] is True
        assert result["winner"] == cand.version_id
        assert result["improvement"] > 0.05

    def test_evaluate_control_wins(self):
        """对照组显著优于候选模型"""
        ctrl = self.registry.register("base", {"lr": 0.01})
        cand = self.registry.register("base", {"lr": 0.02})
        self.ab.start_test(ctrl.version_id, cand.version_id)

        # 对照误差明显更小
        for _ in range(15):
            self.ab.record_prediction(cand.version_id, 0.20, is_candidate=True)
            self.ab.record_prediction(ctrl.version_id, 0.05, is_candidate=False)

        result = self.ab.evaluate(cand.version_id, ctrl.version_id)
        assert result["conclusive"] is True
        assert result["winner"] == ctrl.version_id
        assert result["improvement"] < -0.05

    def test_evaluate_zero_control_error(self):
        """对照组误差为0时improvement应为0"""
        ctrl = self.registry.register("base", {"lr": 0.01})
        cand = self.registry.register("base", {"lr": 0.02})
        self.ab.start_test(ctrl.version_id, cand.version_id)

        for _ in range(15):
            self.ab.record_prediction(cand.version_id, 0.10, is_candidate=True)
            self.ab.record_prediction(ctrl.version_id, 0.00, is_candidate=False)

        result = self.ab.evaluate(cand.version_id, ctrl.version_id)
        assert result["control_avg_error"] == 0.0
        assert result["improvement"] == 0.0
        assert result["conclusive"] is False  # improvement=0不满足阈值

    def test_auto_promote_candidate_wins(self):
        """自动提升优胜候选版本"""
        ctrl = self.registry.register("base", {"lr": 0.01})
        cand = self.registry.register("base", {"lr": 0.02})
        self.ab.start_test(ctrl.version_id, cand.version_id)

        for _ in range(15):
            self.ab.record_prediction(cand.version_id, 0.05, is_candidate=True)
            self.ab.record_prediction(ctrl.version_id, 0.20, is_candidate=False)

        promoted = self.ab.auto_promote(cand.version_id, ctrl.version_id)
        assert promoted is True
        active = self.registry.get_active_version("base")
        assert active.version_id == cand.version_id

    def test_auto_promote_inconclusive(self):
        """结果不显著时不提升"""
        ctrl = self.registry.register("base", {"lr": 0.01})
        cand = self.registry.register("base", {"lr": 0.02})
        self.ab.start_test(ctrl.version_id, cand.version_id)

        # 只记录少量样本
        for _ in range(5):
            self.ab.record_prediction(cand.version_id, 0.10, is_candidate=True)
            self.ab.record_prediction(ctrl.version_id, 0.10, is_candidate=False)

        promoted = self.ab.auto_promote(cand.version_id, ctrl.version_id)
        assert promoted is False

    def test_auto_promote_control_wins(self):
        """对照组获胜时不提升"""
        ctrl = self.registry.register("base", {"lr": 0.01})
        cand = self.registry.register("base", {"lr": 0.02})
        self.ab.start_test(ctrl.version_id, cand.version_id)

        for _ in range(15):
            self.ab.record_prediction(cand.version_id, 0.20, is_candidate=True)
            self.ab.record_prediction(ctrl.version_id, 0.05, is_candidate=False)

        promoted = self.ab.auto_promote(cand.version_id, ctrl.version_id)
        assert promoted is False
        # 对照组应保持未激活（或保持原状态）
        active = self.registry.get_active_version("base")
        assert active is None or active.version_id != cand.version_id

    def test_get_stats(self):
        """获取A/B测试统计信息"""
        ctrl = self.registry.register("base", {"lr": 0.01})
        cand = self.registry.register("base", {"lr": 0.02})
        self.ab.start_test(ctrl.version_id, cand.version_id)

        self.ab.record_prediction(cand.version_id, 0.1, is_candidate=True)
        self.ab.record_prediction(ctrl.version_id, 0.15, is_candidate=False)

        stats = self.ab.get_stats()
        assert stats["active_tests"] == 1
        assert stats["candidate_samples"][cand.version_id] == 1
        assert stats["control_samples"][ctrl.version_id] == 1
        assert stats["improvement_threshold"] == 0.05

    def test_get_stats_empty(self):
        """无活跃测试时统计为空"""
        stats = self.ab.get_stats()
        assert stats["active_tests"] == 0
        assert stats["candidate_samples"] == {}
        assert stats["control_samples"] == {}

    def test_start_test_with_custom_duration(self):
        """自定义测试时长"""
        ctrl = self.registry.register("base", {"lr": 0.01})
        cand = self.registry.register("base", {"lr": 0.02})

        assert self.ab.start_test(ctrl.version_id, cand.version_id, duration=60.0) is True
        assert self.ab._test_duration == 60.0
