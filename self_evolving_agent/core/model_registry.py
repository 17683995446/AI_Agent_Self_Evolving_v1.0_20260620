"""
模型版本管理与A/B测试 —— 预测模型的版本化、对比和自动选择

设计原则（第一性原理）：
1. 模型即资产：每个版本必须可追踪、可恢复
2. 数据驱动：模型优劣由实际预测误差决定
3. 安全切换：新模型验证通过前，旧模型继续服务
"""

import os
import sys
import time
import json
import pickle
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


@dataclass
class ModelVersion:
    """模型版本记录"""
    version_id: str
    model_type: str  # "base", "mid", "high"
    created_at: float
    params: Dict[str, Any]  # 模型参数快照
    metrics: Dict[str, float]  # 评估指标
    checksum: str
    is_active: bool = False
    is_candidate: bool = False  # A/B测试候选

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_type": self.model_type,
            "created_at": self.created_at,
            "metrics": self.metrics,
            "checksum": self.checksum,
            "is_active": self.is_active,
            "is_candidate": self.is_candidate,
        }


class ModelRegistry:
    """模型注册表 —— 管理预测模型的所有版本"""

    def __init__(self, registry_dir: str = "./data/models"):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        self._versions: Dict[str, ModelVersion] = {}
        self._load_existing_versions()
        logger.info("MODEL_REGISTRY", "模型注册表初始化", module="model_registry",
                    extra={"dir": registry_dir, "loaded": len(self._versions)})

    def _compute_checksum(self, params: Dict[str, Any]) -> str:
        content = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _version_path(self, version_id: str) -> str:
        return os.path.join(self.registry_dir, f"model_{version_id}.pkl")

    def _load_existing_versions(self) -> None:
        """加载磁盘上已有的模型版本"""
        if not os.path.exists(self.registry_dir):
            return
        for fname in os.listdir(self.registry_dir):
            if fname.startswith("model_") and fname.endswith(".pkl"):
                path = os.path.join(self.registry_dir, fname)
                try:
                    with open(path, 'rb') as f:
                        version = pickle.load(f)
                    if isinstance(version, ModelVersion):
                        self._versions[version.version_id] = version
                except Exception as e:
                    logger.warning("MODEL_REGISTRY", f"加载旧版本失败: {e}", module="model_registry")

    def register(self, model_type: str, params: Dict[str, Any],
                 metrics: Optional[Dict[str, float]] = None) -> ModelVersion:
        """注册新模型版本"""
        version_id = f"{model_type}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000000) % 1000000:06d}"
        checksum = self._compute_checksum(params)

        version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            created_at=time.time(),
            params=params,
            metrics=metrics or {},
            checksum=checksum,
        )

        # 持久化
        path = self._version_path(version_id)
        try:
            with open(path, 'wb') as f:
                pickle.dump(version, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error("MODEL_REGISTRY", f"模型持久化失败: {e}", module="model_registry")

        self._versions[version_id] = version
        logger.info("MODEL_REGISTRY", f"模型版本注册: {version_id}", module="model_registry",
                    extra={"type": model_type, "metrics": metrics})
        return version

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """获取指定版本"""
        return self._versions.get(version_id)

    def list_versions(self, model_type: Optional[str] = None) -> List[ModelVersion]:
        """列出所有版本（可按类型过滤）"""
        versions = list(self._versions.values())
        if model_type:
            versions = [v for v in versions if v.model_type == model_type]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def get_active_version(self, model_type: str) -> Optional[ModelVersion]:
        """获取当前激活的版本"""
        for v in self._versions.values():
            if v.model_type == model_type and v.is_active:
                return v
        return None

    def activate(self, version_id: str) -> bool:
        """激活指定版本（同类型其他版本自动失效）"""
        version = self._versions.get(version_id)
        if not version:
            return False

        # 同类型其他版本失效
        for v in self._versions.values():
            if v.model_type == version.model_type:
                v.is_active = False

        version.is_active = True
        logger.info("MODEL_REGISTRY", f"模型版本激活: {version_id}", module="model_registry")
        return True

    def delete_version(self, version_id: str) -> bool:
        """删除版本"""
        if version_id not in self._versions:
            return False
        path = self._version_path(version_id)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.warning("MODEL_REGISTRY", f"删除模型文件失败: {e}", module="model_registry")
        del self._versions[version_id]
        logger.info("MODEL_REGISTRY", f"模型版本删除: {version_id}", module="model_registry")
        return True


class ABTestFramework:
    """A/B测试框架 —— 并行运行新旧模型，对比效果"""

    def __init__(self, model_registry: ModelRegistry):
        self.registry = model_registry
        self._candidate_metrics: Dict[str, List[float]] = {}  # version_id -> error_list
        self._control_metrics: Dict[str, List[float]] = {}  # version_id -> error_list
        self._test_duration = 300.0  # 默认测试5分钟
        self._min_samples = 10
        self._improvement_threshold = 0.05  # 5%改善阈值
        logger.info("AB_TEST", "A/B测试框架初始化", module="model_registry")

    def start_test(self, control_version_id: str, candidate_version_id: str,
                   duration: Optional[float] = None) -> bool:
        """启动A/B测试"""
        control = self.registry.get_version(control_version_id)
        candidate = self.registry.get_version(candidate_version_id)
        if not control or not candidate:
            logger.error("AB_TEST", "版本不存在，无法启动A/B测试", module="model_registry")
            return False

        if control.model_type != candidate.model_type:
            logger.error("AB_TEST", "模型类型不匹配", module="model_registry")
            return False

        candidate.is_candidate = True
        self._candidate_metrics[candidate_version_id] = []
        self._control_metrics[control_version_id] = []
        self._test_duration = duration or self._test_duration

        logger.info("AB_TEST", f"A/B测试启动: {control_version_id} vs {candidate_version_id}",
                    module="model_registry", extra={"duration": self._test_duration})
        return True

    def record_prediction(self, version_id: str, actual_error: float, is_candidate: bool = False) -> None:
        """记录预测误差"""
        if is_candidate:
            if version_id in self._candidate_metrics:
                self._candidate_metrics[version_id].append(actual_error)
        else:
            if version_id in self._control_metrics:
                self._control_metrics[version_id].append(actual_error)

    def evaluate(self, candidate_version_id: str, control_version_id: str) -> Dict[str, Any]:
        """评估A/B测试结果"""
        cand_errors = self._candidate_metrics.get(candidate_version_id, [])
        ctrl_errors = self._control_metrics.get(control_version_id, [])

        if len(cand_errors) < self._min_samples or len(ctrl_errors) < self._min_samples:
            return {
                "conclusive": False,
                "reason": f"样本不足 (候选:{len(cand_errors)}, 对照:{len(ctrl_errors)})",
            }

        cand_avg = sum(cand_errors) / len(cand_errors)
        ctrl_avg = sum(ctrl_errors) / len(ctrl_errors)

        if ctrl_avg == 0:
            improvement = 0.0
        else:
            improvement = (ctrl_avg - cand_avg) / ctrl_avg

        conclusive = abs(improvement) >= self._improvement_threshold
        winner = candidate_version_id if improvement > 0 else control_version_id

        result = {
            "conclusive": conclusive,
            "winner": winner if conclusive else None,
            "improvement": improvement,
            "candidate_avg_error": cand_avg,
            "control_avg_error": ctrl_avg,
            "candidate_samples": len(cand_errors),
            "control_samples": len(ctrl_errors),
        }

        logger.info("AB_TEST", "A/B测试评估完成", module="model_registry",
                    extra={"winner": winner, "improvement": improvement})
        return result

    def auto_promote(self, candidate_version_id: str, control_version_id: str) -> bool:
        """自动评估并提升优胜版本"""
        result = self.evaluate(candidate_version_id, control_version_id)
        if not result["conclusive"]:
            return False

        winner = result["winner"]
        if winner == candidate_version_id:
            self.registry.activate(candidate_version_id)
            logger.info("AB_TEST", f"候选版本自动提升: {candidate_version_id}", module="model_registry")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """获取A/B测试统计"""
        return {
            "active_tests": len(self._candidate_metrics),
            "candidate_samples": {k: len(v) for k, v in self._candidate_metrics.items()},
            "control_samples": {k: len(v) for k, v in self._control_metrics.items()},
            "improvement_threshold": self._improvement_threshold,
        }
