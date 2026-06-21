"""
U34：持续学习与元学习 —— 学习CLI辅助

第一性原理：
- 系统名为"自进化"，持续学习与元学习是M3核心能力，但CLI完全空白
- 持续学习防止技能遗忘（EWC+经验回放），元学习学会快速适应（MAML）
- 用户需要通过CLI初始化学习器、执行训练、查看统计、评估适应能力
- 学习管理应轻量级，直接访问控制器，不启动完整Agent

设计：
1. LearnCliHelper：学习CLI辅助器
   - 封装ContinualLearningController和MetaLearningController
   - 支持init/train/stats/evaluate
   - 持续学习支持合成数据训练（演示/测试用）
   - 元学习自动生成任务分布训练
2. 复用既有控制器，不重复造轮子
"""

import numpy as np
from typing import Dict, List, Optional, Any


class LearnCliHelper:
    """学习CLI辅助器

    轻量级访问持续学习与元学习控制器，不启动完整Agent
    """

    def __init__(self):
        self._continual_learner = None
        self._meta_learner = None
        self._input_dim = 4
        self._output_dim = 2

    # ---------- 初始化 ----------

    def init_continual(self, input_dim: int = 4, output_dim: int = 2) -> bool:
        """初始化持续学习控制器

        Args:
            input_dim: 输入维度
            output_dim: 输出维度

        Returns:
            是否成功
        """
        try:
            from core.continual_learning import ContinualLearningController
            self._continual_learner = ContinualLearningController(
                input_dim=input_dim, output_dim=output_dim,
                strategy="ewc_replay",
            )
            self._input_dim = input_dim
            self._output_dim = output_dim
            return True
        except Exception:
            return False

    def init_meta(self, input_dim: int = 4, output_dim: int = 2) -> bool:
        """初始化元学习控制器

        Args:
            input_dim: 输入维度
            output_dim: 输出维度

        Returns:
            是否成功
        """
        try:
            from core.meta_learning import MetaLearningController
            self._meta_learner = MetaLearningController(
                input_dim=input_dim, output_dim=output_dim,
                n_classes=output_dim,
            )
            self._input_dim = input_dim
            self._output_dim = output_dim
            return True
        except Exception:
            return False

    def init_all(self, input_dim: int = 4, output_dim: int = 2) -> Dict[str, bool]:
        """初始化所有学习器"""
        return {
            "continual": self.init_continual(input_dim, output_dim),
            "meta": self.init_meta(input_dim, output_dim),
        }

    # ---------- 状态查询 ----------

    def is_continual_initialized(self) -> bool:
        """持续学习是否已初始化"""
        return self._continual_learner is not None

    def is_meta_initialized(self) -> bool:
        """元学习是否已初始化"""
        return self._meta_learner is not None

    # ---------- 训练 ----------

    def train_continual(self, task_id: str, n_samples: int = 50,
                        n_steps: int = 50) -> Dict[str, Any]:
        """持续学习训练（使用合成数据）

        Args:
            task_id: 任务ID
            n_samples: 样本数
            n_steps: 训练步数

        Returns:
            训练结果
        """
        if self._continual_learner is None:
            return {"error": "持续学习未初始化，请先执行 'learn init'"}
        try:
            data, labels = self._generate_synthetic_data(n_samples)
            return self._continual_learner.learn_task(task_id, data, labels, n_steps=n_steps)
        except Exception as e:
            return {"error": str(e)}

    def train_meta(self, n_iterations: int = 10, batch_size: int = 4) -> Dict[str, Any]:
        """元学习训练

        Args:
            n_iterations: 训练迭代数
            batch_size: 批大小

        Returns:
            训练结果
        """
        if self._meta_learner is None:
            return {"error": "元学习未初始化，请先执行 'learn init'"}
        try:
            return self._meta_learner.meta_train(n_iterations=n_iterations, batch_size=batch_size)
        except Exception as e:
            return {"error": str(e)}

    # ---------- 评估 ----------

    def evaluate_continual(self, task_id: str, n_samples: int = 50) -> Dict[str, Any]:
        """评估持续学习任务性能

        Args:
            task_id: 任务ID
            n_samples: 样本数

        Returns:
            评估结果
        """
        if self._continual_learner is None:
            return {"error": "持续学习未初始化"}
        try:
            data, labels = self._generate_synthetic_data(n_samples)
            return self._continual_learner.evaluate_task(task_id, data, labels)
        except Exception as e:
            return {"error": str(e)}

    def evaluate_meta(self, n_tasks: int = 10) -> Dict[str, Any]:
        """评估元学习适应能力

        Args:
            n_tasks: 评估任务数

        Returns:
            评估结果
        """
        if self._meta_learner is None:
            return {"error": "元学习未初始化"}
        try:
            return self._meta_learner.evaluate_adaptation(n_tasks=n_tasks)
        except Exception as e:
            return {"error": str(e)}

    # ---------- 统计 ----------

    def get_stats(self) -> Dict[str, Any]:
        """获取学习统计"""
        stats = {
            "continual_initialized": self.is_continual_initialized(),
            "meta_initialized": self.is_meta_initialized(),
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
        }
        if self._continual_learner is not None:
            stats["continual"] = self._continual_learner.get_stats()
        if self._meta_learner is not None:
            stats["meta"] = self._meta_learner.get_stats()
        return stats

    # ---------- 内部工具 ----------

    def _generate_synthetic_data(self, n_samples: int) -> tuple:
        """生成合成数据用于持续学习训练/评估"""
        data = np.random.randn(n_samples, self._input_dim).astype(np.float32)
        labels = np.random.randint(0, self._output_dim, size=n_samples).astype(np.float32)
        return data, labels

    # ---------- 格式化方法 ----------

    def format_init_result(self, results: Dict[str, bool], input_dim: int,
                           output_dim: int) -> str:
        """格式化初始化结果"""
        lines = ["=== 学习系统初始化 ==="]
        lines.append(f"  输入维度: {input_dim}")
        lines.append(f"  输出维度: {output_dim}")
        lines.append(f"  持续学习: {'成功' if results.get('continual') else '失败'}")
        lines.append(f"  元学习: {'成功' if results.get('meta') else '失败'}")
        if results.get("continual"):
            lines.append("    策略: EWC + 经验回放")
        if results.get("meta"):
            lines.append("    算法: MAML（模型无关元学习）")
        return "\n".join(lines)

    def format_train_result(self, train_type: str, result: Dict[str, Any]) -> str:
        """格式化训练结果"""
        if "error" in result:
            return f"=== {train_type}训练失败 ===\n  错误: {result['error']}"

        lines = [f"=== {train_type}训练完成 ==="]
        for key, value in result.items():
            if isinstance(value, (list, dict)):
                if isinstance(value, list) and len(value) > 0:
                    lines.append(f"  {key}: {len(value)}项")
                    if len(value) <= 5:
                        for v in value:
                            lines.append(f"    - {v}")
                elif isinstance(value, dict):
                    lines.append(f"  {key}:")
                    for k, v in value.items():
                        lines.append(f"    {k}: {v}")
                else:
                    lines.append(f"  {key}: {value}")
            else:
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def format_stats(self) -> str:
        """格式化统计"""
        stats = self.get_stats()
        lines = ["=== 学习系统统计 ==="]
        lines.append(f"  输入维度: {stats.get('input_dim', 4)}")
        lines.append(f"  输出维度: {stats.get('output_dim', 2)}")
        cl_init = "已初始化" if stats.get("continual_initialized") else "未初始化"
        ml_init = "已初始化" if stats.get("meta_initialized") else "未初始化"
        lines.append(f"  持续学习: {cl_init}")
        lines.append(f"  元学习: {ml_init}")

        if "continual" in stats:
            cl = stats["continual"]
            lines.append(f"\n  --- 持续学习详情 ---")
            lines.append(f"    策略: {cl.get('strategy', 'N/A')}")
            lines.append(f"    任务数: {cl.get('task_count', 0)}")
            lines.append(f"    当前任务: {cl.get('current_task', 'N/A')}")
            lines.append(f"    经验回放大小: {cl.get('replay_size', 0)}")
            lines.append(f"    EWC任务数: {cl.get('ewc_tasks', 0)}")
            forgetting = cl.get("forgetting_summary", {})
            if forgetting:
                lines.append(f"    遗忘摘要: {forgetting}")

        if "meta" in stats:
            ml = stats["meta"]
            lines.append(f"\n  --- 元学习详情 ---")
            lines.append(f"    元训练步数: {ml.get('meta_steps', 0)}")
            lines.append(f"    回放缓冲区: {ml.get('replay_size', 0)}")
            lines.append(f"    模型参数数: {ml.get('model_params', 0)}")

        return "\n".join(lines)

    def format_evaluate_result(self, eval_type: str, result: Dict[str, Any]) -> str:
        """格式化评估结果"""
        if "error" in result:
            return f"=== {eval_type}评估失败 ===\n  错误: {result['error']}"

        lines = [f"=== {eval_type}评估结果 ==="]
        for key, value in result.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        lines.append(f"    {k}: {v:.4f}")
                    else:
                        lines.append(f"    {k}: {v}")
            elif isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)
