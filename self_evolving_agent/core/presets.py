"""
U11：配置预设与场景模板

第一性原理：常见场景应一键启用，而非手写每个参数。
预设是经过验证的配置组合，用户只需选择场景名即可获得合理配置。

预设优先级：--preset > 配置文件 > 默认值
预设只覆盖预设定义的字段，其他字段保持配置文件原值。
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class Preset:
    """配置预设定义"""
    name: str
    label: str
    description: str
    config: Dict[str, Any]


# 预设定义 —— 经过验证的配置组合
PRESETS: Dict[str, Preset] = {
    "balanced": Preset(
        name="balanced",
        label="均衡（推荐）",
        description="适合大多数场景，资源使用与性能平衡",
        config={
            "resource": {
                "max_cpu_percent": 20.0,
                "max_memory_percent": 20.0,
                "max_reconstruction_iterations": 3,
                "storage_retain_ratio": 0.30,
            },
            "drive_weights": {"safe": 0.4, "gain": 0.4, "eff": 0.2},
            "evolution": {"exploration_ratio": 0.30, "idle_replay_interval_hours": 6.0},
            "main_loop_frequency": 1.0,
        },
    ),
    "aggressive": Preset(
        name="aggressive",
        label="激进（高性能机器）",
        description="充分利用资源，最大化探索与收益",
        config={
            "resource": {
                "max_cpu_percent": 50.0,
                "max_memory_percent": 50.0,
                "max_reconstruction_iterations": 5,
                "storage_retain_ratio": 0.40,
            },
            "drive_weights": {"safe": 0.2, "gain": 0.6, "eff": 0.2},
            "evolution": {"exploration_ratio": 0.50, "idle_replay_interval_hours": 4.0},
            "main_loop_frequency": 2.0,
        },
    ),
    "conservative": Preset(
        name="conservative",
        label="保守（资源受限）",
        description="资源占用低，优先安全与稳定",
        config={
            "resource": {
                "max_cpu_percent": 10.0,
                "max_memory_percent": 10.0,
                "max_reconstruction_iterations": 2,
                "storage_retain_ratio": 0.20,
            },
            "drive_weights": {"safe": 0.6, "gain": 0.2, "eff": 0.2},
            "evolution": {"exploration_ratio": 0.20, "idle_replay_interval_hours": 8.0},
            "main_loop_frequency": 0.5,
        },
    ),
    "eco": Preset(
        name="eco",
        label="节能（嵌入式/边缘）",
        description="极低资源占用，适合嵌入式设备",
        config={
            "resource": {
                "max_cpu_percent": 5.0,
                "max_memory_percent": 5.0,
                "max_reconstruction_iterations": 1,
                "storage_retain_ratio": 0.15,
            },
            "drive_weights": {"safe": 0.5, "gain": 0.2, "eff": 0.3},
            "evolution": {"exploration_ratio": 0.15, "idle_replay_interval_hours": 12.0},
            "main_loop_frequency": 0.2,
        },
    ),
}


def list_presets() -> List[Preset]:
    """列出所有可用预设"""
    return list(PRESETS.values())


def get_preset(name: str) -> Optional[Preset]:
    """获取指定名称的预设"""
    return PRESETS.get(name)


def preset_names() -> List[str]:
    """获取所有预设名称"""
    return list(PRESETS.keys())


def apply_preset_to_config(preset_name: str, config) -> None:
    """将预设应用到现有配置对象（原地修改）

    只覆盖预设定义的字段，其他字段保持原值。
    预设优先级高于配置文件，但低于环境变量（环境变量在load时已应用）。

    Args:
        preset_name: 预设名称
        config: SystemConfig 实例

    Raises:
        ValueError: 预设名称不存在
    """
    preset = get_preset(preset_name)
    if preset is None:
        raise ValueError(
            f"未知预设: {preset_name}，可用预设: {preset_names()}"
        )
    cfg = preset.config
    # 资源约束
    if "resource" in cfg:
        rc = cfg["resource"]
        if "max_cpu_percent" in rc:
            config.resource.max_cpu_percent = rc["max_cpu_percent"]
        if "max_memory_percent" in rc:
            config.resource.max_memory_percent = rc["max_memory_percent"]
        if "max_reconstruction_iterations" in rc:
            config.resource.max_reconstruction_iterations = rc["max_reconstruction_iterations"]
        if "storage_retain_ratio" in rc:
            config.resource.storage_retain_ratio = rc["storage_retain_ratio"]
    # 驱力权重
    if "drive_weights" in cfg:
        dw = cfg["drive_weights"]
        if "safe" in dw:
            config.drive_weights.safe = dw["safe"]
        if "gain" in dw:
            config.drive_weights.gain = dw["gain"]
        if "eff" in dw:
            config.drive_weights.eff = dw["eff"]
    # 进化参数
    if "evolution" in cfg:
        ev = cfg["evolution"]
        if "exploration_ratio" in ev:
            config.evolution.exploration_ratio = ev["exploration_ratio"]
        if "idle_replay_interval_hours" in ev:
            config.evolution.idle_replay_interval_hours = ev["idle_replay_interval_hours"]
    # 主循环频率
    if "main_loop_frequency" in cfg:
        config.main_loop_frequency = cfg["main_loop_frequency"]


def format_presets_help() -> str:
    """格式化预设帮助文本"""
    lines = ["可用预设:"]
    for p in list_presets():
        lines.append(f"  {p.name:15s} — {p.label}")
        lines.append(f"  {'':15s}  {p.description}")
    return "\n".join(lines)
