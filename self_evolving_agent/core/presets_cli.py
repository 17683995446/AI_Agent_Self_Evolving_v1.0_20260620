"""
U33：预设与场景发现 —— 预设/场景CLI辅助

第一性原理：
- 预设是经过验证的配置组合，场景是模块+配置+适配器的完整运行模式
- 两者都是"一键启用"能力，但CLI仅通过run --preset/--scenario间接暴露
- 用户需要独立查看预设/场景详情，无需启动Agent即可决策
- 预设可应用到当前配置，场景可查看其配置覆盖

设计：
1. PresetsCliHelper：预设CLI辅助器
   - 封装core.presets的CLI访问
   - 支持list/show/apply
2. ScenariosCliHelper：场景CLI辅助器
   - 封装ScenarioLoader的CLI访问（只读，不应用）
   - 支持list/info
3. 复用既有模块，不重复造轮子
"""

import json
from typing import Dict, List, Optional, Any


class PresetsCliHelper:
    """预设CLI辅助器

    轻量级访问core.presets，不启动完整Agent
    """

    def __init__(self):
        from core.presets import PRESETS
        self._presets = PRESETS

    def list_presets(self) -> List[Dict[str, Any]]:
        """列出所有预设"""
        from core.presets import list_presets as _list
        return [
            {
                "name": p.name,
                "label": p.label,
                "description": p.description,
            }
            for p in _list()
        ]

    def get_preset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取预设详情"""
        from core.presets import get_preset
        preset = get_preset(name)
        if preset is None:
            return None
        return {
            "name": preset.name,
            "label": preset.label,
            "description": preset.description,
            "config": preset.config,
        }

    def apply_preset(self, name: str) -> bool:
        """应用预设到当前配置

        Args:
            name: 预设名称

        Returns:
            是否成功应用
        """
        from core.presets import apply_preset_to_config, get_preset
        if get_preset(name) is None:
            return False
        try:
            from core.config import config_manager
            apply_preset_to_config(name, config_manager.config)
            return True
        except Exception:
            return False

    def get_available_names(self) -> List[str]:
        """获取可用预设名称"""
        return list(self._presets.keys())

    # ---------- 格式化方法 ----------

    def format_list(self) -> str:
        """格式化预设列表"""
        presets = self.list_presets()
        lines = [f"=== 配置预设（{len(presets)}个）==="]
        lines.append(f"{'名称':<15} {'标签':<20} {'描述'}")
        lines.append("-" * 70)
        for p in presets:
            lines.append(f"{p['name']:<15} {p['label']:<20} {p['description']}")
        lines.append("")
        lines.append("使用 'presets show <名称>' 查看详情")
        lines.append("使用 'presets apply <名称>' 应用到当前配置")
        return "\n".join(lines)

    def format_info(self, name: str) -> str:
        """格式化预设详情"""
        info = self.get_preset_info(name)
        if info is None:
            return f"预设 '{name}' 不存在，可用预设: {self.get_available_names()}"

        lines = [f"=== 预设详情: {info['name']} ==="]
        lines.append(f"  名称: {info['name']}")
        lines.append(f"  标签: {info['label']}")
        lines.append(f"  描述: {info['description']}")

        config = info["config"]
        lines.append(f"\n  配置项:")

        if "resource" in config:
            lines.append(f"    资源约束:")
            for k, v in config["resource"].items():
                lines.append(f"      {k}: {v}")

        if "drive_weights" in config:
            lines.append(f"    驱力权重:")
            for k, v in config["drive_weights"].items():
                lines.append(f"      {k}: {v}")

        if "evolution" in config:
            lines.append(f"    进化参数:")
            for k, v in config["evolution"].items():
                lines.append(f"      {k}: {v}")

        if "main_loop_frequency" in config:
            lines.append(f"    主循环频率: {config['main_loop_frequency']}")

        return "\n".join(lines)

    def format_apply_result(self, name: str, success: bool) -> str:
        """格式化应用结果"""
        if success:
            lines = [f"=== 预设 '{name}' 已应用 ==="]
            lines.append("  配置已更新（仅当前会话生效）")
            lines.append("  使用 'config validate' 验证配置")
            lines.append("  使用 'run --preset " + name + "' 在启动时应用")
        else:
            lines = [f"=== 预设 '{name}' 应用失败 ==="]
            lines.append(f"  可用预设: {self.get_available_names()}")
        return "\n".join(lines)


class ScenariosCliHelper:
    """场景CLI辅助器

    轻量级访问ScenarioLoader（只读模式，不应用配置覆盖）
    """

    def __init__(self):
        from core.scenario_loader import ScenarioLoader
        self.loader = ScenarioLoader()

    def list_scenarios(self) -> List[str]:
        """列出所有场景名称"""
        return self.loader.list_scenarios()

    def get_scenario_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取场景信息（不应用）"""
        return self.loader.get_scenario_info(name)

    # ---------- 格式化方法 ----------

    def format_list(self) -> str:
        """格式化场景列表"""
        scenarios = self.list_scenarios()
        if not scenarios:
            return "（无可用场景，请检查 config/scenarios.yaml）"

        lines = [f"=== 运行场景（{len(scenarios)}个）==="]
        lines.append(f"{'名称':<30} {'描述'}")
        lines.append("-" * 70)
        for name in scenarios:
            info = self.get_scenario_info(name)
            desc = info.get("description", "") if info else ""
            lines.append(f"{name:<30} {desc}")
        lines.append("")
        lines.append("使用 'scenarios info <名称>' 查看详情")
        lines.append("使用 'run --scenario <名称>' 加载场景运行")
        return "\n".join(lines)

    def format_info(self, name: str) -> str:
        """格式化场景详情"""
        info = self.get_scenario_info(name)
        if info is None:
            available = self.list_scenarios()
            return f"场景 '{name}' 不存在，可用场景: {available}"

        lines = [f"=== 场景详情: {name} ==="]
        lines.append(f"  名称: {name}")
        lines.append(f"  描述: {info.get('description', '')}")

        modules = info.get("modules", {})
        if modules:
            lines.append(f"\n  模块配置:")
            for k, v in modules.items():
                lines.append(f"    {k}: {v}")

        overrides = info.get("config_overrides", {})
        if overrides:
            lines.append(f"\n  配置覆盖:")
            for k, v in overrides.items():
                if isinstance(v, dict):
                    lines.append(f"    {k}:")
                    for sk, sv in v.items():
                        lines.append(f"      {sk}: {sv}")
                else:
                    lines.append(f"    {k}: {v}")

        adapters = info.get("adapters", [])
        if adapters:
            lines.append(f"\n  适配器（{len(adapters)}个）:")
            for adapter in adapters:
                lines.append(f"    - {adapter}")

        alert_channels = info.get("alert_channels", [])
        if alert_channels:
            lines.append(f"\n  告警通道（{len(alert_channels)}个）:")
            for ch in alert_channels:
                lines.append(f"    - {ch}")

        allowed_services = info.get("allowed_services", [])
        if allowed_services:
            lines.append(f"\n  允许的服务:")
            for svc in allowed_services:
                lines.append(f"    - {svc}")

        allowed_kill = info.get("allowed_kill_patterns", [])
        if allowed_kill:
            lines.append(f"\n  允许的终止模式:")
            for pat in allowed_kill:
                lines.append(f"    - {pat}")

        return "\n".join(lines)
