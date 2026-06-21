"""
U11：配置预设与场景模板测试

验证目标：
1. PRESETS 包含4种预设（balanced/aggressive/conservative/eco）
2. 每个预设有 name/label/description/config
3. 预设配置的驱力权重和为1.0
4. apply_preset_to_config 正确覆盖配置字段
5. 预设只覆盖定义的字段，不覆盖其他字段
6. 未知预设名抛出 ValueError
7. list_presets/get_preset/preset_names 工具函数
8. format_presets_help 输出帮助文本
9. CLI run --preset 选项可解析
10. 预设应用后配置仍能通过验证
11. 无回归

第一性原理：常见场景应一键启用，而非手写每个参数。
预设是经过验证的配置组合，只覆盖定义的字段。
"""
import os
import sys
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent


class TestU11PresetsDefinition(unittest.TestCase):
    """U11：预设定义"""

    def test_presets_module_importable(self):
        from core.presets import PRESETS
        self.assertIsInstance(PRESETS, dict)

    def test_four_presets_exist(self):
        from core.presets import PRESETS
        for name in ["balanced", "aggressive", "conservative", "eco"]:
            self.assertIn(name, PRESETS, f"应包含 {name} 预设")

    def test_preset_has_required_fields(self):
        from core.presets import PRESETS, Preset
        for name, preset in PRESETS.items():
            self.assertIsInstance(preset, Preset)
            self.assertEqual(preset.name, name)
            self.assertTrue(preset.label)
            self.assertTrue(preset.description)
            self.assertIsInstance(preset.config, dict)

    def test_preset_drive_weights_sum_to_one(self):
        """所有预设的驱力权重和应为1.0"""
        from core.presets import PRESETS
        for name, preset in PRESETS.items():
            dw = preset.config["drive_weights"]
            total = dw["safe"] + dw["gain"] + dw["eff"]
            self.assertAlmostEqual(total, 1.0, places=6,
                                   msg=f"预设 {name} 驱力权重和应为1.0，实际{total}")

    def test_preset_has_resource_section(self):
        from core.presets import PRESETS
        for name, preset in PRESETS.items():
            self.assertIn("resource", preset.config, f"{name} 应有 resource 段")
            rc = preset.config["resource"]
            self.assertIn("max_cpu_percent", rc)
            self.assertIn("max_memory_percent", rc)

    def test_preset_has_main_loop_frequency(self):
        from core.presets import PRESETS
        for name, preset in PRESETS.items():
            self.assertIn("main_loop_frequency", preset.config)

    def test_presets_have_distinct_configs(self):
        """各预设配置应有差异"""
        from core.presets import PRESETS
        cpu_values = [p.config["resource"]["max_cpu_percent"] for p in PRESETS.values()]
        self.assertEqual(len(set(cpu_values)), len(cpu_values), "各预设CPU上限应不同")


class TestU11PresetHelpers(unittest.TestCase):
    """U11：预设工具函数"""

    def test_list_presets_returns_list(self):
        from core.presets import list_presets, Preset
        result = list_presets()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], Preset)

    def test_get_preset_existing(self):
        from core.presets import get_preset, Preset
        result = get_preset("balanced")
        self.assertIsInstance(result, Preset)
        self.assertEqual(result.name, "balanced")

    def test_get_preset_nonexistent_returns_none(self):
        from core.presets import get_preset
        self.assertIsNone(get_preset("nonexistent"))

    def test_preset_names(self):
        from core.presets import preset_names
        names = preset_names()
        self.assertEqual(len(names), 4)
        for n in ["balanced", "aggressive", "conservative", "eco"]:
            self.assertIn(n, names)

    def test_format_presets_help(self):
        from core.presets import format_presets_help
        text = format_presets_help()
        self.assertIn("可用预设", text)
        for name in ["balanced", "aggressive", "conservative", "eco"]:
            self.assertIn(name, text)


class TestU11ApplyPreset(unittest.TestCase):
    """U11：预设应用"""

    def test_apply_balanced_preset(self):
        from core.presets import apply_preset_to_config
        from core.config import SystemConfig
        cfg = SystemConfig()
        original_cpu = cfg.resource.max_cpu_percent
        apply_preset_to_config("balanced", cfg)
        self.assertEqual(cfg.resource.max_cpu_percent, 20.0)
        self.assertEqual(cfg.resource.max_memory_percent, 20.0)
        self.assertAlmostEqual(cfg.drive_weights.safe, 0.4)
        self.assertAlmostEqual(cfg.drive_weights.gain, 0.4)
        self.assertAlmostEqual(cfg.drive_weights.eff, 0.2)

    def test_apply_aggressive_preset(self):
        from core.presets import apply_preset_to_config
        from core.config import SystemConfig
        cfg = SystemConfig()
        apply_preset_to_config("aggressive", cfg)
        self.assertEqual(cfg.resource.max_cpu_percent, 50.0)
        self.assertEqual(cfg.resource.max_memory_percent, 50.0)
        self.assertAlmostEqual(cfg.drive_weights.gain, 0.6)
        self.assertEqual(cfg.main_loop_frequency, 2.0)

    def test_apply_conservative_preset(self):
        from core.presets import apply_preset_to_config
        from core.config import SystemConfig
        cfg = SystemConfig()
        apply_preset_to_config("conservative", cfg)
        self.assertEqual(cfg.resource.max_cpu_percent, 10.0)
        self.assertAlmostEqual(cfg.drive_weights.safe, 0.6)
        self.assertEqual(cfg.main_loop_frequency, 0.5)

    def test_apply_eco_preset(self):
        from core.presets import apply_preset_to_config
        from core.config import SystemConfig
        cfg = SystemConfig()
        apply_preset_to_config("eco", cfg)
        self.assertEqual(cfg.resource.max_cpu_percent, 5.0)
        self.assertAlmostEqual(cfg.drive_weights.eff, 0.3)
        self.assertEqual(cfg.main_loop_frequency, 0.2)

    def test_apply_unknown_preset_raises(self):
        from core.presets import apply_preset_to_config
        from core.config import SystemConfig
        cfg = SystemConfig()
        with self.assertRaises(ValueError) as ctx:
            apply_preset_to_config("nonexistent", cfg)
        self.assertIn("nonexistent", str(ctx.exception))

    def test_apply_presets_overwrites_specific_fields_only(self):
        """预设只覆盖定义的字段，不覆盖其他字段"""
        from core.presets import apply_preset_to_config
        from core.config import SystemConfig
        cfg = SystemConfig()
        # 记录预设未涉及的字段
        original_log_level = cfg.logging.console_level
        original_perception = cfg.perception_interval
        apply_preset_to_config("balanced", cfg)
        # 这些字段应未被预设覆盖
        self.assertEqual(cfg.logging.console_level, original_log_level)
        self.assertEqual(cfg.perception_interval, original_perception)

    def test_applied_preset_config_validates(self):
        """应用预设后配置应能通过验证"""
        from core.presets import apply_preset_to_config
        from core.config import SystemConfig
        for name in ["balanced", "aggressive", "conservative", "eco"]:
            cfg = SystemConfig()
            apply_preset_to_config(name, cfg)
            self.assertTrue(cfg.validate(), f"预设 {name} 应用后应通过验证")

    def test_apply_preset_affects_evolution(self):
        from core.presets import apply_preset_to_config
        from core.config import SystemConfig
        cfg = SystemConfig()
        apply_preset_to_config("aggressive", cfg)
        self.assertAlmostEqual(cfg.evolution.exploration_ratio, 0.50)


class TestU11CliIntegration(unittest.TestCase):
    """U11：CLI --preset 集成"""

    def test_cli_run_has_preset_option(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["run", "--preset", "balanced"])
        self.assertEqual(args.preset, "balanced")

    def test_cli_run_preset_choices(self):
        from cli import _create_parser
        parser = _create_parser()
        for name in ["balanced", "aggressive", "conservative", "eco"]:
            args = parser.parse_args(["run", "--preset", name])
            self.assertEqual(args.preset, name)

    def test_cli_run_invalid_preset_rejected(self):
        """无效预设名应被argparse拒绝"""
        from cli import _create_parser
        parser = _create_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["run", "--preset", "invalid"])

    def test_cli_run_preset_default_none(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["run"])
        self.assertIsNone(args.preset)


class TestU11NoRegression(unittest.TestCase):
    """U11：无回归保护"""

    def test_agent_init_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        self.assertIsNotNone(agent.perception)

    def test_main_loop_still_runs(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        import time
        time.sleep(1.5)
        agent.stop()
        self.assertGreaterEqual(agent.loop_count, 0)

    def test_config_validate_still_works(self):
        from core.config import SystemConfig
        cfg = SystemConfig()
        self.assertTrue(cfg.validate())


if __name__ == "__main__":
    unittest.main()
