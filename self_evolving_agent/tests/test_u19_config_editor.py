"""
U19：交互式配置编辑器 —— 引导式配置修改

测试覆盖：
1. FieldMeta数据类（验证、解析、格式化）
2. EDITABLE_FIELDS定义（完整性）
3. ConfigEditor基础（字段列表、获取元数据）
4. 字段值读写（get/set nested attr）
5. 值验证（类型/范围/选择）
6. 变更预览
7. 交互式编辑（IO注入）
8. CLI集成（config edit/get/set/fields）
9. 无回归
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field as dc_field
from typing import Dict

# 路径处理
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@dataclass
class MockResource:
    max_cpu_percent: float = 20.0
    max_memory_percent: float = 20.0


@dataclass
class MockDriveWeights:
    safe: float = 0.4
    gain: float = 0.3
    eff: float = 0.3


@dataclass
class MockLogging:
    console_level: str = "INFO"
    file_level: str = "DEBUG"


@dataclass
class MockConfig:
    resource: MockResource = dc_field(default_factory=MockResource)
    drive_weights: MockDriveWeights = dc_field(default_factory=MockDriveWeights)
    logging: MockLogging = dc_field(default_factory=MockLogging)
    main_loop_frequency: float = 1.0
    log_dir: str = "./logs"


class TestU19FieldMeta(unittest.TestCase):
    """测试FieldMeta数据类"""

    def test_fieldmeta_importable(self):
        from core.config_editor import FieldMeta
        self.assertTrue(callable(FieldMeta))

    def test_fieldmeta_instantiable(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(
            path="test.field",
            label="测试字段",
            description="测试",
            field_type="float",
            default=1.0,
        )
        self.assertEqual(meta.path, "test.field")

    def test_validate_float_valid(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="float",
                         default=1.0, min_value=0.0, max_value=10.0)
        valid, _ = meta.validate_value(5.0)
        self.assertTrue(valid)

    def test_validate_float_out_of_range(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="float",
                         default=1.0, min_value=0.0, max_value=10.0)
        valid, msg = meta.validate_value(15.0)
        self.assertFalse(valid)
        self.assertIn("不能大于", msg)

    def test_validate_float_invalid_type(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="float", default=1.0)
        valid, msg = meta.validate_value("abc")
        self.assertFalse(valid)
        self.assertIn("浮点数", msg)

    def test_validate_int_valid(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="int",
                         default=1, min_value=0, max_value=100)
        valid, _ = meta.validate_value(50)
        self.assertTrue(valid)

    def test_validate_int_invalid(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="int", default=1)
        valid, _ = meta.validate_value("abc")
        self.assertFalse(valid)

    def test_validate_choice_valid(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="choice",
                         default="INFO", choices=["DEBUG", "INFO", "WARNING"])
        valid, _ = meta.validate_value("INFO")
        self.assertTrue(valid)

    def test_validate_choice_invalid(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="choice",
                         default="INFO", choices=["DEBUG", "INFO", "WARNING"])
        valid, msg = meta.validate_value("VERBOSE")
        self.assertFalse(valid)
        self.assertIn("VERBOSE", msg)

    def test_validate_bool_valid(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="bool", default=False)
        valid, _ = meta.validate_value(True)
        self.assertTrue(valid)

    def test_validate_bool_string_true(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="bool", default=False)
        valid, _ = meta.validate_value("true")
        self.assertTrue(valid)

    def test_validate_bool_string_false(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="bool", default=False)
        valid, _ = meta.validate_value("no")
        self.assertTrue(valid)

    def test_validate_bool_invalid(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="bool", default=False)
        valid, _ = meta.validate_value("maybe")
        self.assertFalse(valid)

    def test_parse_value_float(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="float", default=1.0)
        self.assertEqual(meta.parse_value("3.14"), 3.14)

    def test_parse_value_int(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="int", default=1)
        self.assertEqual(meta.parse_value("42"), 42)

    def test_parse_value_bool_true(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="bool", default=False)
        self.assertTrue(meta.parse_value("yes"))

    def test_parse_value_bool_false(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="bool", default=True)
        self.assertFalse(meta.parse_value("no"))

    def test_parse_value_choice(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="choice",
                         default="INFO", choices=["INFO"])
        self.assertEqual(meta.parse_value("DEBUG"), "DEBUG")

    def test_format_value_bool(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="bool", default=False)
        self.assertEqual(meta.format_value(True), "是")
        self.assertEqual(meta.format_value(False), "否")

    def test_format_value_float(self):
        from core.config_editor import FieldMeta
        meta = FieldMeta(path="x", label="x", description="", field_type="float", default=1.0)
        self.assertEqual(meta.format_value(3.14159), "3.14")


class TestU19EditableFields(unittest.TestCase):
    """测试EDITABLE_FIELDS定义"""

    def test_editable_fields_defined(self):
        from core.config_editor import EDITABLE_FIELDS
        self.assertIsInstance(EDITABLE_FIELDS, list)
        self.assertGreater(len(EDITABLE_FIELDS), 0)

    def test_has_resource_fields(self):
        from core.config_editor import EDITABLE_FIELDS
        paths = [f.path for f in EDITABLE_FIELDS]
        self.assertIn("resource.max_cpu_percent", paths)
        self.assertIn("resource.max_memory_percent", paths)

    def test_has_drive_weights_fields(self):
        from core.config_editor import EDITABLE_FIELDS
        paths = [f.path for f in EDITABLE_FIELDS]
        self.assertIn("drive_weights.safe", paths)
        self.assertIn("drive_weights.gain", paths)
        self.assertIn("drive_weights.eff", paths)

    def test_has_main_loop_frequency(self):
        from core.config_editor import EDITABLE_FIELDS
        paths = [f.path for f in EDITABLE_FIELDS]
        self.assertIn("main_loop_frequency", paths)

    def test_has_logging_fields(self):
        from core.config_editor import EDITABLE_FIELDS
        paths = [f.path for f in EDITABLE_FIELDS]
        self.assertIn("logging.console_level", paths)
        self.assertIn("logging.file_level", paths)

    def test_all_fields_have_required_attributes(self):
        from core.config_editor import EDITABLE_FIELDS
        for f in EDITABLE_FIELDS:
            self.assertIsNotNone(f.path)
            self.assertIsNotNone(f.label)
            self.assertIsNotNone(f.description)
            self.assertIsNotNone(f.field_type)
            self.assertIsNotNone(f.default)
            self.assertIsNotNone(f.section)

    def test_all_fields_have_sections(self):
        from core.config_editor import EDITABLE_FIELDS
        sections = {f.section for f in EDITABLE_FIELDS}
        self.assertIn("资源约束", sections)
        self.assertIn("驱力权重", sections)


class TestU19ConfigEditor(unittest.TestCase):
    """测试ConfigEditor"""

    def test_editor_importable(self):
        from core.config_editor import ConfigEditor
        self.assertTrue(callable(ConfigEditor))

    def test_editor_instantiable(self):
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        self.assertIsNotNone(editor)

    def test_list_fields(self):
        from core.config_editor import ConfigEditor, EDITABLE_FIELDS
        editor = ConfigEditor()
        fields = editor.list_fields()
        self.assertEqual(len(fields), len(EDITABLE_FIELDS))

    def test_list_fields_by_section(self):
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        sections = editor.list_fields_by_section()
        self.assertIsInstance(sections, dict)
        self.assertGreater(len(sections), 0)

    def test_get_field_meta_existing(self):
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        meta = editor.get_field_meta("resource.max_cpu_percent")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.path, "resource.max_cpu_percent")

    def test_get_field_meta_nonexistent(self):
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        meta = editor.get_field_meta("nonexistent.field")
        self.assertIsNone(meta)

    def test_get_field_value(self):
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        config = MockConfig()
        value = editor.get_field_value(config, "resource.max_cpu_percent")
        self.assertEqual(value, 20.0)

    def test_get_field_value_nested(self):
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        config = MockConfig()
        value = editor.get_field_value(config, "drive_weights.safe")
        self.assertEqual(value, 0.4)

    def test_get_field_value_top_level(self):
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        config = MockConfig()
        value = editor.get_field_value(config, "main_loop_frequency")
        self.assertEqual(value, 1.0)

    def test_set_field_value(self):
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        config = MockConfig()
        success = editor.set_field_value(config, "resource.max_cpu_percent", 50.0)
        self.assertTrue(success)
        self.assertEqual(config.resource.max_cpu_percent, 50.0)

    def test_set_field_value_nested(self):
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        config = MockConfig()
        success = editor.set_field_value(config, "drive_weights.safe", 0.6)
        self.assertTrue(success)
        self.assertEqual(config.drive_weights.safe, 0.6)

    def test_set_field_value_invalid(self):
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        config = MockConfig()
        # 超出范围的值应失败
        success = editor.set_field_value(config, "resource.max_cpu_percent", 200.0)
        self.assertFalse(success)

    def test_set_field_value_unknown_path(self):
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        config = MockConfig()
        success = editor.set_field_value(config, "nonexistent.field", 1.0)
        self.assertFalse(success)

    def test_preview_change(self):
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        config = MockConfig()
        preview = editor.preview_change(config, "resource.max_cpu_percent", 50.0)
        self.assertIn("变更预览", preview)
        self.assertIn("50.00", preview)
        self.assertIn("20.00", preview)

    def test_preview_change_unknown_field(self):
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        config = MockConfig()
        preview = editor.preview_change(config, "nonexistent", 1.0)
        self.assertIn("未知字段", preview)

    def test_format_fields_list(self):
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        output = editor.format_fields_list()
        self.assertIn("可编辑配置字段", output)
        self.assertIn("资源约束", output)
        self.assertIn("CPU使用上限", output)

    def test_format_fields_list_with_config(self):
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        config = MockConfig()
        output = editor.format_fields_list(config)
        self.assertIn("20.00", output)  # 当前值


class TestU19InteractiveEdit(unittest.TestCase):
    """测试交互式编辑（IO注入）"""

    def test_interactive_edit_quit_immediately(self):
        """立即退出不应修改"""
        from core.config_editor import ConfigEditor
        inputs = iter(["q"])
        editor = ConfigEditor(input_fn=lambda p: next(inputs))
        config = MockConfig()
        original = config.resource.max_cpu_percent
        modified = editor.interactive_edit(config)
        self.assertFalse(modified)
        self.assertEqual(config.resource.max_cpu_percent, original)

    def test_interactive_edit_empty_input(self):
        """空输入应退出"""
        from core.config_editor import ConfigEditor
        inputs = iter([""])
        editor = ConfigEditor(input_fn=lambda p: next(inputs))
        config = MockConfig()
        modified = editor.interactive_edit(config)
        self.assertFalse(modified)

    def test_interactive_edit_by_number(self):
        """通过编号编辑字段"""
        from core.config_editor import ConfigEditor
        # 输入: "1"选择第一个字段, "50"新值, "y"确认, "q"退出
        inputs = iter(["1", "50", "y", "q"])
        confirms = iter([True, False])
        editor = ConfigEditor(
            input_fn=lambda p: next(inputs),
            confirm_fn=lambda p, default=False: next(confirms)
        )
        config = MockConfig()
        modified = editor.interactive_edit(config)
        self.assertTrue(modified)
        self.assertEqual(config.resource.max_cpu_percent, 50.0)

    def test_interactive_edit_by_path(self):
        """通过路径编辑字段"""
        from core.config_editor import ConfigEditor
        inputs = iter(["main_loop_frequency", "5.0", "y", "q"])
        confirms = iter([True])
        editor = ConfigEditor(
            input_fn=lambda p: next(inputs),
            confirm_fn=lambda p, default=False: next(confirms)
        )
        config = MockConfig()
        modified = editor.interactive_edit(config)
        self.assertTrue(modified)
        self.assertEqual(config.main_loop_frequency, 5.0)

    def test_interactive_edit_cancel_change(self):
        """取消修改不应改变配置"""
        from core.config_editor import ConfigEditor
        inputs = iter(["1", "50", "n", "q"])
        confirms = iter([False])  # 不确认
        editor = ConfigEditor(
            input_fn=lambda p: next(inputs),
            confirm_fn=lambda p, default=False: next(confirms)
        )
        config = MockConfig()
        original = config.resource.max_cpu_percent
        modified = editor.interactive_edit(config)
        self.assertFalse(modified)
        self.assertEqual(config.resource.max_cpu_percent, original)

    def test_interactive_edit_empty_new_value(self):
        """空新值应跳过"""
        from core.config_editor import ConfigEditor
        inputs = iter(["1", "", "q"])
        editor = ConfigEditor(input_fn=lambda p: next(inputs))
        config = MockConfig()
        modified = editor.interactive_edit(config)
        self.assertFalse(modified)

    def test_interactive_edit_invalid_number(self):
        """无效编号应继续"""
        from core.config_editor import ConfigEditor
        inputs = iter(["999", "q"])
        editor = ConfigEditor(input_fn=lambda p: next(inputs))
        config = MockConfig()
        modified = editor.interactive_edit(config)
        self.assertFalse(modified)

    def test_interactive_edit_unknown_path(self):
        """未知路径应继续"""
        from core.config_editor import ConfigEditor
        inputs = iter(["unknown.path", "q"])
        editor = ConfigEditor(input_fn=lambda p: next(inputs))
        config = MockConfig()
        modified = editor.interactive_edit(config)
        self.assertFalse(modified)

    def test_interactive_edit_invalid_value(self):
        """无效值应跳过"""
        from core.config_editor import ConfigEditor
        # 输入"abc"作为float字段的值
        inputs = iter(["1", "abc", "q"])
        editor = ConfigEditor(input_fn=lambda p: next(inputs))
        config = MockConfig()
        modified = editor.interactive_edit(config)
        self.assertFalse(modified)

    def test_interactive_edit_out_of_range(self):
        """超出范围的值应跳过"""
        from core.config_editor import ConfigEditor
        inputs = iter(["1", "200", "q"])
        editor = ConfigEditor(input_fn=lambda p: next(inputs))
        config = MockConfig()
        modified = editor.interactive_edit(config)
        self.assertFalse(modified)


class TestU19CliIntegration(unittest.TestCase):
    """测试CLI集成"""

    def test_config_edit_subcommand_exists(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["config", "edit"])
        self.assertEqual(args.command, "config")
        self.assertEqual(args.config_action, "edit")

    def test_config_get_subcommand_exists(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["config", "get", "main_loop_frequency"])
        self.assertEqual(args.config_action, "get")
        self.assertEqual(args.field, "main_loop_frequency")

    def test_config_set_subcommand_exists(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["config", "set", "main_loop_frequency", "5.0"])
        self.assertEqual(args.config_action, "set")
        self.assertEqual(args.field, "main_loop_frequency")
        self.assertEqual(args.value, "5.0")

    def test_config_fields_subcommand_exists(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["config", "fields"])
        self.assertEqual(args.config_action, "fields")

    def test_cli_config_fields_returns_0(self):
        from cli import cli_main
        rc = cli_main(["config", "fields"])
        self.assertEqual(rc, 0)

    def test_cli_config_get_valid_field(self):
        from cli import cli_main
        rc = cli_main(["config", "get", "main_loop_frequency"])
        self.assertEqual(rc, 0)

    def test_cli_config_get_invalid_field(self):
        from cli import cli_main
        rc = cli_main(["config", "get", "nonexistent.field"])
        self.assertEqual(rc, 1)


class TestU19NoRegression(unittest.TestCase):
    """验证无回归"""

    def test_existing_config_commands_still_work(self):
        from cli import _create_parser
        parser = _create_parser()
        for action in ["validate", "show", "reload", "backup", "rollback", "backups", "migrate", "check", "edit", "get", "set", "fields"]:
            args = parser.parse_args(["config", action] if action not in ("get", "set") else
                                     ["config", action, "main_loop_frequency"] + (["1.0"] if action == "set" else []))
            self.assertEqual(args.command, "config")

    def test_other_commands_still_work(self):
        from cli import _create_parser
        parser = _create_parser()
        for cmd in ["run", "status", "health", "snapshot", "init", "doctor", "shell", "alias", "completion", "theme"]:
            args = parser.parse_args([cmd])
            self.assertEqual(args.command, cmd)


if __name__ == "__main__":
    unittest.main()
