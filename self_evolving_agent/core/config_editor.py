"""
U19：交互式配置编辑器 —— 引导式配置修改

第一性原理：
- 用户修改配置时，不应需要记忆YAML语法和字段名
- 应通过引导式交互完成常见配置修改
- 修改前应预览变更，修改后应验证
- 修改应可回滚（通过U12的备份机制）

设计：
1. EDITABLE_FIELDS：可编辑字段元数据（路径/类型/范围/描述/默认值）
2. ConfigEditor：引导式配置编辑器
   - list_fields()：列出可编辑字段
   - get_field()：获取字段当前值
   - set_field()：设置字段值（带验证）
   - interactive_edit()：交互式编辑菜单
   - preview_change()：预览变更
3. IO分离：input_fn/confirm_fn可注入，便于测试
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


@dataclass
class FieldMeta:
    """可编辑字段元数据"""
    path: str  # 字段路径，如 "resource.max_cpu_percent" 或 "main_loop_frequency"
    label: str  # 显示名称
    description: str  # 描述
    field_type: str  # "float" / "int" / "str" / "bool" / "choice"
    default: Any  # 默认值
    min_value: Optional[float] = None  # 最小值（数值类型）
    max_value: Optional[float] = None  # 最大值（数值类型）
    choices: Optional[List[str]] = None  # 可选值（choice类型）
    section: str = "其他"  # 分组

    def validate_value(self, value: Any) -> Tuple[bool, str]:
        """验证值是否合法

        Returns:
            (是否合法, 错误消息)
        """
        if self.field_type == "float":
            try:
                v = float(value)
            except (ValueError, TypeError):
                return False, f"值必须是浮点数，得到: {value}"
            if self.min_value is not None and v < self.min_value:
                return False, f"值不能小于 {self.min_value}，得到: {v}"
            if self.max_value is not None and v > self.max_value:
                return False, f"值不能大于 {self.max_value}，得到: {v}"
            return True, ""

        elif self.field_type == "int":
            try:
                v = int(value)
            except (ValueError, TypeError):
                return False, f"值必须是整数，得到: {value}"
            if self.min_value is not None and v < self.min_value:
                return False, f"值不能小于 {self.min_value}，得到: {v}"
            if self.max_value is not None and v > self.max_value:
                return False, f"值不能大于 {self.max_value}，得到: {v}"
            return True, ""

        elif self.field_type == "str":
            if not isinstance(value, str):
                return False, f"值必须是字符串，得到: {type(value).__name__}"
            return True, ""

        elif self.field_type == "bool":
            if isinstance(value, bool):
                return True, ""
            if isinstance(value, str):
                if value.lower() in ("true", "yes", "1", "on"):
                    return True, ""
                if value.lower() in ("false", "no", "0", "off"):
                    return True, ""
            return False, f"值必须是布尔值（true/false），得到: {value}"

        elif self.field_type == "choice":
            if self.choices and value not in self.choices:
                return False, f"值必须是 {self.choices} 之一，得到: {value}"
            return True, ""

        return True, ""

    def parse_value(self, raw: str) -> Any:
        """将字符串输入解析为目标类型"""
        if self.field_type == "float":
            return float(raw)
        elif self.field_type == "int":
            return int(raw)
        elif self.field_type == "bool":
            return raw.lower() in ("true", "yes", "1", "on")
        elif self.field_type == "choice":
            return raw
        else:  # str
            return raw

    def format_value(self, value: Any) -> str:
        """格式化值用于显示"""
        if self.field_type == "bool":
            return "是" if value else "否"
        elif self.field_type == "float":
            return f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
        else:
            return str(value)


# 可编辑字段定义
EDITABLE_FIELDS: List[FieldMeta] = [
    # 资源约束
    FieldMeta(
        path="resource.max_cpu_percent",
        label="CPU使用上限(%)",
        description="Agent自身CPU使用上限，防止拖垮宿主系统",
        field_type="float",
        default=20.0,
        min_value=1.0,
        max_value=100.0,
        section="资源约束",
    ),
    FieldMeta(
        path="resource.max_memory_percent",
        label="内存使用上限(%)",
        description="Agent自身内存使用上限",
        field_type="float",
        default=20.0,
        min_value=1.0,
        max_value=100.0,
        section="资源约束",
    ),
    # 驱力权重
    FieldMeta(
        path="drive_weights.safe",
        label="安全驱力权重",
        description="安全驱力权重，越高越保守",
        field_type="float",
        default=0.4,
        min_value=0.0,
        max_value=1.0,
        section="驱力权重",
    ),
    FieldMeta(
        path="drive_weights.gain",
        label="收益驱力权重",
        description="收益驱力权重，越高越进取",
        field_type="float",
        default=0.3,
        min_value=0.0,
        max_value=1.0,
        section="驱力权重",
    ),
    FieldMeta(
        path="drive_weights.eff",
        label="效率驱力权重",
        description="效率驱力权重，越高越节能",
        field_type="float",
        default=0.3,
        min_value=0.0,
        max_value=1.0,
        section="驱力权重",
    ),
    # 运行参数
    FieldMeta(
        path="main_loop_frequency",
        label="主循环频率(Hz)",
        description="主循环频率，1.0=每秒1轮，10.0=每秒10轮",
        field_type="float",
        default=1.0,
        min_value=0.1,
        max_value=100.0,
        section="运行参数",
    ),
    FieldMeta(
        path="perception_interval",
        label="感知间隔(秒)",
        description="数据采集间隔，越短越实时但CPU消耗越高",
        field_type="float",
        default=1.0,
        min_value=0.1,
        max_value=3600.0,
        section="运行参数",
    ),
    FieldMeta(
        path="decision_interval",
        label="决策间隔(秒)",
        description="决策间隔，越短越频繁但可能过度反应",
        field_type="float",
        default=1.0,
        min_value=0.1,
        max_value=3600.0,
        section="运行参数",
    ),
    # 日志
    FieldMeta(
        path="logging.console_level",
        label="控制台日志级别",
        description="控制台输出的日志级别",
        field_type="choice",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        section="日志",
    ),
    FieldMeta(
        path="logging.file_level",
        label="文件日志级别",
        description="文件记录的日志级别",
        field_type="choice",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        section="日志",
    ),
    FieldMeta(
        path="log_dir",
        label="日志目录",
        description="日志文件保存目录",
        field_type="str",
        default="./logs",
        section="日志",
    ),
    # 存储
    FieldMeta(
        path="model_snapshot_dir",
        label="模型快照目录",
        description="模型快照保存目录",
        field_type="str",
        default="./data/snapshots",
        section="存储",
    ),
    FieldMeta(
        path="database_path",
        label="数据库路径",
        description="长期记忆SQLite数据库路径",
        field_type="str",
        default="./data/agent_memory.db",
        section="存储",
    ),
]


def _default_input(prompt: str) -> str:
    return input(prompt)


def _default_confirm(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    raw = input(prompt + suffix)
    if not raw.strip():
        return default
    return raw.lower() in ("y", "yes", "true", "1")


class ConfigEditor:
    """交互式配置编辑器

    第一性原理：
    - 用户不需要记忆YAML语法
    - 通过菜单选择字段，引导输入
    - 修改前预览，修改后验证
    - 保存时自动备份
    """

    def __init__(self, config_path: Optional[str] = None,
                 input_fn: Callable[[str], str] = _default_input,
                 confirm_fn: Callable[[str, bool], bool] = _default_confirm):
        self.config_path = config_path or "config/agent.yaml"
        self.input_fn = input_fn
        self.confirm_fn = confirm_fn
        self._fields_by_path = {f.path: f for f in EDITABLE_FIELDS}

    def list_fields(self) -> List[FieldMeta]:
        """列出所有可编辑字段"""
        return list(EDITABLE_FIELDS)

    def list_fields_by_section(self) -> Dict[str, List[FieldMeta]]:
        """按分组列出字段"""
        result: Dict[str, List[FieldMeta]] = {}
        for f in EDITABLE_FIELDS:
            result.setdefault(f.section, []).append(f)
        return result

    def get_field_meta(self, path: str) -> Optional[FieldMeta]:
        """获取字段元数据"""
        return self._fields_by_path.get(path)

    def _get_nested_attr(self, obj: Any, path: str) -> Any:
        """获取嵌套属性值"""
        parts = path.split(".")
        current = obj
        for part in parts:
            current = getattr(current, part)
        return current

    def _set_nested_attr(self, obj: Any, path: str, value: Any):
        """设置嵌套属性值"""
        parts = path.split(".")
        current = obj
        for part in parts[:-1]:
            current = getattr(current, part)
        setattr(current, parts[-1], value)

    def get_field_value(self, config: Any, path: str) -> Any:
        """获取配置中指定字段的当前值"""
        return self._get_nested_attr(config, path)

    def set_field_value(self, config: Any, path: str, value: Any) -> bool:
        """设置配置中指定字段的值

        Returns:
            是否设置成功
        """
        meta = self.get_field_meta(path)
        if meta is None:
            return False
        valid, msg = meta.validate_value(value)
        if not valid:
            return False
        self._set_nested_attr(config, path, value)
        return True

    def preview_change(self, config: Any, path: str, new_value: Any) -> str:
        """预览变更

        Returns:
            预览文本
        """
        meta = self.get_field_meta(path)
        if meta is None:
            return f"未知字段: {path}"

        old_value = self.get_field_value(config, path)
        valid, msg = meta.validate_value(new_value)
        status = "有效" if valid else f"无效({msg})"

        lines = [
            f"=== 变更预览 ===",
            f"  字段: {meta.label} ({path})",
            f"  描述: {meta.description}",
            f"  当前值: {meta.format_value(old_value)}",
            f"  新值: {meta.format_value(new_value)}",
            f"  状态: {status}",
        ]
        if meta.field_type in ("float", "int") and meta.min_value is not None:
            lines.append(f"  范围: [{meta.min_value}, {meta.max_value}]")
        elif meta.field_type == "choice" and meta.choices:
            lines.append(f"  可选: {', '.join(meta.choices)}")
        return "\n".join(lines)

    def format_fields_list(self, config: Optional[Any] = None) -> str:
        """格式化字段列表

        Args:
            config: 若提供，则显示当前值
        """
        lines = ["=== 可编辑配置字段 ==="]
        sections = self.list_fields_by_section()
        for section_name, fields in sections.items():
            lines.append(f"\n[{section_name}]")
            for i, f in enumerate(fields, 1):
                value_str = ""
                if config is not None:
                    try:
                        current = self.get_field_value(config, f.path)
                        value_str = f" = {f.format_value(current)}"
                    except Exception:
                        value_str = " = (未设置)"
                lines.append(f"  {i:2d}. {f.label}{value_str}")
                lines.append(f"      路径: {f.path}")
                lines.append(f"      描述: {f.description}")
                if f.field_type in ("float", "int") and f.min_value is not None:
                    lines.append(f"      范围: [{f.min_value}, {f.max_value}]  默认: {f.default}")
                elif f.field_type == "choice" and f.choices:
                    lines.append(f"      可选: {', '.join(f.choices)}  默认: {f.default}")
                else:
                    lines.append(f"      默认: {f.default}")
        return "\n".join(lines)

    def interactive_edit(self, config: Any) -> bool:
        """交互式编辑配置

        Args:
            config: 配置对象（将被原地修改）

        Returns:
            是否有修改
        """
        modified = False
        while True:
            print("\n" + self.format_fields_list(config))
            print("\n输入字段路径或编号进行编辑（如 resource.max_cpu_percent 或 1）")
            print("输入 'q' 或 'quit' 退出")

            raw = self.input_fn("\n选择字段> ").strip()
            if not raw or raw.lower() in ("q", "quit", "exit"):
                break

            # 解析输入（编号或路径）
            meta = None
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(EDITABLE_FIELDS):
                    meta = EDITABLE_FIELDS[idx]
                else:
                    print(f"无效编号: {raw}")
                    continue
            else:
                meta = self.get_field_meta(raw)
                if meta is None:
                    print(f"未知字段: {raw}")
                    continue

            # 显示当前值并提示输入
            try:
                current = self.get_field_value(config, meta.path)
            except Exception as e:
                print(f"无法读取当前值: {e}")
                continue

            print(f"\n字段: {meta.label}")
            print(f"描述: {meta.description}")
            print(f"当前值: {meta.format_value(current)}")
            if meta.field_type in ("float", "int") and meta.min_value is not None:
                print(f"范围: [{meta.min_value}, {meta.max_value}]")
            elif meta.field_type == "choice" and meta.choices:
                print(f"可选: {', '.join(meta.choices)}")

            if meta.field_type == "bool":
                prompt = f"新值 (true/false) [{meta.format_value(current)}]: "
            else:
                prompt = f"新值 [{meta.format_value(current)}]: "

            new_raw = self.input_fn(prompt).strip()
            if not new_raw:
                print("未修改")
                continue

            # 解析和验证
            try:
                new_value = meta.parse_value(new_raw)
            except ValueError as e:
                print(f"解析失败: {e}")
                continue

            valid, msg = meta.validate_value(new_value)
            if not valid:
                print(f"验证失败: {msg}")
                continue

            # 预览
            print()
            print(self.preview_change(config, meta.path, new_value))

            # 确认
            if not self.confirm_fn("确认修改?", default=True):
                print("已取消")
                continue

            # 应用
            self._set_nested_attr(config, meta.path, new_value)
            print(f"已修改: {meta.label} = {meta.format_value(new_value)}")
            modified = True

        return modified

    def save_config(self, config: Any) -> bool:
        """保存配置到文件（带备份）

        Args:
            config: 配置对象

        Returns:
            是否保存成功
        """
        try:
            # 先备份
            from core.config_backup import ConfigBackupManager
            mgr = ConfigBackupManager(self.config_path)
            mgr.backup(reason="edit")

            # 保存
            from core.config import config_manager
            config_manager.config = config
            config_manager.save()
            return True
        except Exception:
            return False
