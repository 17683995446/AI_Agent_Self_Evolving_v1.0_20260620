"""
U28：插件管理 —— 模块插件CLI辅助

第一性原理：
- 系统支持乐高式插件化，但用户无法通过CLI查看可用插件
- 用户需要知道有哪些插件可用、各插件的作用、如何使用
- 插件管理应轻量级，尽量不依赖完整Agent初始化
- 插件信息应包含类名、文档、模块路径等关键信息

设计：
1. PluginsCliHelper：插件CLI辅助器
   - 封装ModuleRegistry的CLI访问
   - 列出所有模块类型及其注册的插件
   - 查看插件详情（类名/文档/模块路径）
   - 显示如何使用指定插件（配置示例）
2. 优雅处理依赖缺失（如numpy）
"""

import os
import sys
from typing import Dict, List, Optional, Any


class PluginsCliHelper:
    """插件CLI辅助器

    轻量级访问ModuleRegistry，查看和管理模块插件
    """

    # 模块类型到中文名的映射
    MODULE_TYPE_NAMES = {
        "perception": "感知",
        "prediction": "预测",
        "drive_engine": "驱力引擎",
        "decision": "决策",
        "error_eval": "错误评估",
        "optimizer": "优化器",
        "memory": "记忆",
    }

    # 模块类型到配置键的映射
    MODULE_CONFIG_KEYS = {
        "perception": "perception",
        "prediction": "prediction",
        "drive_engine": "drive_engine",
        "decision": "decision",
        "error_eval": "error_eval",
        "optimizer": "optimizer",
        "memory": "memory",
    }

    def __init__(self):
        self._registry = None
        self._load_errors: List[str] = []

    @property
    def registry(self):
        """获取注册表，优雅处理加载失败"""
        if self._registry is None:
            self._registry = self._load_registry()
        return self._registry

    def _load_registry(self):
        """加载注册表，优雅处理依赖缺失"""
        from core.registry import ModuleRegistry
        registry = ModuleRegistry()

        # 尝试加载各模块，跳过依赖缺失的
        load_attempts = [
            ("perception", "default", "modules.perception", "EnvironmentPerception"),
            ("drive_engine", "default", "modules.drive_engine", "DriveEngine"),
            ("prediction", "default", "modules.prediction", "PredictionEngine"),
            ("prediction", "enhanced", "modules.prediction", "EnhancedPredictionEngine"),
            ("decision", "default", "modules.decision", "DecisionEngine"),
            ("error_eval", "default", "modules.error_eval", "ErrorEvaluator"),
            ("optimizer", "default", "modules.optimizer", "Optimizer"),
            ("memory", "default", "modules.memory", "MemorySystem"),
        ]

        for module_type, name, module_path, class_name in load_attempts:
            try:
                __import__(module_path)
                module = sys.modules[module_path]
                cls = getattr(module, class_name)
                register_method = getattr(registry, f"register_{module_type}")
                register_method(name, cls)
            except Exception as e:
                self._load_errors.append(f"{module_path}.{class_name}: {e}")

        return registry

    def list_module_types(self) -> List[str]:
        """列出所有模块类型"""
        return list(self.MODULE_TYPE_NAMES.keys())

    def list_plugins(self, module_type: Optional[str] = None) -> Dict[str, List[str]]:
        """列出所有插件

        Args:
            module_type: 指定模块类型，None返回全部

        Returns:
            {module_type: [plugin_name, ...]}
        """
        result = {}
        types = [module_type] if module_type else self.list_module_types()

        for mt in types:
            registry_dict = self._get_registry_dict(mt)
            if registry_dict is not None:
                result[mt] = list(registry_dict.keys())
            else:
                result[mt] = []

        return result

    def _get_registry_dict(self, module_type: str) -> Optional[Dict]:
        """获取指定模块类型的注册字典"""
        attr_map = {
            "perception": "_perception",
            "prediction": "_prediction",
            "drive_engine": "_drive_engine",
            "decision": "_decision",
            "error_eval": "_error_eval",
            "optimizer": "_optimizer",
            "memory": "_memory",
        }
        attr = attr_map.get(module_type)
        if attr is None:
            return None
        return getattr(self.registry, attr, {})

    def get_plugin_info(self, module_type: str, name: str) -> Optional[Dict[str, Any]]:
        """获取插件详情

        Args:
            module_type: 模块类型
            name: 插件名

        Returns:
            插件信息字典，不存在返回None
        """
        registry_dict = self._get_registry_dict(module_type)
        if registry_dict is None or name not in registry_dict:
            return None

        cls = registry_dict[name]
        info = {
            "module_type": module_type,
            "module_type_name": self.MODULE_TYPE_NAMES.get(module_type, module_type),
            "name": name,
            "class_name": cls.__name__,
            "module_path": cls.__module__,
            "docstring": cls.__doc__ or "（无文档）",
            "is_default": name == "default",
        }

        # 尝试获取更多信息
        try:
            info["bases"] = [b.__name__ for b in cls.__bases__]
        except Exception:
            info["bases"] = []

        return info

    def get_plugin_usage(self, module_type: str, name: str) -> Optional[Dict[str, Any]]:
        """获取插件使用方式

        Args:
            module_type: 模块类型
            name: 插件名

        Returns:
            使用信息字典，不存在返回None
        """
        info = self.get_plugin_info(module_type, name)
        if info is None:
            return None

        config_key = self.MODULE_CONFIG_KEYS.get(module_type, module_type)
        usage = {
            **info,
            "config_key": config_key,
            "config_example": f'module_config = {{"{config_key}": "{name}"}}',
            "env_var": f"AGENT_{config_key.upper()}_PLUGIN",
            "python_example": self._generate_python_example(module_type, name),
        }
        return usage

    def _generate_python_example(self, module_type: str, name: str) -> str:
        """生成Python使用示例"""
        return (
            f"from core.registry import get_default_registry\n"
            f"from main import SelfEvolvingAgent\n\n"
            f"agent = SelfEvolvingAgent(\n"
            f"    module_config={{\"{module_type}\": \"{name}\"}}\n"
            f")"
        )

    def get_load_errors(self) -> List[str]:
        """获取加载错误列表"""
        return self._load_errors

    # ---------- 格式化方法 ----------

    def format_list(self, module_type: Optional[str] = None) -> str:
        """格式化插件列表"""
        plugins = self.list_plugins(module_type)
        lines = ["=== 插件列表 ==="]

        for mt, names in plugins.items():
            type_name = self.MODULE_TYPE_NAMES.get(mt, mt)
            lines.append(f"\n[{type_name}] ({mt})")
            if not names:
                lines.append("  （无可用插件）")
            else:
                for name in names:
                    default_mark = " (默认)" if name == "default" else ""
                    lines.append(f"  - {name}{default_mark}")

        # 显示加载错误
        if self._load_errors:
            lines.append(f"\n（{len(self._load_errors)}个插件加载失败，可能缺少依赖）")

        return "\n".join(lines)

    def format_info(self, module_type: str, name: str) -> str:
        """格式化插件详情"""
        info = self.get_plugin_info(module_type, name)
        if info is None:
            return f"插件不存在: {module_type}/{name}"

        lines = [f"=== 插件详情: {info['module_type_name']}/{name} ==="]
        lines.append(f"  模块类型: {info['module_type']} ({info['module_type_name']})")
        lines.append(f"  插件名: {info['name']}")
        lines.append(f"  类名: {info['class_name']}")
        lines.append(f"  模块路径: {info['module_path']}")
        if info.get("bases"):
            lines.append(f"  继承自: {', '.join(info['bases'])}")
        lines.append(f"  默认插件: {'是' if info['is_default'] else '否'}")
        lines.append(f"  文档:")
        # 缩进文档
        doc = info["docstring"].strip()
        for line in doc.split("\n"):
            lines.append(f"    {line}")
        return "\n".join(lines)

    def format_usage(self, module_type: str, name: str) -> str:
        """格式化插件使用方式"""
        usage = self.get_plugin_usage(module_type, name)
        if usage is None:
            return f"插件不存在: {module_type}/{name}"

        lines = [f"=== 插件使用: {usage['module_type_name']}/{name} ==="]
        lines.append(f"  类: {usage['class_name']}")
        lines.append(f"  配置键: {usage['config_key']}")
        lines.append(f"  环境变量: {usage['env_var']}")
        lines.append(f"\n  配置示例:")
        lines.append(f"    {usage['config_example']}")
        lines.append(f"\n  Python示例:")
        for line in usage["python_example"].split("\n"):
            lines.append(f"    {line}")
        return "\n".join(lines)

    def format_types(self) -> str:
        """格式化模块类型列表"""
        lines = ["=== 模块类型 ==="]
        for mt, name in self.MODULE_TYPE_NAMES.items():
            plugins = self.list_plugins(mt)
            count = len(plugins.get(mt, []))
            lines.append(f"  {mt:<15} {name:<10} ({count}个插件)")
        return "\n".join(lines)
