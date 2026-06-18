#!/usr/bin/env python3
"""
API文档自动生成脚本
运行: python scripts/generate_api_docs.py
输出: docs/API_REFERENCE.md
"""
import os
import sys
import inspect
import importlib
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_doc(obj) -> str:
    """提取对象的文档字符串"""
    doc = obj.__doc__ or "暂无文档"
    return doc.strip()


def generate_interface_docs() -> str:
    """生成接口文档"""
    from core import interfaces

    lines = ["# Agent系统 API 参考文档", "", "> 自动生成于: " + __import__('datetime').datetime.now().isoformat(), ""]

    # 接口定义
    lines.extend(["## 一、核心接口定义", ""])

    for name in dir(interfaces):
        obj = getattr(interfaces, name)
        if inspect.isclass(obj) and name.startswith('I'):
            lines.append(f"### {name}")
            lines.append("")
            lines.append(extract_doc(obj))
            lines.append("")

            # 方法
            for method_name, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                if not method_name.startswith('_'):
                    sig = inspect.signature(method)
                    lines.append(f"- `{method_name}{sig}`")
                    lines.append(f"  - {extract_doc(method)}")
                    lines.append("")

    # 标准键名
    lines.extend(["## 二、标准键名契约", ""])

    for name in ['StateKeys', 'MetricKeys', 'DriveKeys', 'ErrorReportKeys']:
        obj = getattr(interfaces, name, None)
        if obj:
            lines.append(f"### {name}")
            lines.append("")
            lines.append(extract_doc(obj))
            lines.append("")
            for key_name in obj.all_keys():
                lines.append(f"- `{key_name}`")
            lines.append("")

    return "\n".join(lines)


def generate_module_docs() -> str:
    """生成模块文档"""
    modules = [
        ('modules.perception', 'EnvironmentPerception'),
        ('modules.drive_engine', 'DriveEngine'),
        ('modules.decision', 'DecisionEngine'),
        ('modules.prediction', 'PredictionEngine'),
        ('modules.error_eval', 'ErrorEvaluator'),
        ('modules.optimizer', 'Optimizer'),
        ('modules.memory', 'MemorySystem'),
    ]

    lines = ["## 三、模块实现文档", ""]

    for module_name, class_name in modules:
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
            lines.append(f"### {class_name}")
            lines.append("")
            lines.append(f"**模块**: `{module_name}`")
            lines.append("")
            lines.append(extract_doc(cls))
            lines.append("")

            # 公共方法
            for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                if not method_name.startswith('_') or method_name == '__init__':
                    try:
                        sig = inspect.signature(method)
                        lines.append(f"- `{method_name}{sig}`")
                    except ValueError:
                        lines.append(f"- `{method_name}()`")
            lines.append("")
        except Exception as e:
            lines.append(f"### {class_name}")
            lines.append(f"加载失败: {e}")
            lines.append("")

    return "\n".join(lines)


def main():
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
    os.makedirs(docs_dir, exist_ok=True)

    content = generate_interface_docs() + "\n" + generate_module_docs()

    output_path = os.path.join(docs_dir, "API_REFERENCE.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"API文档已生成: {output_path}")
    print(f"文档大小: {len(content)} 字符")


if __name__ == "__main__":
    main()
