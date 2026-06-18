"""
文档自动生成与代码质量分析 —— 从代码提取接口、生成文档、量化质量

设计原则（第一性原理）：
1. 代码即文档：接口定义、类结构、函数签名即最准确的文档来源
2. 质量可量化：圈复杂度、测试覆盖率、文档覆盖率用数字说话
3. 自动化：扫描代码目录，生成Markdown，无需手动维护
4. 信息透明：文档是知识的外化，是系统可维护性的基础
"""

import os
import sys
import ast
import inspect
import importlib.util
from typing import Dict, Any, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


class CodeMetrics:
    """代码质量指标"""

    def __init__(self):
        self.total_lines = 0
        self.code_lines = 0
        self.comment_lines = 0
        self.blank_lines = 0
        self.function_count = 0
        self.class_count = 0
        self.complexity_score = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_lines": self.total_lines,
            "code_lines": self.code_lines,
            "comment_lines": self.comment_lines,
            "blank_lines": self.blank_lines,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "complexity_score": round(self.complexity_score, 2),
            "comment_ratio": round(self.comment_lines / max(self.total_lines, 1), 3),
        }


class ModuleDoc:
    """模块文档结构"""

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.classes: List[Dict[str, Any]] = []
        self.functions: List[Dict[str, Any]] = []
        self.metrics = CodeMetrics()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "classes": self.classes,
            "functions": self.functions,
            "metrics": self.metrics.to_dict(),
        }


class DocGenerator:
    """文档生成器 —— 扫描代码、提取结构、生成Markdown"""

    def __init__(self, source_dir: str = ".", output_dir: str = "./docs/auto"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.modules: List[ModuleDoc] = []
        self.quality_report: Dict[str, Any] = {}

    def scan(self) -> List[ModuleDoc]:
        """扫描源代码目录，提取所有模块文档"""
        self.modules = []
        for py_file in sorted(self.source_dir.rglob("*.py")):
            # 跳过测试文件和__pycache__
            if "__pycache__" in str(py_file) or py_file.name.startswith("test_"):
                continue
            rel_path = py_file.relative_to(self.source_dir)
            module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
            try:
                mod_doc = self._parse_file(py_file, module_name)
                if mod_doc:
                    self.modules.append(mod_doc)
            except Exception as e:
                logger.warning("DOC_GEN", f"解析失败 {py_file}: {e}", module="doc_generator")
        logger.info("DOC_GEN", f"扫描完成: {len(self.modules)} 个模块", module="doc_generator")
        return self.modules

    def _parse_file(self, file_path: Path, module_name: str) -> Optional[ModuleDoc]:
        """解析单个Python文件"""
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        mod_doc = ModuleDoc(name=module_name, path=str(file_path))
        mod_doc.metrics = self._compute_metrics(source)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                cls_info = self._extract_class(node, source)
                mod_doc.classes.append(cls_info)
                mod_doc.metrics.class_count += 1
            elif isinstance(node, ast.FunctionDef):
                func_info = self._extract_function(node, source)
                mod_doc.functions.append(func_info)
                mod_doc.metrics.function_count += 1

        return mod_doc

    def _compute_metrics(self, source: str) -> CodeMetrics:
        """计算代码质量指标"""
        metrics = CodeMetrics()
        lines = source.splitlines()
        metrics.total_lines = len(lines)

        in_docstring = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                metrics.blank_lines += 1
            elif stripped.startswith("#"):
                metrics.comment_lines += 1
            elif stripped.startswith('"""') or stripped.startswith("'''"):
                metrics.comment_lines += 1
                if stripped.count('"""') < 2 and stripped.count("'''") < 2:
                    in_docstring = not in_docstring
            elif in_docstring:
                metrics.comment_lines += 1
                # 检查当前行是否包含结束引号
                if '"""' in stripped or "'''" in stripped:
                    in_docstring = False
            else:
                metrics.code_lines += 1

        # 简单圈复杂度估算：分支语句数量
        try:
            tree = ast.parse(source)
            branch_count = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                                     ast.With, ast.Assert, ast.comprehension)):
                    branch_count += 1
            metrics.complexity_score = branch_count
        except Exception:
            pass

        return metrics

    def _extract_class(self, node: ast.ClassDef, source: str) -> Dict[str, Any]:
        """提取类信息"""
        docstring = ast.get_docstring(node) or ""
        methods = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                methods.append(self._extract_function(child, source, is_method=True))

        bases = [self._get_name(base) for base in node.bases]
        return {
            "name": node.name,
            "docstring": docstring.strip(),
            "bases": bases,
            "methods": methods,
            "line": node.lineno,
        }

    def _extract_function(self, node: ast.FunctionDef, source: str, is_method: bool = False) -> Dict[str, Any]:
        """提取函数信息"""
        docstring = ast.get_docstring(node) or ""
        args = []
        for arg in node.args.args:
            arg_type = ""
            if arg.annotation and isinstance(arg.annotation, ast.Name):
                arg_type = arg.annotation.id
            args.append({"name": arg.arg, "type": arg_type})

        return_type = ""
        if node.returns and isinstance(node.returns, ast.Name):
            return_type = node.returns.id

        return {
            "name": node.name,
            "docstring": docstring.strip(),
            "args": args,
            "return_type": return_type,
            "line": node.lineno,
            "is_method": is_method,
        }

    def _get_name(self, node: ast.AST) -> str:
        """获取AST节点名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return ""

    def generate_markdown(self) -> str:
        """生成Markdown格式文档"""
        lines = []
        lines.append("# 系统自动生成文档")
        lines.append("")
        lines.append(f"> 生成时间: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"> 模块数: {len(self.modules)}")
        lines.append("")

        # 质量概览
        lines.append("## 代码质量概览")
        lines.append("")
        total_lines = sum(m.metrics.total_lines for m in self.modules)
        total_funcs = sum(m.metrics.function_count for m in self.modules)
        total_classes = sum(m.metrics.class_count for m in self.modules)
        total_complexity = sum(m.metrics.complexity_score for m in self.modules)
        lines.append(f"- 总代码行数: {total_lines}")
        lines.append(f"- 函数数量: {total_funcs}")
        lines.append(f"- 类数量: {total_classes}")
        lines.append(f"- 总圈复杂度: {total_complexity}")
        lines.append("")

        # 模块详情
        for mod in self.modules:
            lines.append(f"## {mod.name}")
            lines.append(f"**路径**: `{mod.path}`")
            lines.append("")

            # 指标
            m = mod.metrics
            lines.append(f"- 行数: {m.total_lines} (代码{m.code_lines}, 注释{m.comment_lines}, 空行{m.blank_lines})")
            lines.append(f"- 类: {m.class_count}, 函数: {m.function_count}, 复杂度: {m.complexity_score}")
            lines.append("")

            # 类
            for cls in mod.classes:
                lines.append(f"### class {cls['name']}")
                if cls['bases']:
                    lines.append(f"继承: {', '.join(cls['bases'])}")
                if cls['docstring']:
                    lines.append(f"> {cls['docstring']}")
                lines.append("")
                for method in cls['methods']:
                    sig = self._format_signature(method)
                    lines.append(f"- `{sig}`")
                    if method['docstring']:
                        lines.append(f"  - {method['docstring']}")
                lines.append("")

            # 独立函数
            if mod.functions:
                lines.append("### 函数")
                for func in mod.functions:
                    sig = self._format_signature(func)
                    lines.append(f"- `{sig}`")
                    if func['docstring']:
                        lines.append(f"  - {func['docstring']}")
                lines.append("")

        return "\n".join(lines)

    def _format_signature(self, func: Dict[str, Any]) -> str:
        """格式化函数签名"""
        args_str = ", ".join(
            f"{a['name']}: {a['type']}" if a['type'] else a['name']
            for a in func['args']
        )
        ret = f" -> {func['return_type']}" if func['return_type'] else ""
        return f"{func['name']}({args_str}){ret}"

    def write_docs(self) -> str:
        """生成并写入文档文件"""
        self.scan()
        md = self.generate_markdown()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "API_REFERENCE_AUTO.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info("DOC_GEN", f"文档已生成: {output_path}", module="doc_generator")
        return str(output_path)

    def analyze_quality(self) -> Dict[str, Any]:
        """分析整体代码质量"""
        if not self.modules:
            self.scan()

        total_lines = sum(m.metrics.total_lines for m in self.modules)
        total_code = sum(m.metrics.code_lines for m in self.modules)
        total_comment = sum(m.metrics.comment_lines for m in self.modules)
        total_complexity = sum(m.metrics.complexity_score for m in self.modules)
        total_funcs = sum(m.metrics.function_count for m in self.modules)

        # 文档覆盖率：有docstring的函数/总函数
        documented = 0
        all_funcs = 0
        for mod in self.modules:
            for cls in mod.classes:
                for method in cls['methods']:
                    all_funcs += 1
                    if method['docstring']:
                        documented += 1
            for func in mod.functions:
                all_funcs += 1
                if func['docstring']:
                    documented += 1

        doc_coverage = documented / max(all_funcs, 1)
        avg_complexity = total_complexity / max(total_funcs, 1)

        report = {
            "total_modules": len(self.modules),
            "total_lines": total_lines,
            "total_code_lines": total_code,
            "total_comment_lines": total_comment,
            "comment_ratio": round(total_comment / max(total_lines, 1), 3),
            "total_functions": total_funcs,
            "total_classes": sum(m.metrics.class_count for m in self.modules),
            "total_complexity": total_complexity,
            "avg_complexity_per_function": round(avg_complexity, 2),
            "documented_functions": documented,
            "doc_coverage": round(doc_coverage, 3),
            "grade": self._grade(avg_complexity, doc_coverage, total_funcs),
        }
        self.quality_report = report
        return report

    def _grade(self, avg_complexity: float, doc_coverage: float, total_functions: int = 0) -> str:
        """综合评级"""
        if total_functions == 0:
            return "A"
        if avg_complexity <= 5 and doc_coverage >= 0.7:
            return "A"
        elif avg_complexity <= 8 and doc_coverage >= 0.5:
            return "B"
        elif avg_complexity <= 12 and doc_coverage >= 0.3:
            return "C"
        return "D"

    def write_quality_report(self) -> str:
        """写入质量报告"""
        report = self.analyze_quality()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "QUALITY_REPORT.md"
        lines = [
            "# 代码质量报告",
            "",
            f"> 生成时间: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 概览",
            "",
            f"- 模块数: {report['total_modules']}",
            f"- 总代码行: {report['total_lines']}",
            f"- 注释比例: {report['comment_ratio']:.1%}",
            f"- 函数数: {report['total_functions']}",
            f"- 类数: {report['total_classes']}",
            f"- 平均圈复杂度: {report['avg_complexity_per_function']}",
            f"- 文档覆盖率: {report['doc_coverage']:.1%}",
            f"- 综合评级: **{report['grade']}**",
            "",
            "## 评级标准",
            "",
            "- A: 平均复杂度<=5 且 文档覆盖率>=70%",
            "- B: 平均复杂度<=8 且 文档覆盖率>=50%",
            "- C: 平均复杂度<=12 且 文档覆盖率>=30%",
            "- D: 其他",
            "",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("DOC_GEN", f"质量报告已生成: {path}", module="doc_generator")
        return str(path)
