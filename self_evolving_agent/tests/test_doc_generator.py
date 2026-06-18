"""
文档自动生成与代码质量测试 —— 验证代码扫描、文档生成、质量分析

测试前置：
1. 拆解DocGenerator接口：scan/generate_markdown/write_docs/analyze_quality/write_quality_report
2. 边界条件：空目录、无Python文件、语法错误文件、无docstring代码
"""
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.doc_generator import DocGenerator, CodeMetrics


class TestCodeMetrics:
    """测试代码指标计算"""

    def test_empty_source(self):
        """空源码指标为0"""
        gen = DocGenerator()
        metrics = gen._compute_metrics("")
        assert metrics.total_lines == 0
        assert metrics.code_lines == 0

    def test_comment_detection(self):
        """正确识别注释行"""
        source = "# comment\nx = 1\n"
        gen = DocGenerator()
        metrics = gen._compute_metrics(source)
        assert metrics.comment_lines == 1
        assert metrics.code_lines == 1

    def test_docstring_detection(self):
        """正确识别文档字符串"""
        source = '"""module doc"""\nx = 1\n'
        gen = DocGenerator()
        metrics = gen._compute_metrics(source)
        assert metrics.comment_lines >= 1
        assert metrics.code_lines == 1

    def test_multiline_docstring(self):
        """多行文档字符串识别"""
        source = '"""line1\nline2\nline3"""\nx = 1\n'
        gen = DocGenerator()
        metrics = gen._compute_metrics(source)
        assert metrics.comment_lines == 3
        assert metrics.code_lines == 1

    def test_blank_lines(self):
        """空行识别"""
        source = "x = 1\n\ny = 2\n"
        gen = DocGenerator()
        metrics = gen._compute_metrics(source)
        assert metrics.blank_lines == 1
        assert metrics.total_lines == 3

    def test_complexity_estimation(self):
        """圈复杂度估算"""
        source = "def f():\n    if x:\n        for i in range(3):\n            pass\n"
        gen = DocGenerator()
        metrics = gen._compute_metrics(source)
        assert metrics.complexity_score >= 2  # if + for


class TestModuleParsing:
    """测试模块解析"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.gen = DocGenerator(source_dir=self.temp_dir, output_dir=os.path.join(self.temp_dir, "out"))

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_file(self, name: str, content: str):
        path = os.path.join(self.temp_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_scan_empty_dir(self):
        """空目录扫描返回空列表"""
        modules = self.gen.scan()
        assert modules == []

    def test_scan_single_module(self):
        """扫描单个模块"""
        self._write_file("mod_a.py", "class Foo:\n    def bar(self):\n        pass\n")
        modules = self.gen.scan()
        assert len(modules) == 1
        assert modules[0].name == "mod_a"
        assert modules[0].classes[0]["name"] == "Foo"

    def test_scan_skips_tests(self):
        """跳过测试文件"""
        self._write_file("test_something.py", "def test_x(): pass\n")
        modules = self.gen.scan()
        assert len(modules) == 0

    def test_scan_skips_pycache(self):
        """跳过__pycache__"""
        os.makedirs(os.path.join(self.temp_dir, "__pycache__"))
        self._write_file("__pycache__/cached.py", "x = 1\n")
        modules = self.gen.scan()
        assert len(modules) == 0

    def test_extract_class_with_methods(self):
        """提取类及其方法"""
        self._write_file("cls.py", "\n".join([
            "class Base:",
            '    """Base class."""',
            "    def method(self, x: int) -> str:",
            '        """Do something."""',
            "        return str(x)",
        ]))
        modules = self.gen.scan()
        cls = modules[0].classes[0]
        assert cls["name"] == "Base"
        assert cls["docstring"] == "Base class."
        assert len(cls["methods"]) == 1
        assert cls["methods"][0]["name"] == "method"
        assert cls["methods"][0]["docstring"] == "Do something."

    def test_extract_function(self):
        """提取独立函数"""
        self._write_file("func.py", "\n".join([
            "def add(a, b):",
            '    """Add two numbers."""',
            "    return a + b",
        ]))
        modules = self.gen.scan()
        func = modules[0].functions[0]
        assert func["name"] == "add"
        assert func["docstring"] == "Add two numbers."

    def test_syntax_error_graceful(self):
        """语法错误文件不抛异常"""
        self._write_file("bad.py", "def f(\n")
        modules = self.gen.scan()
        assert len(modules) == 0  # 跳过语法错误文件

    def test_nested_package(self):
        """扫描嵌套包目录"""
        self._write_file("pkg/sub.py", "x = 1\n")
        modules = self.gen.scan()
        assert any("pkg.sub" in m.name for m in modules)


class TestMarkdownGeneration:
    """测试Markdown文档生成"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.gen = DocGenerator(source_dir=self.temp_dir, output_dir=os.path.join(self.temp_dir, "out"))

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_file(self, name: str, content: str):
        path = os.path.join(self.temp_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_markdown_contains_module(self):
        """Markdown包含模块名"""
        self._write_file("m.py", "class A:\n    pass\n")
        self.gen.scan()
        md = self.gen.generate_markdown()
        assert "m" in md
        assert "class A" in md

    def test_markdown_contains_metrics(self):
        """Markdown包含质量指标"""
        self._write_file("m.py", "x = 1\n")
        self.gen.scan()
        md = self.gen.generate_markdown()
        assert "总代码行数" in md

    def test_markdown_contains_function_signature(self):
        """Markdown包含函数签名"""
        self._write_file("m.py", "def foo(a: int) -> str:\n    pass\n")
        self.gen.scan()
        md = self.gen.generate_markdown()
        assert "foo(a: int) -> str" in md

    def test_write_docs_creates_file(self):
        """write_docs创建文件"""
        self._write_file("m.py", "x = 1\n")
        path = self.gen.write_docs()
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "系统自动生成文档" in content


class TestQualityAnalysis:
    """测试质量分析"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.gen = DocGenerator(source_dir=self.temp_dir, output_dir=os.path.join(self.temp_dir, "out"))

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_file(self, name: str, content: str):
        path = os.path.join(self.temp_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_quality_report_basic(self):
        """质量报告基本字段"""
        self._write_file("m.py", "x = 1\n")
        report = self.gen.analyze_quality()
        assert report["total_modules"] == 1
        assert report["total_lines"] >= 1
        assert "grade" in report

    def test_doc_coverage_calculation(self):
        """文档覆盖率计算"""
        self._write_file("m.py", "\n".join([
            "def documented():",
            '    """Has doc."""',
            "    pass",
            "def undocumented():",
            "    pass",
        ]))
        report = self.gen.analyze_quality()
        assert report["documented_functions"] == 1
        assert report["total_functions"] == 2
        assert report["doc_coverage"] == 0.5

    def test_grade_a(self):
        """评级A条件"""
        grade = self.gen._grade(4.0, 0.8)
        assert grade == "A"

    def test_grade_d(self):
        """评级D条件"""
        grade = self.gen._grade(20.0, 0.1, total_functions=10)
        assert grade == "D"

    def test_write_quality_report(self):
        """写入质量报告文件"""
        self._write_file("m.py", "x = 1\n")
        path = self.gen.write_quality_report()
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "代码质量报告" in content
        assert "综合评级" in content

    def test_empty_project_quality(self):
        """空项目质量报告"""
        report = self.gen.analyze_quality()
        assert report["total_modules"] == 0
        assert report["grade"] == "A"  # 无函数时默认A
