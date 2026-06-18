"""
链路追踪测试 —— 验证追踪ID、结构化日志、性能追踪
"""
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tracing import (
    TraceContext, StructuredLogger, PerformanceTracer,
    TracePropagator, structured_logger, performance_tracer,
)


class TestTraceContext:
    """测试追踪上下文"""

    def setup_method(self):
        TraceContext.clear()

    def test_get_trace_id_auto_create(self):
        tid = TraceContext.get_trace_id()
        assert len(tid) == 12
        assert tid == TraceContext.get_trace_id()  # 同一线程应相同

    def test_set_and_get_trace_id(self):
        TraceContext.set_trace_id("custom123456")
        assert TraceContext.get_trace_id() == "custom123456"

    def test_span_stack(self):
        TraceContext.push_span("span1")
        TraceContext.push_span("span2")
        assert TraceContext.get_span_stack() == ["span1", "span2"]
        assert TraceContext.pop_span() == "span2"
        assert TraceContext.get_span_stack() == ["span1"]

    def test_clear(self):
        TraceContext.set_trace_id("abc")
        TraceContext.push_span("s1")
        TraceContext.clear()
        assert TraceContext.get_trace_id() != "abc"
        assert TraceContext.get_span_stack() == []

    def test_thread_isolation(self):
        TraceContext.set_trace_id("main_thread")
        result = {}

        def worker():
            result["tid"] = TraceContext.get_trace_id()

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert result["tid"] != "main_thread"


class TestStructuredLogger:
    """测试结构化日志"""

    def setup_method(self):
        self.logger = StructuredLogger(min_level="DEBUG")

    def test_log_level_filtering(self):
        self.logger.set_level("ERROR")
        assert self.logger._should_log("ERROR") is True
        assert self.logger._should_log("WARNING") is False
        assert self.logger._should_log("INFO") is False

    def test_module_level_override(self):
        self.logger.set_level("DEBUG")
        self.logger.set_level("ERROR", module="test_mod")
        assert self.logger._should_log("INFO", "other_mod") is True
        assert self.logger._should_log("INFO", "test_mod") is False
        assert self.logger._should_log("ERROR", "test_mod") is True

    def test_log_includes_trace_id(self):
        TraceContext.set_trace_id("trace123")
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.logger.info("TEST_EVENT", "test message", module="test")
        output = f.getvalue()
        assert "trace123" in output
        assert "TEST_EVENT" in output
        assert "test message" in output

    def test_log_with_duration(self):
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.logger.info("TEST", "msg", duration_ms=15.5)
        output = f.getvalue()
        assert "15.5" in output or "15.50" in output


class TestPerformanceTracer:
    """测试性能追踪器"""

    def setup_method(self):
        performance_tracer.reset()
        TraceContext.clear()

    def test_trace_timing(self):
        with performance_tracer.trace("test_span"):
            time.sleep(0.01)
        stats = performance_tracer.get_stats("test_span")
        assert stats["count"] == 1
        assert stats["avg_ms"] >= 10.0

    def test_multiple_traces(self):
        for _ in range(5):
            with performance_tracer.trace("multi_span"):
                time.sleep(0.005)
        stats = performance_tracer.get_stats("multi_span")
        assert stats["count"] == 5
        assert stats["avg_ms"] >= 5.0

    def test_get_all_stats(self):
        with performance_tracer.trace("span_a"):
            pass
        with performance_tracer.trace("span_b"):
            pass
        all_stats = performance_tracer.get_stats()
        assert "span_a" in all_stats
        assert "span_b" in all_stats

    def test_slow_spans(self):
        with performance_tracer.trace("fast"):
            pass
        with performance_tracer.trace("slow"):
            time.sleep(0.15)
        slow = performance_tracer.get_slow_spans(threshold_ms=50.0)
        assert len(slow) >= 1
        assert any(s["span"] == "slow" for s in slow)

    def test_span_stack_in_trace(self):
        TraceContext.clear()
        with performance_tracer.trace("outer"):
            with performance_tracer.trace("inner"):
                pass
        stack = TraceContext.get_span_stack()
        assert stack == []  # 两个都弹出了


class TestTracePropagator:
    """测试追踪传播"""

    def setup_method(self):
        TraceContext.clear()

    def test_inject_and_extract(self):
        TraceContext.set_trace_id("abc123")
        TraceContext.push_span("span1")
        headers = TracePropagator.inject()
        assert headers["trace_id"] == "abc123"
        assert "span1" in headers["span_stack"]

        TraceContext.clear()
        TracePropagator.extract(headers)
        assert TraceContext.get_trace_id() == "abc123"
        assert "span1" in TraceContext.get_span_stack()

    def test_extract_empty(self):
        TracePropagator.extract({})
        assert TraceContext.get_trace_id() != ""
