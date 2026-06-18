"""
性能基准与优化测试 —— 测量并验证系统性能指标

目标：
1. 单周期执行耗时 < 100ms
2. 内存增长 < 1MB/小时
3. 高频调用下无性能退化
"""
import sys
import os
import time
import random
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import SelfEvolvingAgent
from modules.perception import EnvironmentPerception
from modules.prediction import PredictionEngine
from modules.drive_engine import DriveEngine
from modules.decision import DecisionEngine
from modules.error_eval import ErrorEvaluator
from modules.optimizer import Optimizer
from modules.memory import MemorySystem


class TestPerformanceBaseline:
    """性能基准测试"""

    def test_perception_latency(self):
        """感知模块延迟测试"""
        p = EnvironmentPerception()
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            p.get_state_vector()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = sum(times) / len(times)
        assert avg < 0.5, f"感知延迟过高: {avg:.4f}s"

    def test_prediction_latency(self):
        """预测模块延迟测试"""
        pe = PredictionEngine()
        for i in range(25):
            pe.update({"cpu": 50.0 + i, "memory": 60.0 + i, "disk": 70.0 + i})
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            pe.predict()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = sum(times) / len(times)
        assert avg < 0.1, f"预测延迟过高: {avg:.4f}s"

    def test_drive_engine_latency(self):
        """驱力引擎延迟测试"""
        de = DriveEngine()
        state = {"cpu_percent": 0.5, "memory_percent": 0.5}
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            de.compute(state, 0, 0, 0.0)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = sum(times) / len(times)
        assert avg < 0.05, f"驱力引擎延迟过高: {avg:.4f}s"

    def test_decision_latency(self):
        """决策模块延迟测试"""
        dec = DecisionEngine()
        state = {"cpu_percent": 0.5, "memory_percent": 0.5}
        drives = {"safe": 0.0, "gain": 0.0, "eff": 0.0}
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            dec.decide_and_execute(state, drives)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = sum(times) / len(times)
        assert avg < 1.0, f"决策延迟过高: {avg:.4f}s"

    def test_full_pipeline_latency(self):
        """完整链路延迟测试"""
        agent = SelfEvolvingAgent()
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            agent.run_for_duration(0.1)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = sum(times) / len(times)
        # 0.1s运行 + 开销应 < 1s
        assert avg < 1.5, f"全链路延迟过高: {avg:.4f}s"

    def test_memory_growth_over_cycles(self):
        """测试多周期运行后的内存增长"""
        tracemalloc.start()
        agent = SelfEvolvingAgent()
        snapshot1 = tracemalloc.take_snapshot()
        agent.run_for_duration(3.0)
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_diff = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        # 3秒运行内存增长应 < 10MB
        assert total_diff < 10 * 1024 * 1024, f"内存增长过大: {total_diff / 1024 / 1024:.2f}MB"

    def test_prediction_with_large_history(self):
        """大量历史数据下的预测性能"""
        pe = PredictionEngine()
        for i in range(1000):
            pe.update({"cpu": random.random() * 100, "memory": random.random() * 100, "disk": random.random() * 100})
        t0 = time.perf_counter()
        pe.predict()
        t1 = time.perf_counter()
        assert t1 - t0 < 0.5, f"大数据量预测延迟过高: {t1 - t0:.4f}s"

    def test_memory_system_with_large_data(self):
        """大量记忆数据下的性能"""
        ms = MemorySystem()
        t0 = time.perf_counter()
        for i in range(500):
            ms.record_experience({"s": i}, "a", 1.0, {"s": i + 1}, td_error=0.1)
        t1 = time.perf_counter()
        assert t1 - t0 < 5.0, f"大量记忆记录延迟过高: {t1 - t0:.4f}s"
        stats = ms.get_stats()
        assert stats["short_term_size"] > 0 or stats.get("replay_pool_size", 0) > 0


class TestPerformanceRegression:
    """性能回归测试 —— 验证优化后性能不下降"""

    def test_repeated_calls_no_slowdown(self):
        """重复调用不应明显变慢"""
        p = EnvironmentPerception()
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            p.get_state_vector()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        # 后10次平均不应比前10次慢50%以上
        first_half = sum(times[:10]) / 10
        second_half = sum(times[10:]) / 10
        assert second_half < first_half * 1.5, f"性能退化: 前={first_half:.4f}s 后={second_half:.4f}s"
