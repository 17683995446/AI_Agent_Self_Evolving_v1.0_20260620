"""
复杂场景压力测试 —— 长时间大规模混乱场景实战
运行: cd /workspace/self_evolving_agent && python tests/test_stress.py
"""
import os
import sys
import time
import random
import threading
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import logger
from core.config import config_manager

from modules.perception import EnvironmentPerception
from modules.drive_engine import DriveEngine
from modules.decision import DecisionEngine
from modules.prediction import PredictionEngine
from modules.error_eval import ErrorEvaluator
from modules.optimizer import Optimizer
from modules.memory import MemorySystem

from main import SelfEvolvingAgent


class ChaosInjector:
    """混乱注入器 —— 模拟各种异常场景"""

    def __init__(self):
        self.injection_count = 0

    def inject_random_state(self):
        """注入随机极端状态"""
        return {
            "cpu_percent": random.choice([0.0, 0.01, 0.5, 0.95, 1.0]),
            "memory_percent": random.choice([0.0, 0.01, 0.5, 0.95, 1.0]),
            "disk_percent": random.choice([0.0, 0.01, 0.5, 0.95, 1.0]),
            "error_rate": random.choice([0.0, 0.1, 0.5, 1.0]),
            "load_avg": random.choice([0.0, 0.1, 1.0, 5.0, 10.0]),
            "model_accuracy": random.uniform(0.0, 1.0),
            "agent_cpu": random.uniform(0.0, 1.0),
            "agent_memory": random.uniform(0.0, 1.0),
            "disk_io": random.uniform(0.0, 1.0),
            "net_io": random.uniform(0.0, 1.0),
        }

    def inject_nan_state(self):
        """注入NaN/Inf状态"""
        import math
        return {
            "cpu_percent": random.choice([float('nan'), float('inf'), -float('inf'), 0.5]),
            "memory_percent": random.choice([float('nan'), float('inf'), 0.5]),
            "disk_percent": 0.5,
            "error_rate": 0.0,
            "load_avg": 0.5,
            "model_accuracy": 0.5,
            "agent_cpu": 0.1,
            "agent_memory": 0.1,
            "disk_io": 0.1,
            "net_io": 0.1,
        }

    def inject_empty_state(self):
        """注入空状态"""
        return {}

    def inject_large_values(self):
        """注入超大值"""
        return {
            "cpu_percent": 1e10,
            "memory_percent": -1e10,
            "disk_percent": 1e-10,
            "error_rate": 1e10,
            "load_avg": 1e10,
            "model_accuracy": 1e10,
            "agent_cpu": 1e10,
            "agent_memory": 1e10,
            "disk_io": 1e10,
            "net_io": 1e10,
        }

    def inject(self):
        """随机选择一种注入方式"""
        self.injection_count += 1
        injectors = [
            self.inject_random_state,
            self.inject_nan_state,
            self.inject_empty_state,
            self.inject_large_values,
        ]
        return random.choice(injectors)()


def stress_test_single_modules(iterations=500):
    """单模块压力测试"""
    print(f"\n[压力测试] 单模块压力测试: {iterations} 轮")
    chaos = ChaosInjector()
    perception = EnvironmentPerception()
    drive_engine = DriveEngine()
    decision = DecisionEngine()
    prediction = PredictionEngine()
    error_eval = ErrorEvaluator()
    optimizer = Optimizer()
    memory = MemorySystem()

    errors = []
    for i in range(iterations):
        try:
            state = chaos.inject()
            drives = drive_engine.compute(state, i % 10, i % 5, random.random())
            drives_dict = {"safe": drives.safety, "gain": drives.gain, "eff": drives.efficiency}

            prediction.update({
                "cpu": random.uniform(-100, 100),
                "memory": random.uniform(-100, 100),
                "disk": random.uniform(-100, 100),
            })
            pred = prediction.predict()
            actual = {"cpu": random.uniform(-100, 100), "memory": random.uniform(-100, 100), "disk": random.uniform(-100, 100)}

            report = error_eval.evaluate(pred.predicted_values, actual)
            opt_result = optimizer.online_optimize(pred.predicted_values, actual, report)
            action_result = decision.decide_and_execute(state, drives_dict)
            memory.record_experience(
                state=state, action=action_result.action.name,
                reward=drives.intrinsic_motivation,
                next_state=state, td_error=report.get("mae", 0.0)
            )
            prediction.online_learn(actual)

        except Exception as e:
            errors.append((i, str(e), traceback.format_exc()))

    memory.close()
    print(f"  完成: {iterations} 轮, 错误: {len(errors)} 个")
    if errors:
        print(f"  前3个错误:")
        for idx, err, tb in errors[:3]:
            print(f"    轮次 {idx}: {err}")
    return len(errors)


def stress_test_concurrent_threads(duration_seconds=30, thread_count=10):
    """并发线程压力测试"""
    print(f"\n[压力测试] 并发线程压力测试: {thread_count} 线程 x {duration_seconds}s")
    errors = []
    success_count = [0]
    lock = threading.Lock()
    stop_event = threading.Event()

    def worker(worker_id):
        perception = EnvironmentPerception()
        drive_engine = DriveEngine()
        decision = DecisionEngine()
        while not stop_event.is_set():
            try:
                state = perception.get_state_vector()
                drives = drive_engine.compute(state, 0, 0, 0.0)
                drives_dict = {"safe": drives.safety, "gain": drives.gain, "eff": drives.efficiency}
                result = decision.decide_and_execute(state, drives_dict)
                with lock:
                    success_count[0] += 1
            except Exception as e:
                with lock:
                    errors.append((worker_id, str(e)))

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(thread_count)]
    for t in threads:
        t.start()

    time.sleep(duration_seconds)
    stop_event.set()

    for t in threads:
        t.join(timeout=5.0)

    print(f"  完成: 成功 {success_count[0]} 次, 错误 {len(errors)} 个")
    if errors:
        print(f"  前3个错误: {errors[:3]}")
    return len(errors)


def stress_test_agent_long_run(duration_seconds=180):
    """Agent长时间运行压力测试"""
    print(f"\n[压力测试] Agent长时间运行: {duration_seconds}s")
    agent = SelfEvolvingAgent()
    agent.start()
    time.sleep(duration_seconds)
    agent.stop()
    print(f"  完成: 总循环 {agent.loop_count} 轮, 运行时间 {time.time() - agent.start_time:.1f}s")
    return agent.loop_count


def stress_test_rapid_start_stop(cycles=20):
    """快速启停压力测试"""
    print(f"\n[压力测试] 快速启停: {cycles} 次")
    errors = []
    for i in range(cycles):
        try:
            agent = SelfEvolvingAgent()
            agent.start()
            time.sleep(0.5)
            agent.stop()
        except Exception as e:
            errors.append((i, str(e)))
    print(f"  完成: {cycles} 次, 错误 {len(errors)} 个")
    return len(errors)


def stress_test_memory_overflow(iterations=2000):
    """记忆溢出压力测试"""
    print(f"\n[压力测试] 记忆溢出: {iterations} 条经验")
    memory = MemorySystem()
    for i in range(iterations):
        memory.record_experience(
            state={"idx": i, "data": "x" * 1000},
            action=f"action_{i % 10}",
            reward=random.uniform(-10, 10),
            next_state={"idx": i + 1, "data": "y" * 1000},
            td_error=random.random(),
        )
    stats = memory.get_stats()
    memory.close()
    print(f"  完成: 短期 {stats['short_term_size']}, 长期 {stats['long_term_count']}, 回放 {stats['replay_pool_size']}")
    return stats


def stress_test_prediction_instability(iterations=200):
    """预测不稳定性压力测试"""
    print(f"\n[压力测试] 预测不稳定性: {iterations} 轮")
    pe = PredictionEngine()
    errors = []
    for i in range(iterations):
        try:
            # 大幅波动的输入
            val = random.choice([0, 1, 100, -100, 1e6, -1e6, float('nan'), float('inf')])
            pe.update({"cpu": val, "memory": val, "disk": val})
            pred = pe.predict()
            actual = {"cpu": random.uniform(-1e6, 1e6), "memory": random.uniform(-1e6, 1e6), "disk": random.uniform(-1e6, 1e6)}
            pe.online_learn(actual)
        except Exception as e:
            errors.append((i, str(e)))
    print(f"  完成: {iterations} 轮, 错误 {len(errors)} 个")
    return len(errors)


def run_all_stress_tests():
    """运行全部压力测试"""
    print("=" * 60)
    print("自驱Agent系统 —— 复杂场景压力测试")
    print("=" * 60)
    start_time = time.time()

    results = {}
    results["single_modules"] = stress_test_single_modules(500)
    results["concurrent_threads"] = stress_test_concurrent_threads(15, 10)
    results["agent_long_run"] = stress_test_agent_long_run(60)
    results["rapid_start_stop"] = stress_test_rapid_start_stop(10)
    results["memory_overflow"] = stress_test_memory_overflow(2000)
    results["prediction_instability"] = stress_test_prediction_instability(200)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("压力测试汇总")
    print("=" * 60)
    # 只统计真正的错误数（单模块、并发、快速启停、预测不稳定性）
    error_tests = ["single_modules", "concurrent_threads", "rapid_start_stop", "prediction_instability"]
    total_errors = sum(results[k] for k in error_tests)
    print(f"总耗时: {elapsed:.1f}s")
    print(f"总错误: {total_errors}")
    for name, result in results.items():
        print(f"  {name}: {result}")
    print("=" * 60)

    if total_errors == 0:
        print("所有压力测试通过，无错误！")
    else:
        print(f"发现 {total_errors} 个错误，需要修复")
    return total_errors


if __name__ == "__main__":
    run_all_stress_tests()
