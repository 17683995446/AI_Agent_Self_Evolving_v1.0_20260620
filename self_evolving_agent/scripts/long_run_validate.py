#!/usr/bin/env python3
"""
长期运行验证脚本 —— 24小时稳定性测试

用法: python scripts/long_run_validate.py [--duration 86400]
"""
import os
import sys
import time
import json
import argparse
import tracemalloc
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import SelfEvolvingAgent
from core.logger import logger


class LongRunValidator:
    """长期运行验证器"""

    def __init__(self, duration: float = 86400.0, sample_interval: float = 60.0):
        self.duration = duration
        self.sample_interval = sample_interval
        self.samples: list = []
        self.start_time = 0.0
        self.agent: SelfEvolvingAgent = None

    def _sample(self) -> dict:
        """采集当前状态样本"""
        state = self.agent.perception.get_state_vector()
        pred = self.agent.prediction.predict()
        stats = self.agent.memory.get_stats()
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()

        return {
            "timestamp": time.time(),
            "elapsed": time.time() - self.start_time,
            "loop_count": self.agent.loop_count,
            "cpu_percent": state.get("cpu_percent", 0),
            "memory_percent": state.get("memory_percent", 0),
            "prediction_confidence": pred.confidence,
            "short_term_size": stats.get("short_term_size", 0),
            "long_term_count": stats.get("long_term_count", 0),
            "process_rss_mb": mem_info.rss / (1024 * 1024),
            "process_vms_mb": mem_info.vms / (1024 * 1024),
        }

    def run(self) -> dict:
        """执行长期运行验证"""
        print(f"[{datetime.now()}] 长期运行验证启动")
        print(f"目标时长: {self.duration / 3600:.1f}小时")
        print(f"采样间隔: {self.sample_interval:.0f}秒")

        self.agent = SelfEvolvingAgent()
        self.start_time = time.time()

        # 启动Agent后台线程
        self.agent.start()

        next_sample = self.start_time
        end_time = self.start_time + self.duration

        try:
            while time.time() < end_time:
                now = time.time()
                if now >= next_sample:
                    sample = self._sample()
                    self.samples.append(sample)
                    self._print_sample(sample)
                    next_sample = now + self.sample_interval

                    # 检查异常指标
                    if sample["process_rss_mb"] > 512:
                        print(f"WARNING: 内存使用过高: {sample['process_rss_mb']:.1f}MB")

                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.agent.stop()

        return self._generate_report()

    def _print_sample(self, sample: dict):
        """打印样本"""
        elapsed = sample["elapsed"]
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        print(f"[{hours:02d}:{minutes:02d}] loops={sample['loop_count']:4d} "
              f"cpu={sample['cpu_percent']*100:5.1f}% mem={sample['memory_percent']*100:5.1f}% "
              f"conf={sample['prediction_confidence']:.2f} "
              f"rss={sample['process_rss_mb']:6.1f}MB")

    def _generate_report(self) -> dict:
        """生成验证报告"""
        if not self.samples:
            return {"error": "无采样数据"}

        rss_values = [s["process_rss_mb"] for s in self.samples]
        conf_values = [s["prediction_confidence"] for s in self.samples]
        loop_counts = [s["loop_count"] for s in self.samples]

        # 计算内存增长趋势（MB/小时）
        if len(rss_values) >= 2:
            duration_hours = (self.samples[-1]["timestamp"] - self.samples[0]["timestamp"]) / 3600
            mem_growth = (rss_values[-1] - rss_values[0]) / duration_hours if duration_hours > 0 else 0
        else:
            mem_growth = 0

        report = {
            "duration_seconds": self.samples[-1]["elapsed"],
            "duration_hours": self.samples[-1]["elapsed"] / 3600,
            "total_loops": self.agent.loop_count if self.agent else 0,
            "sample_count": len(self.samples),
            "memory": {
                "initial_rss_mb": rss_values[0],
                "final_rss_mb": rss_values[-1],
                "growth_per_hour_mb": mem_growth,
                "max_rss_mb": max(rss_values),
            },
            "prediction": {
                "avg_confidence": sum(conf_values) / len(conf_values),
                "min_confidence": min(conf_values),
                "max_confidence": max(conf_values),
            },
            "stability": {
                "loops_per_hour": (loop_counts[-1] - loop_counts[0]) /
                                  ((self.samples[-1]["timestamp"] - self.samples[0]["timestamp"]) / 3600)
                if len(loop_counts) >= 2 and (self.samples[-1]["timestamp"] - self.samples[0]["timestamp"]) > 0 else 0,
            },
            "passed": mem_growth < 10,  # 内存增长 < 10MB/小时视为通过
        }

        print("\n" + "=" * 60)
        print("长期运行验证报告")
        print("=" * 60)
        print(f"运行时长: {report['duration_hours']:.2f}小时")
        print(f"总循环数: {report['total_loops']}")
        print(f"内存增长: {mem_growth:.2f}MB/小时 (初始: {rss_values[0]:.1f}MB, 最终: {rss_values[-1]:.1f}MB)")
        print(f"预测置信度: 平均={report['prediction']['avg_confidence']:.2f}, 范围=[{report['prediction']['min_confidence']:.2f}, {report['prediction']['max_confidence']:.2f}]")
        print(f"结果: {'通过' if report['passed'] else '未通过'}")
        print("=" * 60)

        # 保存详细报告
        report_path = os.path.join(os.path.dirname(__file__), "..", "data", "long_run_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump({"report": report, "samples": self.samples}, f, indent=2, default=str)
        print(f"详细报告已保存: {report_path}")

        return report


def main():
    parser = argparse.ArgumentParser(description="长期运行验证")
    parser.add_argument("--duration", type=float, default=86400.0, help="运行时长（秒），默认24小时")
    parser.add_argument("--interval", type=float, default=60.0, help="采样间隔（秒）")
    args = parser.parse_args()

    validator = LongRunValidator(duration=args.duration, sample_interval=args.interval)
    report = validator.run()
    sys.exit(0 if report.get("passed", False) else 1)


if __name__ == "__main__":
    main()
