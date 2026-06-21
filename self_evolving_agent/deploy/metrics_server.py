#!/usr/bin/env python3
"""
HTTP指标暴露服务 —— 提供Prometheus风格的/metrics端点
用法: python deploy/metrics_server.py [--port 8080]
"""
import os
import sys
import time
import json
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 自动推断Agent根目录（允许通过环境变量AGENT_DIR覆盖）
AGENT_DIR = os.environ.get(
    "AGENT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
PID_FILE = os.path.join(AGENT_DIR, "agent.pid")

# 全局指标缓存
_metrics_cache = {}
_metrics_lock = threading.Lock()
_metrics_timestamp = 0


def collect_metrics():
    """采集系统指标"""
    global _metrics_cache, _metrics_timestamp
    try:
        import psutil
        from main import SelfEvolvingAgent

        agent = SelfEvolvingAgent()
        state = agent.perception.get_state_vector()
        pred = agent.prediction.predict()
        stats = agent.memory.get_stats()

        metrics = {
            "agent_cpu_percent": state.get("cpu_percent", 0.0) * 100,
            "agent_memory_percent": state.get("memory_percent", 0.0) * 100,
            "agent_disk_percent": state.get("disk_percent", 0.0) * 100,
            "prediction_confidence": pred.confidence,
            "short_term_memory_size": stats.get("short_term_size", 0),
            "long_term_memory_size": stats.get("long_term_count", 0),
            "replay_pool_size": stats.get("replay_pool_size", 0),
            "system_cpu_percent": psutil.cpu_percent(interval=0.1),
            "system_memory_percent": psutil.virtual_memory().percent,
            "timestamp": time.time(),
        }

        with _metrics_lock:
            _metrics_cache = metrics
            _metrics_timestamp = time.time()
        return metrics
    except Exception as e:
        return {"error": str(e), "timestamp": time.time()}


def get_prometheus_format(metrics):
    """转换为Prometheus文本格式"""
    lines = []
    lines.append("# HELP agent_cpu_percent Agent感知到的CPU使用率")
    lines.append("# TYPE agent_cpu_percent gauge")
    lines.append(f"agent_cpu_percent {metrics.get('agent_cpu_percent', 0)}")

    lines.append("# HELP agent_memory_percent Agent感知到的内存使用率")
    lines.append("# TYPE agent_memory_percent gauge")
    lines.append(f"agent_memory_percent {metrics.get('agent_memory_percent', 0)}")

    lines.append("# HELP prediction_confidence 预测置信度")
    lines.append("# TYPE prediction_confidence gauge")
    lines.append(f"prediction_confidence {metrics.get('prediction_confidence', 0)}")

    lines.append("# HELP short_term_memory_size 短期记忆大小")
    lines.append("# TYPE short_term_memory_size gauge")
    lines.append(f"short_term_memory_size {metrics.get('short_term_memory_size', 0)}")

    lines.append("# HELP long_term_memory_size 长期记忆大小")
    lines.append("# TYPE long_term_memory_size gauge")
    lines.append(f"long_term_memory_size {metrics.get('long_term_memory_size', 0)}")

    lines.append("# HELP system_cpu_percent 系统CPU使用率")
    lines.append("# TYPE system_cpu_percent gauge")
    lines.append(f"system_cpu_percent {metrics.get('system_cpu_percent', 0)}")

    lines.append("# HELP system_memory_percent 系统内存使用率")
    lines.append("# TYPE system_memory_percent gauge")
    lines.append(f"system_memory_percent {metrics.get('system_memory_percent', 0)}")

    return "\n".join(lines) + "\n"


class MetricsHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # 静默日志，避免刷屏
        pass

    def do_GET(self):
        if self.path == "/metrics":
            # 检查缓存是否过期（5秒）
            with _metrics_lock:
                cache_age = time.time() - _metrics_timestamp
                if cache_age > 5 or not _metrics_cache:
                    metrics = collect_metrics()
                else:
                    metrics = dict(_metrics_cache)

            body = get_prometheus_format(metrics)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())

        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())

        elif self.path == "/api/status":
            metrics = collect_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(metrics, indent=2).encode())

        else:
            self.send_response(404)
            self.end_headers()


def run_server(port=8080):
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    print(f"指标服务启动于 http://0.0.0.0:{port}")
    print(f"  /metrics  - Prometheus格式指标")
    print(f"  /health   - 健康检查")
    print(f"  /api/status - JSON格式状态")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务停止")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Agent指标暴露服务")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    args = parser.parse_args()
    run_server(args.port)


if __name__ == "__main__":
    main()
