"""
可视化监控面板 —— Web仪表板实时展示Agent状态

端点：
- GET /           -> 主页面（HTML）
- GET /api/state  -> JSON格式的当前状态
- GET /api/history -> 历史数据
"""

import os
import sys
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger

# 全局状态缓存（由外部Agent线程更新）
_dashboard_state = {
    "cpu_percent": 0.0,
    "memory_percent": 0.0,
    "disk_percent": 0.0,
    "prediction_confidence": 0.0,
    "safety_drive": 0.0,
    "gain_drive": 0.0,
    "efficiency_drive": 0.0,
    "loop_count": 0,
    "timestamp": 0,
    "history": [],  # 最近100个数据点
}
_state_lock = threading.Lock()


def update_state(**kwargs):
    """外部调用更新仪表板状态"""
    global _dashboard_state
    with _state_lock:
        for k, v in kwargs.items():
            if k in _dashboard_state:
                _dashboard_state[k] = v
        _dashboard_state["timestamp"] = time.time()
        # 维护历史数据
        if "history" not in _dashboard_state:
            _dashboard_state["history"] = []
        point = {k: v for k, v in _dashboard_state.items() if k != "history"}
        _dashboard_state["history"].append(point)
        if len(_dashboard_state["history"]) > 100:
            _dashboard_state["history"] = _dashboard_state["history"][-100:]


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>自驱Agent监控面板</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }
.header { text-align: center; margin-bottom: 24px; }
.header h1 { font-size: 24px; color: #38bdf8; margin-bottom: 8px; }
.header .status { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 12px; }
.status.online { background: #065f46; color: #34d399; }
.status.offline { background: #7f1d1d; color: #f87171; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; max-width: 1200px; margin: 0 auto; }
.card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
.card h3 { font-size: 14px; color: #94a3b8; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
.metric { display: flex; justify-content: space-between; align-items: center; margin: 8px 0; }
.metric .label { color: #cbd5e1; font-size: 14px; }
.metric .value { font-family: 'Courier New', monospace; font-size: 18px; font-weight: bold; }
.value.good { color: #34d399; }
.value.warn { color: #fbbf24; }
.value.danger { color: #f87171; }
.bar { width: 100%; height: 8px; background: #334155; border-radius: 4px; margin-top: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
.chart { height: 120px; margin-top: 12px; position: relative; }
.chart svg { width: 100%; height: 100%; }
.footer { text-align: center; margin-top: 24px; color: #64748b; font-size: 12px; }
.drive-item { display: flex; align-items: center; margin: 6px 0; }
.drive-item .drive-label { width: 60px; font-size: 12px; color: #94a3b8; }
.drive-item .drive-bar-bg { flex: 1; height: 6px; background: #334155; border-radius: 3px; margin: 0 8px; position: relative; }
.drive-item .drive-bar { height: 100%; border-radius: 3px; position: absolute; top: 0; transition: all 0.5s ease; }
.drive-item .drive-value { width: 40px; text-align: right; font-family: monospace; font-size: 12px; }
</style>
</head>
<body>
<div class="header">
    <h1>自驱永续进化Agent监控面板</h1>
    <span class="status online" id="status">运行中</span>
    <div style="margin-top:8px;color:#64748b;font-size:12px;">循环次数: <span id="loop">0</span> | 最后更新: <span id="last-update">--</span></div>
</div>
<div class="grid">
    <div class="card">
        <h3>系统资源</h3>
        <div class="metric"><span class="label">CPU</span><span class="value" id="cpu-val">0%</span></div>
        <div class="bar"><div class="bar-fill" id="cpu-bar" style="width:0%;background:#38bdf8;"></div></div>
        <div class="metric"><span class="label">内存</span><span class="value" id="mem-val">0%</span></div>
        <div class="bar"><div class="bar-fill" id="mem-bar" style="width:0%;background:#a78bfa;"></div></div>
        <div class="metric"><span class="label">磁盘</span><span class="value" id="disk-val">0%</span></div>
        <div class="bar"><div class="bar-fill" id="disk-bar" style="width:0%;background:#f472b6;"></div></div>
    </div>
    <div class="card">
        <h3>预测引擎</h3>
        <div class="metric"><span class="label">置信度</span><span class="value" id="conf-val">0.00</span></div>
        <div class="bar"><div class="bar-fill" id="conf-bar" style="width:0%;background:#34d399;"></div></div>
        <div class="chart" id="pred-chart"></div>
    </div>
    <div class="card">
        <h3>驱力状态</h3>
        <div class="drive-item">
            <span class="drive-label">安全</span>
            <div class="drive-bar-bg"><div class="drive-bar" id="safe-bar" style="left:50%;width:0%;background:#fbbf24;"></div></div>
            <span class="drive-value" id="safe-val">0.0</span>
        </div>
        <div class="drive-item">
            <span class="drive-label">收益</span>
            <div class="drive-bar-bg"><div class="drive-bar" id="gain-bar" style="left:50%;width:0%;background:#34d399;"></div></div>
            <span class="drive-value" id="gain-val">0.0</span>
        </div>
        <div class="drive-item">
            <span class="drive-label">能效</span>
            <div class="drive-bar-bg"><div class="drive-bar" id="eff-bar" style="left:50%;width:0%;background:#38bdf8;"></div></div>
            <span class="drive-value" id="eff-val">0.0</span>
        </div>
    </div>
    <div class="card">
        <h3>历史趋势</h3>
        <div class="chart" id="history-chart"></div>
    </div>
</div>
<div class="footer">自驱永续进化预测修正Agent系统 v1.0 | 基于第一性原理设计</div>
<script>
function getColorClass(val, thresholds) {
    if (val >= thresholds[1]) return 'danger';
    if (val >= thresholds[0]) return 'warn';
    return 'good';
}
function updateDrive(id, val) {
    const bar = document.getElementById(id + '-bar');
    const txt = document.getElementById(id + '-val');
    const pct = Math.abs(val) * 50;
    const left = val < 0 ? (50 - pct) : 50;
    bar.style.left = left + '%';
    bar.style.width = pct + '%';
    bar.style.background = val < 0 ? '#f87171' : '#34d399';
    txt.textContent = val.toFixed(2);
}
function drawSparkline(data, key, color, containerId) {
    const container = document.getElementById(containerId);
    if (!data.length) return;
    const vals = data.map(d => d[key] || 0);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const range = max - min || 1;
    const w = container.clientWidth || 280;
    const h = 120;
    const step = w / (vals.length - 1 || 1);
    let path = '';
    vals.forEach((v, i) => {
        const x = i * step;
        const y = h - ((v - min) / range) * (h - 10) - 5;
        path += (i === 0 ? 'M' : 'L') + x + ',' + y;
    });
    container.innerHTML = '<svg viewBox="0 0 ' + w + ' ' + h + '"><path d="' + path + '" fill="none" stroke="' + color + '" stroke-width="2"/></svg>';
}
async function refresh() {
    try {
        const res = await fetch('/api/state');
        const s = await res.json();
        document.getElementById('loop').textContent = s.loop_count || 0;
        document.getElementById('last-update').textContent = new Date(s.timestamp * 1000).toLocaleTimeString();
        const cpu = (s.cpu_percent || 0) * 100;
        const mem = (s.memory_percent || 0) * 100;
        const disk = (s.disk_percent || 0) * 100;
        document.getElementById('cpu-val').textContent = cpu.toFixed(1) + '%';
        document.getElementById('cpu-val').className = 'value ' + getColorClass(cpu, [50, 80]);
        document.getElementById('cpu-bar').style.width = Math.min(cpu, 100) + '%';
        document.getElementById('mem-val').textContent = mem.toFixed(1) + '%';
        document.getElementById('mem-val').className = 'value ' + getColorClass(mem, [60, 85]);
        document.getElementById('mem-bar').style.width = Math.min(mem, 100) + '%';
        document.getElementById('disk-val').textContent = disk.toFixed(1) + '%';
        document.getElementById('disk-bar').style.width = Math.min(disk, 100) + '%';
        const conf = (s.prediction_confidence || 0) * 100;
        document.getElementById('conf-val').textContent = (s.prediction_confidence || 0).toFixed(2);
        document.getElementById('conf-bar').style.width = conf + '%';
        updateDrive('safe', s.safety_drive || 0);
        updateDrive('gain', s.gain_drive || 0);
        updateDrive('eff', s.efficiency_drive || 0);
        if (s.history && s.history.length > 1) {
            drawSparkline(s.history, 'prediction_confidence', '#34d399', 'pred-chart');
            drawSparkline(s.history, 'cpu_percent', '#38bdf8', 'history-chart');
        }
        document.getElementById('status').textContent = '运行中';
        document.getElementById('status').className = 'status online';
    } catch(e) {
        document.getElementById('status').textContent = '离线';
        document.getElementById('status').className = 'status offline';
    }
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())

        elif path == "/api/state":
            global _dashboard_state
            with _state_lock:
                data = dict(_dashboard_state)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        else:
            self.send_response(404)
            self.end_headers()


def start_dashboard_server(port: int = 8080):
    """启动仪表板服务器"""
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    logger.info("DASHBOARD", f"监控面板启动于 http://0.0.0.0:{port}", module="dashboard")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    start_dashboard_server()
