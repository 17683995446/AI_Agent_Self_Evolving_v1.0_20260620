#!/usr/bin/env python3
"""
6小时持续运行启动脚本 —— 使用start()而非run_for_duration()，主线程阻塞等待
自动脱离终端，防止SSH断开导致进程终止
"""
import os
import sys
import time
import signal

# 脱离终端控制（创建新会话）
try:
    os.setsid()
except OSError:
    pass  # 已经是没有控制终端的进程

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import SelfEvolvingAgent

agent = None
running = True

def signal_handler(signum, frame):
    global running
    print(f"[SIGNAL] 收到信号 {signum}，准备优雅退出")
    running = False
    if agent:
        agent.stop()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGHUP, signal.SIG_IGN)  # 忽略挂起信号

print("========================================")
print("自驱Agent —— 6小时持续运行模式")
print("========================================")

agent = SelfEvolvingAgent()
agent.start()

# 将真实Python进程PID写入文件
pid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent.pid")
pid_path = os.path.abspath(pid_path)
with open(pid_path, "w") as f:
    f.write(str(os.getpid()))

print(f"[INFO] Agent已启动 (PID: {os.getpid()})")
print(f"[INFO] PID文件: {pid_path}")
print("[INFO] 主线程进入阻塞等待，6小时后自动停止")

# 主线程阻塞6小时（21600秒），每10秒检查一次停止信号
TOTAL_SECONDS = 21600
CHECK_INTERVAL = 10
elapsed = 0

while running and elapsed < TOTAL_SECONDS:
    time.sleep(CHECK_INTERVAL)
    elapsed += CHECK_INTERVAL
    if elapsed % 600 == 0:  # 每10分钟打印一次状态
        minutes = elapsed // 60
        print(f"[STATUS] 已运行 {minutes} 分钟 / 360 分钟")

print(f"[INFO] 运行时间到达，准备停止 (总运行: {elapsed}秒)")
agent.stop()
print("[INFO] Agent已停止")
print("========================================")
