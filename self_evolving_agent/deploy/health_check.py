#!/usr/bin/env python3
"""
健康检查脚本 —— 检查Agent系统是否正常运行
支持模块级健康检查和详细诊断
用法: python deploy/health_check.py [--detail]
"""
import os
import sys
import time
import sqlite3
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AGENT_DIR = "/workspace/self_evolving_agent"
PID_FILE = os.path.join(AGENT_DIR, "agent.pid")
LOG_DIR = os.path.join(AGENT_DIR, "logs")
DATA_DIR = os.path.join(AGENT_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "agent_memory.db")


def check_process():
    """检查进程是否运行"""
    if not os.path.exists(PID_FILE):
        return False, "PID文件不存在"

    with open(PID_FILE) as f:
        pid = f.read().strip()

    try:
        os.kill(int(pid), 0)
        return True, f"进程运行中 (PID: {pid})"
    except (OSError, ValueError):
        return False, f"进程不存在 (PID: {pid})"


def check_logs():
    """检查日志是否更新"""
    if not os.path.isdir(LOG_DIR):
        return False, "日志目录不存在"
    log_files = [f for f in os.listdir(LOG_DIR) if f.startswith("agent_") and f.endswith(".log")]
    if not log_files:
        return False, "无日志文件"

    latest = max(log_files, key=lambda f: os.path.getmtime(os.path.join(LOG_DIR, f)))
    latest_path = os.path.join(LOG_DIR, latest)
    mtime = os.path.getmtime(latest_path)
    age = time.time() - mtime

    if age > 300:  # 5分钟未更新视为异常
        return False, f"日志 {age:.0f} 秒未更新"

    # 检查最近是否有ERROR/CRITICAL
    try:
        with open(latest_path, 'r') as f:
            lines = f.readlines()
        recent = lines[-200:] if len(lines) > 200 else lines
        errors = [l for l in recent if 'ERROR' in l or 'CRITICAL' in l]
        if errors:
            return False, f"最近发现 {len(errors)} 条错误日志"
    except Exception:
        pass

    return True, f"日志正常更新 ({latest}, {age:.0f}s前)"


def check_database():
    """检查数据库"""
    if not os.path.exists(DB_PATH):
        return True, "数据库文件不存在（首次运行）"

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM experiences")
        count = cursor.fetchone()[0]
        conn.close()
        return True, f"数据库正常，{count} 条经验"
    except Exception as e:
        return False, f"数据库异常: {e}"


def check_modules():
    """模块级健康检查 —— 验证各核心模块可导入和实例化"""
    try:
        from main import SelfEvolvingAgent
        from core.container import DIContainer
        from core.registry import get_default_registry

        # 验证注册表
        registry = get_default_registry()
        modules_ok = []
        modules_fail = []
        for mtype in ["perception", "prediction", "drive_engine", "decision", "error_eval", "optimizer", "memory"]:
            if registry.is_registered(mtype, "default"):
                modules_ok.append(mtype)
            else:
                modules_fail.append(mtype)

        # 快速实例化测试
        agent = SelfEvolvingAgent()
        state = agent.perception.get_state_vector()
        pred = agent.prediction.predict()
        drive = agent.drive_engine.compute(state, 0, 0, 0.0)

        detail = f"模块注册: {len(modules_ok)}/7, 快速实例化通过"
        if modules_fail:
            detail += f", 缺失: {', '.join(modules_fail)}"
        return len(modules_fail) == 0, detail
    except Exception as e:
        return False, f"模块检查异常: {e}"


def check_resources():
    """检查系统资源是否在安全范围内"""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent

        warnings = []
        if cpu > 80:
            warnings.append(f"CPU过高({cpu:.1f}%)")
        if mem > 90:
            warnings.append(f"内存过高({mem:.1f}%)")
        if disk > 90:
            warnings.append(f"磁盘过高({disk:.1f}%)")

        if warnings:
            return False, "; ".join(warnings)
        return True, f"CPU:{cpu:.1f}% 内存:{mem:.1f}% 磁盘:{disk:.1f}%"
    except Exception as e:
        return True, f"资源检查跳过: {e}"


def main():
    parser = argparse.ArgumentParser(description="Agent系统健康检查")
    parser.add_argument("--detail", action="store_true", help="显示详细诊断信息")
    args = parser.parse_args()

    print("=" * 50)
    print("自驱Agent系统 —— 健康检查")
    print("=" * 50)

    checks = [
        ("进程状态", check_process),
        ("日志状态", check_logs),
        ("数据库状态", check_database),
        ("模块健康", check_modules),
        ("系统资源", check_resources),
    ]

    all_ok = True
    results = {}
    for name, check_fn in checks:
        ok, msg = check_fn()
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {msg}")
        results[name] = {"ok": ok, "msg": msg}
        if not ok:
            all_ok = False

    if args.detail:
        print("\n[详细诊断]")
        try:
            from main import SelfEvolvingAgent
            agent = SelfEvolvingAgent()
            print(f"  感知模块: {type(agent.perception).__name__}")
            print(f"  预测模块: {type(agent.prediction).__name__}")
            print(f"  驱力模块: {type(agent.drive_engine).__name__}")
            print(f"  决策模块: {type(agent.decision).__name__}")
            print(f"  记忆模块: {type(agent.memory).__name__}")
            print(f"  优化模块: {type(agent.optimizer).__name__}")
            state = agent.perception.get_state_vector()
            print(f"  当前状态: CPU={state.get('cpu_percent', 'N/A'):.2f} 内存={state.get('memory_percent', 'N/A'):.2f}")
        except Exception as e:
            print(f"  详细诊断失败: {e}")

    print("=" * 50)
    if all_ok:
        print("健康检查通过，系统运行正常")
        return 0
    else:
        print("健康检查未通过，请检查系统状态")
        return 1


if __name__ == "__main__":
    sys.exit(main())
