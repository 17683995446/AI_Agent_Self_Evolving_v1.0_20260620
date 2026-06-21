#!/bin/bash
# 自驱Agent系统停止脚本

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="${AGENT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PID_FILE="${PID_FILE:-$AGENT_DIR/agent.pid}"

echo "========================================"
echo "自驱Agent系统 —— 停止"
echo "========================================"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "发送停止信号到Agent (PID: $PID)..."
        kill -TERM "$PID"
        
        # 等待优雅退出
        for i in {1..15}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo "Agent已停止"
                rm -f "$PID_FILE"
                exit 0
            fi
            echo "等待Agent退出... ($i/15)"
            sleep 1
        done
        
        # 强制终止
        echo "强制终止Agent..."
        kill -9 "$PID" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "Agent已强制停止"
    else
        echo "Agent进程不存在"
        rm -f "$PID_FILE"
    fi
else
    echo "PID文件不存在，Agent可能未运行"
    # 尝试查找并终止
    pkill -f "python.*main.py" 2>/dev/null && echo "已终止残留的Agent进程" || echo "未找到Agent进程"
fi

echo "========================================"
