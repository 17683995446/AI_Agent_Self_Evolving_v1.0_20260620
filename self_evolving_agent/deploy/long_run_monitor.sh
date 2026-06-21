#!/bin/bash
# 5小时长时间运行监控脚本
# 每30分钟记录一次状态，共10次

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="${AGENT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOG_FILE="$AGENT_DIR/logs/long_run_monitor.log"
PID_FILE="${PID_FILE:-$AGENT_DIR/agent.pid}"
DB_PATH="$AGENT_DIR/data/agent_memory.db"

mkdir -p "$AGENT_DIR/logs"

echo "========================================" >> "$LOG_FILE"
echo "5小时长时间运行监控开始: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

for i in $(seq 1 10); do
    sleep 1800  # 30分钟
    
    MINUTES=$((i * 30))
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "" >> "$LOG_FILE"
    echo "=== $TIMESTAMP - ${MINUTES}分钟 ===" >> "$LOG_FILE"
    
    # 进程状态
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE" 2>/dev/null)
        if kill -0 "$PID" 2>/dev/null; then
            ps -p "$PID" -o pid,etime,%cpu,%mem,rss 2>/dev/null >> "$LOG_FILE"
        else
            echo "[ERROR] 进程已停止 (PID: $PID)" >> "$LOG_FILE"
        fi
    else
        echo "[ERROR] PID文件不存在" >> "$LOG_FILE"
    fi
    
    # 数据库记录数
    COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM experiences;" 2>/dev/null)
    echo "经验记录数: $COUNT" >> "$LOG_FILE"
    
    # 日志文件大小
    LATEST_LOG=$(ls -t "$AGENT_DIR/logs"/agent_*.log 2>/dev/null | head -n 1)
    if [ -n "$LATEST_LOG" ]; then
        LOG_SIZE=$(du -h "$LATEST_LOG" 2>/dev/null | cut -f1)
        LOG_LINES=$(wc -l < "$LATEST_LOG" 2>/dev/null)
        echo "日志: $(basename "$LATEST_LOG") | 大小: $LOG_SIZE | 行数: $LOG_LINES" >> "$LOG_FILE"
    fi
    
    # 检查最近是否有ERROR
    if [ -n "$LATEST_LOG" ]; then
        ERROR_COUNT=$(grep -c "ERROR\|CRITICAL" "$LATEST_LOG" 2>/dev/null || echo 0)
        echo "累计错误数: $ERROR_COUNT" >> "$LOG_FILE"
    fi
    
    # 快照数量
    SNAP_COUNT=$(ls -1 "$AGENT_DIR/data/snapshots"/*.pkl 2>/dev/null | wc -l)
    echo "模型快照数: $SNAP_COUNT" >> "$LOG_FILE"
done

echo "" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "5小时监控结束: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
