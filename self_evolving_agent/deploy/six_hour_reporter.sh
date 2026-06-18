#!/bin/bash
# 6小时运行完成后自动生成报告
# 此脚本应在Agent启动后运行，它会等待6小时然后生成报告

AGENT_DIR="/workspace/self_evolving_agent"
REPORT_FILE="$AGENT_DIR/logs/six_hour_final_report.txt"
LOG_FILE="$AGENT_DIR/logs/six_hour_run.log"
PID_FILE="$AGENT_DIR/agent.pid"
DB_PATH="$AGENT_DIR/data/agent_memory.db"

# 等待6小时
echo "[REPORTER] 等待6小时运行完成..."
sleep 21600

# 生成报告
echo "========================================" > "$REPORT_FILE"
echo "自驱Agent 6小时运行报告" >> "$REPORT_FILE"
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"
echo "========================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# 进程状态
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "[进程状态] 运行中 (PID: $PID)" >> "$REPORT_FILE"
        ps -p "$PID" -o pid,etime,%cpu,%mem,rss,cmd >> "$REPORT_FILE" 2>/dev/null
    else
        echo "[进程状态] 已停止 (PID: $PID)" >> "$REPORT_FILE"
    fi
else
    echo "[进程状态] PID文件不存在" >> "$REPORT_FILE"
fi

echo "" >> "$REPORT_FILE"

# 日志统计
if [ -f "$LOG_FILE" ]; then
    LOG_LINES=$(wc -l < "$LOG_FILE" 2>/dev/null)
    LOG_SIZE=$(du -h "$LOG_FILE" 2>/dev/null | cut -f1)
    ERROR_COUNT=$(grep -c "ERROR\|CRITICAL" "$LOG_FILE" 2>/dev/null || echo 0)
    WARNING_COUNT=$(grep -c "WARNING" "$LOG_FILE" 2>/dev/null || echo 0)
    echo "[日志统计]" >> "$REPORT_FILE"
    echo "  文件: $LOG_FILE" >> "$REPORT_FILE"
    echo "  行数: $LOG_LINES" >> "$REPORT_FILE"
    echo "  大小: $LOG_SIZE" >> "$REPORT_FILE"
    echo "  ERROR数: $ERROR_COUNT" >> "$REPORT_FILE"
    echo "  WARNING数: $WARNING_COUNT" >> "$REPORT_FILE"
fi

echo "" >> "$REPORT_FILE"

# 数据库统计
if [ -f "$DB_PATH" ]; then
    DB_SIZE=$(du -h "$DB_PATH" 2>/dev/null | cut -f1)
    echo "[数据库统计]" >> "$REPORT_FILE"
    echo "  文件: $DB_PATH" >> "$REPORT_FILE"
    echo "  大小: $DB_SIZE" >> "$REPORT_FILE"
    if command -v sqlite3 &> /dev/null; then
        EXP_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM experiences;" 2>/dev/null || echo "N/A")
        echo "  经验记录: $EXP_COUNT" >> "$REPORT_FILE"
    fi
fi

echo "" >> "$REPORT_FILE"

# 快照统计
SNAP_COUNT=$(ls -1 "$AGENT_DIR/data/snapshots"/*.pkl 2>/dev/null | wc -l)
echo "[模型快照] 数量: $SNAP_COUNT" >> "$REPORT_FILE"

echo "" >> "$REPORT_FILE"
echo "========================================" >> "$REPORT_FILE"
echo "报告生成完毕" >> "$REPORT_FILE"
echo "========================================" >> "$REPORT_FILE"

echo "[REPORTER] 报告已生成: $REPORT_FILE"
