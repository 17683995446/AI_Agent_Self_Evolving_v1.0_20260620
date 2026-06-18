#!/bin/bash
# 自驱Agent系统状态检查脚本

AGENT_DIR="/workspace/self_evolving_agent"
PID_FILE="$AGENT_DIR/agent.pid"
LOG_DIR="$AGENT_DIR/logs"
DATA_DIR="$AGENT_DIR/data"

echo "========================================"
echo "自驱Agent系统 —— 运行状态"
echo "========================================"

# 检查进程状态
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if kill -0 "$PID" 2>/dev/null; then
        echo "[进程状态] 运行中 (PID: $PID)"
        
        # 获取进程资源使用
        CPU=$(ps -p "$PID" -o %cpu= 2>/dev/null | tr -d ' ')
        MEM=$(ps -p "$PID" -o %mem= 2>/dev/null | tr -d ' ')
        RSS=$(ps -p "$PID" -o rss= 2>/dev/null | tr -d ' ')
        ELAPSED=$(ps -p "$PID" -o etime= 2>/dev/null | tr -d ' ')
        
        echo "  CPU使用率: ${CPU:-未知}%"
        echo "  内存使用率: ${MEM:-未知}%"
        echo "  内存占用: ${RSS:-未知}KB"
        echo "  运行时长: ${ELAPSED:-未知}"
    else
        echo "[进程状态] 未运行 (PID文件残留)"
        rm -f "$PID_FILE"
    fi
else
    echo "[进程状态] 未运行"
fi

echo ""

# 检查日志
echo "[日志文件]"
if [ -d "$LOG_DIR" ]; then
    LATEST_LOG=$(ls -t "$LOG_DIR"/agent_*.log 2>/dev/null | head -n 1)
    if [ -n "$LATEST_LOG" ]; then
        LOG_SIZE=$(du -h "$LATEST_LOG" 2>/dev/null | cut -f1)
        LOG_LINES=$(wc -l < "$LATEST_LOG" 2>/dev/null)
        echo "  最新日志: $(basename "$LATEST_LOG")"
        echo "  大小: $LOG_SIZE"
        echo "  行数: $LOG_LINES"
        
        # 最近5条关键日志
        echo ""
        echo "[最近关键日志]"
        grep -E "(INFO|WARNING|ERROR)" "$LATEST_LOG" 2>/dev/null | tail -n 5 | sed 's/^/  /'
    else
        echo "  暂无日志文件"
    fi
else
    echo "  日志目录不存在"
fi

echo ""

# 检查数据库
echo "[记忆数据库]"
if [ -f "$DATA_DIR/agent_memory.db" ]; then
    DB_SIZE=$(du -h "$DATA_DIR/agent_memory.db" 2>/dev/null | cut -f1)
    echo "  文件: agent_memory.db"
    echo "  大小: $DB_SIZE"
    
    # 尝试查询记录数
    COUNT=$(sqlite3 "$DATA_DIR/agent_memory.db" "SELECT COUNT(*) FROM experiences;" 2>/dev/null)
    if [ -n "$COUNT" ]; then
        echo "  经验记录数: $COUNT"
    fi
else
    echo "  数据库文件不存在"
fi

echo ""

# 检查快照
echo "[模型快照]"
if [ -d "$DATA_DIR/snapshots" ]; then
    SNAP_COUNT=$(ls -1 "$DATA_DIR/snapshots"/*.pkl 2>/dev/null | wc -l)
    echo "  快照数量: $SNAP_COUNT"
    if [ "$SNAP_COUNT" -gt 0 ]; then
        LATEST_SNAP=$(ls -t "$DATA_DIR/snapshots"/*.pkl 2>/dev/null | head -n 1)
        echo "  最新快照: $(basename "$LATEST_SNAP")"
    fi
else
    echo "  快照目录不存在"
fi

echo ""

# 系统资源
echo "[系统资源]"
SYS_CPU=$(top -bn1 | grep "Cpu(s)" 2>/dev/null | awk '{print $2}' | cut -d'%' -f1)
SYS_MEM=$(free | grep Mem 2>/dev/null | awk '{printf "%.1f", $3/$2 * 100.0}')
echo "  系统CPU: ${SYS_CPU:-未知}%"
echo "  系统内存: ${SYS_MEM:-未知}%"

echo ""
echo "========================================"
echo "操作命令:"
echo "  查看实时日志: tail -f $LOG_DIR/agent_deploy.log"
echo "  重启: ./deploy/stop.sh && ./deploy/start.sh"
echo "  停止: ./deploy/stop.sh"
echo "========================================"
