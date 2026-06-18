#!/bin/bash
# 自驱Agent系统部署启动脚本
# 用法: ./deploy/start.sh [supervisor|nohup]

set -e

AGENT_DIR="/workspace/self_evolving_agent"
LOG_DIR="$AGENT_DIR/logs"
DATA_DIR="$AGENT_DIR/data"
PID_FILE="$AGENT_DIR/agent.pid"
MODE="${1:-nohup}"

echo "========================================"
echo "自驱永续进化Agent系统 —— 部署启动"
echo "========================================"
echo "启动模式: $MODE"
echo "Agent目录: $AGENT_DIR"

# 创建必要目录
mkdir -p "$LOG_DIR" "$DATA_DIR/snapshots" "$AGENT_DIR/config"

# 检查依赖
echo "[1/4] 检查Python依赖..."
cd "$AGENT_DIR"
python -c "import numpy, psutil, yaml" 2>/dev/null || {
    echo "安装依赖..."
    pip install numpy psutil pyyaml -q
}
echo "依赖检查通过"

# 创建默认配置文件（如果不存在）
if [ ! -f "$AGENT_DIR/config/agent.yaml" ]; then
echo "[2/4] 创建默认配置文件..."
cat > "$AGENT_DIR/config/agent.yaml" << 'EOF'
resource:
  max_cpu_percent: 20.0
  max_memory_percent: 20.0
  max_reconstruction_iterations: 3
  storage_retain_ratio: 0.30

evolution:
  exploration_ratio: 0.30
  idle_replay_interval_hours: 6.0
  steady_state_error_threshold: 0.01
  exploration_boost_on_steady: 0.10

drive_weights:
  safe: 0.4
  gain: 0.4
  eff: 0.2

main_loop_frequency: 1.0
perception_interval: 1.0
decision_interval: 1.0
log_dir: "./logs"
database_path: "./data/agent_memory.db"
model_snapshot_dir: "./data/snapshots"
EOF
echo "配置文件创建完成"
else
    echo "[2/4] 配置文件已存在，跳过"
fi

# 检查是否已有实例在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null)
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[警告] Agent已在运行 (PID: $OLD_PID)"
        echo "如需重启，请先运行: ./deploy/stop.sh"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# 启动Agent
echo "[3/4] 启动Agent..."
if [ "$MODE" = "supervisor" ]; then
    # Supervisor模式
    if command -v supervisorctl &> /dev/null; then
        cp "$AGENT_DIR/deploy/supervisor_agent.conf" /app/supervisord.conf.d/ 2>/dev/null || \
        cp "$AGENT_DIR/deploy/supervisor_agent.conf" /etc/supervisor/conf.d/ 2>/dev/null || {
            echo "无法找到supervisor配置目录，回退到nohup模式"
            MODE="nohup"
        }
        if [ "$MODE" = "supervisor" ]; then
            supervisorctl reread
            supervisorctl update
            supervisorctl start self_evolving_agent
            echo "Supervisor模式启动完成"
        fi
    else
        echo "supervisorctl不可用，回退到nohup模式"
        MODE="nohup"
    fi
fi

if [ "$MODE" = "nohup" ]; then
    # Nohup模式
    nohup python "$AGENT_DIR/main.py" 9999999 > "$LOG_DIR/agent_deploy.log" 2>&1 &
    AGENT_PID=$!
    echo $AGENT_PID > "$PID_FILE"
    echo "Nohup模式启动完成 (PID: $AGENT_PID)"
fi

# 等待启动
sleep 3

echo "[4/4] 验证启动状态..."
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Agent运行中 (PID: $PID)"
        echo "日志文件: $LOG_DIR/agent_deploy.log"
        echo "数据库: $DATA_DIR/agent_memory.db"
        echo ""
        echo "查看实时日志: tail -f $LOG_DIR/agent_deploy.log"
        echo "查看状态: ./deploy/status.sh"
        echo "停止: ./deploy/stop.sh"
    else
        echo "[错误] Agent启动失败，请检查日志"
        exit 1
    fi
fi

echo "========================================"
echo "部署完成！"
echo "========================================"
