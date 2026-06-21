# 故障排除

系统设计遵循"快速失败"原则：遇到问题立即报错并给出修复建议，不静默降级。
本文档将常见错误映射到解决方案。

## 错误诊断流程

```
Agent启动失败
    │
    ├─ ConfigurationException → 检查配置文件（见下文）
    │
    ├─ ModuleNotFoundError → 检查依赖安装
    │
    ├─ PermissionError → 检查文件/目录权限
    │
    └─ 其他异常 → 查看日志中的 traceback 字段
```

## 配置错误

### `ConfigurationException: 配置文件YAML语法错误`

**原因**: `config/agent.yaml` 存在YAML语法错误。

**诊断**:
```bash
python main.py config validate
```

**修复**:
- 检查缩进（YAML不允许Tab，必须用空格）
- 检查冒号后是否有空格（`key: value` 而非 `key:value`）
- 检查字符串是否需要引号

### `ConfigurationException: CPU 约束必须在 0-100 之间`

**原因**: `resource.max_cpu_percent` 超出范围。

**修复**:
```yaml
# config/agent.yaml
resource:
  max_cpu_percent: 20.0  # 必须 (0, 100]
```

或通过环境变量：
```bash
AGENT_MAX_CPU=20.0 python main.py run
```

### `ConfigurationException: 驱力权重之和必须等于 1.0`

**原因**: `safe + gain + eff ≠ 1.0`。

**修复**:
```yaml
drive_weights:
  safe: 0.4
  gain: 0.4
  eff: 0.2  # 0.4 + 0.4 + 0.2 = 1.0
```

### `ConfigurationException: 环境变量 AGENT_MAX_CPU=... 无法转换为 float`

**原因**: 环境变量值不是有效数字。

**修复**:
```bash
# 错误
export AGENT_MAX_CPU="abc"

# 正确
export AGENT_MAX_CPU="30.0"
```

## 依赖错误

### `ModuleNotFoundError: No module named 'yaml'`

**修复**:
```bash
pip install pyyaml
```

### `ModuleNotFoundError: No module named 'psutil'`

**修复**:
```bash
pip install psutil
```

### `ModuleNotFoundError: No module named 'numpy'`

**修复**:
```bash
pip install numpy
```

或一次性安装全部依赖：
```bash
pip install -r requirements.txt
```

## 权限错误

### `PermissionError: [Errno 13] Permission denied: './logs/...'`

**原因**: 日志目录无写权限。

**修复**:
```bash
# 方案1：修改目录权限
chmod 755 logs/ data/ data/snapshots/

# 方案2：指定其他目录
AGENT_LOG_DIR=/tmp/agent_logs python main.py run
```

### `PermissionError: [Errno 13] Permission denied: './data/agent_memory.db'`

**原因**: 数据库文件无写权限。

**修复**:
```bash
chmod 644 data/agent_memory.db
# 或指定其他路径
AGENT_DB_PATH=/tmp/agent_memory.db python main.py run
```

## 端口冲突

### `OSError: [Errno 98] Address already in use`

**原因**: REST API 或 Dashboard 端口被占用。

**诊断**:
```bash
# 查看端口占用
lsof -i :9090  # REST API
lsof -i :8080  # Dashboard
```

**修复**:
```bash
# 方案1：使用其他端口
python main.py run --rest-port 9091 --dashboard-port 8081

# 方案2：终止占用进程
kill -9 $(lsof -t -i :9090)
```

## 运行时问题

### Agent运行但不出日志

**诊断**:
```bash
# 1. 检查日志级别
python -c "from core.logger import logger; print('Console:', logger.console_level)"

# 2. 检查日志文件是否存在
ls -la logs/

# 3. 以DEBUG级别运行
python main.py run --log-level DEBUG
```

### Agent启动后立即退出

**诊断**:
```bash
# 查看最近的错误日志
grep "ERROR\|CRITICAL" logs/agent_*.log | tail -20

# 查看异常堆栈（U6后未知异常保留完整堆栈）
grep -A 20 "traceback" logs/agent_*.log | tail -50
```

### 主循环卡住不执行

**诊断**:
```bash
# 1. 检查主循环频率
python main.py config show

# 2. 检查是否资源受限
python main.py health --detail

# 3. 检查是否有死锁（查看线程状态）
python -c "
from main import SelfEvolvingAgent
import threading
agent = SelfEvolvingAgent()
agent.start()
import time; time.sleep(2)
print('Threads:', threading.enumerate())
agent.stop()
"
```

## 部署问题

### Docker构建失败

**诊断**:
```bash
# 查看构建日志
docker compose build --no-cache 2>&1 | tail -30

# 常见原因：
# 1. requirements.txt 中的包版本不存在
# 2. 网络问题导致pip下载失败
```

### Docker容器健康检查失败

**诊断**:
```bash
# 查看容器状态
docker compose ps

# 查看健康检查日志
docker inspect self-evolving-agent | grep -A 10 "Health"

# 进入容器排查
docker compose exec agent python main.py health
```

### systemd服务无法启动

**诊断**:
```bash
# 查看服务状态
sudo systemctl status self-evolving-agent

# 查看详细日志
sudo journalctl -u self-evolving-agent -n 50 --no-pager

# 常见原因：
# 1. WorkingDirectory路径错误
# 2. Python路径错误
# 3. 配置文件不存在
```

## 环境变量参考

| 环境变量 | 类型 | 说明 | 默认值 |
|---------|------|------|-------|
| `AGENT_CONFIG` | str | 配置文件路径 | `./config/agent.yaml` |
| `AGENT_ENV` | str | 运行环境（dev/prod） | `dev` |
| `AGENT_MAX_CPU` | float | CPU使用上限(%) | `20.0` |
| `AGENT_MAX_MEMORY` | float | 内存使用上限(%) | `20.0` |
| `AGENT_MAX_RECONSTRUCT` | int | 模型重构最大轮次 | `3` |
| `AGENT_STORAGE_RETAIN` | float | 快照保留比例 | `0.30` |
| `AGENT_EXPLORATION_RATIO` | float | 探索比例 | `0.30` |
| `AGENT_REPLAY_INTERVAL` | float | 离线复盘间隔(小时) | `6.0` |
| `AGENT_SAFE_WEIGHT` | float | 安全驱力权重 | `0.4` |
| `AGENT_GAIN_WEIGHT` | float | 收益驱力权重 | `0.4` |
| `AGENT_EFF_WEIGHT` | float | 能效驱力权重 | `0.2` |
| `AGENT_PERCEPTION_INTERVAL` | float | 感知间隔(秒) | `1.0` |
| `AGENT_DECISION_INTERVAL` | float | 决策间隔(秒) | `1.0` |
| `AGENT_LOOP_FREQUENCY` | float | 主循环频率(Hz) | `1.0` |
| `AGENT_LOG_DIR` | str | 日志目录 | `./logs` |
| `AGENT_DB_PATH` | str | 数据库路径 | `./data/agent_memory.db` |
| `AGENT_SNAPSHOT_DIR` | str | 快照目录 | `./data/snapshots` |
| `AGENT_LOG_LEVEL` | str | 控制台日志级别 | `INFO` |
| `AGENT_REST_PORT` | int | REST API端口 | `9090` |
| `AGENT_DASHBOARD_PORT` | int | Dashboard端口 | `8080` |

> 优先级：环境变量 > 配置文件 > 默认值

## 获取帮助

如果以上文档无法解决你的问题：

1. 查看完整日志：`logs/agent_*.log`
2. 运行健康检查：`python main.py health --detail`
3. 查看异常的 `remediation` 字段（U6后所有异常都包含修复建议）
4. 查看 [FAQ.md](FAQ.md)
