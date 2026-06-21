# 快速入门

5分钟内让自驱永续进化Agent系统跑起来。

## 1. 安装

```bash
# 克隆项目后进入目录
cd self_evolving_agent

# 安装依赖（仅需3个核心包）
pip install -r requirements.txt
```

依赖清单（全部是轻量级包）：
- `pyyaml` — 配置文件解析
- `psutil` — 系统资源感知
- `numpy` — 数值计算
- `pytest` — 测试（开发时）

> REST API 与 Dashboard 使用 Python 标准库 `http.server`，无需额外 Web 框架。

## 2. 验证安装

```bash
# 运行健康检查
python main.py health

# 验证配置
python main.py config validate

# 运行核心测试（约2分钟）
python -m pytest tests/test_u1_cli_framework.py tests/test_u5_config_usability.py -v
```

## 3. 启动Agent

### 基本启动（永久运行）

```bash
# 默认永久运行，按 Ctrl+C 停止
python main.py run
```

### 启用全部能力

```bash
# 多Agent编排 + 永续运行 + REST API + Dashboard
python main.py run \
  --enable-multi-agent \
  --enable-forever \
  --rest-port 9090 \
  --dashboard-port 8080
```

启动后可访问：
- Dashboard: http://localhost:8080
- REST API: http://localhost:9090

### 定时运行

```bash
# 运行60秒后自动停止
python main.py run --duration 60
```

## 4. 常用命令

```bash
# 查看状态
python main.py status

# 健康检查（详细）
python main.py health --detail

# 保存快照
python main.py snapshot save

# 列出快照
python main.py snapshot list

# 查看当前配置
python main.py config show

# 重新加载配置（热更新）
python main.py config reload
```

## 5. 配置

### 方式一：配置文件

编辑 `config/agent.yaml`：

```yaml
resource:
  max_cpu_percent: 20.0
  max_memory_percent: 20.0

drive_weights:
  safe: 0.4
  gain: 0.4
  eff: 0.2

main_loop_frequency: 1.0
```

### 方式二：环境变量（优先级更高）

```bash
# 通过环境变量覆盖配置
export AGENT_MAX_CPU=30.0
export AGENT_LOG_LEVEL=DEBUG
export AGENT_LOOP_FREQUENCY=2.0

python main.py run
```

常用环境变量：

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| `AGENT_CONFIG` | 配置文件路径 | `./config/agent.yaml` |
| `AGENT_ENV` | 运行环境 | `dev` |
| `AGENT_MAX_CPU` | CPU使用上限(%) | `20.0` |
| `AGENT_MAX_MEMORY` | 内存使用上限(%) | `20.0` |
| `AGENT_LOG_LEVEL` | 控制台日志级别 | `INFO` |
| `AGENT_LOG_DIR` | 日志目录 | `./logs` |
| `AGENT_LOOP_FREQUENCY` | 主循环频率(Hz) | `1.0` |

完整环境变量列表见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md#环境变量参考)。

## 6. Docker 部署

```bash
# 一键启动（含Agent + 可选监控栈）
docker compose up -d

# 启用监控栈（Prometheus + Grafana）
docker compose --profile monitoring up -d
```

详见 [部署指南](../deploy/)。

## 7. 日志调试

```bash
# DEBUG级别运行（输出详细日志）
python main.py run --log-level DEBUG

# 同时控制文件日志级别
python main.py run --log-level INFO --log-file-level DEBUG

# 查看实时日志
tail -f logs/agent_*.log
```

## 下一步

- [架构文档](ARCHITECTURE.md)
- [API参考](API_REFERENCE.md)
- [常见问题](FAQ.md)
- [故障排除](TROUBLESHOOTING.md)
