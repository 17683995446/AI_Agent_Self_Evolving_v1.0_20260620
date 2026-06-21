# 自驱永续进化预测修正Agent系统

一个具备自主学会操作多个Agent、增强自身技能、自驱完成长任务、永续运行能力的智能Agent系统。

## 核心能力

1. **多Agent编排** — 自主spawn/delegate/recover多个专用子Agent
2. **任务分解与工作流** — 将高层目标自动分解为DAG工作流并调度执行
3. **技能学习与进化** — 从工作流模式中提炼技能，技能可进化/淘汰
4. **长任务自驱** — 驱力阈值控制的自驱推进，支持暂停/恢复/重试
5. **状态持久化与恢复** — 快照保存/恢复 + 看门狗故障检测
6. **永续运行** — 健康监控 + 资源回收 + 分级自愈

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动Agent（默认永久运行）
python main.py run

# 启用全部能力（多Agent + 永续运行 + REST API + Dashboard）
python main.py run \
  --enable-multi-agent \
  --enable-forever \
  --rest-port 9090 \
  --dashboard-port 8080
```

> 详细入门指南见 [docs/QUICKSTART.md](docs/QUICKSTART.md)

## CLI 命令

```bash
python main.py run          # 启动Agent（默认永久运行）
python main.py run --duration 60  # 运行60秒
python main.py status       # 查看状态
python main.py health       # 健康检查
python main.py snapshot save    # 保存快照
python main.py snapshot list    # 列出快照
python main.py config validate  # 验证配置
python main.py config show      # 显示配置
python main.py config reload    # 热重载配置
```

常用选项：
- `--log-level DEBUG|INFO|WARNING|ERROR` — 控制台日志级别
- `--log-file-level DEBUG|INFO|WARNING|ERROR` — 文件日志级别
- `--enable-multi-agent` — 启用多Agent编排
- `--enable-forever` — 启用永续运行（健康监控+自愈）
- `--rest-port 9090` — REST API端口
- `--dashboard-port 8080` — Dashboard端口

## 配置

### 配置文件

主配置：`config/agent.yaml`

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

### 环境变量覆盖

环境变量优先级高于配置文件：

```bash
AGENT_MAX_CPU=30.0 AGENT_LOG_LEVEL=DEBUG python main.py run
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
| `AGENT_EXPLORATION_RATIO` | 探索比例 | `0.30` |

完整环境变量列表见 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md#环境变量参考)。

## 部署

### Docker

```bash
# 一键启动
docker compose up -d

# 启用监控栈（Prometheus + Grafana）
docker compose --profile monitoring up -d
```

### systemd

```bash
sudo cp deploy/self-evolving-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now self-evolving-agent
```

### nohup / supervisor

```bash
./deploy/start.sh           # nohup模式
./deploy/start.sh supervisor  # supervisor模式
./deploy/stop.sh            # 停止
./deploy/status.sh          # 查看状态
```

## 目录结构

```
self_evolving_agent/
├── main.py                 # 主入口，SelfEvolvingAgent主类
├── cli.py                  # CLI命令行接口（U1）
├── core/                   # 核心模块
│   ├── config.py           # 配置管理（U5：环境变量覆盖+拒绝启动）
│   ├── exceptions.py       # 异常层次结构（U6：remediation字段）
│   ├── interfaces.py       # 抽象接口
│   ├── logger.py           # 结构化日志（U6：exception方法+console_level属性）
│   ├── agent_orchestrator.py   # M1: 多Agent编排
│   ├── task_decomposer.py      # M2: 任务分解
│   ├── workflow_engine.py      # M2: 工作流引擎
│   ├── skill_library.py        # M3: 技能库
│   ├── long_task_planner.py    # M4: 长任务规划
│   ├── persistence.py          # M5: 状态持久化
│   └── forever_runner.py       # M6: 永续运行
├── modules/                # 业务实现
│   ├── perception.py       # 环境感知
│   ├── prediction.py       # 预测引擎
│   ├── decision.py         # 决策引擎
│   ├── drive_engine.py     # 驱力引擎
│   ├── error_eval.py       # 误差评估
│   ├── optimizer.py        # 优化器
│   └── memory.py           # 记忆系统
├── api/rest_server.py      # REST API（U2）
├── dashboard/dashboard_server.py  # Dashboard（U3）
├── config/                 # 配置文件
│   ├── agent.yaml          # 主配置
│   └── scenarios.yaml      # 场景模板
├── deploy/                 # 部署脚本（U4：无硬编码路径）
│   ├── start.sh / stop.sh / status.sh
│   ├── health_check.py / metrics_server.py
│   ├── supervisor_agent.conf
│   ├── self-evolving-agent.service  # systemd单元
│   └── prometheus.yml      # Prometheus配置
├── docs/                   # 文档
│   ├── QUICKSTART.md       # 快速入门
│   ├── FAQ.md              # 常见问题
│   ├── TROUBLESHOOTING.md  # 故障排除
│   ├── ARCHITECTURE.md     # 架构文档
│   └── API_REFERENCE.md    # API参考
├── tests/                  # 测试
├── Dockerfile              # Docker镜像（U4）
├── docker-compose.yml      # Docker Compose编排（U4）
├── requirements.txt        # 依赖清单
└── .env.example            # 环境变量模板
```

## 测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行特定阶段测试
python -m pytest tests/test_m1_multi_agent_orchestration.py -v  # 多Agent
python -m pytest tests/test_o1_config_order.py -v               # 秩序化
python -m pytest tests/test_u1_cli_framework.py -v              # 易用性
python -m pytest tests/test_u5_config_usability.py -v           # 配置易用性
python -m pytest tests/test_u6_debug_usability.py -v            # 调试易用性
```

## 文档

- [快速入门](docs/QUICKSTART.md)
- [常见问题](docs/FAQ.md)
- [故障排除](docs/TROUBLESHOOTING.md)
- [架构文档](docs/ARCHITECTURE.md)
- [API参考](docs/API_REFERENCE.md)

## 架构

详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
