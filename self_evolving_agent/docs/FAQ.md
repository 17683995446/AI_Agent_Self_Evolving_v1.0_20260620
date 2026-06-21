# 常见问题（FAQ）

## 启动相关

### Q: Agent启动时报 `ConfigurationException: 配置文件YAML语法错误`

**A**: 配置文件 `config/agent.yaml` 存在YAML语法错误。系统不会静默降级到默认配置，而是拒绝启动。

解决方法：
```bash
# 1. 验证配置
python main.py config validate

# 2. 常见语法错误：
#    - 缩进不一致（Tab和空格混用）
#    - 冒号后缺少空格（key:value 应为 key: value）
#    - 字符串未引号但含特殊字符

# 3. 临时绕过：删除配置文件，使用默认配置
mv config/agent.yaml config/agent.yaml.bak
python main.py run
```

### Q: Agent启动时报 `ConfigurationException: 驱力权重之和必须等于 1.0`

**A**: `drive_weights` 的 `safe + gain + eff` 必须等于 1.0。

```yaml
# 错误示例
drive_weights:
  safe: 0.5
  gain: 0.5
  eff: 0.5  # 总和 1.5，不等于 1.0

# 正确示例
drive_weights:
  safe: 0.4
  gain: 0.4
  eff: 0.2  # 总和 1.0
```

### Q: 如何不修改配置文件就改变Agent行为？

**A**: 使用环境变量。环境变量优先级高于配置文件。

```bash
# 临时调整CPU上限
AGENT_MAX_CPU=50.0 python main.py run

# 调整日志级别
AGENT_LOG_LEVEL=DEBUG python main.py run

# 同时调整多个参数
AGENT_MAX_CPU=30.0 AGENT_LOOP_FREQUENCY=2.0 AGENT_LOG_LEVEL=DEBUG python main.py run
```

### Q: 默认运行模式是什么？

**A**: 默认永久运行（`--forever`）。按 `Ctrl+C` 优雅停止。

如需定时运行：`python main.py run --duration 60`（运行60秒）。

## 日志相关

### Q: `--log-level` 似乎不生效？

**A**: U6已修复此问题。确保使用最新版本：

```bash
# 正确用法（U6修复后）
python main.py run --log-level DEBUG

# 验证当前级别
python -c "from core.logger import logger; print(logger.console_level)"
```

### Q: 日志文件在哪里？

**A**: 默认在 `./logs/` 目录下：
- `agent_YYYYMMDD_HHMMSS.log` — 结构化日志（DEBUG级别）
- `audit_YYYYMMDD.log` — 审计日志（关键操作）
- `agent_deploy.log` — 部署启动日志

可通过 `AGENT_LOG_DIR` 环境变量修改。

### Q: 如何减少日志噪声？

**A**:
```bash
# 提高日志级别（只显示WARNING及以上）
python main.py run --log-level WARNING

# 或通过环境变量
AGENT_LOG_LEVEL=WARNING python main.py run
```

## 部署相关

### Q: Docker部署后无法访问Dashboard？

**A**: 检查端口映射：

```bash
# 确认端口映射
docker compose ps

# 默认端口
# Dashboard: http://localhost:8080
# REST API: http://localhost:9090

# 自定义端口
AGENT_DASHBOARD_PORT=9080 docker compose up -d
```

### Q: systemd部署后Agent频繁重启？

**A**: 检查日志：

```bash
# 查看Agent日志
sudo journalctl -u self-evolving-agent -f --no-pager

# 常见原因：
# 1. 配置文件路径错误 → 检查 AGENT_DIR 环境变量
# 2. Python路径错误 → 检查 PYTHON_BIN 环境变量
# 3. 资源限制过严 → 调整 MemoryMax/CPUQuota
```

### Q: deploy/脚本中的路径如何适配我的环境？

**A**: U4已修复所有硬编码路径。脚本会自动推断路径，也可通过 `AGENT_DIR` 环境变量覆盖：

```bash
# 自动推断（推荐）
./deploy/start.sh

# 指定路径
AGENT_DIR=/opt/self-evolving-agent ./deploy/start.sh
```

## 异常处理

### Q: 异常信息中的 `remediation` 字段是什么？

**A**: U6为所有异常增加了 `remediation`（修复建议）字段，告诉你如何解决问题：

```python
from core.exceptions import ConfigurationException
try:
    raise ConfigurationException("配置错误")
except ConfigurationException as e:
    print(f"错误: {e}")
    print(f"修复建议: {e.remediation}")
    # 输出: 请检查配置文件语法与取值范围。
    #       可运行 'python main.py config validate' 验证配置...
```

### Q: 未知异常的堆栈在哪里查看？

**A**: U6修复后，未知异常的完整堆栈会记录到日志的 `traceback` 字段：

```bash
# 在日志文件中搜索 traceback
grep "traceback" logs/agent_*.log
```

## 性能相关

### Q: Agent占用CPU过高？

**A**:
```bash
# 1. 降低主循环频率
AGENT_LOOP_FREQUENCY=0.5 python main.py run  # 0.5Hz = 每2秒1轮

# 2. 降低CPU上限
AGENT_MAX_CPU=10.0 python main.py run

# 3. 关闭函数追踪
LOG_DISABLE_FUNC_TRACE=1 python main.py run
```

### Q: 内存持续增长？

**A**: 检查记忆系统：

```bash
# 查看数据库大小
ls -lh data/agent_memory.db

# 查看快照数量
ls data/snapshots/ | wc -l

# 调整存储保留比例（默认0.30，即保留30%旧快照）
# 在 config/agent.yaml 中：
# resource:
#   storage_retain_ratio: 0.15  # 保留15%，更激进清理
```
