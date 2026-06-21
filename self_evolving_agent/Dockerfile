# 自驱永续进化Agent系统 Dockerfile
# U4：多阶段构建，减小镜像体积

FROM python:3.11-slim as builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 运行阶段
FROM python:3.11-slim

LABEL maintainer="Self-Evolving-Agent"
LABEL description="自驱永续进化预测修正Agent系统"

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 复制Python依赖
COPY --from=builder /root/.local /root/.local

# 复制项目代码
WORKDIR /app
COPY . .

# 确保路径
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV AGENT_ENV=prod

# 创建数据目录
RUN mkdir -p /app/logs /app/data/snapshots

# 暴露端口（REST API和Dashboard）
EXPOSE 9090 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from cli import cli_main; cli_main(['health'])" || exit 1

# 默认启动命令：永久运行，启用REST API和Dashboard
CMD ["python", "main.py", "run", "--forever", "--enable-multi-agent", "--enable-forever", "--rest-port", "9090", "--dashboard-port", "8080"]
