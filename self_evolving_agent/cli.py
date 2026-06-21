"""
CLI命令行接口 —— 提供友好的命令行交互体验

U1：CLI框架重构
设计原则：
1. 子命令结构清晰：run / status / health / snapshot / config
2. --help 全覆盖，每个子命令有示例
3. 默认行为合理：无参数=forever模式（而非30秒测试）
4. 错误信息友好：包含修复建议
"""

import argparse
import sys
import os
import time
import json
import signal
from typing import Optional, List

# 路径处理
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.logger import logger


def _create_parser() -> argparse.ArgumentParser:
    """创建CLI参数解析器"""
    parser = argparse.ArgumentParser(
        prog="self-evolving-agent",
        description="自驱永续进化预测修正Agent系统 —— 自主学会操作多Agent、增强技能、自驱完成长任务、永续运行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 永久运行（默认）
  python main.py run

  # 运行60秒
  python main.py run --duration 60

  # 启用全部能力（多Agent+永续运行+REST API）
  python main.py run --enable-multi-agent --enable-forever --rest-port 9090

  # 查看状态
  python main.py status

  # 健康检查
  python main.py health --detail

  # 保存快照
  python main.py snapshot save

  # 验证配置
  python main.py config validate
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # run 子命令
    run_parser = subparsers.add_parser("run", help="启动Agent", formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py run --duration 60 --enable-multi-agent\n  python main.py run --forever")
    run_parser.add_argument("--duration", type=float, default=None,
        help="运行时长（秒），不指定则永久运行")
    run_parser.add_argument("--forever", action="store_true", default=True,
        help="永久运行（默认行为，与不指定--duration等价）")
    run_parser.add_argument("--scenario", type=str, default=None,
        help="场景名称（如system_monitor、dev_env_optimizer）")
    run_parser.add_argument("--preset", type=str, default=None,
        choices=["balanced", "aggressive", "conservative", "eco"],
        help="配置预设（覆盖配置文件中的对应字段）")
    run_parser.add_argument("--enable-multi-agent", action="store_true",
        help="启用多Agent编排")
    run_parser.add_argument("--enable-forever", action="store_true",
        help="启用永续运行（健康监控+资源回收+自愈）")
    run_parser.add_argument("--rest-port", type=int, default=None,
        help="REST API端口（如9090），不指定则不启动REST服务")
    run_parser.add_argument("--dashboard-port", type=int, default=None,
        help="Dashboard端口（如8080），不指定则不启动Dashboard")
    run_parser.add_argument("--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="控制台日志级别（默认INFO）")
    run_parser.add_argument("--log-file-level", type=str, default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="文件日志级别（默认DEBUG，保留排查细节）")
    run_parser.add_argument("--progress", action="store_true", default=True,
        help="显示运行时进度条（默认开启，--no-progress关闭）")
    run_parser.add_argument("--no-progress", dest="progress", action="store_false",
        help="关闭进度条显示")
    run_parser.add_argument("--progress-interval", type=float, default=1.0,
        help="进度条刷新间隔（秒，默认1.0）")

    # status 子命令
    status_parser = subparsers.add_parser("status", help="查看Agent运行状态")
    status_parser.add_argument("--json", action="store_true", help="以JSON格式输出")
    status_parser.add_argument("--watch", action="store_true",
        help="持续刷新状态（每2秒），按Ctrl+C退出")

    # health 子命令
    health_parser = subparsers.add_parser("health", help="健康检查")
    health_parser.add_argument("--detail", action="store_true", help="显示详细信息")

    # snapshot 子命令
    snapshot_parser = subparsers.add_parser("snapshot", help="状态快照管理")
    snapshot_sub = snapshot_parser.add_subparsers(dest="snapshot_action", help="快照操作")
    snapshot_sub.add_parser("save", help="保存当前状态快照")
    snapshot_sub.add_parser("list", help="列出所有快照")
    snapshot_restore = snapshot_sub.add_parser("restore", help="从快照恢复状态")
    snapshot_restore.add_argument("--id", type=str, required=True, help="快照ID（路径/文件名/序号/时间戳）")
    # U26：快照管理增强
    snapshot_info = snapshot_sub.add_parser("info", help="查看快照详情")
    snapshot_info.add_argument("id", type=str, help="快照ID（路径/文件名/序号/时间戳）")
    snapshot_diff = snapshot_sub.add_parser("diff", help="对比两个快照差异")
    snapshot_diff.add_argument("id1", type=str, help="快照1 ID")
    snapshot_diff.add_argument("id2", type=str, help="快照2 ID")
    snapshot_delete = snapshot_sub.add_parser("delete", help="删除快照")
    snapshot_delete.add_argument("id", type=str, help="快照ID")
    snapshot_export = snapshot_sub.add_parser("export", help="导出快照为JSON")
    snapshot_export.add_argument("id", type=str, help="快照ID")
    snapshot_export.add_argument("--output", type=str, default=None, help="输出文件路径（不指定则打印到stdout）")

    # config 子命令
    config_parser = subparsers.add_parser("config", help="配置管理")
    config_sub = config_parser.add_subparsers(dest="config_action", help="配置操作")
    config_sub.add_parser("validate", help="验证配置文件")
    config_sub.add_parser("show", help="显示当前配置")
    config_sub.add_parser("reload", help="重新加载配置")
    # U12：备份与回滚
    config_backup = config_sub.add_parser("backup", help="备份当前配置文件")
    config_backup.add_argument("--reason", type=str, default="manual",
        help="备份原因（manual/init/reload/preset/rollback）")
    config_rollback = config_sub.add_parser("rollback", help="回滚到之前的配置备份")
    config_rollback.add_argument("--name", type=str, default=None,
        help="指定备份文件名（不指定则回滚到最近一个）")
    config_sub.add_parser("backups", help="列出所有配置备份")
    # U14：迁移与版本管理
    config_migrate = config_sub.add_parser("migrate", help="迁移配置到最新版本")
    config_migrate.add_argument("--dry-run", action="store_true",
        help="仅模拟迁移，不修改文件")
    config_sub.add_parser("check", help="检查配置兼容性（版本/字段）")
    # U19：交互式配置编辑器
    config_sub.add_parser("edit", help="交互式编辑配置（引导式）")
    config_get = config_sub.add_parser("get", help="获取配置字段值")
    config_get.add_argument("field", type=str, help="字段路径（如 resource.max_cpu_percent）")
    config_set = config_sub.add_parser("set", help="设置配置字段值")
    config_set.add_argument("field", type=str, help="字段路径（如 resource.max_cpu_percent）")
    config_set.add_argument("value", type=str, help="字段值")
    config_sub.add_parser("fields", help="列出所有可编辑字段")

    # U8：init 子命令 —— 交互式初始化向导
    init_parser = subparsers.add_parser("init", help="交互式初始化向导（首次配置）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py init                  # 交互式向导\n  python main.py init --non-interactive  # 使用默认配置\n  python main.py init --force            # 覆盖现有配置")
    init_parser.add_argument("--non-interactive", action="store_true",
        help="非交互模式，使用默认配置（便于脚本化）")
    init_parser.add_argument("--force", action="store_true",
        help="覆盖现有配置文件")
    init_parser.add_argument("--config-path", type=str, default="config/agent.yaml",
        help="配置文件输出路径（默认 config/agent.yaml）")

    # U9：doctor 子命令 —— 智能诊断
    doctor_parser = subparsers.add_parser("doctor", help="一键诊断（检查环境/依赖/配置/端口）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py doctor                # 完整体检\n  python main.py doctor --json          # JSON输出\n  python main.py doctor --rest-port 9090 --dashboard-port 8080")
    doctor_parser.add_argument("--json", action="store_true",
        help="以JSON格式输出诊断结果")
    doctor_parser.add_argument("--rest-port", type=int, default=None,
        help="检查REST API端口是否可用")
    doctor_parser.add_argument("--dashboard-port", type=int, default=None,
        help="检查Dashboard端口是否可用")

    # U15：shell 子命令 —— 交互式Shell
    shell_parser = subparsers.add_parser("shell", help="交互式Shell（REPL环境）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py shell                 # 进入交互式Shell\n  # 在Shell中:\n  agent> status\n  agent> health --detail\n  agent> config show\n  agent> exit")

    # U16：alias 子命令 —— 别名管理
    alias_parser = subparsers.add_parser("alias", help="命令别名管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py alias list            # 列出所有别名\n  python main.py alias add myrun 'run --forever'\n  python main.py alias remove myrun\n  # 使用别名: python main.py s  等价于  python main.py status")
    alias_sub = alias_parser.add_subparsers(dest="alias_action", help="别名操作")
    alias_sub.add_parser("list", help="列出所有别名")
    alias_add = alias_sub.add_parser("add", help="添加自定义别名")
    alias_add.add_argument("name", type=str, help="别名名称")
    alias_add.add_argument("command", type=str, help="别名对应的完整命令")
    alias_remove = alias_sub.add_parser("remove", help="删除自定义别名")
    alias_remove.add_argument("name", type=str, help="要删除的别名名称")

    # U17：completion 子命令 —— 智能补全
    completion_parser = subparsers.add_parser("completion", help="生成shell补全脚本（bash/zsh）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py completion bash            # 输出bash补全脚本\n  python main.py completion zsh             # 输出zsh补全脚本\n  python main.py completion bash --install  # 安装到bash-completion目录\n  python main.py completion zsh --install   # 安装到zsh fpath目录\n  python main.py completion --list          # 列出支持的shell")
    completion_parser.add_argument("shell", type=str, nargs="?", default=None,
        choices=["bash", "zsh"], help="shell类型（bash/zsh）")
    completion_parser.add_argument("--install", action="store_true",
        help="安装补全脚本到系统目录")
    completion_parser.add_argument("--force", action="store_true",
        help="强制覆盖已存在的补全脚本")
    completion_parser.add_argument("--list", action="store_true",
        help="列出支持的shell类型")

    # U18：theme 子命令 —— 主题管理
    theme_parser = subparsers.add_parser("theme", help="输出主题管理（着色/预览）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py theme list                # 列出所有主题\n  python main.py theme show                 # 显示当前主题\n  python main.py theme preview              # 预览主题颜色\n  python main.py theme set dark              # 设置主题为dark\n  python main.py theme set plain             # 设置无颜色主题")
    theme_sub = theme_parser.add_subparsers(dest="theme_action", help="主题操作")
    theme_sub.add_parser("list", help="列出所有可用主题")
    theme_sub.add_parser("show", help="显示当前主题设置")
    theme_sub.add_parser("preview", help="预览当前主题的各级别颜色")
    theme_set = theme_sub.add_parser("set", help="设置主题")
    theme_set.add_argument("name", type=str, help="主题名称（default/dark/light/plain）")

    # U20：history 子命令 —— 命令历史
    history_parser = subparsers.add_parser("history", help="命令历史管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py history list               # 列出最近20条历史\n  python main.py history list --limit 50     # 列出最近50条\n  python main.py history search status       # 搜索包含status的历史\n  python main.py history clear               # 清空历史")
    history_sub = history_parser.add_subparsers(dest="history_action", help="历史操作")
    history_list = history_sub.add_parser("list", help="列出命令历史")
    history_list.add_argument("--limit", type=int, default=20, help="显示条数（默认20）")
    history_search = history_sub.add_parser("search", help="搜索命令历史")
    history_search.add_argument("keyword", type=str, help="搜索关键字")
    history_sub.add_parser("clear", help="清空所有历史")

    # U20：bookmark 子命令 —— 书签管理
    bookmark_parser = subparsers.add_parser("bookmark", help="命令书签管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py bookmark list              # 列出所有书签\n  python main.py bookmark add myrun 'run --forever'  # 添加书签\n  python main.py bookmark remove myrun       # 删除书签\n  python main.py bookmark run start          # 执行书签'start'")
    bookmark_sub = bookmark_parser.add_subparsers(dest="bookmark_action", help="书签操作")
    bookmark_sub.add_parser("list", help="列出所有书签")
    bookmark_add = bookmark_sub.add_parser("add", help="添加书签")
    bookmark_add.add_argument("name", type=str, help="书签名称")
    bookmark_add.add_argument("command", type=str, help="书签对应的命令")
    bookmark_add.add_argument("--desc", type=str, default="", help="书签描述")
    bookmark_remove = bookmark_sub.add_parser("remove", help="删除书签")
    bookmark_remove.add_argument("name", type=str, help="要删除的书签名称")
    bookmark_run = bookmark_sub.add_parser("run", help="执行书签对应的命令")
    bookmark_run.add_argument("name", type=str, help="要执行的书签名称")

    # U21：help 子命令 —— 上下文感知帮助
    help_parser = subparsers.add_parser("help", help="上下文感知帮助（基于系统状态）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py help                # 上下文感知帮助\n  python main.py help --context       # 显示系统状态上下文\n  python main.py help --troubleshoot  # 故障排除指南\n  python main.py help --quick-start   # 快速入门")
    help_parser.add_argument("--context", action="store_true",
        help="显示系统状态上下文")
    help_parser.add_argument("--troubleshoot", action="store_true",
        help="显示故障排除指南")
    help_parser.add_argument("--quick-start", action="store_true",
        help="显示快速入门指南")

    # U22：logs 子命令 —— 日志查看与过滤
    logs_parser = subparsers.add_parser("logs", help="日志查看与过滤",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py logs tail                  # 查看最新日志最后20行\n  python main.py logs tail --lines 50        # 最后50行\n  python main.py logs list                   # 列出所有日志文件\n  python main.py logs search ERROR           # 搜索包含ERROR的日志\n  python main.py logs search CONFIG --level DEBUG  # 搜索CONFIG且级别DEBUG\n  python main.py logs clear --keep 3         # 清理日志，保留最近3个")
    logs_sub = logs_parser.add_subparsers(dest="logs_action", help="日志操作")
    logs_tail = logs_sub.add_parser("tail", help="查看最新日志的末尾")
    logs_tail.add_argument("--lines", type=int, default=20, help="显示行数（默认20）")
    logs_tail.add_argument("--file", type=str, default=None, help="指定日志文件（默认最新）")
    logs_tail.add_argument("--level", type=str, default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="按级别过滤")
    logs_tail.add_argument("--no-color", action="store_true", help="禁用颜色")
    logs_list_sub = logs_sub.add_parser("list", help="列出所有日志文件")
    logs_list_sub.add_argument("--type", type=str, default="agent",
        choices=["agent", "audit", "all"], help="日志类型")
    logs_search = logs_sub.add_parser("search", help="搜索日志")
    logs_search.add_argument("keyword", type=str, help="搜索关键字")
    logs_search.add_argument("--file", type=str, default=None, help="指定日志文件")
    logs_search.add_argument("--level", type=str, default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="按级别过滤")
    logs_search.add_argument("--no-color", action="store_true", help="禁用颜色")
    logs_clear = logs_sub.add_parser("clear", help="清理日志文件")
    logs_clear.add_argument("--type", type=str, default="agent",
        choices=["agent", "audit", "all"], help="日志类型")
    logs_clear.add_argument("--keep", type=int, default=1, help="保留最近N个文件（默认1）")

    # U23：alerts 子命令 —— 告警管理
    alerts_parser = subparsers.add_parser("alerts", help="告警管理（规则/历史/通道/静默/测试）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py alerts rules                # 列出告警规则\n  python main.py alerts channels             # 列出告警通道\n  python main.py alerts history              # 查看告警历史\n  python main.py alerts stats                # 告警统计\n  python main.py alerts silence cpu_high 300 # 静默cpu_high规则300秒\n  python main.py alerts test cpu_high 0.9    # 测试cpu_high规则在值0.9时是否触发")
    alerts_sub = alerts_parser.add_subparsers(dest="alerts_action", help="告警操作")
    alerts_sub.add_parser("rules", help="列出告警规则")
    alerts_sub.add_parser("channels", help="列出告警通道")
    alerts_history = alerts_sub.add_parser("history", help="查看告警历史")
    alerts_history.add_argument("--limit", type=int, default=50, help="显示条数（默认50）")
    alerts_sub.add_parser("stats", help="告警统计")
    alerts_silence = alerts_sub.add_parser("silence", help="静默规则")
    alerts_silence.add_argument("name", type=str, help="规则名称")
    alerts_silence.add_argument("duration", type=float, help="静默时长（秒）")
    alerts_test = alerts_sub.add_parser("test", help="测试规则")
    alerts_test.add_argument("name", type=str, help="规则名称")
    alerts_test.add_argument("value", type=float, help="模拟指标值")

    # U24：perf 子命令 —— 性能分析
    perf_parser = subparsers.add_parser("perf", help="性能分析（统计/慢操作/报告/重置）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py perf stats                # 查看所有操作统计\n  python main.py perf stats perceive       # 查看单操作统计\n  python main.py perf slow                 # 查看慢操作\n  python main.py perf slow --threshold 0.2 # 指定慢操作阈值0.2秒\n  python main.py perf summary              # 性能摘要\n  python main.py perf report               # 完整性能报告\n  python main.py perf reset                # 重置性能统计")
    perf_sub = perf_parser.add_subparsers(dest="perf_action", help="性能操作")
    perf_stats = perf_sub.add_parser("stats", help="查看操作统计")
    perf_stats.add_argument("name", type=str, nargs="?", default=None, help="操作名（可选，不指定则查看全部）")
    perf_slow = perf_sub.add_parser("slow", help="查看慢操作")
    perf_slow.add_argument("--threshold", type=float, default=None, help="慢操作阈值（秒）")
    perf_sub.add_parser("summary", help="性能摘要")
    perf_sub.add_parser("report", help="完整性能报告")
    perf_sub.add_parser("reset", help="重置性能统计")

    # U25：agents 子命令 —— 多Agent控制台
    agents_parser = subparsers.add_parser("agents", help="多Agent控制台（状态/列表/创建/委派/重启/健康）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py agents status                 # 集群状态\n  python main.py agents list                   # 列出子Agent\n  python main.py agents spawn perception       # 创建感知Agent\n  python main.py agents spawn-default          # 创建默认4个子Agent\n  python main.py agents delegate predict '{}'  # 委派预测任务\n  python main.py agents restart perception-1   # 重启Agent\n  python main.py agents health                 # 健康检查\n  python main.py agents recover                # 自动恢复故障Agent\n  python main.py agents stop perception-1      # 停止Agent\n  python main.py agents history                # 委派历史")
    agents_sub = agents_parser.add_subparsers(dest="agents_action", help="Agent操作")
    agents_sub.add_parser("status", help="集群状态")
    agents_sub.add_parser("list", help="列出子Agent")
    agents_spawn = agents_sub.add_parser("spawn", help="创建子Agent")
    agents_spawn.add_argument("role", type=str,
        choices=["perception", "prediction", "decision", "execution"],
        help="角色: perception/prediction/decision/execution")
    agents_spawn.add_argument("--id", type=str, default=None, help="自定义Agent ID")
    agents_sub.add_parser("spawn-default", help="创建默认4个子Agent")
    agents_delegate = agents_sub.add_parser("delegate", help="委派任务")
    agents_delegate.add_argument("task_type", type=str,
        choices=["perceive", "predict", "decide", "execute", "analyze"],
        help="任务类型")
    agents_delegate.add_argument("payload", type=str, help="任务参数（JSON字符串）")
    agents_delegate.add_argument("--timeout", type=float, default=None, help="超时秒数")
    agents_restart = agents_sub.add_parser("restart", help="重启子Agent")
    agents_restart.add_argument("agent_id", type=str, help="Agent ID")
    agents_sub.add_parser("health", help="健康检查")
    agents_sub.add_parser("recover", help="自动恢复故障Agent")
    agents_stop = agents_sub.add_parser("stop", help="停止子Agent")
    agents_stop.add_argument("agent_id", type=str, help="Agent ID")
    agents_history = agents_sub.add_parser("history", help="委派历史")
    agents_history.add_argument("--limit", type=int, default=20, help="显示条数（默认20）")

    # U27：export 子命令 —— 数据导出
    export_parser = subparsers.add_parser("export", help="数据导出（JSON/CSV）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py export status                       # 导出状态数据(JSON)\n  python main.py export health --csv                 # 导出健康数据(CSV)\n  python main.py export alerts --output alerts.json # 导出告警到文件\n  python main.py export perf --csv --output perf.csv # 导出性能到CSV文件\n  python main.py export snapshot                     # 导出快照列表\n  python main.py export --list                       # 列出可用数据源")
    export_parser.add_argument("source", type=str, nargs="?", default=None,
        choices=["status", "health", "alerts", "perf", "snapshot"],
        help="数据源: status/health/alerts/perf/snapshot")
    export_parser.add_argument("--format", type=str, default="json",
        choices=["json", "csv"], help="输出格式（默认json）")
    export_parser.add_argument("--output", type=str, default=None, help="输出文件路径（不指定则打印到stdout）")
    export_parser.add_argument("--list", action="store_true", help="列出可用数据源")

    # U28：plugins 子命令 —— 插件管理
    plugins_parser = subparsers.add_parser("plugins", help="插件管理（列表/详情/使用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py plugins list                    # 列出所有插件\n  python main.py plugins list prediction         # 列出预测模块插件\n  python main.py plugins info prediction enhanced # 查看enhanced预测插件详情\n  python main.py plugins use prediction enhanced  # 查看如何使用enhanced插件\n  python main.py plugins types                    # 列出模块类型")
    plugins_sub = plugins_parser.add_subparsers(dest="plugins_action", help="插件操作")
    plugins_list = plugins_sub.add_parser("list", help="列出插件")
    plugins_list.add_argument("module_type", type=str, nargs="?", default=None,
        choices=["perception", "prediction", "drive_engine", "decision", "error_eval", "optimizer", "memory"],
        help="模块类型（可选，不指定则列出全部）")
    plugins_info = plugins_sub.add_parser("info", help="查看插件详情")
    plugins_info.add_argument("module_type", type=str,
        choices=["perception", "prediction", "drive_engine", "decision", "error_eval", "optimizer", "memory"],
        help="模块类型")
    plugins_info.add_argument("name", type=str, help="插件名")
    plugins_use = plugins_sub.add_parser("use", help="查看插件使用方式")
    plugins_use.add_argument("module_type", type=str,
        choices=["perception", "prediction", "drive_engine", "decision", "error_eval", "optimizer", "memory"],
        help="模块类型")
    plugins_use.add_argument("name", type=str, help="插件名")
    plugins_sub.add_parser("types", help="列出模块类型")

    # U29：skill 子命令 —— 技能库管理
    skill_parser = subparsers.add_parser("skill", help="技能库管理（列表/详情/搜索/进化/统计）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py skill list                      # 列出所有技能\n  python main.py skill list --status active      # 列出活跃技能\n  python main.py skill info abc12345              # 查看技能详情\n  python main.py skill search clean               # 搜索clean类型技能\n  python main.py skill evolve abc12345 step1,step2 # 进化技能\n  python main.py skill stats                      # 技能库统计\n  python main.py skill clear                      # 清空技能库")
    skill_sub = skill_parser.add_subparsers(dest="skill_action", help="技能操作")
    skill_list = skill_sub.add_parser("list", help="列出技能")
    skill_list.add_argument("--status", type=str, default=None,
        choices=["draft", "active", "deprecated", "evolving"], help="状态过滤")
    skill_info = skill_sub.add_parser("info", help="查看技能详情")
    skill_info.add_argument("id", type=str, help="技能ID")
    skill_search = skill_sub.add_parser("search", help="按任务类型搜索技能")
    skill_search.add_argument("task_type", type=str, help="任务类型")
    skill_evolve = skill_sub.add_parser("evolve", help="进化技能")
    skill_evolve.add_argument("id", type=str, help="技能ID")
    skill_evolve.add_argument("steps", type=str, help="新步骤（逗号分隔）")
    skill_evolve.add_argument("--params", type=str, default=None, help="新参数（JSON字符串）")
    skill_sub.add_parser("stats", help="技能库统计")
    skill_sub.add_parser("clear", help="清空技能库")

    # U30：long-task 子命令 —— 长任务管理
    longtask_parser = subparsers.add_parser("long-task", help="长任务管理（规划/列表/状态/推进/暂停/恢复）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py long-task plan system_optimization     # 规划系统优化任务\n  python main.py long-task plan system_optimization --name my_task # 指定任务名\n  python main.py long-task list                           # 列出所有任务\n  python main.py long-task status abc12345                # 查看任务状态\n  python main.py long-task advance abc12345               # 推进任务\n  python main.py long-task advance abc12345 --drive 0.9   # 指定驱力推进\n  python main.py long-task pause abc12345                 # 暂停任务\n  python main.py long-task resume abc12345                # 恢复任务\n  python main.py long-task stats                          # 任务统计\n  python main.py long-task templates                      # 可用模板")
    longtask_sub = longtask_parser.add_subparsers(dest="longtask_action", help="长任务操作")
    longtask_plan = longtask_sub.add_parser("plan", help="规划长任务")
    longtask_plan.add_argument("template", type=str, help="模板名称")
    longtask_plan.add_argument("--name", type=str, default="", help="任务名称")
    longtask_plan.add_argument("--desc", type=str, default="", help="任务描述")
    longtask_sub.add_parser("list", help="列出所有任务")
    longtask_status = longtask_sub.add_parser("status", help="查看任务状态")
    longtask_status.add_argument("id", type=str, help="任务ID")
    longtask_advance = longtask_sub.add_parser("advance", help="推进任务")
    longtask_advance.add_argument("id", type=str, help="任务ID")
    longtask_advance.add_argument("--drive", type=float, default=0.8, help="驱力水平[0-1]（默认0.8）")
    longtask_pause = longtask_sub.add_parser("pause", help="暂停任务")
    longtask_pause.add_argument("id", type=str, help="任务ID")
    longtask_resume = longtask_sub.add_parser("resume", help="恢复任务")
    longtask_resume.add_argument("id", type=str, help="任务ID")
    longtask_sub.add_parser("stats", help="任务统计")
    longtask_sub.add_parser("templates", help="可用模板")

    # U31：workflow 子命令 —— 工作流管理
    workflow_parser = subparsers.add_parser("workflow", help="工作流管理（提交/状态/统计/列表）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py workflow submit monitor_and_alert              # 提交监控告警工作流\n  python main.py workflow submit optimize_resource --desc '优化CPU' # 指定描述\n  python main.py workflow submit custom --params '{\"key\":\"value\"}'  # 带参数\n  python main.py workflow list                                    # 列出所有工作流\n  python main.py workflow status abc12345                         # 查看工作流状态\n  python main.py workflow stats                                   # 引擎统计\n  python main.py workflow goals                                   # 可用目标类型")
    workflow_sub = workflow_parser.add_subparsers(dest="workflow_action", help="工作流操作")
    workflow_submit = workflow_sub.add_parser("submit", help="提交目标工作流")
    workflow_submit.add_argument("goal_type", type=str,
        choices=["monitor_and_alert", "optimize_resource", "clean_and_recover", "analyze_and_decide", "custom"],
        help="目标类型")
    workflow_submit.add_argument("--desc", type=str, default="", help="工作流描述")
    workflow_submit.add_argument("--params", type=str, default=None, help="参数（JSON字符串）")
    workflow_sub.add_parser("list", help="列出所有工作流")
    workflow_status = workflow_sub.add_parser("status", help="查看工作流状态")
    workflow_status.add_argument("id", type=str, help="工作流ID")
    workflow_sub.add_parser("stats", help="引擎统计")
    workflow_sub.add_parser("goals", help="可用目标类型")

    # U32：forever 子命令 —— 永续运行与自愈控制
    forever_parser = subparsers.add_parser("forever", help="永续运行与自愈控制（启用/禁用/统计/健康/自愈/清理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py forever enable                                    # 启用永续运行\n  python main.py forever enable --health-interval 60 --cleanup-interval 600  # 自定义间隔\n  python main.py forever disable                                   # 禁用永续运行\n  python main.py forever stats                                     # 永续统计\n  python main.py forever health                                    # 健康报告\n  python main.py forever history                                   # 健康历史\n  python main.py forever history --limit 20                        # 指定条数\n  python main.py forever self-heal                                 # 手动触发自愈\n  python main.py forever cleanup                                   # 手动触发资源回收\n  python main.py forever auto-recovery                             # 查看自动恢复状态\n  python main.py forever auto-recovery --enable                    # 启用自动恢复\n  python main.py forever auto-recovery --disable                   # 禁用自动恢复")
    forever_sub = forever_parser.add_subparsers(dest="forever_action", help="永续运行操作")
    forever_enable = forever_sub.add_parser("enable", help="启用永续运行")
    forever_enable.add_argument("--health-interval", type=float, default=30.0, help="健康检查间隔（秒，默认30）")
    forever_enable.add_argument("--cleanup-interval", type=float, default=300.0, help="资源回收间隔（秒，默认300）")
    forever_sub.add_parser("disable", help="禁用永续运行")
    forever_sub.add_parser("stats", help="永续运行统计")
    forever_sub.add_parser("health", help="健康报告")
    forever_history = forever_sub.add_parser("history", help="健康历史")
    forever_history.add_argument("--limit", type=int, default=10, help="返回条数（默认10）")
    forever_sub.add_parser("self-heal", help="手动触发自愈")
    forever_sub.add_parser("cleanup", help="手动触发资源回收")
    forever_ar = forever_sub.add_parser("auto-recovery", help="自动恢复状态/设置")
    forever_ar_group = forever_ar.add_mutually_exclusive_group()
    forever_ar_group.add_argument("--enable", action="store_true", help="启用自动恢复")
    forever_ar_group.add_argument("--disable", action="store_true", help="禁用自动恢复")

    # U33：presets 子命令 —— 配置预设管理
    presets_parser = subparsers.add_parser("presets", help="配置预设管理（列表/详情/应用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py presets list                  # 列出所有预设\n  python main.py presets show balanced         # 查看预设详情\n  python main.py presets apply aggressive      # 应用预设到当前配置")
    presets_sub = presets_parser.add_subparsers(dest="presets_action", help="预设操作")
    presets_sub.add_parser("list", help="列出所有预设")
    presets_show = presets_sub.add_parser("show", help="查看预设详情")
    presets_show.add_argument("name", type=str, help="预设名称")
    presets_apply = presets_sub.add_parser("apply", help="应用预设到当前配置")
    presets_apply.add_argument("name", type=str, help="预设名称")

    # U33：scenarios 子命令 —— 运行场景发现
    scenarios_parser = subparsers.add_parser("scenarios", help="运行场景发现（列表/详情）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py scenarios list                # 列出所有场景\n  python main.py scenarios info system_monitor # 查看场景详情")
    scenarios_sub = scenarios_parser.add_subparsers(dest="scenarios_action", help="场景操作")
    scenarios_sub.add_parser("list", help="列出所有场景")
    scenarios_info = scenarios_sub.add_parser("info", help="查看场景详情")
    scenarios_info.add_argument("name", type=str, help="场景名称")

    # U34：learn 子命令 —— 持续学习与元学习
    learn_parser = subparsers.add_parser("learn", help="持续学习与元学习（初始化/训练/统计/评估）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py learn init                              # 初始化学习系统\n  python main.py learn init --input-dim 8 --output-dim 3 # 自定义维度\n  python main.py learn train --type meta --iterations 20  # 元学习训练\n  python main.py learn train --type continual --task task1 # 持续学习训练\n  python main.py learn stats                             # 学习统计\n  python main.py learn evaluate --type meta               # 评估元学习\n  python main.py learn evaluate --type continual --task task1 # 评估持续学习")
    learn_sub = learn_parser.add_subparsers(dest="learn_action", help="学习操作")
    learn_init = learn_sub.add_parser("init", help="初始化学习系统")
    learn_init.add_argument("--input-dim", type=int, default=4, help="输入维度（默认4）")
    learn_init.add_argument("--output-dim", type=int, default=2, help="输出维度（默认2）")
    learn_train = learn_sub.add_parser("train", help="训练")
    learn_train.add_argument("--type", choices=["meta", "continual"], default="meta",
        help="训练类型：meta（元学习）/ continual（持续学习），默认meta")
    learn_train.add_argument("--iterations", type=int, default=10, help="元学习训练迭代数（默认10）")
    learn_train.add_argument("--task", type=str, default="task_1", help="持续学习任务ID")
    learn_train.add_argument("--samples", type=int, default=50, help="合成数据样本数（默认50）")
    learn_train.add_argument("--steps", type=int, default=50, help="持续学习训练步数（默认50）")
    learn_sub.add_parser("stats", help="学习统计")
    learn_eval = learn_sub.add_parser("evaluate", help="评估")
    learn_eval.add_argument("--type", choices=["meta", "continual"], default="meta",
        help="评估类型：meta / continual，默认meta")
    learn_eval.add_argument("--task", type=str, default="task_1", help="持续学习任务ID")
    learn_eval.add_argument("--n-tasks", type=int, default=10, help="元学习评估任务数（默认10）")
    learn_eval.add_argument("--samples", type=int, default=50, help="合成数据样本数（默认50）")

    # U35：collab 子命令 —— 多Agent协作
    collab_parser = subparsers.add_parser("collab", help="多Agent协作（统计/分享/经验）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py collab stats                                    # 协作统计\n  python main.py collab share --action optimize --reward 0.8     # 分享经验\n  python main.py collab share --action optimize --reward 0.8 --state '{\"cpu\":80}'  # 带状态\n  python main.py collab experiences                               # 接收的经验\n  python main.py collab experiences --n 20                        # 指定条数")
    collab_sub = collab_parser.add_subparsers(dest="collab_action", help="协作操作")
    collab_sub.add_parser("stats", help="协作统计")
    collab_share = collab_sub.add_parser("share", help="分享经验")
    collab_share.add_argument("--action", type=str, required=True, help="动作名称")
    collab_share.add_argument("--reward", type=float, default=0.0, help="奖励值")
    collab_share.add_argument("--state", type=str, default=None, help="状态（JSON字符串）")
    collab_share.add_argument("--next-state", type=str, default=None, help="下一状态（JSON字符串）")
    collab_share.add_argument("--td-error", type=float, default=0.0, help="TD误差")
    collab_share.add_argument("--tags", type=str, default=None, help="标签（逗号分隔）")
    collab_experiences = collab_sub.add_parser("experiences", help="接收到的共享经验")
    collab_experiences.add_argument("--n", type=int, default=10, help="返回条数（默认10）")

    # U35：consensus 子命令 —— 共识投票
    consensus_parser = subparsers.add_parser("consensus", help="共识投票（投票/结果/统计）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py consensus vote --topic deploy --agent agent1 --proposal yes   # 投票\n  python main.py consensus vote --topic deploy --agent agent2 --proposal yes --confidence 0.9  # 带置信度\n  python main.py consensus result --topic deploy                               # 查看共识结果\n  python main.py consensus stats                                               # 共识统计")
    consensus_sub = consensus_parser.add_subparsers(dest="consensus_action", help="共识操作")
    consensus_vote = consensus_sub.add_parser("vote", help="投票")
    consensus_vote.add_argument("--topic", type=str, required=True, help="议题")
    consensus_vote.add_argument("--agent", type=str, required=True, help="投票者ID")
    consensus_vote.add_argument("--proposal", type=str, required=True, help="提案")
    consensus_vote.add_argument("--confidence", type=float, default=1.0, help="置信度（默认1.0）")
    consensus_result = consensus_sub.add_parser("result", help="查看共识结果")
    consensus_result.add_argument("--topic", type=str, required=True, help="议题")
    consensus_sub.add_parser("stats", help="共识统计")

    return parser


def _cmd_run(args) -> int:
    """执行run子命令"""
    from main import SelfEvolvingAgent, _agent_instance as _

    # U6：设置日志级别 —— 通过 set_console_level 实际生效
    if args.log_level:
        import core.logger as lg
        lg.logger.set_console_level(args.log_level)
        logger.info("CLI", f"控制台日志级别设置为: {args.log_level} (当前: {lg.logger.console_level})", module="cli")
    if getattr(args, "log_file_level", None):
        import core.logger as lg
        lg.logger.set_file_level(args.log_file_level)
        logger.info("CLI", f"文件日志级别设置为: {args.log_file_level}", module="cli")

    # 注册信号处理
    def signal_handler(sig, frame):
        logger.info("CLI", "收到停止信号，正在优雅关闭...", module="cli")
        if _agent_instance_global:
            _agent_instance_global.stop()
        sys.exit(0)

    global _agent_instance_global
    _agent_instance_global = None

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 创建Agent
    agent = SelfEvolvingAgent()
    _agent_instance_global = agent

    # U11：应用配置预设（优先级高于配置文件，低于环境变量）
    if args.preset:
        try:
            from core.presets import apply_preset_to_config
            apply_preset_to_config(args.preset, agent.config)
            logger.info("CLI", f"已应用预设: {args.preset}", module="cli")
        except Exception as e:
            logger.warning("CLI", f"应用预设失败: {e}", module="cli")

    # 应用场景配置
    if args.scenario:
        try:
            from core.scenario_loader import ScenarioLoader
            loader = ScenarioLoader()
            config = loader.load_scenario(args.scenario)
            if config:
                logger.info("CLI", f"已加载场景: {args.scenario}", module="cli")
            else:
                logger.warning("CLI", f"场景 {args.scenario} 不存在，使用默认配置", module="cli")
        except Exception as e:
            logger.warning("CLI", f"加载场景失败: {e}，使用默认配置", module="cli")

    # 启动Agent
    agent.start()
    logger.info("CLI", "Agent已启动", module="cli")

    # 启用多Agent
    if args.enable_multi_agent:
        try:
            ids = agent.enable_multi_agent(spawn_default=True)
            logger.info("CLI", f"已启用多Agent编排，创建{len(ids)}个子Agent", module="cli")
        except Exception as e:
            logger.warning("CLI", f"启用多Agent失败: {e}", module="cli")

    # 启用永续运行
    if args.enable_forever:
        try:
            agent.enable_forever_running()
            logger.info("CLI", "已启用永续运行", module="cli")
        except Exception as e:
            logger.warning("CLI", f"启用永续运行失败: {e}", module="cli")

    # 启动REST API
    if args.rest_port:
        try:
            from api.rest_server import start_rest_server, set_agent
            set_agent(agent)
            start_rest_server(port=args.rest_port, block=False)
            logger.info("CLI", f"REST API已启动: http://localhost:{args.rest_port}", module="cli")
        except Exception as e:
            logger.warning("CLI", f"启动REST API失败: {e}", module="cli")

    # 启动Dashboard
    if args.dashboard_port:
        try:
            from dashboard.dashboard_server import DashboardServer
            dashboard = DashboardServer(port=args.dashboard_port)
            dashboard.start()
            logger.info("CLI", f"Dashboard已启动: http://localhost:{args.dashboard_port}", module="cli")
        except Exception as e:
            logger.warning("CLI", f"启动Dashboard失败: {e}", module="cli")

    # 运行模式
    if args.duration is not None:
        logger.info("CLI", f"运行模式: 定时 {args.duration}秒", module="cli")
        agent.run_for_duration(args.duration)
    else:
        logger.info("CLI", "运行模式: 永久运行（按Ctrl+C停止）", module="cli")
        # U13：进度条集成
        progress_bar = None
        if args.progress:
            from core.progress import ProgressBar, ProgressConfig
            from core.status_formatter import StatusFormatter
            prog_config = ProgressConfig(refresh_interval=args.progress_interval)
            progress_bar = ProgressBar(config=prog_config)
        try:
            while agent.running:
                if progress_bar:
                    status = StatusFormatter.collect_status(agent)
                    progress_bar.update(status)
                time.sleep(args.progress_interval if progress_bar else 1.0)
        except KeyboardInterrupt:
            agent.stop()
        finally:
            if progress_bar:
                progress_bar.finish()

    agent.stop()
    logger.info("CLI", "Agent已停止", module="cli")
    return 0


def _cmd_status(args) -> int:
    """U10：执行status子命令 —— 人类友好 + JSON双模式"""
    from main import SelfEvolvingAgent
    from core.status_formatter import StatusFormatter
    agent = SelfEvolvingAgent()
    try:
        def render():
            status = StatusFormatter.collect_status(agent)
            fmt = "json" if args.json else "human"
            print(StatusFormatter.format(status, fmt))

        if args.watch:
            try:
                while True:
                    render()
                    time.sleep(2.0)
            except KeyboardInterrupt:
                print("\n已退出watch模式")
        else:
            render()
    finally:
        agent.stop()
    return 0


def _cmd_health(args) -> int:
    """执行health子命令"""
    from main import SelfEvolvingAgent
    agent = SelfEvolvingAgent()
    try:
        report = agent.get_health_report()
        if report:
            print("=== 健康报告 ===")
            print(f"  级别: {report.get('level', 'unknown')}")
            print(f"  分数: {report.get('score', 0):.2f}")
            if args.detail:
                issues = report.get("issues", [])
                if issues:
                    print("  问题:")
                    for issue in issues:
                        print(f"    - {issue}")
                recommendations = report.get("recommendations", [])
                if recommendations:
                    print("  建议:")
                    for rec in recommendations:
                        print(f"    - {rec}")
                metrics = report.get("metrics", {})
                if metrics:
                    print("  指标:")
                    for k, v in metrics.items():
                        print(f"    {k}: {v}")
        else:
            print("无法获取健康报告")
            return 1
    finally:
        agent.stop()
    return 0


def _cmd_snapshot(args) -> int:
    """执行snapshot子命令"""
    # U26：info/diff/delete/export 使用轻量级SnapshotCliHelper，不需要启动Agent
    if args.snapshot_action in ("info", "diff", "delete", "export"):
        from core.snapshot_cli import SnapshotCliHelper
        try:
            helper = SnapshotCliHelper()
        except Exception as e:
            print(f"初始化快照管理器失败: {e}")
            return 1

        if args.snapshot_action == "info":
            print(helper.format_info(args.id))
            return 0
        elif args.snapshot_action == "diff":
            print(helper.format_diff(args.id1, args.id2))
            return 0
        elif args.snapshot_action == "delete":
            success = helper.delete_snapshot(args.id)
            if success:
                print(f"快照 '{args.id}' 已删除")
                return 0
            else:
                print(f"删除失败: 快照 '{args.id}' 不存在")
                return 1
        elif args.snapshot_action == "export":
            result = helper.export_snapshot(args.id, args.output)
            if result is None:
                print(f"导出失败: 快照 '{args.id}' 不存在或加载失败")
                return 1
            if args.output:
                print(f"快照已导出到: {result}")
            else:
                print(result)
            return 0
        return 0

    # save/list/restore 需要启动Agent
    from main import SelfEvolvingAgent
    agent = SelfEvolvingAgent()
    try:
        if args.snapshot_action == "save":
            path = agent.save_state()
            print(f"快照已保存: {path}")
        elif args.snapshot_action == "list":
            # U26：使用增强的列表格式
            from core.snapshot_cli import SnapshotCliHelper
            try:
                helper = SnapshotCliHelper()
                print(helper.format_list())
            except Exception:
                # 回退到简单格式
                snapshots = agent.list_snapshots()
                print(f"=== 快照列表 ({len(snapshots)}个) ===")
                for s in snapshots:
                    print(f"  {s}")
        elif args.snapshot_action == "restore":
            # U26修复：将--id参数传递给load_state
            from core.snapshot_cli import SnapshotCliHelper
            try:
                helper = SnapshotCliHelper()
                path = helper.resolve_snapshot_id(args.id)
            except Exception:
                path = None

            if path is None:
                # 回退：直接使用args.id作为路径
                path = args.id

            result = agent.load_state(path)
            print(f"状态恢复: {'成功' if result else '失败'}")
            return 0 if result else 1
        else:
            print("请指定快照操作: save / list / restore / info / diff / delete / export")
            return 1
    finally:
        agent.stop()
    return 0


def _cmd_config(args) -> int:
    """执行config子命令"""
    from core.config import config_manager
    if args.config_action == "validate":
        try:
            result = config_manager.config.validate()
            if result:
                print("配置验证通过")
                return 0
            else:
                print("配置验证失败")
                return 1
        except Exception as e:
            print(f"配置验证异常: {e}")
            return 1
    elif args.config_action == "show":
        config = config_manager.config
        print("=== 当前配置 ===")
        print(f"  资源约束: CPU<{config.resource.max_cpu_percent}% 内存<{config.resource.max_memory_percent}%")
        print(f"  驱力权重: safe={config.drive_weights.safe} gain={config.drive_weights.gain} eff={config.drive_weights.eff}")
        print(f"  主循环频率: {config.main_loop_frequency}Hz")
        print(f"  日志目录: {config.log_dir}")
        return 0
    elif args.config_action == "reload":
        try:
            # U12：reload前自动备份
            from core.config_backup import ConfigBackupManager
            mgr_backup = ConfigBackupManager()
            backup = mgr_backup.backup(reason="reload")
            if backup:
                logger.info("CLI", f"配置已备份: {backup.name}", module="cli")
            config_manager.load()
            print("配置已重新加载")
            return 0
        except Exception as e:
            print(f"配置重载失败: {e}")
            return 1
    elif args.config_action == "backup":
        # U12：手动备份
        from core.config_backup import ConfigBackupManager
        mgr_backup = ConfigBackupManager()
        entry = mgr_backup.backup(reason=args.reason)
        if entry:
            print(f"配置已备份: {entry.name}")
            print(f"  时间: {entry.time_str}  原因: {entry.reason}  大小: {entry.size}B")
            return 0
        else:
            print("配置文件不存在，无法备份")
            return 1
    elif args.config_action == "rollback":
        # U12：回滚配置
        from core.config_backup import ConfigBackupManager
        mgr_backup = ConfigBackupManager()
        try:
            pre_backup = mgr_backup.rollback(backup_name=args.name)
            if pre_backup:
                print(f"回滚前已备份当前配置: {pre_backup.name}")
            print("配置已回滚")
            return 0
        except FileNotFoundError as e:
            print(f"回滚失败: {e}")
            return 1
        except Exception as e:
            print(f"回滚失败: {e}")
            return 1
    elif args.config_action == "backups":
        # U12：列出备份
        from core.config_backup import ConfigBackupManager
        mgr_backup = ConfigBackupManager()
        print(mgr_backup.format_backups_list())
        return 0
    elif args.config_action == "migrate":
        # U14：迁移配置
        from core.config_migrator import ConfigMigrator
        migrator = ConfigMigrator()
        if not migrator.needs_migration():
            print("配置已是最新版本，无需迁移")
            return 0
        # 迁移前自动备份
        from core.config_backup import ConfigBackupManager
        ConfigBackupManager().backup(reason="pre_migrate")
        result = migrator.migrate(dry_run=args.dry_run)
        if result.success:
            print(f"迁移成功: {result.message}")
            if result.migrated_fields:
                print(f"  迁移字段: {', '.join(result.migrated_fields)}")
            if args.dry_run:
                print("  （dry-run模式，未实际修改文件）")
            return 0
        else:
            print(f"迁移失败: {result.message}")
            return 1
    elif args.config_action == "check":
        # U14：兼容性检查
        from core.config_migrator import ConfigMigrator
        migrator = ConfigMigrator()
        print(migrator.format_compatibility_report())
        report = migrator.check_compatibility()
        return 1 if report["missing_fields"] else 0
    elif args.config_action == "edit":
        # U19：交互式编辑
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        modified = editor.interactive_edit(config_manager.config)
        if modified:
            try:
                editor.save_config(config_manager.config)
                print("配置已保存")
                return 0
            except Exception as e:
                print(f"保存失败: {e}")
                return 1
        else:
            print("未修改")
            return 0
    elif args.config_action == "get":
        # U19：获取字段值
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        meta = editor.get_field_meta(args.field)
        if meta is None:
            print(f"未知字段: {args.field}")
            print("使用 'config fields' 查看可编辑字段")
            return 1
        try:
            value = editor.get_field_value(config_manager.config, args.field)
            print(f"{meta.label} ({args.field})")
            print(f"  值: {meta.format_value(value)}")
            print(f"  描述: {meta.description}")
            return 0
        except Exception as e:
            print(f"读取失败: {e}")
            return 1
    elif args.config_action == "set":
        # U19：设置字段值
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        meta = editor.get_field_meta(args.field)
        if meta is None:
            print(f"未知字段: {args.field}")
            print("使用 'config fields' 查看可编辑字段")
            return 1
        # 解析值
        try:
            new_value = meta.parse_value(args.value)
        except ValueError as e:
            print(f"值解析失败: {e}")
            return 1
        # 验证
        valid, msg = meta.validate_value(new_value)
        if not valid:
            print(f"值无效: {msg}")
            return 1
        # 预览
        print(editor.preview_change(config_manager.config, args.field, new_value))
        # 应用
        if not editor.set_field_value(config_manager.config, args.field, new_value):
            print("设置失败")
            return 1
        # 保存
        try:
            editor.save_config(config_manager.config)
            print("配置已保存")
            return 0
        except Exception as e:
            print(f"保存失败: {e}")
            return 1
    elif args.config_action == "fields":
        # U19：列出可编辑字段
        from core.config_editor import ConfigEditor
        editor = ConfigEditor()
        print(editor.format_fields_list(config_manager.config))
        return 0
    else:
        print("请指定配置操作: validate / show / reload")
        return 1


# 全局Agent实例引用（用于信号处理）
_agent_instance_global = None


def _cmd_init(args) -> int:
    """U8：执行init子命令 —— 交互式初始化向导"""
    from core.init_wizard import InitWizard
    wizard = InitWizard(config_path=args.config_path)
    return wizard.run(force=args.force, non_interactive=args.non_interactive)


def _cmd_doctor(args) -> int:
    """U9：执行doctor子命令 —— 智能诊断"""
    from core.doctor import Doctor
    doctor = Doctor()
    doctor.run_all_checks(
        rest_port=args.rest_port,
        dashboard_port=args.dashboard_port,
    )
    if args.json:
        print(json.dumps(doctor.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(doctor.format_report())
    summary = doctor.get_summary()
    # 退出码：有FAIL返回1，仅WARN返回0（警告不阻断）
    return 1 if summary["failed"] > 0 else 0


def _cmd_shell(args) -> int:
    """U15：执行shell子命令 —— 交互式Shell"""
    from core.interactive_shell import InteractiveShell
    shell = InteractiveShell()
    return shell.run()


def _cmd_alias(args) -> int:
    """U16：执行alias子命令 —— 别名管理"""
    from core.aliases import AliasManager
    mgr = AliasManager()
    if args.alias_action == "list":
        print(mgr.format_aliases_list())
        return 0
    elif args.alias_action == "add":
        if mgr.add(args.name, args.command):
            print(f"别名已添加: {args.name} -> {args.command}")
            return 0
        else:
            print(f"添加失败: '{args.name}' 与预定义别名冲突")
            return 1
    elif args.alias_action == "remove":
        if mgr.remove(args.name):
            print(f"别名已删除: {args.name}")
            return 0
        else:
            print(f"删除失败: '{args.name}' 不存在或为预定义别名")
            return 1
    else:
        print("请指定操作: list / add / remove")
        return 1


def _cmd_completion(args) -> int:
    """U17：执行completion子命令 —— 生成shell补全脚本"""
    from core.completion import CompletionManager
    mgr = CompletionManager()

    # --list 优先
    if args.list:
        print(mgr.format_list())
        return 0

    # 必须指定shell
    if args.shell is None:
        print("请指定shell类型: bash / zsh")
        print("使用 --list 查看支持的shell")
        return 1

    # 安装模式
    if args.install:
        success, message = mgr.install(args.shell, force=args.force)
        print(message)
        return 0 if success else 1

    # 输出脚本到stdout
    try:
        script = mgr.generate(args.shell)
        print(script)
        return 0
    except ValueError as e:
        print(f"错误: {e}")
        return 1


def _cmd_theme(args) -> int:
    """U18：执行theme子命令 —— 主题管理"""
    from core.themes import get_theme_manager, THEMES, reset_theme_manager
    mgr = get_theme_manager()

    if args.theme_action == "list":
        print(mgr.format_themes_list())
        return 0
    elif args.theme_action == "show":
        print(f"当前主题: {mgr.theme_name}")
        print(f"颜色启用: {'是' if mgr.color_enabled else '否'}")
        print(f"描述: {mgr.theme.description}")
        return 0
    elif args.theme_action == "preview":
        print(mgr.preview())
        return 0
    elif args.theme_action == "set":
        if args.name not in THEMES:
            print(f"未知主题: {args.name}")
            print(f"可用主题: {', '.join(THEMES.keys())}")
            return 1
        # 设置全局主题
        reset_theme_manager()
        from core.themes import ThemeManager
        new_mgr = ThemeManager(theme_name=args.name, color_enabled=True)
        print(f"主题已设置为: {args.name}")
        print(f"描述: {new_mgr.theme.description}")
        # 预览
        print("")
        print(new_mgr.preview())
        return 0
    else:
        print("请指定操作: list / show / preview / set")
        return 1


def _cmd_history(args) -> int:
    """U20：执行history子命令 —— 命令历史管理"""
    from core.history import CommandHistory
    history = CommandHistory()

    if args.history_action == "list":
        print(history.format_list(limit=args.limit))
        return 0
    elif args.history_action == "search":
        results = history.search(args.keyword)
        if not results:
            print(f"未找到包含 '{args.keyword}' 的历史记录")
            return 0
        print(f"=== 搜索结果（{len(results)}条）===")
        for i, entry in enumerate(results, 1):
            time_str = entry.format_time()
            print(f"  {i:3d}. [{time_str}] {entry.command}")
        return 0
    elif args.history_action == "clear":
        history.clear()
        print("历史记录已清空")
        return 0
    else:
        print("请指定操作: list / search / clear")
        return 1


def _cmd_bookmark(args) -> int:
    """U20：执行bookmark子命令 —— 书签管理"""
    from core.history import BookmarkManager
    mgr = BookmarkManager()

    if args.bookmark_action == "list":
        print(mgr.format_list())
        return 0
    elif args.bookmark_action == "add":
        if mgr.add(args.name, args.command, description=args.desc):
            print(f"书签已添加: {args.name} -> {args.command}")
            return 0
        else:
            print(f"添加失败: '{args.name}' 与预定义书签冲突")
            return 1
    elif args.bookmark_action == "remove":
        if mgr.remove(args.name):
            print(f"书签已删除: {args.name}")
            return 0
        else:
            print(f"删除失败: '{args.name}' 不存在或为预定义书签")
            return 1
    elif args.bookmark_action == "run":
        command = mgr.get_command(args.name)
        if command is None:
            print(f"书签不存在: {args.name}")
            return 1
        print(f"执行书签: {args.name} -> {command}")
        # 递归调用cli_main执行书签命令
        return cli_main(command.split())
    else:
        print("请指定操作: list / add / remove / run")
        return 1


def _cmd_help(args) -> int:
    """U21：执行help子命令 —— 上下文感知帮助"""
    from core.context_help import ContextDetector, ContextualHelpProvider
    detector = ContextDetector()
    ctx = detector.detect()
    provider = ContextualHelpProvider()

    if args.context:
        print(provider.format_context_report(ctx))
        return 0
    elif args.troubleshoot:
        print(provider.get_troubleshooting(ctx))
        return 0
    elif args.quick_start:
        print(provider.get_quick_start(ctx))
        return 0
    else:
        print(provider.get_help(ctx))
        return 0


def _cmd_logs(args) -> int:
    """U22：执行logs子命令 —— 日志查看与过滤"""
    from core.log_viewer import LogManager, LogReader, format_log_entries
    from core.themes import reset_theme_manager

    log_dir = "./logs"
    mgr = LogManager(log_dir)

    if args.logs_action == "list":
        print(mgr.format_list(args.type))
        return 0
    elif args.logs_action == "tail":
        # 确定日志文件
        if args.file:
            file_path = args.file
        else:
            file_path = mgr.get_latest_log()
            if file_path is None:
                print("未找到日志文件")
                return 1
        reader = LogReader(file_path)
        if not reader.exists():
            print(f"日志文件不存在: {file_path}")
            return 1
        entries = reader.read_tail(args.lines)
        # 级别过滤
        if args.level:
            entries = reader.filter_by_level(entries, args.level)
        # 着色输出
        color_enabled = not args.no_color
        if color_enabled:
            reset_theme_manager()
        print(f"=== {os.path.basename(file_path)}（最后{len(entries)}行）===")
        print(format_log_entries(entries, color_enabled=color_enabled))
        return 0
    elif args.logs_action == "search":
        # 确定日志文件
        if args.file:
            file_path = args.file
        else:
            file_path = mgr.get_latest_log()
            if file_path is None:
                print("未找到日志文件")
                return 1
        reader = LogReader(file_path)
        if not reader.exists():
            print(f"日志文件不存在: {file_path}")
            return 1
        results = reader.search(args.keyword)
        # 级别过滤
        if args.level:
            results = reader.filter_by_level(results, args.level)
        if not results:
            print(f"未找到包含 '{args.keyword}' 的日志")
            return 0
        color_enabled = not args.no_color
        if color_enabled:
            reset_theme_manager()
        print(f"=== 搜索结果（{len(results)}条，文件: {os.path.basename(file_path)}）===")
        print(format_log_entries(results, color_enabled=color_enabled, show_line_numbers=True))
        return 0
    elif args.logs_action == "clear":
        deleted = mgr.clear_logs(log_type=args.type, keep_latest=args.keep)
        print(f"已清理 {deleted} 个日志文件（保留最近{args.keep}个）")
        return 0
    else:
        print("请指定操作: tail / list / search / clear")
        return 1


def _cmd_alerts(args) -> int:
    """U23：执行alerts子命令 —— 告警管理"""
    from core.alert_cli import AlertCliHelper
    try:
        helper = AlertCliHelper()
    except Exception as e:
        print(f"初始化告警管理器失败: {e}")
        return 1

    if args.alerts_action == "rules":
        print(helper.format_rules_list())
        return 0
    elif args.alerts_action == "channels":
        print(helper.format_channels_list())
        return 0
    elif args.alerts_action == "history":
        print(helper.format_history(args.limit))
        return 0
    elif args.alerts_action == "stats":
        print(helper.format_stats())
        return 0
    elif args.alerts_action == "silence":
        success = helper.silence_rule(args.name, args.duration)
        if success:
            print(f"规则 '{args.name}' 已静默 {args.duration} 秒")
            return 0
        else:
            print(f"静默失败: 规则 '{args.name}' 不存在")
            return 1
    elif args.alerts_action == "test":
        result = helper.test_rule(args.name, args.value)
        print(helper.format_test_result(result))
        return 0 if result.get("exists") else 1
    else:
        print("请指定操作: rules / channels / history / stats / silence / test")
        return 1


def _cmd_perf(args) -> int:
    """U24：执行perf子命令 —— 性能分析"""
    from core.perf_cli import PerfCliHelper
    try:
        helper = PerfCliHelper()
    except Exception as e:
        print(f"初始化性能监控器失败: {e}")
        return 1

    if args.perf_action == "stats":
        print(helper.format_stats(args.name))
        return 0
    elif args.perf_action == "slow":
        print(helper.format_slow(args.threshold))
        return 0
    elif args.perf_action == "summary":
        print(helper.format_summary())
        return 0
    elif args.perf_action == "report":
        print(helper.format_report())
        return 0
    elif args.perf_action == "reset":
        helper.reset()
        print("性能统计已重置")
        return 0
    else:
        print("请指定操作: stats / slow / summary / report / reset")
        return 1


def _cmd_agents(args) -> int:
    """U25：执行agents子命令 —— 多Agent控制台"""
    import json
    from core.agents_cli import AgentsCliHelper
    try:
        helper = AgentsCliHelper()
    except Exception as e:
        print(f"初始化多Agent编排器失败: {e}")
        return 1

    try:
        if args.agents_action == "status":
            print(helper.format_status())
            return 0
        elif args.agents_action == "list":
            print(helper.format_list())
            return 0
        elif args.agents_action == "spawn":
            agent_id = helper.spawn_agent(args.role, args.id)
            print(helper.format_spawn_result(agent_id, args.role))
            return 0
        elif args.agents_action == "spawn-default":
            ids = helper.spawn_default_agents()
            print(f"已创建默认子Agent集群: {ids}")
            return 0
        elif args.agents_action == "delegate":
            try:
                payload = json.loads(args.payload)
            except json.JSONDecodeError as e:
                print(f"payload JSON解析失败: {e}")
                return 1
            result = helper.delegate_task(args.task_type, payload, args.timeout)
            print(helper.format_delegation_result(result))
            return 0 if result.get("success") else 1
        elif args.agents_action == "restart":
            success = helper.restart_agent(args.agent_id)
            if success:
                print(f"Agent '{args.agent_id}' 已重启")
                return 0
            else:
                print(f"重启失败: Agent '{args.agent_id}' 不存在")
                return 1
        elif args.agents_action == "health":
            print(helper.format_health())
            return 0
        elif args.agents_action == "recover":
            recovered = helper.auto_recover()
            print(f"已恢复 {recovered} 个故障Agent")
            return 0
        elif args.agents_action == "stop":
            success = helper.stop_agent(args.agent_id)
            if success:
                print(f"Agent '{args.agent_id}' 已停止")
                return 0
            else:
                print(f"停止失败: Agent '{args.agent_id}' 不存在")
                return 1
        elif args.agents_action == "history":
            print(helper.format_history(args.limit))
            return 0
        else:
            print("请指定操作: status / list / spawn / spawn-default / delegate / restart / health / recover / stop / history")
            return 1
    finally:
        # 清理：停止所有子Agent，避免资源泄漏
        try:
            helper.stop_all()
        except Exception:
            pass


def _cmd_export(args) -> int:
    """U27：执行export子命令 —— 数据导出"""
    from core.export_cli import ExportCliHelper
    try:
        helper = ExportCliHelper()
    except Exception as e:
        print(f"初始化导出器失败: {e}")
        return 1

    # 列出可用数据源
    if args.list:
        print(helper.format_sources_list())
        return 0

    # 检查数据源
    if args.source is None:
        print("请指定数据源，或使用 --list 查看可用数据源")
        print("用法: export <status|health|alerts|perf|snapshot> [--format json|csv] [--output FILE]")
        return 1

    # 导出
    result = helper.export(args.source, args.format, args.output)
    if result is None:
        print(f"导出失败: 不支持的数据源 '{args.source}'")
        return 1

    if args.output:
        print(f"数据已导出到: {result}")
    else:
        print(result)
    return 0


def _cmd_plugins(args) -> int:
    """U28：执行plugins子命令 —— 插件管理"""
    from core.plugins_cli import PluginsCliHelper
    try:
        helper = PluginsCliHelper()
    except Exception as e:
        print(f"初始化插件管理器失败: {e}")
        return 1

    if args.plugins_action == "list":
        print(helper.format_list(args.module_type))
        return 0
    elif args.plugins_action == "info":
        print(helper.format_info(args.module_type, args.name))
        return 0
    elif args.plugins_action == "use":
        print(helper.format_usage(args.module_type, args.name))
        return 0
    elif args.plugins_action == "types":
        print(helper.format_types())
        return 0
    else:
        print("请指定操作: list / info / use / types")
        return 1


def _cmd_skill(args) -> int:
    """U29：执行skill子命令 —— 技能库管理"""
    import json
    from core.skill_cli import SkillCliHelper
    try:
        helper = SkillCliHelper()
    except Exception as e:
        print(f"初始化技能库失败: {e}")
        return 1

    if args.skill_action == "list":
        print(helper.format_list(args.status))
        return 0
    elif args.skill_action == "info":
        print(helper.format_info(args.id))
        return 0
    elif args.skill_action == "search":
        print(helper.format_search(args.task_type))
        return 0
    elif args.skill_action == "evolve":
        steps = [s.strip() for s in args.steps.split(",") if s.strip()]
        params = None
        if args.params:
            try:
                params = json.loads(args.params)
            except json.JSONDecodeError as e:
                print(f"params JSON解析失败: {e}")
                return 1
        result = helper.evolve_skill(args.id, steps, params)
        print(helper.format_evolve_result(result))
        return 0 if result is not None else 1
    elif args.skill_action == "stats":
        print(helper.format_stats())
        return 0
    elif args.skill_action == "clear":
        helper.clear()
        print("技能库已清空")
        return 0
    else:
        print("请指定操作: list / info / search / evolve / stats / clear")
        return 1


def _cmd_long_task(args) -> int:
    """U30：执行long-task子命令 —— 长任务管理"""
    from core.long_task_cli import LongTaskCliHelper
    try:
        helper = LongTaskCliHelper()
    except Exception as e:
        print(f"初始化长任务规划器失败: {e}")
        return 1

    if args.longtask_action == "plan":
        result = helper.plan_task(args.template, args.name, args.desc)
        print(helper.format_plan_result(result))
        return 0
    elif args.longtask_action == "list":
        print(helper.format_list())
        return 0
    elif args.longtask_action == "status":
        print(helper.format_status(args.id))
        return 0
    elif args.longtask_action == "advance":
        result = helper.advance_task(args.id, args.drive)
        print(helper.format_advance_result(result))
        if result.get("error"):
            return 1
        return 0
    elif args.longtask_action == "pause":
        success = helper.pause_task(args.id)
        if success:
            print(f"任务 '{args.id}' 已暂停")
            return 0
        else:
            print(f"暂停失败: 任务 '{args.id}' 不存在或非运行状态")
            return 1
    elif args.longtask_action == "resume":
        success = helper.resume_task(args.id)
        if success:
            print(f"任务 '{args.id}' 已恢复")
            return 0
        else:
            print(f"恢复失败: 任务 '{args.id}' 不存在或非暂停状态")
            return 1
    elif args.longtask_action == "stats":
        print(helper.format_stats())
        return 0
    elif args.longtask_action == "templates":
        print(helper.format_templates())
        return 0
    else:
        print("请指定操作: plan / list / status / advance / pause / resume / stats / templates")
        return 1


def _cmd_workflow(args) -> int:
    """U31：执行workflow子命令 —— 工作流管理"""
    import json
    from core.workflow_cli import WorkflowCliHelper
    try:
        helper = WorkflowCliHelper()
    except Exception as e:
        print(f"初始化工作流引擎失败: {e}")
        return 1

    if args.workflow_action == "submit":
        params = None
        if args.params:
            try:
                params = json.loads(args.params)
            except json.JSONDecodeError as e:
                print(f"params JSON解析失败: {e}")
                return 1
        workflow_id = helper.submit_goal(args.goal_type, args.desc, params)
        print(helper.format_submit_result(workflow_id, args.goal_type))
        return 0
    elif args.workflow_action == "list":
        print(helper.format_list())
        return 0
    elif args.workflow_action == "status":
        print(helper.format_status(args.id))
        return 0
    elif args.workflow_action == "stats":
        print(helper.format_stats())
        return 0
    elif args.workflow_action == "goals":
        print(helper.format_goal_types())
        return 0
    else:
        print("请指定操作: submit / list / status / stats / goals")
        return 1


def _cmd_forever(args) -> int:
    """U32：执行forever子命令 —— 永续运行与自愈控制"""
    from core.forever_cli import ForeverCliHelper
    try:
        helper = ForeverCliHelper()
    except Exception as e:
        print(f"初始化永续运行器失败: {e}")
        return 1

    if args.forever_action == "enable":
        success = helper.enable(args.health_interval, args.cleanup_interval)
        print(helper.format_enable_result(success, args.health_interval, args.cleanup_interval))
        return 0
    elif args.forever_action == "disable":
        success = helper.disable()
        print(helper.format_disable_result(success))
        return 0
    elif args.forever_action == "stats":
        print(helper.format_stats())
        return 0
    elif args.forever_action == "health":
        print(helper.format_health_report())
        return 0
    elif args.forever_action == "history":
        print(helper.format_history(args.limit))
        return 0
    elif args.forever_action == "self-heal":
        report = helper.trigger_self_heal()
        print(helper.format_self_heal_result(report))
        return 0
    elif args.forever_action == "cleanup":
        cleanup_count = helper.trigger_cleanup()
        print(helper.format_cleanup_result(cleanup_count))
        return 0
    elif args.forever_action == "auto-recovery":
        if args.enable:
            helper.set_auto_recovery(True)
            print(helper.format_auto_recovery_set(True))
        elif args.disable:
            helper.set_auto_recovery(False)
            print(helper.format_auto_recovery_set(False))
        else:
            print(helper.format_auto_recovery())
        return 0
    else:
        print("请指定操作: enable / disable / stats / health / history / self-heal / cleanup / auto-recovery")
        return 1


def _cmd_presets(args) -> int:
    """U33：执行presets子命令 —— 配置预设管理"""
    from core.presets_cli import PresetsCliHelper
    try:
        helper = PresetsCliHelper()
    except Exception as e:
        print(f"初始化预设管理器失败: {e}")
        return 1

    if args.presets_action == "list":
        print(helper.format_list())
        return 0
    elif args.presets_action == "show":
        print(helper.format_info(args.name))
        return 0
    elif args.presets_action == "apply":
        success = helper.apply_preset(args.name)
        print(helper.format_apply_result(args.name, success))
        return 0 if success else 1
    else:
        print("请指定操作: list / show / apply")
        return 1


def _cmd_scenarios(args) -> int:
    """U33：执行scenarios子命令 —— 运行场景发现"""
    from core.presets_cli import ScenariosCliHelper
    try:
        helper = ScenariosCliHelper()
    except Exception as e:
        print(f"初始化场景加载器失败: {e}")
        return 1

    if args.scenarios_action == "list":
        print(helper.format_list())
        return 0
    elif args.scenarios_action == "info":
        print(helper.format_info(args.name))
        return 0
    else:
        print("请指定操作: list / info")
        return 1


def _cmd_learn(args) -> int:
    """U34：执行learn子命令 —— 持续学习与元学习"""
    from core.learn_cli import LearnCliHelper
    try:
        helper = LearnCliHelper()
    except Exception as e:
        print(f"初始化学习系统失败: {e}")
        return 1

    if args.learn_action == "init":
        results = helper.init_all(args.input_dim, args.output_dim)
        print(helper.format_init_result(results, args.input_dim, args.output_dim))
        return 0
    elif args.learn_action == "train":
        if args.type == "meta":
            result = helper.train_meta(n_iterations=args.iterations)
        else:
            result = helper.train_continual(task_id=args.task, n_samples=args.samples,
                                            n_steps=args.steps)
        print(helper.format_train_result(
            "元学习" if args.type == "meta" else "持续学习", result))
        return 0 if "error" not in result else 1
    elif args.learn_action == "stats":
        print(helper.format_stats())
        return 0
    elif args.learn_action == "evaluate":
        if args.type == "meta":
            result = helper.evaluate_meta(n_tasks=args.n_tasks)
        else:
            result = helper.evaluate_continual(task_id=args.task, n_samples=args.samples)
        print(helper.format_evaluate_result(
            "元学习" if args.type == "meta" else "持续学习", result))
        return 0 if "error" not in result else 1
    else:
        print("请指定操作: init / train / stats / evaluate")
        return 1


def _cmd_collab(args) -> int:
    """U35：执行collab子命令 —— 多Agent协作"""
    import json
    from core.collab_cli import CollabCliHelper
    try:
        helper = CollabCliHelper()
    except Exception as e:
        print(f"初始化协作中心失败: {e}")
        return 1

    if args.collab_action == "stats":
        print(helper.format_stats())
        return 0
    elif args.collab_action == "share":
        state = {}
        if args.state:
            try:
                state = json.loads(args.state)
            except json.JSONDecodeError as e:
                print(f"state JSON解析失败: {e}")
                return 1
        next_state = {}
        if args.next_state:
            try:
                next_state = json.loads(args.next_state)
            except json.JSONDecodeError as e:
                print(f"next-state JSON解析失败: {e}")
                return 1
        tags = args.tags.split(",") if args.tags else None
        exp = helper.share_experience(
            state=state, action=args.action, reward=args.reward,
            next_state=next_state, td_error=args.td_error, tags=tags,
        )
        print(helper.format_share_result(exp))
        return 0
    elif args.collab_action == "experiences":
        print(helper.format_experiences(args.n))
        return 0
    else:
        print("请指定操作: stats / share / experiences")
        return 1


def _cmd_consensus(args) -> int:
    """U35：执行consensus子命令 —— 共识投票"""
    from core.collab_cli import ConsensusCliHelper
    try:
        helper = ConsensusCliHelper()
    except Exception as e:
        print(f"初始化共识投票器失败: {e}")
        return 1

    if args.consensus_action == "vote":
        helper.vote(args.topic, args.agent, args.proposal, args.confidence)
        print(helper.format_vote_result(args.topic, args.agent, args.proposal))
        return 0
    elif args.consensus_action == "result":
        print(helper.format_consensus_result(args.topic))
        return 0
    elif args.consensus_action == "stats":
        print(helper.format_stats())
        return 0
    else:
        print("请指定操作: vote / result / stats")
        return 1


def cli_main(argv: Optional[List[str]] = None) -> int:
    """CLI主入口

    Args:
        argv: 命令行参数，None则使用sys.argv

    Returns:
        退出码：0成功，非0失败
    """
    if argv is None:
        argv = sys.argv[1:]
    # U16：别名解析（在参数解析前展开）
    if argv:
        try:
            from core.aliases import AliasManager
            alias_mgr = AliasManager()
            argv = alias_mgr.resolve(argv)
        except Exception:
            pass  # 别名解析失败不影响正常流程

    parser = _create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # 无子命令时显示帮助
        parser.print_help()
        return 0

    # U20：记录命令历史（history/bookmark命令自身不记录，避免递归）
    should_record_history = args.command not in ("history", "bookmark")
    if should_record_history:
        try:
            from core.history import CommandHistory
            history = CommandHistory()
            history.add(" ".join(argv))
        except Exception:
            pass

    try:
        if args.command == "run":
            return _cmd_run(args)
        elif args.command == "status":
            return _cmd_status(args)
        elif args.command == "health":
            return _cmd_health(args)
        elif args.command == "snapshot":
            return _cmd_snapshot(args)
        elif args.command == "config":
            return _cmd_config(args)
        elif args.command == "init":
            return _cmd_init(args)
        elif args.command == "doctor":
            return _cmd_doctor(args)
        elif args.command == "shell":
            return _cmd_shell(args)
        elif args.command == "alias":
            return _cmd_alias(args)
        elif args.command == "completion":
            return _cmd_completion(args)
        elif args.command == "theme":
            return _cmd_theme(args)
        elif args.command == "history":
            return _cmd_history(args)
        elif args.command == "bookmark":
            return _cmd_bookmark(args)
        elif args.command == "help":
            return _cmd_help(args)
        elif args.command == "logs":
            return _cmd_logs(args)
        elif args.command == "alerts":
            return _cmd_alerts(args)
        elif args.command == "perf":
            return _cmd_perf(args)
        elif args.command == "agents":
            return _cmd_agents(args)
        elif args.command == "export":
            return _cmd_export(args)
        elif args.command == "plugins":
            return _cmd_plugins(args)
        elif args.command == "skill":
            return _cmd_skill(args)
        elif args.command == "long-task":
            return _cmd_long_task(args)
        elif args.command == "workflow":
            return _cmd_workflow(args)
        elif args.command == "forever":
            return _cmd_forever(args)
        elif args.command == "presets":
            return _cmd_presets(args)
        elif args.command == "scenarios":
            return _cmd_scenarios(args)
        elif args.command == "learn":
            return _cmd_learn(args)
        elif args.command == "collab":
            return _cmd_collab(args)
        elif args.command == "consensus":
            return _cmd_consensus(args)
        else:
            parser.print_help()
            return 0
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except Exception as e:
        logger.error("CLI", f"命令执行失败: {e}", module="cli")
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())
