"""
U21：上下文感知帮助 —— 基于状态的帮助

第一性原理：
- 帮助信息应根据当前系统状态提供有针对性的建议
- 首次使用应引导初始化
- 配置问题应提供诊断建议
- 错误状态应提供故障排除步骤
- 帮助应是动态的，而非静态的通用文本

设计：
1. HelpContext：描述当前系统状态
2. ContextDetector：检测系统状态
3. ContextualHelpProvider：根据上下文生成帮助
   - 首次使用引导
   - 配置问题诊断
   - 运行状态建议
   - 错误故障排除
4. CLI集成：help子命令增强
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class HelpContext:
    """帮助上下文 —— 描述当前系统状态"""
    is_first_run: bool = False  # 是否首次使用（无配置文件）
    config_exists: bool = False  # 配置文件是否存在
    config_valid: bool = False  # 配置是否有效
    has_backups: bool = False  # 是否有配置备份
    has_history: bool = False  # 是否有命令历史
    has_bookmarks: bool = False  # 是否有用户书签
    agent_running: bool = False  # Agent是否在运行
    has_errors: bool = False  # 是否有错误
    error_type: Optional[str] = None  # 错误类型
    config_version: Optional[int] = None  # 配置版本
    needs_migration: bool = False  # 是否需要迁移

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_first_run": self.is_first_run,
            "config_exists": self.config_exists,
            "config_valid": self.config_valid,
            "has_backups": self.has_backups,
            "has_history": self.has_history,
            "has_bookmarks": self.has_bookmarks,
            "agent_running": self.agent_running,
            "has_errors": self.has_errors,
            "error_type": self.error_type,
            "config_version": self.config_version,
            "needs_migration": self.needs_migration,
        }


class ContextDetector:
    """上下文检测器

    检测当前系统状态，构建HelpContext
    """

    def __init__(self, config_path: str = "config/agent.yaml"):
        self.config_path = config_path

    def detect(self) -> HelpContext:
        """检测系统状态"""
        ctx = HelpContext()

        # 配置文件存在性
        ctx.config_exists = os.path.exists(self.config_path)
        ctx.is_first_run = not ctx.config_exists

        # 配置有效性
        if ctx.config_exists:
            try:
                from core.config import config_manager
                config_manager.load()
                ctx.config_valid = config_manager.config.validate()
            except Exception:
                ctx.config_valid = False
                ctx.has_errors = True
                ctx.error_type = "config_invalid"

        # 配置版本和迁移
        if ctx.config_exists:
            try:
                from core.config_migrator import ConfigMigrator
                migrator = ConfigMigrator()
                ctx.needs_migration = migrator.needs_migration()
            except Exception:
                pass

        # 备份
        try:
            from core.config_backup import ConfigBackupManager
            mgr = ConfigBackupManager(self.config_path)
            backups = mgr.list_backups()
            ctx.has_backups = len(backups) > 0
        except Exception:
            pass

        # 历史命令
        try:
            from core.history import CommandHistory
            history = CommandHistory()
            ctx.has_history = history.count() > 0
        except Exception:
            pass

        # 用户书签
        try:
            from core.history import BookmarkManager
            mgr_bm = BookmarkManager()
            ctx.has_bookmarks = len(mgr_bm.list_user()) > 0
        except Exception:
            pass

        return ctx


class ContextualHelpProvider:
    """上下文感知帮助提供者

    根据HelpContext生成有针对性的帮助内容
    """

    def __init__(self, program_name: str = "python main.py"):
        self.program_name = program_name

    def get_help(self, ctx: HelpContext) -> str:
        """根据上下文生成帮助"""
        sections = []

        # 首次使用引导
        if ctx.is_first_run:
            sections.append(self._first_run_help())
        else:
            # 通用命令参考
            sections.append(self._general_help())

        # 配置问题
        if ctx.config_exists and not ctx.config_valid:
            sections.append(self._config_problem_help())

        # 需要迁移
        if ctx.needs_migration:
            sections.append(self._migration_help())

        # 错误状态
        if ctx.has_errors:
            sections.append(self._error_help(ctx))

        # 个性化建议
        suggestions = self.get_suggestions(ctx)
        if suggestions:
            sections.append(self._format_suggestions(suggestions))

        return "\n\n".join(sections)

    def _first_run_help(self) -> str:
        """首次使用帮助"""
        return f"""=== 欢迎使用自驱永续进化Agent系统 ===

检测到您是首次使用，以下是快速入门指南：

1. 初始化配置（推荐）:
   {self.program_name} init
   交互式向导将引导您完成配置

2. 或使用非交互模式:
   {self.program_name} init --non-interactive

3. 初始化后启动Agent:
   {self.program_name} run

4. 查看运行状态:
   {self.program_name} status

5. 健康检查:
   {self.program_name} doctor

更多帮助:
   {self.program_name} --help              # 查看所有命令
   {self.program_name} <command> --help     # 查看特定命令帮助
   {self.program_name} help                 # 上下文感知帮助"""

    def _general_help(self) -> str:
        """通用帮助"""
        return f"""=== 命令参考 ===

常用命令:
  {self.program_name} run                    # 启动Agent（默认永久运行）
  {self.program_name} status                 # 查看运行状态
  {self.program_name} health                 # 健康检查
  {self.program_name} doctor                 # 一键诊断
  {self.program_name} snapshot save          # 保存快照

配置管理:
  {self.program_name} config show            # 查看配置
  {self.program_name} config edit            # 交互式编辑配置
  {self.program_name} config validate        # 验证配置
  {self.program_name} config backup          # 备份配置

便捷功能:
  {self.program_name} shell                  # 交互式Shell
  {self.program_name} alias list             # 命令别名
  {self.program_name} bookmark list          # 命令书签
  {self.program_name} history list           # 命令历史
  {self.program_name} completion bash        # 生成补全脚本
  {self.program_name} theme list             # 输出主题"""

    def _config_problem_help(self) -> str:
        """配置问题帮助"""
        return f"""=== 配置问题排查 ===

检测到配置可能存在问题，建议:

1. 运行诊断:
   {self.program_name} doctor

2. 验证配置:
   {self.program_name} config validate

3. 检查配置兼容性:
   {self.program_name} config check

4. 如需重新配置:
   {self.program_name} init --force

5. 回滚到之前的配置:
   {self.program_name} config rollback"""

    def _migration_help(self) -> str:
        """迁移帮助"""
        return f"""=== 配置需要迁移 ===

检测到配置文件版本较旧，建议迁移到最新版本:

1. 查看兼容性报告:
   {self.program_name} config check

2. 模拟迁移（不修改文件）:
   {self.program_name} config migrate --dry-run

3. 执行迁移（自动备份）:
   {self.program_name} config migrate"""

    def _error_help(self, ctx: HelpContext) -> str:
        """错误帮助"""
        lines = ["=== 故障排除 ==="]

        if ctx.error_type == "config_invalid":
            lines.append("配置文件无效，可能原因:")
            lines.append("  - YAML语法错误")
            lines.append("  - 字段值超出范围")
            lines.append("  - 缺少必需字段")
            lines.append("")
            lines.append("建议步骤:")
            lines.append(f"  1. 运行诊断: {self.program_name} doctor")
            lines.append(f"  2. 检查配置: {self.program_name} config check")
            lines.append(f"  3. 重新初始化: {self.program_name} init --force")
            lines.append(f"  4. 回滚配置: {self.program_name} config rollback")
        else:
            lines.append("系统检测到异常，建议:")
            lines.append(f"  1. 运行诊断: {self.program_name} doctor")
            lines.append(f"  2. 查看状态: {self.program_name} status")
            lines.append(f"  3. 查看日志: 检查logs目录")

        return "\n".join(lines)

    def get_suggestions(self, ctx: HelpContext) -> List[str]:
        """获取个性化建议"""
        suggestions = []

        if ctx.is_first_run:
            suggestions.append("运行 'init' 开始初始化配置")
        else:
            if not ctx.config_valid:
                suggestions.append("运行 'doctor' 诊断配置问题")
            if ctx.needs_migration:
                suggestions.append("运行 'config migrate' 迁移配置到最新版本")
            if not ctx.has_backups:
                suggestions.append("运行 'config backup' 创建首个配置备份")
            if not ctx.has_history:
                suggestions.append("开始使用系统，命令将被自动记录到历史")
            if not ctx.has_bookmarks:
                suggestions.append("运行 'bookmark list' 查看预定义书签，简化常用操作")
            suggestions.append("运行 'completion bash --install' 安装Tab补全")
            suggestions.append("运行 'theme preview' 预览输出颜色主题")

        return suggestions

    def _format_suggestions(self, suggestions: List[str]) -> str:
        """格式化建议"""
        lines = ["=== 下一步建议 ==="]
        for i, s in enumerate(suggestions, 1):
            lines.append(f"  {i}. {s}")
        return "\n".join(lines)

    def get_troubleshooting(self, ctx: HelpContext) -> str:
        """获取故障排除指南"""
        if not ctx.has_errors and ctx.config_valid:
            return "系统状态正常，无需故障排除。"

        return self._error_help(ctx)

    def get_quick_start(self, ctx: HelpContext) -> str:
        """获取快速入门指南"""
        if ctx.is_first_run:
            return self._first_run_help()
        return self._general_help()

    def format_context_report(self, ctx: HelpContext) -> str:
        """格式化上下文报告"""
        lines = ["=== 系统状态上下文 ==="]
        status_map = {
            "is_first_run": ("首次使用", ctx.is_first_run),
            "config_exists": ("配置文件存在", ctx.config_exists),
            "config_valid": ("配置有效", ctx.config_valid),
            "has_backups": ("有配置备份", ctx.has_backups),
            "has_history": ("有命令历史", ctx.has_history),
            "has_bookmarks": ("有用户书签", ctx.has_bookmarks),
            "needs_migration": ("需要迁移", ctx.needs_migration),
            "has_errors": ("有错误", ctx.has_errors),
        }
        for key, (label, value) in status_map.items():
            marker = "是" if value else "否"
            lines.append(f"  {label}: {marker}")
        if ctx.config_version is not None:
            lines.append(f"  配置版本: {ctx.config_version}")
        if ctx.error_type:
            lines.append(f"  错误类型: {ctx.error_type}")
        return "\n".join(lines)
