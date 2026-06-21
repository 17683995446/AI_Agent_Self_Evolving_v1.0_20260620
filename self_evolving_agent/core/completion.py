"""
U17：智能补全 —— bash/zsh补全脚本生成

第一性原理：
- 用户在终端使用CLI时，应能通过Tab键自动补全子命令和参数
- 减少记忆负担和输入错误
- 补全内容应包括：子命令、子命令的子命令、常用选项、别名

设计：
1. CompletionGenerator：从argparse解析器提取命令结构
2. BashCompletionGenerator：生成bash补全脚本（complete -F）
3. ZshCompletionGenerator：生成zsh补全脚本（#compdef）
4. CompletionManager：管理补全脚本生成和安装
"""

import os
from typing import Dict, List, Optional, Tuple


# 子命令结构定义（与cli.py保持一致）
# 格式: {命令名: (描述, [子命令列表], [常用选项列表])}
COMMAND_STRUCTURE: Dict[str, Tuple[str, List[str], List[str]]] = {
    "run": (
        "启动Agent",
        [],
        ["--duration", "--forever", "--scenario", "--preset",
         "--enable-multi-agent", "--enable-forever",
         "--rest-port", "--dashboard-port",
         "--log-level", "--log-file-level",
         "--progress", "--no-progress", "--progress-interval", "--help"],
    ),
    "status": (
        "查看Agent运行状态",
        [],
        ["--json", "--watch", "--help"],
    ),
    "health": (
        "健康检查",
        [],
        ["--detail", "--help"],
    ),
    "snapshot": (
        "状态快照管理",
        ["save", "list", "restore"],
        ["--id", "--help"],
    ),
    "config": (
        "配置管理",
        ["validate", "show", "reload", "backup", "rollback", "backups", "migrate", "check"],
        ["--reason", "--name", "--dry-run", "--help"],
    ),
    "init": (
        "交互式初始化向导",
        [],
        ["--non-interactive", "--force", "--config-path", "--help"],
    ),
    "doctor": (
        "一键诊断",
        [],
        ["--json", "--rest-port", "--dashboard-port", "--help"],
    ),
    "shell": (
        "交互式Shell",
        [],
        ["--help"],
    ),
    "alias": (
        "命令别名管理",
        ["list", "add", "remove"],
        ["--help"],
    ),
    "completion": (
        "生成shell补全脚本",
        ["bash", "zsh"],
        ["--install", "--list", "--help"],
    ),
}

# 顶层命令名列表
TOP_LEVEL_COMMANDS: List[str] = list(COMMAND_STRUCTURE.keys())


class CompletionGenerator:
    """补全脚本生成器基类"""

    def __init__(self, program_name: str = "self-evolving-agent"):
        self.program_name = program_name

    def get_aliases(self) -> Dict[str, str]:
        """获取用户自定义别名（失败时返回空字典）"""
        try:
            from core.aliases import AliasManager, DEFAULT_ALIASES
            mgr = AliasManager()
            # 合并预定义别名和用户自定义别名
            aliases = dict(DEFAULT_ALIASES)
            aliases.update(mgr.aliases)
            return aliases
        except Exception:
            return {}

    def generate(self) -> str:
        """生成补全脚本（子类实现）"""
        raise NotImplementedError


class BashCompletionGenerator(CompletionGenerator):
    """Bash补全脚本生成器

    生成使用 complete -F 的bash补全脚本
    """

    def generate(self) -> str:
        aliases = self.get_aliases()
        alias_names = sorted(aliases.keys())

        # 构建子命令选项映射
        subcommand_options = {}
        subcommand_subcommands = {}
        for cmd, (_, subs, opts) in COMMAND_STRUCTURE.items():
            if subs:
                subcommand_subcommands[cmd] = " ".join(subs)
            if opts:
                subcommand_options[cmd] = " ".join(opts)

        script = f"""#!/usr/bin/env bash
# {self.program_name} bash补全脚本
# 生成方式: {self.program_name} completion bash
# 安装方式: {self.program_name} completion bash --install
#           或手动: source <({self.program_name} completion bash)

_{self.program_name.replace('-', '_')}_completion() {{
    local cur prev words cword
    _init_completion || return

    local cmds="{ ' '.join(TOP_LEVEL_COMMANDS) }"
    local aliases="{ ' '.join(alias_names) }"

    # 第一级补全：子命令或别名
    if [[ $cword -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "$cmds $aliases" -- "$cur") )
        return 0
    fi

    local cmd="${{words[1]}}"

    # 别名展开后的命令
    case "$cmd" in
"""
        # 为每个别名添加展开提示
        for alias_name, target in sorted(aliases.items()):
            target_cmd = target.split()[0]
            if target_cmd in COMMAND_STRUCTURE:
                script += f"        {alias_name})\n"
                script += f"            cmd=\"{target_cmd}\"\n"
                script += f"            # 别名展开: {alias_name} -> {target}\n"
                script += f"            ;;\n"

        script += "    esac\n\n"

        # 第二级补全：子命令的子命令
        script += "    # 第二级补全：子命令的子命令\n"
        script += "    if [[ $cword -eq 2 ]]; then\n"
        script += "        case \"$cmd\" in\n"
        for cmd, subs_str in subcommand_subcommands.items():
            script += f"            {cmd})\n"
            script += f"                COMPREPLY=( $(compgen -W \"{subs_str}\" -- \"$cur\") )\n"
            script += f"                return 0\n"
            script += f"                ;;\n"
        script += "        esac\n"
        script += "    fi\n\n"

        # 选项补全
        script += "    # 选项补全\n"
        script += "    case \"$cmd\" in\n"
        for cmd, opts_str in subcommand_options.items():
            script += f"        {cmd})\n"
            script += f"            COMPREPLY=( $(compgen -W \"{opts_str}\" -- \"$cur\") )\n"
            script += f"            return 0\n"
            script += f"            ;;\n"
        script += "    esac\n\n"

        # 文件名补全（默认）
        script += "    # 默认：文件名补全\n"
        script += "    _filedir\n"
        script += "    return 0\n"
        script += "}\n\n"

        # 注册补全
        script += f"# 注册补全\n"
        script += f"complete -F _{self.program_name.replace('-', '_')}_completion {self.program_name}\n"
        script += f"complete -F _{self.program_name.replace('-', '_')}_completion agent\n"

        return script


class ZshCompletionGenerator(CompletionGenerator):
    """Zsh补全脚本生成器

    生成使用 #compdef 的zsh补全脚本
    """

    def generate(self) -> str:
        aliases = self.get_aliases()
        alias_names = sorted(aliases.keys())

        # 构建子命令描述
        subcommands_with_desc = []
        for cmd, (desc, _, _) in COMMAND_STRUCTURE.items():
            subcommands_with_desc.append(f"'{cmd}:{desc}'")

        # 构建别名描述
        alias_with_desc = []
        for alias_name in alias_names:
            target = aliases.get(alias_name, "")
            alias_with_desc.append(f"'{alias_name}:别名->{target}'")

        script = f"""#compdef {self.program_name} agent
# {self.program_name} zsh补全脚本
# 生成方式: {self.program_name} completion zsh
# 安装方式: {self.program_name} completion zsh --install
#           或手动: source <({self.program_name} completion zsh)

_{self.program_name.replace('-', '_')}() {{
    local context state line
    typeset -A opt_args

    _arguments -C \\
        '1: :->cmd' \\
        '*::arg:->args'

    case $state in
        cmd)
            local cmds=(
                {' '.join(subcommands_with_desc)}
            )
            local aliases=(
                {' '.join(alias_with_desc)}
            )
            _values 'command' $cmds $aliases
            ;;
        args)
            local cmd="${{line[1]}}"

            # 别名展开
            case "$cmd" in
"""
        # 别名展开
        for alias_name, target in sorted(aliases.items()):
            target_cmd = target.split()[0]
            if target_cmd in COMMAND_STRUCTURE:
                script += f"                {alias_name}) cmd=\"{target_cmd}\" ;;\n"

        script += "            esac\n\n"

        # 子命令补全
        script += "            case \"$cmd\" in\n"

        for cmd, (desc, subs, opts) in COMMAND_STRUCTURE.items():
            if subs:
                script += f"                {cmd})\n"
                sub_descs = []
                for sub in subs:
                    sub_descs.append(f"'{sub}'")
                script += f"                    _values '{desc}' {' '.join(sub_descs)}\n"
                script += f"                    ;;\n"
            else:
                # 直接选项补全
                if opts:
                    opts_str = " ".join(opts)
                    script += f"                {cmd})\n"
                    script += f"                    _arguments '*: :->opts'\n"
                    script += f"                    local opts=({opts_str})\n"
                    script += f"                    _values 'options' $opts\n"
                    script += f"                    ;;\n"

        script += "            esac\n"
        script += "            ;;\n"
        script += "    esac\n\n"

        script += "    return 0\n"
        script += "}\n\n"

        # 注册补全
        script += f"_{self.program_name.replace('-', '_')} \"$@\"\n"

        return script


class CompletionManager:
    """补全脚本管理器

    负责生成、安装、列出补全脚本
    """

    SUPPORTED_SHELLS = ["bash", "zsh"]

    def __init__(self, program_name: str = "self-evolving-agent"):
        self.program_name = program_name
        self._generators = {
            "bash": BashCompletionGenerator(program_name),
            "zsh": ZshCompletionGenerator(program_name),
        }

    def get_generator(self, shell: str) -> Optional[CompletionGenerator]:
        """获取指定shell的生成器"""
        return self._generators.get(shell)

    def generate(self, shell: str) -> str:
        """生成指定shell的补全脚本"""
        gen = self.get_generator(shell)
        if gen is None:
            raise ValueError(f"不支持的shell: {shell}（支持: {', '.join(self.SUPPORTED_SHELLS)}）")
        return gen.generate()

    def list_shells(self) -> List[str]:
        """列出支持的shell类型"""
        return list(self.SUPPORTED_SHELLS)

    def get_install_path(self, shell: str) -> str:
        """获取补全脚本安装路径"""
        if shell == "bash":
            # bash: ~/.local/share/bash-completion/completions/
            base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
            return os.path.join(base, "bash-completion", "completions", self.program_name)
        elif shell == "zsh":
            # zsh: ~/.zsh/completions/ 或 fpath中的目录
            base = os.environ.get("ZSH", os.path.expanduser("~/.zsh"))
            return os.path.join(base, "completions", f"_{self.program_name}")
        else:
            raise ValueError(f"不支持的shell: {shell}")

    def get_rc_file(self, shell: str) -> str:
        """获取shell配置文件路径"""
        if shell == "bash":
            return os.path.expanduser("~/.bashrc")
        elif shell == "zsh":
            return os.path.expanduser("~/.zshrc")
        else:
            raise ValueError(f"不支持的shell: {shell}")

    def install(self, shell: str, force: bool = False) -> Tuple[bool, str]:
        """安装补全脚本

        Args:
            shell: shell类型（bash/zsh）
            force: 是否强制覆盖

        Returns:
            (成功与否, 消息)
        """
        if shell not in self.SUPPORTED_SHELLS:
            return False, f"不支持的shell: {shell}"

        # 生成脚本
        script = self.generate(shell)

        # 安装路径
        install_path = self.get_install_path(shell)
        install_dir = os.path.dirname(install_path)

        # 检查是否已存在
        if os.path.exists(install_path) and not force:
            return False, f"补全脚本已存在: {install_path}（使用 --force 覆盖）"

        # 创建目录
        try:
            os.makedirs(install_dir, exist_ok=True)
        except Exception as e:
            return False, f"创建目录失败: {install_dir} - {e}"

        # 写入文件
        try:
            with open(install_path, "w", encoding="utf-8") as f:
                f.write(script)
            os.chmod(install_path, 0o644)
        except Exception as e:
            return False, f"写入文件失败: {install_path} - {e}"

        # 生成提示信息
        rc_file = self.get_rc_file(shell)
        if shell == "bash":
            hint = (
                f"已安装bash补全脚本到: {install_path}\n"
                f"请确保在 {rc_file} 中加载bash-completion，或手动source:\n"
                f"  source {install_path}\n"
                f"然后重启shell或执行: source {rc_file}"
            )
        else:  # zsh
            hint = (
                f"已安装zsh补全脚本到: {install_path}\n"
                f"请确保 {install_dir} 在fpath中，可在 {rc_file} 中添加:\n"
                f"  fpath=({install_dir} $fpath)\n"
                f"然后重启shell或执行: source {rc_file}\n"
                f"并执行: rm -f ~/.zcompdump* && compinit"
            )

        return True, hint

    def format_list(self) -> str:
        """格式化支持的shell列表"""
        lines = ["=== 支持的Shell补全 ==="]
        for shell in self.SUPPORTED_SHELLS:
            install_path = self.get_install_path(shell)
            exists = "已安装" if os.path.exists(install_path) else "未安装"
            lines.append(f"  {shell:8s} [{exists}]  安装路径: {install_path}")
        lines.append("")
        lines.append("使用方式:")
        lines.append(f"  {self.program_name} completion bash          # 输出bash补全脚本")
        lines.append(f"  {self.program_name} completion zsh           # 输出zsh补全脚本")
        lines.append(f"  {self.program_name} completion bash --install  # 安装bash补全")
        lines.append(f"  {self.program_name} completion zsh --install   # 安装zsh补全")
        return "\n".join(lines)
