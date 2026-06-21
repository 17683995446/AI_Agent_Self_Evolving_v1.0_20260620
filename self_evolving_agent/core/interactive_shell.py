"""
U15：交互式Shell

第一性原理：探索式使用应支持REPL，而非每次敲完整命令。
- 提供持续会话，避免重复输入 `python main.py` 前缀
- 支持所有CLI子命令
- 支持shell元命令（help/exit/history/clear）
- 命令历史持久化（跨会话）
- 可注入IO函数，便于测试

设计原则：
1. REPL是CLI的超集：所有CLI命令可用
2. 元命令不与CLI命令冲突（以.前缀或专用名）
3. 历史持久化到 ~/.self_evolving_agent_history
4. 优雅退出：exit/quit/Ctrl+D
"""

import os
import sys
import readline
import atexit
from typing import Optional, Callable, List, Dict, Any
from pathlib import Path


HISTORY_FILE = Path.home() / ".self_evolving_agent_history"
HISTORY_MAX_LENGTH = 1000


class InteractiveShell:
    """交互式Shell

    将输入循环与IO分离：input_fn 可注入，便于单元测试。
    """

    def __init__(
        self,
        input_fn: Optional[Callable[[str], str]] = None,
        output=None,
        cli_dispatch: Optional[Callable[[List[str]], int]] = None,
        history_file: Optional[str] = None,
    ):
        self.input_fn = input_fn or self._default_input
        self.output = output or sys.stdout
        self.cli_dispatch = cli_dispatch or self._default_cli_dispatch
        self.history_file = history_file or str(HISTORY_FILE)
        self.history: List[str] = []
        self._running = False
        self._welcome_shown = False

    def _default_input(self, prompt: str) -> str:
        try:
            return input(prompt)
        except EOFError:
            return "exit"

    def _default_cli_dispatch(self, args: List[str]) -> int:
        from cli import cli_main
        return cli_main(args)

    def _load_history(self) -> None:
        """加载命令历史"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                self.history = lines[-HISTORY_MAX_LENGTH:]
                # 加载到readline
                for line in self.history:
                    try:
                        readline.add_history(line)
                    except Exception:
                        pass
        except Exception:
            pass

    def _save_history(self) -> None:
        """保存命令历史"""
        try:
            os.makedirs(os.path.dirname(self.history_file) or ".", exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as f:
                for line in self.history[-HISTORY_MAX_LENGTH:]:
                    f.write(line + "\n")
        except Exception:
            pass

    def _write(self, text: str) -> None:
        self.output.write(text)
        self.output.flush()

    def show_welcome(self) -> None:
        """显示欢迎信息"""
        if self._welcome_shown:
            return
        self._write("=" * 50 + "\n")
        self._write("  自驱Agent系统 —— 交互式Shell\n")
        self._write("=" * 50 + "\n")
        self._write("输入命令运行，输入 help 查看帮助，输入 exit 退出\n\n")
        self._welcome_shown = True

    def show_help(self) -> None:
        """显示帮助 —— 动态从CLI解析器同步可用命令"""
        self._write("\n=== Shell帮助 ===\n")
        # 动态获取CLI子命令列表
        cli_commands = self._get_cli_commands()
        if cli_commands:
            self._write("CLI命令（与 python main.py 等价）:\n")
            for cmd_name, cmd_help in cli_commands:
                help_text = cmd_help[:40] if cmd_help else ""
                self._write(f"  {cmd_name:<20} {help_text}\n")
        else:
            self._write("CLI命令（与 python main.py 等价）:\n")
            self._write("  run [选项]          启动Agent\n")
            self._write("  status [--json]     查看状态\n")
            self._write("  health [--detail]   健康检查\n")
        self._write("\nShell元命令:\n")
        self._write("  help                显示此帮助\n")
        self._write("  exit / quit         退出Shell\n")
        self._write("  history             显示命令历史\n")
        self._write("  clear               清屏\n")
        self._write("  !<command>          执行系统命令\n")
        self._write("\n")

    def _get_cli_commands(self) -> List[tuple]:
        """从CLI解析器动态获取子命令列表

        Returns:
            [(command_name, help_text), ...] 列表，获取失败返回空列表
        """
        try:
            from cli import _create_parser
            parser = _create_parser()
            # 从子解析器中提取命令名和帮助文本
            commands = []
            for action in parser._actions:
                if hasattr(action, 'choices') and isinstance(action.choices, dict):
                    for name, subparser in action.choices.items():
                        help_text = subparser.description or ""
                        # 优先使用prog的简短描述
                        if not help_text and subparser.prog:
                            help_text = ""
                        commands.append((name, help_text))
                    break
            return commands
        except Exception:
            return []

    def show_history(self) -> None:
        """显示命令历史"""
        self._write("\n=== 命令历史 ===\n")
        for i, line in enumerate(self.history[-20:], 1):
            self._write(f"  {i:3d}. {line}\n")
        self._write("\n")

    def clear_screen(self) -> None:
        """清屏"""
        self._write("\033[2J\033[H")

    def execute_system_command(self, command: str) -> int:
        """执行系统命令（!前缀）"""
        import subprocess
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.stdout:
                self._write(result.stdout)
            if result.stderr:
                self._write(result.stderr)
            return result.returncode
        except Exception as e:
            self._write(f"执行失败: {e}\n")
            return 1

    def process_command(self, line: str) -> int:
        """处理单条命令

        Returns:
            退出码：0成功，非0失败，-1表示应退出Shell
        """
        line = line.strip()
        if not line:
            return 0
        # 记录历史
        self.history.append(line)
        # 元命令
        lower = line.lower()
        if lower in ("exit", "quit"):
            return -1
        if lower == "help":
            self.show_help()
            return 0
        if lower == "history":
            self.show_history()
            return 0
        if lower == "clear":
            self.clear_screen()
            return 0
        # 系统命令
        if line.startswith("!"):
            return self.execute_system_command(line[1:])
        # CLI命令
        args = line.split()
        try:
            return self.cli_dispatch(args)
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else 0
        except Exception as e:
            self._write(f"命令执行失败: {e}\n")
            return 1

    def run(self) -> int:
        """运行Shell主循环

        Returns:
            退出码
        """
        self._load_history()
        atexit.register(self._save_history)
        self.show_welcome()
        self._running = True
        try:
            while self._running:
                try:
                    line = self.input_fn("agent> ")
                except KeyboardInterrupt:
                    self._write("\n")
                    continue
                except EOFError:
                    self._write("\n")
                    break
                result = self.process_command(line)
                if result == -1:
                    break
        finally:
            self._running = False
            self._save_history()
        self._write("再见！\n")
        return 0
