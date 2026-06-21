"""
U8：交互式初始化向导

第一性原理：首次使用应被引导，而非阅读文档。
向导通过问答生成配置文件、创建目录、验证环境，
让用户在30秒内完成首次配置。

设计原则：
1. 交互式：逐步询问，提供合理默认值（方括号内）
2. 非交互式：--non-interactive 使用预设默认值，便于脚本化
3. 幂等：重复运行不破坏现有配置（除非 --force）
4. 可测试：核心逻辑与IO分离，_ask 抽象为可注入的输入函数
"""

import os
import sys
import yaml
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path


# 预设场景模板（与U11对齐）
SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "label": "均衡（推荐）",
        "resource": {"max_cpu_percent": 20.0, "max_memory_percent": 20.0},
        "drive_weights": {"safe": 0.4, "gain": 0.4, "eff": 0.2},
        "main_loop_frequency": 1.0,
    },
    "aggressive": {
        "label": "激进（高性能机器）",
        "resource": {"max_cpu_percent": 50.0, "max_memory_percent": 50.0},
        "drive_weights": {"safe": 0.2, "gain": 0.6, "eff": 0.2},
        "main_loop_frequency": 2.0,
    },
    "conservative": {
        "label": "保守（资源受限）",
        "resource": {"max_cpu_percent": 10.0, "max_memory_percent": 10.0},
        "drive_weights": {"safe": 0.6, "gain": 0.2, "eff": 0.2},
        "main_loop_frequency": 0.5,
    },
    "eco": {
        "label": "节能（嵌入式/边缘）",
        "resource": {"max_cpu_percent": 5.0, "max_memory_percent": 5.0},
        "drive_weights": {"safe": 0.5, "gain": 0.2, "eff": 0.3},
        "main_loop_frequency": 0.2,
    },
}


def _default_input(prompt: str, default: str) -> str:
    """默认输入函数：从stdin读取，回车使用默认值"""
    try:
        raw = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        return default
    return raw if raw else default


def _default_confirm(prompt: str, default: bool) -> bool:
    """默认确认函数：y/n 问答"""
    hint = "Y/n" if default else "y/N"
    try:
        raw = input(f"{prompt} ({hint}): ").strip().lower()
    except EOFError:
        return default
    if not raw:
        return default
    return raw in ("y", "yes", "是")


class InitWizard:
    """交互式初始化向导

    将问答逻辑与IO分离：ask_fn/confirm_fn 可注入，便于单元测试。
    """

    def __init__(
        self,
        config_path: str = "config/agent.yaml",
        project_root: Optional[str] = None,
        ask_fn: Optional[Callable[[str, str], str]] = None,
        confirm_fn: Optional[Callable[[str, bool], bool]] = None,
    ):
        self.project_root = Path(project_root or os.getcwd())
        self.config_path = self.project_root / config_path
        self.ask = ask_fn or _default_input
        self.confirm = confirm_fn or _default_confirm
        self.answers: Dict[str, Any] = {}

    def check_existing_config(self, force: bool = False) -> bool:
        """检查配置是否已存在

        Returns:
            True 表示可以继续，False 表示应中止
        """
        if self.config_path.exists() and not force:
            overwrite = self.confirm(
                f"配置文件已存在: {self.config_path}\n是否覆盖?", False
            )
            if not overwrite:
                print("已取消，保留现有配置。")
                return False
        return True

    def ask_scenario(self) -> str:
        """询问运行场景预设"""
        print("\n--- 选择运行场景 ---")
        keys = list(SCENARIO_PRESETS.keys())
        for i, key in enumerate(keys, 1):
            print(f"  {i}. {key} — {SCENARIO_PRESETS[key]['label']}")
        choice = self.ask("请选择 (1-4)", "1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                scenario = keys[idx]
            else:
                scenario = "balanced"
        except ValueError:
            # 直接输入名称
            scenario = choice if choice in SCENARIO_PRESETS else "balanced"
        self.answers["scenario"] = scenario
        print(f"  已选择: {scenario}")
        return scenario

    def ask_resource_limits(self, scenario: str) -> Dict[str, float]:
        """询问资源约束"""
        print("\n--- 资源约束 ---")
        preset = SCENARIO_PRESETS[scenario]
        default_cpu = str(preset["resource"]["max_cpu_percent"])
        default_mem = str(preset["resource"]["max_memory_percent"])

        cpu_str = self.ask("CPU使用上限 (%)", default_cpu)
        mem_str = self.ask("内存使用上限 (%)", default_mem)
        try:
            cpu = float(cpu_str)
        except ValueError:
            cpu = float(default_cpu)
        try:
            mem = float(mem_str)
        except ValueError:
            mem = float(default_mem)

        self.answers["max_cpu"] = cpu
        self.answers["max_memory"] = mem
        return {"max_cpu_percent": cpu, "max_memory_percent": mem}

    def ask_features(self) -> Dict[str, bool]:
        """询问启用的能力"""
        print("\n--- 能力启用 ---")
        enable_multi = self.confirm("启用多Agent编排?", True)
        enable_forever = self.confirm("启用永续运行（健康监控+自愈）?", True)
        enable_rest = self.confirm("启用REST API?", True)
        enable_dashboard = self.confirm("启用Dashboard?", True)

        self.answers.update({
            "enable_multi_agent": enable_multi,
            "enable_forever": enable_forever,
            "enable_rest": enable_rest,
            "enable_dashboard": enable_dashboard,
        })
        return self.answers

    def ask_ports(self) -> Dict[str, int]:
        """询问端口"""
        if not (self.answers.get("enable_rest") or self.answers.get("enable_dashboard")):
            return {}
        print("\n--- 端口配置 ---")
        ports = {}
        if self.answers.get("enable_rest"):
            rest_port = int(self.ask("REST API端口", "9090"))
            ports["rest_port"] = rest_port
            self.answers["rest_port"] = rest_port
        if self.answers.get("enable_dashboard"):
            dash_port = int(self.ask("Dashboard端口", "8080"))
            ports["dashboard_port"] = dash_port
            self.answers["dashboard_port"] = dash_port
        return ports

    def ask_log_level(self) -> str:
        """询问日志级别"""
        print("\n--- 日志级别 ---")
        print("  1. DEBUG  2. INFO  3. WARNING  4. ERROR")
        choice = self.ask("请选择 (1-4)", "2")
        levels = {"1": "DEBUG", "2": "INFO", "3": "WARNING", "4": "ERROR"}
        level = levels.get(choice, "INFO")
        self.answers["log_level"] = level
        return level

    def build_config_dict(self) -> Dict[str, Any]:
        """根据回答构建配置字典"""
        scenario = self.answers.get("scenario", "balanced")
        preset = SCENARIO_PRESETS[scenario]
        cfg = {
            "resource": {
                "max_cpu_percent": self.answers.get("max_cpu", preset["resource"]["max_cpu_percent"]),
                "max_memory_percent": self.answers.get("max_memory", preset["resource"]["max_memory_percent"]),
                "max_reconstruction_iterations": 3,
                "storage_retain_ratio": 0.30,
            },
            "safety": {
                "max_consecutive_exploration_failures": 5,
                "performance_degradation_threshold": 0.3,
            },
            "evolution": {
                "exploration_ratio": 0.30,
                "idle_replay_interval_hours": 6.0,
            },
            "drive_weights": dict(preset["drive_weights"]),
            "logging": {
                "console_level": self.answers.get("log_level", "INFO"),
                "file_level": "DEBUG",
                "perf_slow_threshold_sec": 1.0,
                "perf_report_interval_loops": 100,
            },
            "main_loop_frequency": preset["main_loop_frequency"],
            "perception_interval": 1.0,
            "decision_interval": 1.0,
        }
        return cfg

    def create_directories(self) -> List[str]:
        """创建必要目录"""
        created = []
        for dir_name in ["logs", "data", "data/snapshots", "config"]:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                created.append(str(dir_path))
        return created

    def write_config(self, config_dict: Dict[str, Any]) -> str:
        """写入配置文件"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write("# 自驱永续进化Agent系统配置 —— 由 init 向导生成\n")
            f.write(f"# 场景预设: {self.answers.get('scenario', 'balanced')}\n")
            f.write("# 可通过环境变量覆盖（如 AGENT_MAX_CPU、AGENT_LOG_LEVEL）\n\n")
            yaml.safe_dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return str(self.config_path)

    def generate_run_command(self) -> str:
        """根据回答生成启动命令"""
        parts = ["python main.py run"]
        if self.answers.get("enable_multi_agent"):
            parts.append("--enable-multi-agent")
        if self.answers.get("enable_forever"):
            parts.append("--enable-forever")
        if self.answers.get("enable_rest"):
            parts.append(f"--rest-port {self.answers.get('rest_port', 9090)}")
        if self.answers.get("enable_dashboard"):
            parts.append(f"--dashboard-port {self.answers.get('dashboard_port', 8080)}")
        parts.append(f"--log-level {self.answers.get('log_level', 'INFO')}")
        return " ".join(parts)

    def print_summary(self, config_path: str, created_dirs: List[str], run_cmd: str) -> None:
        """打印配置摘要与下一步指引"""
        print("\n" + "=" * 50)
        print("  初始化完成！")
        print("=" * 50)
        print(f"\n配置文件: {config_path}")
        if created_dirs:
            print(f"已创建目录: {', '.join(created_dirs)}")
        print(f"\n场景预设: {self.answers.get('scenario', 'balanced')}")
        print(f"资源约束: CPU<{self.answers.get('max_cpu', 20)}% 内存<{self.answers.get('max_memory', 20)}%")
        print(f"日志级别: {self.answers.get('log_level', 'INFO')}")
        print(f"\n启用的能力:")
        print(f"  多Agent编排: {'是' if self.answers.get('enable_multi_agent') else '否'}")
        print(f"  永续运行: {'是' if self.answers.get('enable_forever') else '否'}")
        print(f"  REST API: {'是' if self.answers.get('enable_rest') else '否'}"
              + (f" (端口 {self.answers.get('rest_port', 9090)})" if self.answers.get('enable_rest') else ""))
        print(f"  Dashboard: {'是' if self.answers.get('enable_dashboard') else '否'}"
              + (f" (端口 {self.answers.get('dashboard_port', 8080)})" if self.answers.get('enable_dashboard') else ""))
        print("\n下一步：")
        print(f"  {run_cmd}")
        print("\n或验证配置：")
        print("  python main.py config validate")
        print("\n查看状态：")
        print("  python main.py status")

    def run(self, force: bool = False, non_interactive: bool = False) -> int:
        """执行完整初始化向导

        Args:
            force: 覆盖现有配置
            non_interactive: 非交互模式，使用默认值

        Returns:
            0 成功，非0 失败
        """
        print("=" * 50)
        print("  自驱Agent系统 —— 初始化向导")
        print("=" * 50)

        if not self.check_existing_config(force=force):
            return 0

        if non_interactive:
            # 非交互模式：使用默认值
            self.answers["scenario"] = "balanced"
            preset = SCENARIO_PRESETS["balanced"]
            self.answers["max_cpu"] = preset["resource"]["max_cpu_percent"]
            self.answers["max_memory"] = preset["resource"]["max_memory_percent"]
            self.answers["enable_multi_agent"] = True
            self.answers["enable_forever"] = True
            self.answers["enable_rest"] = True
            self.answers["enable_dashboard"] = True
            self.answers["rest_port"] = 9090
            self.answers["dashboard_port"] = 8080
            self.answers["log_level"] = "INFO"
            print("（非交互模式，使用默认配置）")
        else:
            scenario = self.ask_scenario()
            self.ask_resource_limits(scenario)
            self.ask_features()
            self.ask_ports()
            self.ask_log_level()

        # 构建并写入配置
        config_dict = self.build_config_dict()
        created_dirs = self.create_directories()
        config_path = self.write_config(config_dict)
        run_cmd = self.generate_run_command()

        self.print_summary(config_path, created_dirs, run_cmd)
        return 0
