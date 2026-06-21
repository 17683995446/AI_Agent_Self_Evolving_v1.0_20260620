"""
U9：智能诊断命令

第一性原理：用户不应手动排查环境问题。
`python main.py doctor` 一键体检，检查：
1. 依赖完整性（pyyaml/psutil/numpy）
2. 配置文件有效性
3. 目录权限（logs/data/snapshots可写）
4. 端口可用性（REST/Dashboard端口未被占用）
5. 磁盘空间（数据目录有足够空间）
6. Python版本兼容性
7. 系统资源基线

每项检查返回 CheckResult（status/message/remediation），
汇总为诊断报告，退出码反映整体健康度。
"""

import os
import sys
import socket
import shutil
import platform
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    """单项检查结果"""
    name: str
    status: CheckStatus
    message: str
    remediation: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == CheckStatus.PASS

    @property
    def is_error(self) -> bool:
        return self.status == CheckStatus.FAIL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "remediation": self.remediation,
            "detail": self.detail,
        }


class Doctor:
    """智能诊断器

    将每项检查实现为独立方法，便于测试与扩展。
    检查项可注入依赖（如端口检测函数），便于单元测试。
    """

    def __init__(
        self,
        project_root: Optional[str] = None,
        config_path: str = "config/agent.yaml",
        port_checker: Optional[Callable[[int], bool]] = None,
    ):
        self.project_root = os.path.abspath(project_root or os.getcwd())
        self.config_path = os.path.join(self.project_root, config_path)
        self._port_checker = port_checker or self._default_port_check
        self.results: List[CheckResult] = []

    @staticmethod
    def _default_port_check(port: int) -> bool:
        """默认端口检测：尝试绑定，成功表示端口可用"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("", port))
                return True
        except (OSError, socket.error):
            return False

    # ---------- 各项检查 ----------

    def check_python_version(self) -> CheckResult:
        """检查Python版本兼容性（要求3.8+）"""
        version = sys.version_info
        detail = {"major": version.major, "minor": version.minor, "micro": version.micro}
        if version >= (3, 8):
            return CheckResult(
                name="python_version",
                status=CheckStatus.PASS,
                message=f"Python {version.major}.{version.minor}.{version.micro}",
                detail=detail,
            )
        return CheckResult(
            name="python_version",
            status=CheckStatus.FAIL,
            message=f"Python {version.major}.{version.minor} 版本过低，需要 3.8+",
            remediation="升级 Python 到 3.8 或更高版本",
            detail=detail,
        )

    def check_dependencies(self) -> CheckResult:
        """检查核心依赖是否安装"""
        missing = []
        detail = {}
        for mod_name, pip_name in [("yaml", "pyyaml"), ("psutil", "psutil"), ("numpy", "numpy")]:
            try:
                __import__(mod_name)
                detail[mod_name] = "installed"
            except ImportError:
                missing.append(pip_name)
                detail[mod_name] = "missing"
        if not missing:
            return CheckResult(
                name="dependencies",
                status=CheckStatus.PASS,
                message="所有核心依赖已安装",
                detail=detail,
            )
        return CheckResult(
            name="dependencies",
            status=CheckStatus.FAIL,
            message=f"缺少依赖: {', '.join(missing)}",
            remediation=f"运行: pip install {' '.join(missing)}",
            detail=detail,
        )

    def check_config_file(self) -> CheckResult:
        """检查配置文件是否存在且有效"""
        if not os.path.exists(self.config_path):
            return CheckResult(
                name="config_file",
                status=CheckStatus.WARN,
                message=f"配置文件不存在: {self.config_path}",
                remediation="运行 'python main.py init' 生成配置文件，或使用默认配置",
                detail={"path": self.config_path, "exists": False},
            )
        try:
            import yaml
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return CheckResult(
                    name="config_file",
                    status=CheckStatus.FAIL,
                    message="配置文件顶层不是字典",
                    remediation="检查配置文件格式，顶层应为键值对",
                    detail={"path": self.config_path, "type": type(data).__name__},
                )
            return CheckResult(
                name="config_file",
                status=CheckStatus.PASS,
                message=f"配置文件有效: {self.config_path}",
                detail={"path": self.config_path, "keys": list(data.keys())},
            )
        except Exception as e:
            return CheckResult(
                name="config_file",
                status=CheckStatus.FAIL,
                message=f"配置文件解析失败: {e}",
                remediation="检查YAML语法，或运行 'python main.py config validate'",
                detail={"path": self.config_path, "error": str(e)},
            )

    def check_config_validation(self) -> CheckResult:
        """检查配置是否能通过验证"""
        if not os.path.exists(self.config_path):
            return CheckResult(
                name="config_validation",
                status=CheckStatus.SKIP,
                message="配置文件不存在，跳过验证",
            )
        try:
            from core.config import ConfigManager
            ConfigManager._instance = None
            ConfigManager._initialized = False
            mgr = ConfigManager(config_path=self.config_path)
            mgr.config.validate()
            return CheckResult(
                name="config_validation",
                status=CheckStatus.PASS,
                message="配置验证通过",
            )
        except Exception as e:
            return CheckResult(
                name="config_validation",
                status=CheckStatus.FAIL,
                message=f"配置验证失败: {e}",
                remediation=getattr(e, "remediation", "检查配置项取值范围"),
            )
        finally:
            try:
                from core.config import ConfigManager
                ConfigManager._instance = None
                ConfigManager._initialized = False
            except Exception:
                pass

    def check_directories(self) -> CheckResult:
        """检查必要目录是否存在且可写"""
        dirs_to_check = ["logs", "data", "data/snapshots"]
        issues = []
        detail = {}
        for dir_name in dirs_to_check:
            dir_path = os.path.join(self.project_root, dir_name)
            detail[dir_name] = {"path": dir_path}
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    detail[dir_name]["status"] = "created"
                except OSError as e:
                    issues.append(f"{dir_name}: 无法创建 ({e})")
                    detail[dir_name]["status"] = "create_failed"
                    continue
            # 测试可写性
            test_file = os.path.join(dir_path, ".doctor_write_test")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                detail[dir_name]["writable"] = True
            except OSError as e:
                issues.append(f"{dir_name}: 不可写 ({e})")
                detail[dir_name]["writable"] = False
        if not issues:
            return CheckResult(
                name="directories",
                status=CheckStatus.PASS,
                message="所有目录存在且可写",
                detail=detail,
            )
        return CheckResult(
            name="directories",
            status=CheckStatus.FAIL,
            message="目录问题: " + "; ".join(issues),
            remediation="检查目录权限: chmod 755 logs/ data/ data/snapshots/",
            detail=detail,
        )

    def check_disk_space(self) -> CheckResult:
        """检查磁盘空间（至少100MB可用）"""
        try:
            usage = shutil.disk_usage(self.project_root)
            free_mb = usage.free / (1024 * 1024)
            detail = {
                "free_mb": round(free_mb, 2),
                "total_mb": round(usage.total / (1024 * 1024), 2),
            }
            if free_mb < 100:
                return CheckResult(
                    name="disk_space",
                    status=CheckStatus.WARN,
                    message=f"磁盘空间不足: {free_mb:.1f}MB 可用（建议至少100MB）",
                    remediation="清理磁盘空间或更换数据目录（AGENT_DB_PATH/AGENT_LOG_DIR）",
                    detail=detail,
                )
            return CheckResult(
                name="disk_space",
                status=CheckStatus.PASS,
                message=f"磁盘空间充足: {free_mb:.1f}MB 可用",
                detail=detail,
            )
        except Exception as e:
            return CheckResult(
                name="disk_space",
                status=CheckStatus.SKIP,
                message=f"无法检测磁盘空间: {e}",
            )

    def check_port(self, port: int, name: str) -> CheckResult:
        """检查端口是否可用"""
        if port is None:
            return CheckResult(
                name=f"port_{name}",
                status=CheckStatus.SKIP,
                message=f"{name} 端口未配置，跳过",
            )
        available = self._port_checker(port)
        detail = {"port": port, "name": name, "available": available}
        if available:
            return CheckResult(
                name=f"port_{name}",
                status=CheckStatus.PASS,
                message=f"{name} 端口 {port} 可用",
                detail=detail,
            )
        return CheckResult(
            name=f"port_{name}",
            status=CheckStatus.WARN,
            message=f"{name} 端口 {port} 被占用",
            remediation=f"使用其他端口: --{name.replace('rest', 'rest').replace('dashboard','dashboard')}-port {port+1}",
            detail=detail,
        )

    def check_system_resources(self) -> CheckResult:
        """检查系统资源基线"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory()
            detail = {
                "cpu_percent": cpu_percent,
                "memory_percent": mem.percent,
                "memory_total_gb": round(mem.total / (1024**3), 2),
            }
            warnings = []
            if cpu_percent > 80:
                warnings.append(f"CPU使用率高: {cpu_percent}%")
            if mem.percent > 80:
                warnings.append(f"内存使用率高: {mem.percent}%")
            if not warnings:
                return CheckResult(
                    name="system_resources",
                    status=CheckStatus.PASS,
                    message=f"系统资源正常: CPU {cpu_percent}% 内存 {mem.percent}%",
                    detail=detail,
                )
            return CheckResult(
                name="system_resources",
                status=CheckStatus.WARN,
                message="; ".join(warnings),
                remediation="降低 Agent 资源约束（AGENT_MAX_CPU/AGENT_MAX_MEMORY）",
                detail=detail,
            )
        except ImportError:
            return CheckResult(
                name="system_resources",
                status=CheckStatus.SKIP,
                message="psutil 未安装，跳过系统资源检查",
            )

    # ---------- 汇总 ----------

    def run_all_checks(self, rest_port: int = None, dashboard_port: int = None) -> List[CheckResult]:
        """运行所有检查"""
        self.results = []
        self.results.append(self.check_python_version())
        self.results.append(self.check_dependencies())
        self.results.append(self.check_config_file())
        self.results.append(self.check_config_validation())
        self.results.append(self.check_directories())
        self.results.append(self.check_disk_space())
        if rest_port:
            self.results.append(self.check_port(rest_port, "rest"))
        if dashboard_port:
            self.results.append(self.check_port(dashboard_port, "dashboard"))
        self.results.append(self.check_system_resources())
        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """获取诊断摘要"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == CheckStatus.PASS)
        warnings = sum(1 for r in self.results if r.status == CheckStatus.WARN)
        failed = sum(1 for r in self.results if r.status == CheckStatus.FAIL)
        skipped = sum(1 for r in self.results if r.status == CheckStatus.SKIP)
        if failed > 0:
            overall = "FAIL"
        elif warnings > 0:
            overall = "WARN"
        else:
            overall = "PASS"
        return {
            "overall": overall,
            "total": total,
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "skipped": skipped,
        }

    def format_report(self) -> str:
        """格式化诊断报告为可读文本"""
        lines = []
        lines.append("=" * 60)
        lines.append("  自驱Agent系统 —— 诊断报告")
        lines.append("=" * 60)
        status_icon = {
            CheckStatus.PASS: "[OK]",
            CheckStatus.WARN: "[!!]",
            CheckStatus.FAIL: "[XX]",
            CheckStatus.SKIP: "[--]",
        }
        for r in self.results:
            icon = status_icon[r.status]
            lines.append(f"\n{icon} {r.name}: {r.message}")
            if r.remediation and r.status in (CheckStatus.WARN, CheckStatus.FAIL):
                lines.append(f"    修复建议: {r.remediation}")
        summary = self.get_summary()
        lines.append("\n" + "=" * 60)
        lines.append(f"  总计: {summary['total']}  通过: {summary['passed']}  "
                     f"警告: {summary['warnings']}  失败: {summary['failed']}  "
                     f"跳过: {summary['skipped']}")
        lines.append(f"  整体状态: {summary['overall']}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典（JSON友好）"""
        return {
            "summary": self.get_summary(),
            "checks": [r.to_dict() for r in self.results],
        }
