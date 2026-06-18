"""
外部动作执行器适配器 —— 支持HTTP回调和脚本执行

功能：
1. HTTP回调：发送POST请求到外部系统
2. 脚本执行：执行本地shell脚本
3. 安全限制：超时控制、命令白名单
"""

import os
import json
import time
import subprocess
from typing import Dict, Any, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError
from core.logger import logger


class ExternalActionAdapter:
    """外部动作适配器 —— 将Agent决策转换为外部系统操作"""

    def __init__(self, timeout: float = 10.0, script_whitelist: Optional[list] = None):
        self.timeout = timeout
        self.script_whitelist = script_whitelist or []
        self.execution_history: list = []
        logger.info("ACTION_ADAPTER", "外部动作适配器初始化", module="action_adapter",
                    extra={"timeout": timeout})

    def execute_http(self, url: str, payload: Dict[str, Any],
                     headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """执行HTTP POST回调"""
        logger.info("ACTION_ADAPTER", f"HTTP回调: {url}", module="action_adapter",
                    extra={"url": url, "payload_keys": list(payload.keys())})

        default_headers = {"Content-Type": "application/json"}
        if headers:
            default_headers.update(headers)

        try:
            data = json.dumps(payload).encode("utf-8")
            req = Request(url, data=data, headers=default_headers, method="POST")
            with urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                status = resp.status
            result = {
                "success": 200 <= status < 300,
                "status_code": status,
                "response": body[:500],  # 截断
                "elapsed": 0.0,
            }
            logger.info("ACTION_ADAPTER", f"HTTP回调完成: {status}", module="action_adapter",
                        extra={"status": status})
        except URLError as e:
            result = {
                "success": False,
                "error": f"URL错误: {e.reason}",
                "status_code": 0,
            }
            logger.error("ACTION_ADAPTER", f"HTTP回调失败: {e}", module="action_adapter")
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "status_code": 0,
            }
            logger.error("ACTION_ADAPTER", f"HTTP回调异常: {e}", module="action_adapter")

        self.execution_history.append({"type": "http", "url": url, "result": result, "time": time.time()})
        return result

    def execute_script(self, script_path: str, args: Optional[list] = None,
                       env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """执行本地脚本"""
        logger.info("ACTION_ADAPTER", f"脚本执行: {script_path}", module="action_adapter",
                    extra={"args": args})

        # 安全检查
        if self.script_whitelist and script_path not in self.script_whitelist:
            logger.error("ACTION_ADAPTER", f"脚本不在白名单中: {script_path}", module="action_adapter")
            return {"success": False, "error": "脚本不在白名单中", "returncode": -1}

        if not os.path.exists(script_path):
            logger.error("ACTION_ADAPTER", f"脚本不存在: {script_path}", module="action_adapter")
            return {"success": False, "error": "脚本不存在", "returncode": -1}

        try:
            cmd = [script_path] + (args or [])
            run_env = os.environ.copy()
            if env:
                run_env.update(env)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=run_env,
            )

            output = {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout[:500],
                "stderr": result.stderr[:500],
            }
            logger.info("ACTION_ADAPTER", f"脚本执行完成: rc={result.returncode}", module="action_adapter")
        except subprocess.TimeoutExpired:
            output = {"success": False, "error": "执行超时", "returncode": -1}
            logger.error("ACTION_ADAPTER", "脚本执行超时", module="action_adapter")
        except Exception as e:
            output = {"success": False, "error": str(e), "returncode": -1}
            logger.error("ACTION_ADAPTER", f"脚本执行异常: {e}", module="action_adapter")

        self.execution_history.append({"type": "script", "path": script_path, "result": output, "time": time.time()})
        return output

    def execute_command(self, command: str, shell: bool = False) -> Dict[str, Any]:
        """执行命令（默认禁用shell模式以保障安全）"""
        logger.info("ACTION_ADAPTER", f"命令执行: {command}", module="action_adapter",
                    extra={"shell": shell})

        if shell:
            logger.warning("ACTION_ADAPTER", "shell模式已启用，存在安全风险", module="action_adapter")

        try:
            result = subprocess.run(
                command if shell else command.split(),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                shell=shell,
            )
            output = {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout[:500],
                "stderr": result.stderr[:500],
            }
        except subprocess.TimeoutExpired:
            output = {"success": False, "error": "执行超时", "returncode": -1}
        except Exception as e:
            output = {"success": False, "error": str(e), "returncode": -1}

        self.execution_history.append({"type": "command", "command": command, "result": output, "time": time.time()})
        return output

    def get_history(self) -> list:
        """获取执行历史"""
        return self.execution_history

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self.execution_history)
        success = sum(1 for h in self.execution_history if h["result"].get("success"))
        return {
            "total_executions": total,
            "success_count": success,
            "failure_count": total - success,
            "success_rate": success / total if total > 0 else 0.0,
        }
