"""
RESTful API服务器 —— 暴露Agent所有模块接口

端点：
- GET  /api/v1/state          -> 当前状态
- GET  /api/v1/prediction     -> 预测结果
- GET  /api/v1/drives         -> 驱力状态
- POST /api/v1/action         -> 执行动作
- GET  /api/v1/memory/stats   -> 记忆统计
- GET  /api/v1/config         -> 当前配置
- POST /api/v1/config/reload  -> 重载配置
- GET  /api/v1/health         -> 健康检查
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger
from core.security import PermissionManager, PermissionLevel, ActionGuard

# 全局Agent引用（由外部设置）
_agent_instance: Optional[Any] = None
_permission_manager = PermissionManager()


def set_agent(agent):
    """设置全局Agent实例"""
    global _agent_instance
    _agent_instance = agent


def set_default_tokens():
    """设置默认访问令牌（仅用于测试）"""
    _permission_manager.register_token("read_token", PermissionLevel.READ)
    _permission_manager.register_token("execute_token", PermissionLevel.EXECUTE)
    _permission_manager.register_token("admin_token", PermissionLevel.ADMIN)


class RESTHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())

    def _get_token(self) -> str:
        """从请求头获取token"""
        return self.headers.get("X-API-Token", "")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        token = self._get_token()

        if path == "/api/v1/health":
            self._send_json({"status": "ok", "timestamp": time.time()})
            return

        if not _permission_manager.check_permission(token, "get_state"):
            self._send_json({"error": "Unauthorized"}, 401)
            return

        if path == "/api/v1/state":
            self._handle_get_state()
        elif path == "/api/v1/prediction":
            self._handle_get_prediction()
        elif path == "/api/v1/drives":
            self._handle_get_drives()
        elif path == "/api/v1/memory/stats":
            self._handle_get_memory_stats()
        elif path == "/api/v1/config":
            self._handle_get_config()
        elif path == "/api/v1/modules":
            self._handle_get_modules()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        token = self._get_token()

        # 读取请求体
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode() if content_length > 0 else "{}"
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        if path == "/api/v1/action":
            self._handle_post_action(token, payload)
        elif path == "/api/v1/config/reload":
            self._handle_reload_config(token, payload)
        else:
            self._send_json({"error": "Not found"}, 404)

    def _handle_get_state(self):
        if _agent_instance is None:
            self._send_json({"error": "Agent not initialized"}, 503)
            return
        try:
            state = _agent_instance.perception.get_state_vector()
            self._send_json({
                "status": "ok",
                "state": state,
                "loop_count": _agent_instance.loop_count,
                "timestamp": time.time(),
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_get_prediction(self):
        if _agent_instance is None:
            self._send_json({"error": "Agent not initialized"}, 503)
            return
        try:
            pred = _agent_instance.prediction.predict()
            self._send_json({
                "status": "ok",
                "prediction": {
                    "values": pred.predicted_values,
                    "confidence": pred.confidence,
                    "layer": pred.layer,
                    "trend": pred.trend_direction,
                },
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_get_drives(self):
        if _agent_instance is None:
            self._send_json({"error": "Agent not initialized"}, 503)
            return
        try:
            state = _agent_instance.perception.get_state_vector()
            drives = _agent_instance.drive_engine.compute(state, 0, 0, 0.0)
            self._send_json({
                "status": "ok",
                "drives": drives.to_dict(),
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_get_memory_stats(self):
        if _agent_instance is None:
            self._send_json({"error": "Agent not initialized"}, 503)
            return
        try:
            stats = _agent_instance.memory.get_stats()
            self._send_json({"status": "ok", "memory": stats})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_get_config(self):
        from core.config import config_manager
        try:
            self._send_json({
                "status": "ok",
                "config": config_manager.config.to_dict() if hasattr(config_manager.config, 'to_dict') else {},
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_get_modules(self):
        if _agent_instance is None:
            self._send_json({"error": "Agent not initialized"}, 503)
            return
        try:
            modules = {
                "perception": type(_agent_instance.perception).__name__,
                "prediction": type(_agent_instance.prediction).__name__,
                "drive_engine": type(_agent_instance.drive_engine).__name__,
                "decision": type(_agent_instance.decision).__name__,
                "error_eval": type(_agent_instance.error_eval).__name__,
                "optimizer": type(_agent_instance.optimizer).__name__,
                "memory": type(_agent_instance.memory).__name__,
            }
            self._send_json({"status": "ok", "modules": modules})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_post_action(self, token: str, payload: Dict[str, Any]):
        if not _permission_manager.check_permission(token, "execute_action"):
            self._send_json({"error": "Permission denied"}, 403)
            return
        if _agent_instance is None:
            self._send_json({"error": {"code": "AGENT_NOT_INITIALIZED",
                                       "message": "Agent未初始化",
                                       "remediation": "通过set_agent()注入Agent实例"}}, 503)
            return
        try:
            action_name = payload.get("action", "sleep")
            # 简单映射动作名称
            from modules.decision import ActionType, Action
            action_map = {
                "sleep": ActionType.SLEEP,
                "collect": ActionType.COLLECT_DATA,
                "reduce_freq": ActionType.REDUCE_COLLECT_FREQ,
                "release": ActionType.RELEASE_RESOURCES,
                "learn": ActionType.LEARN,
            }
            # U2：未知动作返回400而非静默映射为SLEEP
            if action_name not in action_map:
                self._send_json({"error": {"code": "UNKNOWN_ACTION",
                                           "message": f"未知动作: {action_name}",
                                           "remediation": f"合法动作: {list(action_map.keys())}"}}, 400)
                return
            atype = action_map[action_name]
            action = Action(atype, action_name, "API调用")
            result = _agent_instance.decision.executor.execute(action)
            self._send_json({
                "status": "ok",
                "result": {
                    "success": result.success,
                    "message": result.message,
                    "elapsed": result.elapsed_time,
                },
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_reload_config(self, token: str, payload: Dict[str, Any]):
        if not _permission_manager.check_permission(token, "reload_config"):
            self._send_json({"error": {"code": "PERMISSION_DENIED",
                                       "message": "权限不足",
                                       "remediation": "使用具有reload_config权限的令牌"}}, 403)
            return
        try:
            from core.config import config_manager
            # U2：修复reload断裂 —— ConfigManager没有reload()方法，使用load()
            config_manager.load()
            self._send_json({"status": "ok", "message": "配置已重载"})
        except Exception as e:
            self._send_json({"error": {"code": "CONFIG_RELOAD_FAILED",
                                       "message": str(e),
                                       "remediation": "检查config/agent.yaml格式是否正确"}}, 500)


def start_rest_server(port: int = 9090, block: bool = True):
    """启动REST API服务器

    U2：支持非阻塞模式，便于main.py集成

    Args:
        port: 监听端口
        block: 是否阻塞主线程。False时在后台线程启动
    """
    set_default_tokens()
    server = HTTPServer(("0.0.0.0", port), RESTHandler)
    logger.info("REST_API", f"REST API启动于 http://0.0.0.0:{port}", module="rest_api")

    if block:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
    else:
        import threading
        thread = threading.Thread(target=server.serve_forever, name="RESTServer", daemon=True)
        thread.start()
        logger.info("REST_API", "REST API在后台线程运行", module="rest_api")
        return server

    return server


if __name__ == "__main__":
    start_rest_server()
