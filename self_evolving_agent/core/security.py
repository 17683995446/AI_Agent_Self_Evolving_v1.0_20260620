"""
安全加固模块 —— 权限控制、审计日志、配置加密

设计原则（第一性原理）：
1. 最小权限原则：任何操作必须有明确授权
2. 不可抵赖原则：所有敏感操作必须留痕
3. 纵深防御：多层校验，单点突破不导致整体失守
"""

import os
import sys
import time
import hashlib
import hmac
import secrets
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


class PermissionLevel(Enum):
    """权限等级"""
    READ = "read"           # 只读
    EXECUTE = "execute"     # 执行普通动作
    ADMIN = "admin"         # 执行管理动作
    SYSTEM = "system"       # 执行系统级敏感操作


@dataclass
class AuditRecord:
    """审计记录"""
    timestamp: float
    action: str
    actor: str
    permission_level: PermissionLevel
    params: Dict[str, Any]
    result: str
    ip_address: str = ""
    session_id: str = ""
    risk_score: float = 0.0  # 0-1，越高越可疑


class PermissionManager:
    """权限管理器 —— 控制谁可以执行什么操作"""

    def __init__(self):
        # 操作 -> 所需最低权限等级
        self._operation_permissions: Dict[str, PermissionLevel] = {
            "get_state": PermissionLevel.READ,
            "predict": PermissionLevel.READ,
            "get_stats": PermissionLevel.READ,
            "execute_action": PermissionLevel.EXECUTE,
            "optimize": PermissionLevel.EXECUTE,
            "share_experience": PermissionLevel.EXECUTE,
            "reload_config": PermissionLevel.ADMIN,
            "restore_snapshot": PermissionLevel.ADMIN,
            "stop_agent": PermissionLevel.ADMIN,
            "execute_shell": PermissionLevel.SYSTEM,
            "modify_safety_limits": PermissionLevel.SYSTEM,
        }
        # 已授权的token -> 权限等级
        self._tokens: Dict[str, PermissionLevel] = {}
        logger.info("SECURITY", "权限管理器初始化", module="security")

    def register_token(self, token: str, level: PermissionLevel) -> None:
        """注册访问令牌"""
        self._tokens[token] = level
        logger.info("SECURITY", f"令牌注册: {token[:8]}... 权限: {level.value}", module="security")

    def revoke_token(self, token: str) -> None:
        """撤销访问令牌"""
        if token in self._tokens:
            del self._tokens[token]
            logger.info("SECURITY", f"令牌撤销: {token[:8]}...", module="security")

    def check_permission(self, token: str, operation: str) -> bool:
        """检查令牌是否有权限执行操作"""
        if token not in self._tokens:
            logger.warning("SECURITY", f"未授权令牌尝试操作: {operation}", module="security")
            return False

        required = self._operation_permissions.get(operation, PermissionLevel.SYSTEM)
        granted = self._tokens[token]

        # 权限等级比较
        level_order = [PermissionLevel.READ, PermissionLevel.EXECUTE,
                       PermissionLevel.ADMIN, PermissionLevel.SYSTEM]
        if level_order.index(granted) >= level_order.index(required):
            return True

        logger.warning("SECURITY", f"权限不足: {granted.value} < {required.value} for {operation}", module="security")
        return False

    def get_token_level(self, token: str) -> Optional[PermissionLevel]:
        """获取令牌权限等级"""
        return self._tokens.get(token)


class AuditLogger:
    """审计日志 —— 记录所有敏感操作，不可篡改"""

    def __init__(self, log_dir: str = "./data/audit"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._records: List[AuditRecord] = []
        self._chain_hash: str = "0" * 64  # 简单的链式哈希，防止篡改
        logger.info("SECURITY", "审计日志初始化", module="security", extra={"log_dir": log_dir})

    def _compute_record_hash(self, record: AuditRecord) -> str:
        """计算记录哈希（包含前一条哈希，形成链）"""
        content = f"{self._chain_hash}|{record.timestamp}|{record.action}|{record.actor}|{record.result}"
        return hashlib.sha256(content.encode()).hexdigest()

    def log(self, action: str, actor: str, permission_level: PermissionLevel,
            params: Optional[Dict[str, Any]] = None, result: str = "", ip_address: str = "",
            session_id: str = "", risk_score: float = 0.0) -> AuditRecord:
        """记录审计日志"""
        record = AuditRecord(
            timestamp=time.time(),
            action=action,
            actor=actor,
            permission_level=permission_level,
            params=params or {},
            result=result,
            ip_address=ip_address,
            session_id=session_id,
            risk_score=risk_score,
        )

        # 更新链式哈希
        self._chain_hash = self._compute_record_hash(record)
        record._chain_hash = self._chain_hash  # 附加到记录

        self._records.append(record)

        # 写入持久化日志
        self._persist_record(record)

        logger.info("AUDIT", f"审计记录: {action} by {actor}", module="security",
                    extra={"action": action, "actor": actor, "result": result, "risk": risk_score})
        return record

    def _persist_record(self, record: AuditRecord) -> None:
        """持久化单条记录到文件"""
        try:
            date_str = time.strftime("%Y%m%d", time.localtime(record.timestamp))
            log_file = os.path.join(self.log_dir, f"audit_{date_str}.log")
            with open(log_file, 'a') as f:
                f.write(f"{record.timestamp:.6f}|{record.action}|{record.actor}|"
                        f"{record.permission_level.value}|{record.result}|"
                        f"{record.risk_score:.4f}|{getattr(record, '_chain_hash', '')[:16]}\n")
        except Exception as e:
            logger.error("AUDIT", f"审计持久化失败: {e}", module="security")

    def query(self, since: float = 0.0, action_filter: Optional[str] = None,
              actor_filter: Optional[str] = None, min_risk: float = 0.0) -> List[AuditRecord]:
        """查询审计记录"""
        results = []
        for r in self._records:
            if r.timestamp < since:
                continue
            if action_filter and r.action != action_filter:
                continue
            if actor_filter and r.actor != actor_filter:
                continue
            if r.risk_score < min_risk:
                continue
            results.append(r)
        return results

    def verify_integrity(self) -> bool:
        """验证审计链完整性"""
        chain = "0" * 64
        for record in self._records:
            expected = hashlib.sha256(
                f"{chain}|{record.timestamp}|{record.action}|{record.actor}|{record.result}".encode()
            ).hexdigest()
            actual = getattr(record, '_chain_hash', '')
            if actual != expected:
                logger.error("AUDIT", "审计链完整性验证失败", module="security",
                            extra={"expected": expected[:16], "actual": actual[:16]})
                return False
            chain = actual
        return True

    def get_stats(self) -> Dict[str, Any]:
        """获取审计统计"""
        total = len(self._records)
        high_risk = sum(1 for r in self._records if r.risk_score > 0.7)
        return {
            "total_records": total,
            "high_risk_records": high_risk,
            "integrity_verified": self.verify_integrity(),
        }


class SecureConfigStore:
    """安全配置存储 —— 敏感配置加密存储"""

    def __init__(self, key: Optional[str] = None):
        # 使用环境变量或随机生成的密钥
        self._key = (key or os.environ.get("AGENT_CONFIG_KEY", "")).encode()
        if not self._key:
            self._key = secrets.token_bytes(32)
            logger.warning("SECURITY", "使用随机生成的配置密钥（重启后失效）", module="security")
        self._store: Dict[str, str] = {}  # key -> encrypted_value

    def _encrypt(self, plaintext: str) -> str:
        """简单加密（XOR + HMAC）—— 生产环境应使用专业加密库"""
        data = plaintext.encode()
        # 使用密钥派生
        key_stream = hashlib.sha256(self._key).digest()
        encrypted = bytes(b ^ key_stream[i % len(key_stream)] for i, b in enumerate(data))
        # HMAC签名
        sig = hmac.new(self._key, encrypted, hashlib.sha256).hexdigest()[:16]
        return f"{sig}:{encrypted.hex()}"

    def _decrypt(self, ciphertext: str) -> Optional[str]:
        """解密"""
        try:
            sig_hex, data_hex = ciphertext.split(":", 1)
            encrypted = bytes.fromhex(data_hex)
            # 验证签名
            expected_sig = hmac.new(self._key, encrypted, hashlib.sha256).hexdigest()[:16]
            if not hmac.compare_digest(sig_hex, expected_sig):
                logger.error("SECURITY", "配置解密签名验证失败", module="security")
                return None
            key_stream = hashlib.sha256(self._key).digest()
            decrypted = bytes(b ^ key_stream[i % len(key_stream)] for i, b in enumerate(encrypted))
            return decrypted.decode()
        except Exception as e:
            logger.error("SECURITY", f"配置解密失败: {e}", module="security")
            return None

    def set(self, key: str, value: str, encrypt: bool = True) -> None:
        """存储配置值"""
        if encrypt:
            self._store[key] = self._encrypt(value)
        else:
            self._store[key] = f"plain:{value}"

    def get(self, key: str, default: str = "") -> str:
        """获取配置值"""
        if key not in self._store:
            return default
        stored = self._store[key]
        if stored.startswith("plain:"):
            return stored[6:]
        decrypted = self._decrypt(stored)
        return decrypted if decrypted is not None else default

    def delete(self, key: str) -> None:
        """删除配置"""
        if key in self._store:
            del self._store[key]

    def list_keys(self) -> List[str]:
        """列出所有配置键"""
        return list(self._store.keys())


class ActionGuard:
    """动作守卫 —— 敏感操作的二次确认和风险评估"""

    def __init__(self, permission_manager: Optional[PermissionManager] = None,
                 audit_logger: Optional[AuditLogger] = None):
        self.pm = permission_manager or PermissionManager()
        self.audit = audit_logger or AuditLogger()
        # 高风险操作列表
        self._high_risk_actions = {
            "execute_shell", "modify_safety_limits", "stop_agent",
            "delete_snapshot", "restore_snapshot",
        }
        logger.info("SECURITY", "动作守卫初始化", module="security")

    def evaluate_risk(self, action: str, params: Dict[str, Any]) -> float:
        """评估操作风险分数 (0.0 - 1.0)"""
        risk = 0.0

        # 基础风险：操作类型
        if action in self._high_risk_actions:
            risk += 0.5

        # 参数风险
        if "shell" in str(params).lower() or "rm" in str(params).lower():
            risk += 0.3
        if "force" in str(params).lower():
            risk += 0.1
        if "all" in str(params).lower():
            risk += 0.1

        return min(risk, 1.0)

    def check_and_log(self, token: str, action: str, params: Optional[Dict[str, Any]] = None,
                      actor: str = "unknown", require_confirm: bool = False) -> Dict[str, Any]:
        """检查权限、评估风险、记录审计，返回是否允许执行"""
        params = params or {}

        # 1. 权限检查
        if not self.pm.check_permission(token, action):
            self.audit.log(
                action=action, actor=actor, permission_level=PermissionLevel.READ,
                params=params, result="DENIED: permission denied", risk_score=0.3
            )
            return {"allowed": False, "reason": "权限不足", "risk_score": 0.0}

        # 2. 风险评估
        risk = self.evaluate_risk(action, params)

        # 3. 高风险操作需要二次确认
        if risk >= 0.5 and require_confirm:
            self.audit.log(
                action=action, actor=actor,
                permission_level=self.pm.get_token_level(token) or PermissionLevel.READ,
                params=params, result="PENDING: require confirmation", risk_score=risk
            )
            return {"allowed": False, "reason": "需要二次确认", "risk_score": risk, "require_confirm": True}

        # 4. 记录审计（允许执行）
        self.audit.log(
            action=action, actor=actor,
            permission_level=self.pm.get_token_level(token) or PermissionLevel.READ,
            params=params, result="ALLOWED", risk_score=risk
        )

        return {"allowed": True, "risk_score": risk}

    def confirm_execution(self, token: str, action: str, params: Optional[Dict[str, Any]] = None,
                          actor: str = "unknown") -> Dict[str, Any]:
        """二次确认后执行"""
        params = params or {}
        if not self.pm.check_permission(token, action):
            return {"allowed": False, "reason": "权限不足"}

        risk = self.evaluate_risk(action, params)
        self.audit.log(
            action=action, actor=actor,
            permission_level=self.pm.get_token_level(token) or PermissionLevel.READ,
            params=params, result="ALLOWED after confirmation", risk_score=risk
        )
        return {"allowed": True, "risk_score": risk, "confirmed": True}
