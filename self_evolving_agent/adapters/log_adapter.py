"""
日志适配器 —— 接入真实日志文件进行异常检测

功能：
1. 实时监控日志文件变化
2. 提取错误/警告模式
3. 计算日志熵（异常度指标）
"""

import os
import re
import math
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple
from core.logger import logger


class LogFileAdapter:
    """日志文件适配器 —— 将外部日志文件转换为系统可理解的指标"""

    def __init__(self, log_path: str, error_patterns: Optional[List[str]] = None):
        self.log_path = log_path
        self.error_patterns = error_patterns or [
            r"ERROR",
            r"FATAL",
            r"CRITICAL",
            r"Exception",
            r"Traceback",
        ]
        self.warning_patterns = [
            r"WARNING",
            r"WARN",
        ]
        self._compiled_errors = [re.compile(p, re.IGNORECASE) for p in self.error_patterns]
        self._compiled_warnings = [re.compile(p, re.IGNORECASE) for p in self.warning_patterns]
        self._last_position = 0
        self._last_check_time = 0.0
        logger.info("LOG_ADAPTER", f"日志适配器初始化: {log_path}", module="log_adapter")

    def _read_new_lines(self) -> List[str]:
        """读取自上次检查以来的新日志行"""
        if not os.path.exists(self.log_path):
            return []
        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self._last_position)
                lines = f.readlines()
                self._last_position = f.tell()
                return lines
        except Exception as e:
            logger.warning("LOG_ADAPTER", f"读取日志失败: {e}", module="log_adapter")
            return []

    def analyze(self) -> Tuple[float, float, List[str]]:
        """分析日志，返回 (错误率, 日志熵, 新错误列表)"""
        lines = self._read_new_lines()
        if not lines:
            return 0.0, 0.0, []

        total = len(lines)
        errors = []
        warnings = 0

        for line in lines:
            is_error = any(p.search(line) for p in self._compiled_errors)
            is_warning = any(p.search(line) for p in self._compiled_warnings)
            if is_error:
                errors.append(line.strip()[:200])  # 截断避免过长
            elif is_warning:
                warnings += 1

        error_rate = len(errors) / total if total > 0 else 0.0
        warning_rate = warnings / total if total > 0 else 0.0

        # 计算日志熵（基于日志行的模式分布）
        entropy = self._compute_entropy(lines)

        logger.debug("LOG_ADAPTER", f"日志分析完成", module="log_adapter",
                    extra={"total": total, "errors": len(errors), "warnings": warnings,
                           "error_rate": error_rate, "entropy": entropy})

        return error_rate, entropy, errors

    def _compute_entropy(self, lines: List[str]) -> float:
        """计算日志行的香农熵 —— 熵越高表示日志模式越混乱（可能异常）"""
        if not lines:
            return 0.0

        # 简化：按日志行首单词分类
        categories = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                # 提取日志级别作为类别
                cat = parts[0] if len(parts) > 0 else "UNKNOWN"
                categories.append(cat)
            else:
                categories.append("EMPTY")

        counter = Counter(categories)
        total = len(categories)
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def reset(self) -> None:
        """重置读取位置到文件开头"""
        self._last_position = 0
        logger.debug("LOG_ADAPTER", "日志读取位置重置", module="log_adapter")

    def get_stats(self) -> Dict[str, any]:
        """获取适配器统计信息"""
        return {
            "log_path": self.log_path,
            "last_position": self._last_position,
            "error_patterns": self.error_patterns,
        }
