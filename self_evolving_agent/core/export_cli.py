"""
U27：数据导出 —— 统一数据导出CLI辅助

第一性原理：
- 系统各模块数据分散，用户需要统一导出入口
- 数据导出应支持JSON（结构化）和CSV（表格）两种格式
- 导出应轻量级，不启动完整Agent
- 导出数据应可写入文件或打印到stdout

设计：
1. ExportCliHelper：统一数据导出辅助器
   - 聚合各CLI辅助器的数据源
   - 支持status/health/snapshot/alerts/perf数据源
   - 支持JSON和CSV两种格式
   - 支持输出到文件或stdout
2. 复用既有CLI辅助器，不重复造轮子
"""

import os
import csv
import io
import json
import time
from typing import Dict, List, Optional, Any, Union


class ExportCliHelper:
    """统一数据导出CLI辅助器

    轻量级聚合各数据源，支持JSON/CSV导出
    """

    def __init__(self):
        self._alert_helper = None
        self._perf_helper = None
        self._agents_helper = None
        self._snapshot_helper = None

    @property
    def alert_helper(self):
        if self._alert_helper is None:
            from core.alert_cli import AlertCliHelper
            self._alert_helper = AlertCliHelper()
        return self._alert_helper

    @property
    def perf_helper(self):
        if self._perf_helper is None:
            from core.perf_cli import PerfCliHelper
            self._perf_helper = PerfCliHelper()
        return self._perf_helper

    @property
    def agents_helper(self):
        if self._agents_helper is None:
            from core.agents_cli import AgentsCliHelper
            self._agents_helper = AgentsCliHelper()
        return self._agents_helper

    @property
    def snapshot_helper(self):
        if self._snapshot_helper is None:
            from core.snapshot_cli import SnapshotCliHelper
            self._snapshot_helper = SnapshotCliHelper()
        return self._snapshot_helper

    # ---------- 数据采集 ----------

    def collect_status(self) -> Dict[str, Any]:
        """采集状态数据"""
        return {
            "export_time": time.time(),
            "export_time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "perf_summary": self.perf_helper.get_summary(),
            "agents_status": self.agents_helper.get_cluster_status(),
        }

    def collect_health(self) -> Dict[str, Any]:
        """采集健康数据"""
        return {
            "export_time": time.time(),
            "export_time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agents_health": self.agents_helper.health_check(),
            "perf_slow_operations": self.perf_helper.get_slow_operations(),
            "alert_stats": self.alert_helper.get_stats(),
        }

    def collect_alerts(self) -> Dict[str, Any]:
        """采集告警数据"""
        return {
            "export_time": time.time(),
            "export_time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rules": self.alert_helper.list_rules(),
            "channels": self.alert_helper.list_channels(),
            "history": self.alert_helper.get_history(),
            "stats": self.alert_helper.get_stats(),
        }

    def collect_perf(self) -> Dict[str, Any]:
        """采集性能数据"""
        return {
            "export_time": time.time(),
            "export_time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": self.perf_helper.get_summary(),
            "stats": self.perf_helper.get_stats(),
            "slow_operations": self.perf_helper.get_slow_operations(),
        }

    def collect_snapshot_list(self) -> Dict[str, Any]:
        """采集快照列表数据"""
        snapshots = self.snapshot_helper.list_snapshots()
        snapshot_infos = []
        for i, path in enumerate(snapshots, 1):
            info = {
                "index": i,
                "path": path,
                "filename": os.path.basename(path),
            }
            try:
                info["size_bytes"] = os.path.getsize(path)
            except Exception:
                info["size_bytes"] = 0
            snapshot_infos.append(info)

        return {
            "export_time": time.time(),
            "export_time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "snapshot_count": len(snapshots),
            "snapshots": snapshot_infos,
        }

    # ---------- 格式化 ----------

    def to_json(self, data: Any) -> str:
        """转换为JSON字符串"""
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)

    def to_csv(self, data: Any) -> str:
        """转换为CSV字符串

        支持的数据结构：
        - List[Dict]：直接作为表格行
        - Dict containing List[Dict]：提取第一个List[Dict]
        - Dict：键值对表格
        """
        # 找到适合CSV的数据
        csv_data = self._extract_csv_data(data)
        if csv_data is None:
            return self.to_json(data)  # 回退到JSON

        output = io.StringIO()
        if isinstance(csv_data, list) and csv_data and isinstance(csv_data[0], dict):
            # List[Dict] -> 表格
            writer = csv.DictWriter(output, fieldnames=csv_data[0].keys(),
                                    extrasaction='ignore')
            writer.writeheader()
            for row in csv_data:
                # 展平嵌套结构
                flat_row = {}
                for k, v in row.items():
                    if isinstance(v, (dict, list)):
                        flat_row[k] = json.dumps(v, ensure_ascii=False, default=str)
                    else:
                        flat_row[k] = v
                writer.writerow(flat_row)
        elif isinstance(csv_data, dict):
            # Dict -> 键值对
            writer = csv.writer(output)
            writer.writerow(["key", "value"])
            for k, v in csv_data.items():
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, ensure_ascii=False, default=str)
                writer.writerow([k, v])
        else:
            return self.to_json(data)

        return output.getvalue()

    def _extract_csv_data(self, data: Any) -> Optional[Any]:
        """从数据中提取适合CSV的部分"""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # 优先查找List[Dict]类型的值
            for key in ["rules", "channels", "history", "stats", "snapshots",
                        "slow_operations", "agents", "unhealthy"]:
                if key in data and isinstance(data[key], list) and data[key]:
                    if isinstance(data[key][0], dict):
                        return data[key]
            # 如果没有List[Dict]，返回dict本身
            return data
        return None

    # ---------- 导出 ----------

    def export(self, source: str, fmt: str = "json",
               output_path: Optional[str] = None) -> Optional[str]:
        """导出数据

        Args:
            source: 数据源（status/health/alerts/perf/snapshot）
            fmt: 格式（json/csv）
            output_path: 输出文件路径，None则返回字符串

        Returns:
            output_path为None时返回字符串，否则返回文件路径
        """
        # 采集数据
        data = self._collect(source)
        if data is None:
            return None

        # 格式化
        if fmt.lower() == "csv":
            content = self.to_csv(data)
        else:
            content = self.to_json(data)

        # 输出
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return output_path
            except Exception:
                return None
        return content

    def _collect(self, source: str) -> Optional[Dict[str, Any]]:
        """采集指定数据源"""
        if source == "status":
            return self.collect_status()
        elif source == "health":
            return self.collect_health()
        elif source == "alerts":
            return self.collect_alerts()
        elif source == "perf":
            return self.collect_perf()
        elif source == "snapshot":
            return self.collect_snapshot_list()
        else:
            return None

    def get_available_sources(self) -> List[str]:
        """获取可用数据源列表"""
        return ["status", "health", "alerts", "perf", "snapshot"]

    def format_sources_list(self) -> str:
        """格式化可用数据源列表"""
        sources = self.get_available_sources()
        lines = ["=== 可用导出数据源 ==="]
        source_desc = {
            "status": "系统状态（性能摘要+多Agent集群状态）",
            "health": "健康数据（Agent健康+慢操作+告警统计）",
            "alerts": "告警数据（规则+通道+历史+统计）",
            "perf": "性能数据（摘要+统计+慢操作）",
            "snapshot": "快照列表（文件名+大小+路径）",
        }
        for s in sources:
            desc = source_desc.get(s, "")
            lines.append(f"  {s:<12} {desc}")
        return "\n".join(lines)
