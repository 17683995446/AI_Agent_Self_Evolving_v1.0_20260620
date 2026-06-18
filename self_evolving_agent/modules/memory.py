"""
记忆系统模块 —— 实现短期记忆、长期记忆、工作记忆
支持记忆编码、检索、巩固、遗忘等操作
"""
import os
import sys
import time
import json
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import deque
from datetime import datetime

# 仅在直接运行时添加路径（避免副作用）
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger, log_execution
from core.config import config_manager
from core.interfaces import IMemory


@dataclass
class MemoryItem:
    """记忆条目数据类"""
    content: Any
    timestamp: float
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.last_accessed == 0.0:
            self.last_accessed = self.timestamp
        logger.debug("MEMORY_ITEM_INIT", "MemoryItem 初始化", module="memory",
                    extra={"importance": self.importance, "tags": self.tags})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @log_execution
    def access(self) -> None:
        """访问记忆条目，更新访问计数和时间"""
        logger.debug("MEMORY_ITEM_ACCESS", "访问记忆条目", module="memory",
                    extra={"old_access_count": self.access_count})
        self.access_count += 1
        self.last_accessed = time.time()
        logger.info("MEMORY_ITEM_ACCESS", "记忆条目访问完成", module="memory",
                    extra={"new_access_count": self.access_count, "last_accessed": self.last_accessed})


# ============================================================
# 兼容旧测试的辅助类
# ============================================================

@dataclass
class Experience:
    """经验条目 —— 兼容旧测试"""
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    td_error: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "timestamp": self.timestamp,
            "td_error": self.td_error,
        }


class PrioritizedReplay:
    """优先经验回放 —— 兼容旧测试"""

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
        self.alpha = 0.6
        self.beta = 0.4

    def add(self, priority: float, experience: Experience) -> None:
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> List[Experience]:
        if len(self.buffer) == 0:
            return []
        probs = np.array(self.priorities) ** self.alpha
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False, p=probs)
        return [self.buffer[i] for i in indices]

    def update_priority(self, index: int, priority: float) -> None:
        if 0 <= index < len(self.priorities):
            self.priorities[index] = priority

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority

    def size(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()
        self.priorities.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class ShortTermMemory:
    """短期记忆 —— 容量有限，存储最近的经验"""

    def __init__(self, capacity: int = 100, max_size: Optional[int] = None) -> None:
        if max_size is not None:
            capacity = max_size
        logger.info("STM_INIT", "ShortTermMemory 初始化", module="memory", extra={"capacity": capacity})
        self.capacity = capacity
        self.items: deque = deque(maxlen=capacity)
        logger.debug("STM_INIT", "短期记忆缓冲区创建完成", module="memory")

    @log_execution
    def store(self, content: Any, importance: float = 0.5, tags: Optional[List[str]] = None) -> str:
        """存储新记忆"""
        logger.debug("STM_STORE", "存储短期记忆", module="memory",
                    extra={"content_type": type(content).__name__, "importance": importance})
        item = MemoryItem(
            content=content,
            timestamp=time.time(),
            importance=importance,
            tags=tags or [],
        )
        self.items.append(item)
        # 使用可哈希的标识：优先使用id+timestamp，避免不可序列化对象导致异常
        try:
            content_str = str(content)
        except Exception:
            content_str = str(id(content))
        item_id = hashlib.md5(f"{content_str}{item.timestamp}".encode()).hexdigest()[:8]
        logger.info("STM_STORE", "短期记忆存储完成", module="memory",
                    extra={"item_id": item_id, "current_size": len(self.items)})
        return item_id

    # 兼容旧测试的方法
    def add(self, experience: Experience) -> None:
        self.items.append(MemoryItem(
            content=experience,
            timestamp=experience.timestamp,
            importance=abs(experience.reward) + 0.5,
            tags=["experience"],
        ))

    def get_recent(self, n: int = 10) -> List[Experience]:
        recent = list(self.items)[-n:]
        return [item.content for item in recent if isinstance(item.content, Experience)]

    def get_all(self) -> List[Experience]:
        return [item.content for item in self.items if isinstance(item.content, Experience)]

    def __len__(self) -> int:
        return len(self.items)

    @log_execution
    def retrieve(self, n: int = 10) -> List[MemoryItem]:
        """检索最近N条记忆"""
        logger.debug("STM_RETRIEVE", f"检索最近 {n} 条短期记忆", module="memory")
        recent = list(self.items)[-n:]
        for item in recent:
            item.access()
        logger.info("STM_RETRIEVE", "短期记忆检索完成", module="memory", extra={"retrieved": len(recent)})
        return recent

    @log_execution
    def retrieve_by_tag(self, tag: str) -> List[MemoryItem]:
        """按标签检索记忆"""
        logger.debug("STM_RETRIEVE_TAG", f"按标签检索短期记忆: {tag}", module="memory")
        matched = [item for item in self.items if tag in item.tags]
        for item in matched:
            item.access()
        logger.info("STM_RETRIEVE_TAG", "标签检索完成", module="memory",
                    extra={"tag": tag, "matched": len(matched)})
        return matched

    @log_execution
    def clear(self) -> None:
        """清空短期记忆"""
        logger.debug("STM_CLEAR", "清空短期记忆", module="memory")
        self.items.clear()
        logger.info("STM_CLEAR", "短期记忆已清空", module="memory")

    @log_execution
    def get_size(self) -> int:
        logger.debug("STM_SIZE", "获取短期记忆大小", module="memory")
        size = len(self.items)
        logger.info("STM_SIZE", f"短期记忆大小: {size}", module="memory")
        return size


class LongTermMemory:
    """长期记忆 —— 容量较大，存储重要的、经过巩固的记忆"""

    def __init__(self, capacity: int = 1000, db_path: Optional[str] = None) -> None:
        logger.info("LTM_INIT", "LongTermMemory 初始化", module="memory", extra={"capacity": capacity})
        self.capacity = capacity
        self.items: Dict[str, MemoryItem] = {}
        self.index: Dict[str, Set[str]] = {}  # 标签索引
        self._next_id = 1
        self._db_path = db_path
        logger.debug("LTM_INIT", "长期记忆存储和索引初始化完成", module="memory")

    @log_execution
    def store(self, item_id: str, item: MemoryItem) -> bool:
        """存储长期记忆"""
        logger.debug("LTM_STORE", f"存储长期记忆: {item_id}", module="memory",
                    extra={"current_size": len(self.items)})
        if len(self.items) >= self.capacity:
            logger.warning("LTM_STORE", "长期记忆容量已满，执行遗忘", module="memory")
            self._forget_least_important()
        self.items[item_id] = item
        # 更新索引
        for tag in item.tags:
            if tag not in self.index:
                self.index[tag] = set()
            self.index[tag].add(item_id)
            logger.debug("LTM_STORE", f"更新标签索引: {tag}", module="memory")
        logger.info("LTM_STORE", "长期记忆存储完成", module="memory",
                    extra={"item_id": item_id, "current_size": len(self.items)})
        return True

    # 兼容旧测试的方法
    def add(self, experience: Experience) -> int:
        item_id = str(self._next_id)
        self._next_id += 1
        self.store(item_id, MemoryItem(
            content=experience,
            timestamp=experience.timestamp,
            importance=abs(experience.reward) + 0.5,
            tags=["experience", experience.action],
        ))
        return int(item_id)

    def get_count(self) -> int:
        return len(self.items)

    def query_by_action(self, action: str) -> List[Experience]:
        results = []
        for item in self.items.values():
            if isinstance(item.content, Experience) and item.content.action == action:
                results.append(item.content)
        return results

    def update_td_error(self, row_id: int, td_error: float) -> None:
        item = self.items.get(str(row_id))
        if item and isinstance(item.content, Experience):
            item.content.td_error = td_error

    @log_execution
    def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """根据ID检索记忆"""
        logger.debug("LTM_RETRIEVE", f"检索长期记忆: {item_id}", module="memory")
        item = self.items.get(item_id)
        if item:
            item.access()
            logger.info("LTM_RETRIEVE", "长期记忆检索成功", module="memory", extra={"item_id": item_id})
        else:
            logger.warning("LTM_RETRIEVE", "长期记忆未找到", module="memory", extra={"item_id": item_id})
        return item

    @log_execution
    def retrieve_by_tag(self, tag: str) -> List[MemoryItem]:
        """按标签检索记忆"""
        logger.debug("LTM_RETRIEVE_TAG", f"按标签检索长期记忆: {tag}", module="memory")
        item_ids = self.index.get(tag, set())
        matched = []
        for item_id in item_ids:
            item = self.items.get(item_id)
            if item:
                item.access()
                matched.append(item)
        logger.info("LTM_RETRIEVE_TAG", "标签检索完成", module="memory",
                    extra={"tag": tag, "matched": len(matched)})
        return matched

    @log_execution
    def retrieve_similar(self, content: Any, top_k: int = 5) -> List[Tuple[str, MemoryItem, float]]:
        """基于内容相似度检索记忆（简化版：使用标签匹配）"""
        logger.debug("LTM_RETRIEVE_SIM", "基于相似度检索长期记忆", module="memory",
                    extra={"content_type": type(content).__name__, "top_k": top_k})
        if isinstance(content, dict) and "tags" in content:
            query_tags = set(content["tags"])
        elif isinstance(content, str):
            query_tags = {content}
        else:
            query_tags = set()
        scores = []
        for item_id, item in self.items.items():
            item_tags = set(item.tags)
            if item_tags:
                similarity = len(query_tags & item_tags) / len(query_tags | item_tags)
            else:
                similarity = 0.0
            scores.append((item_id, item, similarity))
        scores.sort(key=lambda x: x[2], reverse=True)
        top = scores[:top_k]
        for _, item, _ in top:
            item.access()
        logger.info("LTM_RETRIEVE_SIM", "相似度检索完成", module="memory", extra={"top_k": len(top)})
        return top

    @log_execution
    def _forget_least_important(self) -> None:
        """遗忘最不重要的记忆"""
        logger.debug("LTM_FORGET", "开始遗忘最不重要的记忆", module="memory",
                    extra={"current_size": len(self.items)})
        if not self.items:
            logger.warning("LTM_FORGET", "无记忆可遗忘", module="memory")
            return
        # 计算重要性分数（考虑重要性、访问次数、最近访问时间）
        current_time = time.time()
        scores = []
        for item_id, item in self.items.items():
            recency = 1.0 / (1.0 + (current_time - item.last_accessed) / 3600.0)
            score = item.importance * 0.4 + (item.access_count / 100.0) * 0.3 + recency * 0.3
            scores.append((item_id, score))
            logger.debug("LTM_FORGET", f"记忆 {item_id} 重要性评分: {score:.4f}", module="memory")
        scores.sort(key=lambda x: x[1])
        # 遗忘最低分的10%
        n_forget = max(1, len(scores) // 10)
        for item_id, score in scores[:n_forget]:
            del self.items[item_id]
            # 清理索引
            for tag, ids in self.index.items():
                ids.discard(item_id)
            logger.info("LTM_FORGET", f"遗忘记忆 {item_id}", module="memory", extra={"score": score})
        logger.info("LTM_FORGET", "遗忘完成", module="memory", extra={"forgotten": n_forget})

    @log_execution
    def clear(self) -> None:
        """清空长期记忆"""
        logger.debug("LTM_CLEAR", "清空长期记忆", module="memory")
        self.items.clear()
        self.index.clear()
        logger.info("LTM_CLEAR", "长期记忆已清空", module="memory")

    @log_execution
    def get_size(self) -> int:
        logger.debug("LTM_SIZE", "获取长期记忆大小", module="memory")
        size = len(self.items)
        logger.info("LTM_SIZE", f"长期记忆大小: {size}", module="memory")
        return size


class WorkingMemory:
    """工作记忆 —— 当前关注的记忆，容量最小（7±2条）"""

    def __init__(self, capacity: int = 7) -> None:
        logger.info("WM_INIT", "WorkingMemory 初始化", module="memory", extra={"capacity": capacity})
        self.capacity = capacity
        self.items: Dict[str, MemoryItem] = {}
        self.focus: Optional[str] = None
        logger.debug("WM_INIT", "工作记忆初始化完成", module="memory")

    @log_execution
    def load(self, item_id: str, item: MemoryItem) -> bool:
        """加载记忆到工作记忆"""
        logger.debug("WM_LOAD", f"加载记忆到工作记忆: {item_id}", module="memory",
                    extra={"current_size": len(self.items)})
        if len(self.items) >= self.capacity:
            logger.warning("WM_LOAD", "工作记忆已满，移除最早加载的记忆", module="memory")
            self._unload_oldest()
        self.items[item_id] = item
        self.focus = item_id
        item.access()
        logger.info("WM_LOAD", "记忆加载到工作记忆完成", module="memory",
                    extra={"item_id": item_id, "current_size": len(self.items)})
        return True

    @log_execution
    def unload(self, item_id: str) -> bool:
        """从工作记忆卸载记忆"""
        logger.debug("WM_UNLOAD", f"从工作记忆卸载: {item_id}", module="memory")
        if item_id in self.items:
            del self.items[item_id]
            if self.focus == item_id:
                self.focus = None
            logger.info("WM_UNLOAD", "记忆卸载完成", module="memory", extra={"item_id": item_id})
            return True
        logger.warning("WM_UNLOAD", "记忆不在工作记忆中", module="memory", extra={"item_id": item_id})
        return False

    @log_execution
    def get_focused(self) -> Optional[MemoryItem]:
        """获取当前聚焦的记忆"""
        logger.debug("WM_FOCUS", "获取当前聚焦记忆", module="memory")
        if self.focus and self.focus in self.items:
            item = self.items[self.focus]
            item.access()
            logger.info("WM_FOCUS", "获取聚焦记忆成功", module="memory", extra={"focus": self.focus})
            return item
        logger.warning("WM_FOCUS", "无聚焦记忆", module="memory")
        return None

    @log_execution
    def set_focus(self, item_id: str) -> bool:
        """设置聚焦记忆"""
        logger.debug("WM_SET_FOCUS", f"设置聚焦记忆: {item_id}", module="memory")
        if item_id in self.items:
            self.focus = item_id
            self.items[item_id].access()
            logger.info("WM_SET_FOCUS", "聚焦记忆设置完成", module="memory", extra={"focus": item_id})
            return True
        logger.warning("WM_SET_FOCUS", "记忆不在工作记忆中，无法聚焦", module="memory", extra={"item_id": item_id})
        return False

    @log_execution
    def _unload_oldest(self) -> None:
        """卸载最早加载的记忆"""
        logger.debug("WM_UNLOAD_OLD", "卸载最早加载的工作记忆", module="memory")
        if not self.items:
            logger.warning("WM_UNLOAD_OLD", "工作记忆为空", module="memory")
            return
        oldest_id = min(self.items.keys(), key=lambda k: self.items[k].timestamp)
        del self.items[oldest_id]
        if self.focus == oldest_id:
            self.focus = None
        logger.info("WM_UNLOAD_OLD", "最早记忆已卸载", module="memory", extra={"unloaded": oldest_id})

    @log_execution
    def clear(self) -> None:
        """清空工作记忆"""
        logger.debug("WM_CLEAR", "清空工作记忆", module="memory")
        self.items.clear()
        self.focus = None
        logger.info("WM_CLEAR", "工作记忆已清空", module="memory")

    @log_execution
    def get_size(self) -> int:
        logger.debug("WM_SIZE", "获取工作记忆大小", module="memory")
        size = len(self.items)
        logger.info("WM_SIZE", f"工作记忆大小: {size}", module="memory")
        return size


class MemorySystem(IMemory):
    """记忆系统主类 —— 整合短期、长期、工作记忆"""

    def __init__(self, stm_capacity: int = 100, ltm_capacity: int = 1000, wm_capacity: int = 7) -> None:
        logger.info("MEMORY_INIT", "MemorySystem 初始化", module="memory",
                    extra={"stm": stm_capacity, "ltm": ltm_capacity, "wm": wm_capacity})
        self.short_term = ShortTermMemory(capacity=stm_capacity)
        self.long_term = LongTermMemory(capacity=ltm_capacity)
        self.working = WorkingMemory(capacity=wm_capacity)
        self.consolidation_threshold = 0.7
        self._replay = PrioritizedReplay(capacity=1000)
        logger.debug("MEMORY_INIT", "MemorySystem 子模块初始化完成", module="memory")

    @log_execution
    def store(self, content: Any, importance: float = 0.5, tags: Optional[List[str]] = None) -> str:
        """存储记忆 —— 先存入短期记忆，重要记忆会巩固到长期记忆"""
        logger.info("MEMORY_STORE", "存储记忆", module="memory",
                    extra={"content_type": type(content).__name__, "importance": importance, "tags": tags})
        item_id = self.short_term.store(content, importance, tags)
        logger.debug("MEMORY_STORE", "记忆已存入短期记忆", module="memory", extra={"item_id": item_id})

        # 重要记忆直接巩固到长期记忆
        if importance >= self.consolidation_threshold:
            logger.info("MEMORY_STORE", "记忆重要性高，触发巩固", module="memory",
                        extra={"importance": importance, "threshold": self.consolidation_threshold})
            recent = self.short_term.retrieve(n=1)
            if recent:
                self.long_term.store(item_id, recent[0])
                logger.info("MEMORY_STORE", "记忆已巩固到长期记忆", module="memory", extra={"item_id": item_id})

        return item_id

    @log_execution
    def retrieve(self, n: int = 10) -> List[MemoryItem]:
        """检索最近记忆（优先工作记忆，然后短期记忆）"""
        logger.info("MEMORY_RETRIEVE", f"检索最近 {n} 条记忆", module="memory")
        results = []
        # 先检索工作记忆
        wm_items = list(self.working.items.values())
        logger.debug("MEMORY_RETRIEVE", f"工作记忆检索到 {len(wm_items)} 条", module="memory")
        results.extend(wm_items)
        # 再检索短期记忆
        remaining = n - len(results)
        if remaining > 0:
            stm_items = self.short_term.retrieve(n=remaining)
            logger.debug("MEMORY_RETRIEVE", f"短期记忆检索到 {len(stm_items)} 条", module="memory")
            results.extend(stm_items)
        logger.info("MEMORY_RETRIEVE", "记忆检索完成", module="memory", extra={"total": len(results)})
        return results[:n]

    @log_execution
    def retrieve_by_tag(self, tag: str) -> List[MemoryItem]:
        """按标签检索记忆（优先长期记忆）"""
        logger.info("MEMORY_RETRIEVE_TAG", f"按标签检索记忆: {tag}", module="memory")
        results = []
        # 先检索长期记忆
        ltm_items = self.long_term.retrieve_by_tag(tag)
        logger.debug("MEMORY_RETRIEVE_TAG", f"长期记忆检索到 {len(ltm_items)} 条", module="memory")
        results.extend(ltm_items)
        # 再检索短期记忆
        stm_items = self.short_term.retrieve_by_tag(tag)
        logger.debug("MEMORY_RETRIEVE_TAG", f"短期记忆检索到 {len(stm_items)} 条", module="memory")
        results.extend(stm_items)
        logger.info("MEMORY_RETRIEVE_TAG", "标签检索完成", module="memory", extra={"total": len(results)})
        return results

    @log_execution
    def consolidate(self) -> int:
        """记忆巩固 —— 将短期记忆中重要的记忆转移到长期记忆"""
        logger.info("MEMORY_CONSOLIDATE", "开始记忆巩固", module="memory",
                    extra={"stm_size": self.short_term.get_size()})
        consolidated = 0
        for item in list(self.short_term.items):
            if item.importance >= self.consolidation_threshold or item.access_count >= 3:
                try:
                    content_str = str(item.content)
                except Exception:
                    content_str = str(id(item.content))
                item_id = hashlib.md5(f"{content_str}{item.timestamp}".encode()).hexdigest()[:8]
                self.long_term.store(item_id, item)
                consolidated += 1
                logger.debug("MEMORY_CONSOLIDATE", f"巩固记忆: {item_id}", module="memory",
                            extra={"importance": item.importance, "access_count": item.access_count})
        logger.info("MEMORY_CONSOLIDATE", "记忆巩固完成", module="memory", extra={"consolidated": consolidated})
        return consolidated

    @log_execution
    def forget(self, item_id: str) -> bool:
        """主动遗忘记忆"""
        logger.info("MEMORY_FORGET", f"主动遗忘记忆: {item_id}", module="memory")
        # 从长期记忆中删除
        if item_id in self.long_term.items:
            del self.long_term.items[item_id]
            for tag, ids in self.long_term.index.items():
                ids.discard(item_id)
            logger.info("MEMORY_FORGET", "长期记忆已遗忘", module="memory", extra={"item_id": item_id})
            return True
        # 从工作记忆中删除
        if item_id in self.working.items:
            self.working.unload(item_id)
            logger.info("MEMORY_FORGET", "工作记忆已遗忘", module="memory", extra={"item_id": item_id})
            return True
        logger.warning("MEMORY_FORGET", "记忆未找到", module="memory", extra={"item_id": item_id})
        return False

    @log_execution
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆系统统计信息"""
        logger.debug("MEMORY_STATS", "获取记忆系统统计", module="memory")
        stats = {
            "short_term_size": self.short_term.get_size(),
            "long_term_size": self.long_term.get_size(),
            "long_term_count": self.long_term.get_size(),  # 兼容旧测试代码
            "working_size": self.working.get_size(),
            "replay_pool_size": getattr(self, '_replay', None) and len(self._replay) or 0,
            "total": self.short_term.get_size() + self.long_term.get_size() + self.working.get_size(),
            "consolidation_threshold": self.consolidation_threshold,
        }
        logger.info("MEMORY_STATS", "记忆系统统计获取完成", module="memory", extra=stats)
        return stats

    @log_execution
    def clear(self) -> None:
        """清空所有记忆"""
        logger.info("MEMORY_CLEAR", "清空所有记忆", module="memory")
        self.short_term.clear()
        self.long_term.clear()
        self.working.clear()
        logger.info("MEMORY_CLEAR", "所有记忆已清空", module="memory")

    # 兼容旧测试的方法
    @log_execution
    def record_experience(self, state: Dict[str, Any], action: str,
                          reward: float, next_state: Dict[str, Any],
                          td_error: float = 0.0) -> None:
        """记录经验 —— 兼容旧测试，同时存入短期记忆和优先回放"""
        logger.info("MEMORY_RECORD", "记录经验", module="memory",
                    extra={"action": action, "reward": reward})
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            td_error=td_error,
        )
        # 存入短期记忆
        self.short_term.store(experience, importance=abs(reward) + 0.5)
        # 存入优先回放
        self._replay.add(abs(reward) + 1.0, experience)
        logger.info("MEMORY_RECORD", "经验记录完成", module="memory")

    @log_execution
    def sample_for_training(self, batch_size: int = 32) -> List[Experience]:
        """采样训练数据 —— 兼容旧测试"""
        logger.info("MEMORY_SAMPLE", "采样训练数据", module="memory", extra={"batch_size": batch_size})
        batch = self._replay.sample(batch_size)
        logger.info("MEMORY_SAMPLE", "采样完成", module="memory", extra={"sampled": len(batch)})
        return batch

    @log_execution
    def offline_review(self, batch_size: int = 32) -> Dict[str, Any]:
        """离线回顾 —— 兼容旧测试"""
        logger.info("MEMORY_REVIEW", "离线回顾", module="memory", extra={"batch_size": batch_size})
        batch = self._replay.sample(batch_size)
        report = {
            "sampled_experiences": batch,
            "improvement_suggestions": [],
        }
        logger.info("MEMORY_REVIEW", "离线回顾完成", module="memory", extra={"sampled": len(batch)})
        return report

    @log_execution
    def close(self) -> None:
        """关闭记忆系统"""
        logger.info("MEMORY_CLOSE", "关闭记忆系统", module="memory")
        self.clear()
        logger.info("MEMORY_CLOSE", "记忆系统已关闭", module="memory")
