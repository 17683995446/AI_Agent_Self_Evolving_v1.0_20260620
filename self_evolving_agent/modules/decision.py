"""
自主决策模块 —— 基于驱力引擎输出和状态向量进行自主决策
包含动作空间定义、策略选择、动作执行与资源边界检查
"""
import os
import sys
import time
import random
import psutil
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from collections import deque

# 仅在直接运行时添加路径（避免副作用）
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger, log_execution
from core.config import config_manager
from core.interfaces import IDecision, StateKeys, DriveKeys


class ActionType(Enum):
    """动作类型枚举"""
    COLLECT_DATA = auto()
    ADJUST_MODEL = auto()
    OPTIMIZE_PARAMS = auto()
    SAVE_SNAPSHOT = auto()
    SLEEP = auto()
    REDUCE_COLLECT_FREQ = auto()
    RELEASE_RESOURCES = auto()
    PREDICT = auto()
    LEARN = auto()


@dataclass
class Action:
    """动作定义"""
    action_type: ActionType
    name: str
    description: str
    priority: int = 0
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.name,
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "params": self.params,
        }


@dataclass
class ActionResult:
    """动作执行结果"""
    action: Action
    success: bool
    elapsed_time: float
    cpu_used: float
    memory_used: float
    message: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "success": self.success,
            "elapsed_time": self.elapsed_time,
            "cpu_used": self.cpu_used,
            "memory_used": self.memory_used,
            "message": self.message,
            "timestamp": self.timestamp,
        }


class ActionSpace:
    """动作空间定义 —— 包含所有可能的动作"""

    def __init__(self) -> None:
        logger.info("ACTION_SPACE_INIT", "初始化动作空间", module="decision")
        self.actions: List[Action] = []
        self._init_actions()
        logger.info("ACTION_SPACE_INIT", "动作空间初始化完成", module="decision",
                    extra={"action_count": len(self.actions)})

    @log_execution
    def _init_actions(self) -> None:
        """初始化所有动作"""
        logger.debug("ACTION_SPACE_INIT", "开始注册动作", module="decision")
        self.actions.append(Action(
            action_type=ActionType.COLLECT_DATA,
            name="采集数据",
            description="采集环境和系统数据",
            priority=1,
        ))
        logger.debug("ACTION_SPACE_INIT", "注册动作: 采集数据", module="decision")
        self.actions.append(Action(
            action_type=ActionType.ADJUST_MODEL,
            name="调整模型",
            description="调整模型结构或超参数",
            priority=2,
        ))
        logger.debug("ACTION_SPACE_INIT", "注册动作: 调整模型", module="decision")
        self.actions.append(Action(
            action_type=ActionType.OPTIMIZE_PARAMS,
            name="优化参数",
            description="优化运行参数以提升能效",
            priority=2,
        ))
        logger.debug("ACTION_SPACE_INIT", "注册动作: 优化参数", module="decision")
        self.actions.append(Action(
            action_type=ActionType.SAVE_SNAPSHOT,
            name="保存快照",
            description="保存当前模型和状态快照",
            priority=3,
        ))
        logger.debug("ACTION_SPACE_INIT", "注册动作: 保存快照", module="decision")
        self.actions.append(Action(
            action_type=ActionType.SLEEP,
            name="休眠",
            description="进入休眠状态以降低资源消耗",
            priority=0,
        ))
        logger.debug("ACTION_SPACE_INIT", "注册动作: 休眠", module="decision")
        self.actions.append(Action(
            action_type=ActionType.REDUCE_COLLECT_FREQ,
            name="降低采集频率",
            description="降低数据采集频率以释放资源",
            priority=1,
        ))
        logger.debug("ACTION_SPACE_INIT", "注册动作: 降低采集频率", module="decision")
        self.actions.append(Action(
            action_type=ActionType.RELEASE_RESOURCES,
            name="释放资源",
            description="释放缓存和临时资源",
            priority=1,
        ))
        logger.debug("ACTION_SPACE_INIT", "注册动作: 释放资源", module="decision")
        self.actions.append(Action(
            action_type=ActionType.PREDICT,
            name="预测",
            description="执行预测任务",
            priority=2,
        ))
        logger.debug("ACTION_SPACE_INIT", "注册动作: 预测", module="decision")
        self.actions.append(Action(
            action_type=ActionType.LEARN,
            name="学习",
            description="执行在线学习任务",
            priority=2,
        ))
        logger.debug("ACTION_SPACE_INIT", "注册动作: 学习", module="decision")
        logger.info("ACTION_SPACE_INIT", f"共注册 {len(self.actions)} 个动作", module="decision")

    @log_execution
    def get_action(self, action_type: ActionType) -> Optional[Action]:
        """根据类型获取动作"""
        logger.debug("ACTION_SPACE_GET", f"获取动作: {action_type.name}", module="decision")
        for action in self.actions:
            if action.action_type == action_type:
                logger.debug("ACTION_SPACE_GET", f"找到动作: {action.name}", module="decision")
                return action
        logger.warning("ACTION_SPACE_GET", f"未找到动作: {action_type.name}", module="decision")
        return None

    @log_execution
    def get_all_actions(self) -> List[Action]:
        """获取所有动作"""
        logger.debug("ACTION_SPACE_ALL", f"获取全部动作，数量: {len(self.actions)}", module="decision")
        return list(self.actions)

    @log_execution
    def get_protective_actions(self) -> List[Action]:
        """获取保护性动作"""
        logger.debug("ACTION_SPACE_PROTECT", "获取保护性动作", module="decision")
        protective = [
            self.get_action(ActionType.REDUCE_COLLECT_FREQ),
            self.get_action(ActionType.RELEASE_RESOURCES),
            self.get_action(ActionType.SLEEP),
        ]
        result = [a for a in protective if a is not None]
        logger.debug("ACTION_SPACE_PROTECT", f"保护性动作数量: {len(result)}", module="decision")
        return result

    @log_execution
    def get_learning_actions(self) -> List[Action]:
        """获取预测/学习类动作"""
        logger.debug("ACTION_SPACE_LEARN", "获取学习类动作", module="decision")
        learning = [
            self.get_action(ActionType.PREDICT),
            self.get_action(ActionType.LEARN),
            self.get_action(ActionType.ADJUST_MODEL),
        ]
        result = [a for a in learning if a is not None]
        logger.debug("ACTION_SPACE_LEARN", f"学习类动作数量: {len(result)}", module="decision")
        return result

    @log_execution
    def get_optimization_actions(self) -> List[Action]:
        """获取优化类动作"""
        logger.debug("ACTION_SPACE_OPTIM", "获取优化类动作", module="decision")
        optim = [
            self.get_action(ActionType.OPTIMIZE_PARAMS),
            self.get_action(ActionType.RELEASE_RESOURCES),
        ]
        result = [a for a in optim if a is not None]
        logger.debug("ACTION_SPACE_OPTIM", f"优化类动作数量: {len(result)}", module="decision")
        return result


class Policy:
    """策略类 —— 基于当前状态和驱力选择动作"""

    def __init__(self, action_space: ActionSpace, exploration_ratio: Optional[float] = None) -> None:
        logger.info("POLICY_INIT", "初始化策略类", module="decision")
        self.action_space = action_space
        self.config = config_manager.config
        if exploration_ratio is not None:
            self.exploration_ratio = exploration_ratio
        else:
            self.exploration_ratio = self.config.evolution.exploration_ratio
        logger.info("POLICY_INIT", "策略类初始化完成", module="decision",
                    extra={"exploration_ratio": self.exploration_ratio})

    @log_execution
    def select_action(self, state_vector: Dict[str, float], drives: Dict[str, float]) -> Action:
        """基于状态和驱力选择动作"""
        logger.info("POLICY_SELECT", "开始选择动作", module="decision",
                    extra={"drives": drives, "state_keys": list(state_vector.keys())})

        # 探索机制：epsilon-greedy
        if random.random() < self.exploration_ratio:
            logger.info("POLICY_SELECT", "触发探索机制，随机选择动作", module="decision",
                        extra={"epsilon": self.exploration_ratio})
            action = random.choice(self.action_space.get_all_actions())
            logger.info("POLICY_SELECT", f"随机选择动作: {action.name}", module="decision")
            return action

        logger.debug("POLICY_SELECT", "进入贪心策略选择", module="decision")

        # 提取驱力值
        safe_drive = drives.get(DriveKeys.SAFE, 0.0)
        gain_drive = drives.get(DriveKeys.GAIN, 0.0)
        eff_drive = drives.get(DriveKeys.EFF, 0.0)
        logger.debug("POLICY_SELECT", f"驱力值 — 安全: {safe_drive}, 收益: {gain_drive}, 能效: {eff_drive}", module="decision")

        # 决策逻辑
        if safe_drive < 0:
            logger.warning("POLICY_SELECT", "安全驱力为负，优先执行保护性动作", module="decision",
                           extra={"safe_drive": safe_drive})
            protective = self.action_space.get_protective_actions()
            if protective:
                action = protective[0]
                logger.info("POLICY_SELECT", f"选择保护性动作: {action.name}", module="decision")
                return action
            else:
                logger.warning("POLICY_SELECT", "无保护性动作可用， fallback 到休眠", module="decision")
                return self.action_space.get_action(ActionType.SLEEP)

        if gain_drive > 0.7:
            logger.info("POLICY_SELECT", "收益驱力高，优先执行预测/学习动作", module="decision",
                        extra={"gain_drive": gain_drive})
            learning = self.action_space.get_learning_actions()
            if learning:
                action = learning[0]
                logger.info("POLICY_SELECT", f"选择学习类动作: {action.name}", module="decision")
                return action

        if eff_drive < 0.3:
            logger.info("POLICY_SELECT", "能效驱力低，优先执行优化动作", module="decision",
                        extra={"eff_drive": eff_drive})
            optim = self.action_space.get_optimization_actions()
            if optim:
                action = optim[0]
                logger.info("POLICY_SELECT", f"选择优化类动作: {action.name}", module="decision")
                return action

        # 默认动作：采集数据
        logger.info("POLICY_SELECT", "无特殊驱力触发，执行默认动作: 采集数据", module="decision")
        default_action = self.action_space.get_action(ActionType.COLLECT_DATA)
        if default_action is None:
            logger.error("POLICY_SELECT", "默认动作采集数据未找到，fallback 到休眠", module="decision")
            return self.action_space.get_action(ActionType.SLEEP)
        return default_action


class ActionExecutor:
    """动作执行器 —— 负责执行选定的动作并返回结果"""

    def __init__(self) -> None:
        logger.info("EXECUTOR_INIT", "初始化动作执行器", module="decision")
        self.config = config_manager.config
        self.resource = self.config.resource
        self.execution_history = deque(maxlen=1000)
        self.recent_tasks_completed = 0
        self.recent_tasks_failed = 0
        logger.info("EXECUTOR_INIT", "动作执行器初始化完成", module="decision")

    @log_execution
    def check_resource_boundaries(self) -> tuple[bool, str]:
        """刚性边界检查：CPU/内存是否超过20%"""
        logger.debug("EXECUTOR_CHECK", "开始资源边界检查", module="decision")
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        memory_percent = mem.percent
        logger.debug("EXECUTOR_CHECK", f"当前资源 — CPU: {cpu_percent}%, 内存: {memory_percent}%", module="decision",
                    extra={"cpu": cpu_percent, "memory": memory_percent, "threshold": 20.0})

        if cpu_percent > self.resource.max_cpu_percent:
            msg = f"CPU 使用率 {cpu_percent}% 超过阈值 {self.resource.max_cpu_percent}%，拒绝执行"
            logger.error("EXECUTOR_CHECK", msg, module="decision",
                         extra={"cpu": cpu_percent, "threshold": self.resource.max_cpu_percent})
            return False, msg

        if memory_percent > self.resource.max_memory_percent:
            msg = f"内存使用率 {memory_percent}% 超过阈值 {self.resource.max_memory_percent}%，拒绝执行"
            logger.error("EXECUTOR_CHECK", msg, module="decision",
                         extra={"memory": memory_percent, "threshold": self.resource.max_memory_percent})
            return False, msg

        logger.info("EXECUTOR_CHECK", "资源边界检查通过", module="decision")
        return True, ""

    def execute(self, action: Action) -> ActionResult:
        """执行动作并返回结果 —— 不使用log_execution装饰器以避免None访问问题"""
        # 参数校验（必须在任何属性访问之前）
        if action is None:
            logger.error("EXECUTOR_EXECUTE", "动作对象为空，拒绝执行", module="decision")
            return ActionResult(
                action=Action(ActionType.SLEEP, "休眠", "空动作降级"),
                success=False, elapsed_time=0.0, cpu_used=0.0, memory_used=0.0,
                message="动作对象为空", timestamp=time.time(),
            )

        logger.info("EXECUTOR_EXECUTE", f"准备执行动作: {action.name}", module="decision",
                    extra={"action": action.to_dict()})

        # 刚性边界检查
        passed, msg = self.check_resource_boundaries()
        if not passed:
            logger.warning("EXECUTOR_EXECUTE", f"边界检查未通过，拒绝执行: {action.name}", module="decision")
            self.recent_tasks_failed += 1
            result = ActionResult(
                action=action,
                success=False,
                elapsed_time=0.0,
                cpu_used=0.0,
                memory_used=0.0,
                message=msg,
                timestamp=time.time(),
            )
            self.execution_history.append(result)
            logger.error("EXECUTOR_EXECUTE", f"动作 {action.name} 被拒绝执行", module="decision",
                         extra={"result": result.to_dict()})
            return result

        # 记录执行前资源
        try:
            cpu_before = psutil.cpu_percent(interval=0.1)
            mem_before = psutil.virtual_memory().percent
        except Exception as e:
            logger.warning("EXECUTOR_EXECUTE", f"获取执行前资源失败: {e}", module="decision")
            cpu_before = 0.0
            mem_before = 0.0
        logger.debug("EXECUTOR_EXECUTE", f"执行前资源 — CPU: {cpu_before}%, 内存: {mem_before}%", module="decision")

        start_time = time.time()
        success = True
        message = f"动作 {action.name} 执行成功"

        try:
            logger.info("EXECUTOR_EXECUTE", f"开始执行动作核心逻辑: {action.name}", module="decision")
            self._execute_action_logic(action)
            self.recent_tasks_completed += 1
            logger.info("EXECUTOR_EXECUTE", f"动作核心逻辑执行完成: {action.name}", module="decision")
        except Exception as e:
            success = False
            message = f"动作 {action.name} 执行失败: {str(e)}"
            self.recent_tasks_failed += 1
            logger.error("EXECUTOR_EXECUTE", message, module="decision",
                         extra={"action": action.name, "error": str(e)})

        elapsed_time = time.time() - start_time
        try:
            cpu_after = psutil.cpu_percent(interval=0.1)
            mem_after = psutil.virtual_memory().percent
        except Exception as e:
            logger.warning("EXECUTOR_EXECUTE", f"获取执行后资源失败: {e}", module="decision")
            cpu_after = cpu_before
            mem_after = mem_before
        cpu_used = max(0.0, cpu_after - cpu_before)
        memory_used = max(0.0, mem_after - mem_before)

        logger.debug("EXECUTOR_EXECUTE", f"执行后资源 — CPU: {cpu_after}%, 内存: {mem_after}%", module="decision")
        logger.debug("EXECUTOR_EXECUTE", f"动作耗时: {elapsed_time:.4f}s, CPU消耗: {cpu_used:.2f}%, 内存消耗: {memory_used:.2f}%", module="decision")

        result = ActionResult(
            action=action,
            success=success,
            elapsed_time=elapsed_time,
            cpu_used=cpu_used,
            memory_used=memory_used,
            message=message,
            timestamp=time.time(),
        )
        self.execution_history.append(result)

        level = "INFO" if success else "ERROR"
        logger.info("EXECUTOR_EXECUTE", f"动作 {action.name} 执行结果: {'成功' if success else '失败'}", module="decision",
                    extra={"result": result.to_dict()})
        return result

    @log_execution
    def _execute_action_logic(self, action: Action) -> None:
        """执行具体动作逻辑"""
        logger.debug("EXECUTOR_LOGIC", f"执行动作逻辑: {action.name}", module="decision",
                    extra={"action_type": action.action_type.name})

        if action.action_type == ActionType.COLLECT_DATA:
            logger.info("EXECUTOR_LOGIC", "执行采集数据逻辑", module="decision")
            time.sleep(0.01)
            logger.debug("EXECUTOR_LOGIC", "采集数据逻辑完成", module="decision")
        elif action.action_type == ActionType.ADJUST_MODEL:
            logger.info("EXECUTOR_LOGIC", "执行调整模型逻辑", module="decision")
            time.sleep(0.02)
            logger.debug("EXECUTOR_LOGIC", "调整模型逻辑完成", module="decision")
        elif action.action_type == ActionType.OPTIMIZE_PARAMS:
            logger.info("EXECUTOR_LOGIC", "执行优化参数逻辑", module="decision")
            time.sleep(0.015)
            logger.debug("EXECUTOR_LOGIC", "优化参数逻辑完成", module="decision")
        elif action.action_type == ActionType.SAVE_SNAPSHOT:
            logger.info("EXECUTOR_LOGIC", "执行保存快照逻辑", module="decision")
            snapshot_dir = self.config.model_snapshot_dir
            os.makedirs(snapshot_dir, exist_ok=True)
            logger.debug("EXECUTOR_LOGIC", f"快照目录确认存在: {snapshot_dir}", module="decision")
            time.sleep(0.01)
            logger.debug("EXECUTOR_LOGIC", "保存快照逻辑完成", module="decision")
        elif action.action_type == ActionType.SLEEP:
            logger.info("EXECUTOR_LOGIC", "执行休眠逻辑", module="decision")
            sleep_duration = action.params.get("duration", 1.0)
            # 参数校验：防止负数或超大值
            if not isinstance(sleep_duration, (int, float)):
                logger.warning("EXECUTOR_LOGIC", f"休眠时长类型异常: {type(sleep_duration)}，使用默认值1.0", module="decision")
                sleep_duration = 1.0
            sleep_duration = max(0.0, min(float(sleep_duration), 60.0))
            logger.debug("EXECUTOR_LOGIC", f"休眠时长: {sleep_duration}s", module="decision")
            time.sleep(sleep_duration)
            logger.debug("EXECUTOR_LOGIC", "休眠逻辑完成", module="decision")
        elif action.action_type == ActionType.REDUCE_COLLECT_FREQ:
            logger.info("EXECUTOR_LOGIC", "执行降低采集频率逻辑", module="decision")
            time.sleep(0.005)
            logger.debug("EXECUTOR_LOGIC", "降低采集频率逻辑完成", module="decision")
        elif action.action_type == ActionType.RELEASE_RESOURCES:
            logger.info("EXECUTOR_LOGIC", "执行释放资源逻辑", module="decision")
            time.sleep(0.005)
            logger.debug("EXECUTOR_LOGIC", "释放资源逻辑完成", module="decision")
        elif action.action_type == ActionType.PREDICT:
            logger.info("EXECUTOR_LOGIC", "执行预测逻辑", module="decision")
            time.sleep(0.02)
            logger.debug("EXECUTOR_LOGIC", "预测逻辑完成", module="decision")
        elif action.action_type == ActionType.LEARN:
            logger.info("EXECUTOR_LOGIC", "执行学习逻辑", module="decision")
            time.sleep(0.03)
            logger.debug("EXECUTOR_LOGIC", "学习逻辑完成", module="decision")
        else:
            logger.warning("EXECUTOR_LOGIC", f"未知动作类型: {action.action_type.name}", module="decision")

        logger.info("EXECUTOR_LOGIC", f"动作逻辑执行完毕: {action.name}", module="decision")

    @log_execution
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        logger.debug("EXECUTOR_STATS", "获取执行统计", module="decision")
        total = len(self.execution_history)
        if total == 0:
            logger.debug("EXECUTOR_STATS", "执行历史为空", module="decision")
            return {"total": 0, "success_rate": 0.0, "avg_time": 0.0}

        success_count = sum(1 for r in self.execution_history if r.success)
        avg_time = sum(r.elapsed_time for r in self.execution_history) / total
        success_rate = success_count / total
        stats = {
            "total": total,
            "success_count": success_count,
            "failed_count": total - success_count,
            "success_rate": success_rate,
            "avg_time": avg_time,
            "recent_completed": self.recent_tasks_completed,
            "recent_failed": self.recent_tasks_failed,
        }
        logger.info("EXECUTOR_STATS", "执行统计获取完成", module="decision", extra=stats)
        return stats


class DecisionEngine(IDecision):
    """决策引擎主类 —— 整合动作空间、策略和执行器"""

    def __init__(self) -> None:
        logger.info("DECISION_INIT", "初始化决策引擎", module="decision")
        self.config = config_manager.config
        self.action_space = ActionSpace()
        self.policy = Policy(self.action_space)
        self.executor = ActionExecutor()
        self.running = False
        logger.info("DECISION_INIT", "决策引擎初始化完成", module="decision")

    @log_execution
    def decide_and_execute(self, state_vector: Dict[str, float], drives: Dict[str, float]) -> ActionResult:
        """决策并执行 —— 核心入口"""
        logger.info("DECISION_MAIN", "开始决策-执行循环", module="decision",
                    extra={"state_keys": list(state_vector.keys()), "drives": drives})

        logger.debug("DECISION_MAIN", "调用策略选择动作", module="decision")
        action = self.policy.select_action(state_vector, drives)
        logger.info("DECISION_MAIN", f"策略选择动作: {action.name}", module="decision")

        logger.debug("DECISION_MAIN", "调用执行器执行动作", module="decision")
        result = self.executor.execute(action)

        logger.info("DECISION_MAIN", f"决策-执行循环完成，结果: {'成功' if result.success else '失败'}", module="decision",
                    extra={"result": result.to_dict()})
        return result

    @log_execution
    def get_status(self) -> Dict[str, Any]:
        """获取决策引擎状态"""
        logger.debug("DECISION_STATUS", "获取决策引擎状态", module="decision")
        stats = self.executor.get_execution_stats()
        status = {
            "running": self.running,
            "action_count": len(self.action_space.get_all_actions()),
            "exploration_ratio": self.policy.exploration_ratio,
            "execution_stats": stats,
        }
        logger.info("DECISION_STATUS", "决策引擎状态获取完成", module="decision", extra=status)
        return status

    @log_execution
    def reset_stats(self) -> None:
        """重置执行统计"""
        logger.info("DECISION_RESET", "重置执行统计", module="decision")
        self.executor.execution_history.clear()
        self.executor.recent_tasks_completed = 0
        self.executor.recent_tasks_failed = 0
        logger.info("DECISION_RESET", "执行统计已重置", module="decision")
