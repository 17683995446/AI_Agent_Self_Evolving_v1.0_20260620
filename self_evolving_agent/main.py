"""
自驱永续进化预测修正Agent系统 —— 主入口
整合所有模块，实现双闭环运行：在线实时预测修正 + 离线复盘模型进化

乐高式模块化设计：通过 ModuleRegistry 实现插件化，所有模块可配置替换
"""
import os
import sys
import time
import signal
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List

# 仅在直接运行时添加路径（避免副作用）
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.logger import logger, log_execution
from core.config import config_manager
from core.interfaces import StateKeys, MetricKeys, DriveKeys
from core.registry import ModuleRegistry, get_default_registry
from core.container import DIContainer
from core.performance import perf_monitor
from core.agent_orchestrator import AgentOrchestrator, TaskType, DelegationResult
from core.workflow_engine import WorkflowEngine, Workflow, WorkflowTask, TaskState, WorkflowState
from core.task_decomposer import TaskDecomposer, Goal, GoalType
from core.skill_library import SkillLibrary, Skill, SkillStatus
from core.long_task_planner import LongTaskPlanner, LongTask, SubGoal, LongTaskState
from core.persistence import PersistenceManager, AgentStateSerializer, Snapshot
from core.forever_runner import ForeverRunner, HealthLevel, HealthReport
from core.exceptions import (
    AgentException, PerceptionException, PredictionException,
    DriveEngineException, DecisionException, ErrorEvalException,
    OptimizerException, MemoryException, ModuleInitException,
    ConfigurationException, ResourceExhaustedException,
    WorkflowException, SkillLibraryException, PersistenceException,
    ForeverRunnerException, AlertingException, MultiAgentException, LongTaskException,
)


class SelfEvolvingAgent:
    """自驱永续进化Agent主类 —— 支持乐高式模块替换和依赖注入"""

    def __init__(self, registry: Optional[ModuleRegistry] = None,
                 module_config: Optional[Dict[str, str]] = None,
                 container: Optional[DIContainer] = None,
                 modules: Optional[Dict[str, Any]] = None):
        """
        Args:
            registry: 模块注册中心，传 None 使用默认注册表（向后兼容）
            module_config: 模块选择配置，如 {"prediction": "my_plugin"}（向后兼容）
            container: 依赖注入容器，传 None 则内部创建
            modules: 预构建的模块字典，传 None 则通过容器构建
        """
        logger.info("AGENT_INIT", "自驱Agent系统初始化开始", module="main")
        self.config = config_manager.config
        self._module_config = module_config or {}

        # 依赖注入：优先使用传入的模块，其次使用容器构建，最后回退到直接注册表（向后兼容）
        if modules is not None:
            logger.debug("AGENT_INIT", "使用预注入的模块", module="main")
            self._modules = modules
            self._container = None
        elif container is not None:
            logger.debug("AGENT_INIT", "使用依赖注入容器构建模块", module="main")
            self._container = container
            self._modules = container.build_agent(self._module_config)
        else:
            # 向后兼容：直接通过注册表创建
            logger.debug("AGENT_INIT", "使用注册表直接创建模块（向后兼容模式）", module="main")
            reg = registry or get_default_registry()
            self._container = None
            self._modules = {
                "perception": reg.create_perception(self._module_config.get("perception", "default")),
                "prediction": reg.create_prediction(self._module_config.get("prediction", "default")),
                "error_eval": reg.create_error_eval(self._module_config.get("error_eval", "default")),
                "optimizer": reg.create_optimizer(self._module_config.get("optimizer", "default")),
                "drive_engine": reg.create_drive_engine(self._module_config.get("drive_engine", "default")),
                "decision": reg.create_decision(self._module_config.get("decision", "default")),
                "memory": reg.create_memory(self._module_config.get("memory", "default")),
            }

        # 解包模块引用（保持属性访问接口不变）
        self.perception = self._modules["perception"]
        self.prediction = self._modules["prediction"]
        self.error_eval = self._modules["error_eval"]
        self.optimizer = self._modules["optimizer"]
        self.drive_engine = self._modules["drive_engine"]
        self.decision = self._modules["decision"]
        self.memory = self._modules["memory"]

        logger.debug("AGENT_INIT", "所有模块引用解包完成", module="main",
                    extra={"modules": list(self._modules.keys())})

        # 运行状态
        self.running = False
        self._stopped = False  # O2：幂等保护标志，显式初始化
        self._main_thread = None
        self._offline_thread = None
        self._shutdown_event = threading.Event()
        logger.debug("AGENT_INIT", "运行状态初始化完成", module="main")

        # 统计
        self.loop_count = 0
        self.start_time = None

        # H2：决策-执行接通 —— 将各模块真实方法注入ActionExecutor的回调表
        # 使决策产生的动作能真正调用对应模块执行（而非空壳）
        self._init_alert_manager()
        self._wire_action_callbacks()

        # M1：多Agent编排器 —— 主Agent通过它操作子Agent
        self.orchestrator = AgentOrchestrator()
        self._orchestrator_enabled = False  # 默认关闭，按需启用避免测试噪声

        # M2：任务分解与工作流引擎
        self.task_decomposer = TaskDecomposer()
        self.workflow_engine = WorkflowEngine(max_workers=4)
        self._register_workflow_handlers()
        self._workflow_engine_started = False

        # M3：技能库与持续学习
        self.skill_library = SkillLibrary()
        self._continual_learner = None  # 懒加载，避免numpy依赖影响基础功能
        self._meta_learner = None       # 懒加载
        self._skills_learned_this_session = 0

        # M4：长任务规划与自驱执行
        self.long_task_planner = LongTaskPlanner()
        self._active_long_task_id: Optional[str] = None

        # M5：状态持久化与故障恢复
        self.persistence = PersistenceManager(
            snapshot_dir=os.path.join(self.config.log_dir, "snapshots"),
            max_snapshots=5,
        )
        self._watchdog_thread: Optional[threading.Thread] = None
        self._last_loop_heartbeat: float = time.time()
        self._watchdog_timeout: float = 60.0  # 主循环60秒无心跳视为卡死
        self._auto_recovery_enabled: bool = True

        # M6：永续运行器 —— 健康监控+资源回收+分级自愈
        # 默认关闭自动启动，避免测试噪声；通过 enable_forever_running() 显式启用
        self.forever_runner = ForeverRunner(
            health_check_interval=30.0,
            resource_cleanup_interval=300.0,
        )
        self._forever_running_enabled: bool = False

        # U3：Dashboard监控面板 —— 默认关闭，通过 enable_dashboard() 显式启用
        self.dashboard = None
        self._dashboard_enabled: bool = False

        # H6：应用日志与性能监控配置
        self._apply_logging_config()
        logger.info("AGENT_INIT", "自驱Agent系统初始化完成", module="main")

    def _apply_logging_config(self):
        """H6：从config应用日志级别、采样率和性能监控设置"""
        log_cfg = self.config.logging

        # 应用日志级别
        logger.set_console_level(log_cfg.console_level)
        logger.set_file_level(log_cfg.file_level)

        # 应用函数追踪开关
        logger.set_func_trace_enabled(log_cfg.func_trace_enabled)

        # 应用采样率
        for event_type, every_n in log_cfg.sample_rates.items():
            logger.set_sample_rate(event_type, every_n)

        # 应用性能监控设置
        perf_monitor.set_enabled(log_cfg.perf_monitor_enabled)
        perf_monitor.set_slow_threshold(log_cfg.perf_slow_threshold_sec)

        logger.info("AGENT_INIT", "日志与性能配置已应用", module="main",
                    extra={"console_level": log_cfg.console_level,
                           "file_level": log_cfg.file_level,
                           "func_trace": log_cfg.func_trace_enabled,
                           "perf_enabled": log_cfg.perf_monitor_enabled,
                           "sample_rates": log_cfg.sample_rates})

    def _init_alert_manager(self):
        """H3：初始化告警管理器，从配置加载告警通道"""
        from core.alerting import AlertManager, AlertRule, AlertLevel
        self.alert_manager = AlertManager()

        # 从配置文件加载告警通道（如果配置了）
        alert_channels_config = getattr(self.config, "alert_channels", [])
        if alert_channels_config:
            count = self.alert_manager.setup_channels_from_config(alert_channels_config)
            logger.info("AGENT_INIT", f"从配置加载 {count} 个告警通道", module="main")

        # 注册默认告警规则（CPU/内存超限）
        try:
            self.alert_manager.register_rule(AlertRule(
                name="cpu_high",
                metric_path="cpu_percent",
                threshold=0.8,
                operator=">",
                duration_sec=10,
                level=AlertLevel.WARNING,
                channels=["log"],
            ))
            self.alert_manager.register_rule(AlertRule(
                name="memory_high",
                metric_path="memory_percent",
                threshold=0.8,
                operator=">",
                duration_sec=10,
                level=AlertLevel.ERROR,
                channels=["log"],
            ))
        except Exception as e:
            logger.warning("AGENT_INIT", f"默认告警规则注册失败: {e}", module="main")

        logger.info("AGENT_INIT", "告警管理器初始化完成", module="main",
                    extra={"channels": list(self.alert_manager._channels.keys())})

    def _wire_action_callbacks(self):
        """H2：将各模块方法注入决策执行器的回调表，实现决策-执行闭环"""
        if not hasattr(self.decision, "executor"):
            logger.warning("AGENT_INIT", "决策模块无executor属性，跳过回调注入", module="main")
            return

        executor = self.decision.executor
        wired_count = 0

        # 预测回调：让PREDICT动作真正调用prediction模块
        if hasattr(self.prediction, "predict"):
            def predict_cb(params):
                try:
                    return self.prediction.predict()
                except Exception as e:
                    logger.error("ACTION_CB", f"预测回调异常: {e}", module="main")
                    return None
            executor.register_callback("predict", predict_cb)
            wired_count += 1

        # 在线学习回调：让LEARN动作真正调用prediction.online_learn
        if hasattr(self.prediction, "online_learn"):
            def learn_cb(params):
                try:
                    # 用最近一次实际值做在线学习
                    actual = params.get("actual", {})
                    if actual:
                        self.prediction.online_learn(actual)
                except Exception as e:
                    logger.error("ACTION_CB", f"学习回调异常: {e}", module="main")
            executor.register_callback("learn", learn_cb)
            wired_count += 1

        # 模型调整回调：让ADJUST_MODEL动作真正调用optimizer
        if hasattr(self.optimizer, "online_optimize"):
            def adjust_model_cb(params):
                try:
                    predicted = params.get("predicted", {})
                    actual = params.get("actual", {})
                    error_report = params.get("error_report", {})
                    if predicted and actual:
                        self.optimizer.online_optimize(predicted, actual, error_report)
                except Exception as e:
                    logger.error("ACTION_CB", f"模型调整回调异常: {e}", module="main")
            executor.register_callback("adjust_model", adjust_model_cb)
            wired_count += 1

        # 参数优化回调：复用optimizer
        if hasattr(self.optimizer, "online_optimize"):
            executor.register_callback("optimize_params", adjust_model_cb)
            wired_count += 1

        # 降低采集频率回调：让REDUCE_COLLECT_FREQ动作真正调整perception
        if hasattr(self.perception, "set_interval"):
            def reduce_freq_cb(params):
                try:
                    new_interval = params.get("interval", 2.0)
                    self.perception.set_interval(new_interval)
                    logger.info("ACTION_CB", f"采集频率调整为: {new_interval}s", module="main")
                except Exception as e:
                    logger.error("ACTION_CB", f"降低频率回调异常: {e}", module="main")
            executor.register_callback("reduce_freq", reduce_freq_cb)
            wired_count += 1

        # 告警回调：让ALERT_NOTIFY动作触发AlertManager
        if hasattr(self, "alert_manager") and self.alert_manager is not None:
            def alert_cb(info):
                try:
                    from core.alerting import AlertEvent, AlertLevel
                    level_map = {
                        "info": AlertLevel.INFO,
                        "warning": AlertLevel.WARNING,
                        "error": AlertLevel.ERROR,
                        "critical": AlertLevel.CRITICAL,
                    }
                    event = AlertEvent(
                        rule_name="action_trigger",
                        level=level_map.get(info.get("level", "warning"), AlertLevel.WARNING),
                        message=info.get("message", ""),
                        timestamp=time.time(),
                        metric_value=0.0,
                        threshold=0.0,
                    )
                    self.alert_manager.dispatch(event)
                except Exception as e:
                    logger.error("ACTION_CB", f"告警回调异常: {e}", module="main")
            executor.register_callback("alert", alert_cb)
            wired_count += 1

        logger.info("AGENT_INIT", f"决策-执行回调注入完成: {wired_count}个回调已连接",
                    module="main", extra={"wired_count": wired_count})

    def _inject_action_context(self, predicted, actual, error_report, state_vector):
        """H2：把当前循环上下文注入执行器，供内部动作回调使用"""
        if not hasattr(self.decision, "executor"):
            return
        executor = self.decision.executor
        # 把上下文存到执行器的_last_context属性，回调闭包可读取
        executor._last_context = {
            "predicted": predicted,
            "actual": actual,
            "error_report": error_report,
            "state_vector": state_vector,
            "timestamp": time.time(),
        }
        # 同时更新learn/adjust回调使用的默认参数源
        # 通过覆盖回调使它们读取_last_context
        self._refresh_context_aware_callbacks()

    def _refresh_context_aware_callbacks(self):
        """刷新依赖上下文的回调，使其读取最新的_last_context"""
        executor = self.decision.executor
        ctx_holder = executor  # 上下文持有者

        # 重新注册learn回调，从_last_context取actual
        if hasattr(self.prediction, "online_learn"):
            def learn_cb(params):
                try:
                    ctx = getattr(ctx_holder, "_last_context", {})
                    actual = params.get("actual") or ctx.get("actual", {})
                    if actual:
                        self.prediction.online_learn(actual)
                except Exception as e:
                    logger.error("ACTION_CB", f"学习回调异常: {e}", module="main")
            executor.register_callback("learn", learn_cb)

        # 重新注册adjust_model回调，从_last_context取数据
        if hasattr(self.optimizer, "online_optimize"):
            def adjust_model_cb(params):
                try:
                    ctx = getattr(ctx_holder, "_last_context", {})
                    predicted = params.get("predicted") or ctx.get("predicted", {})
                    actual = params.get("actual") or ctx.get("actual", {})
                    error_report = params.get("error_report") or ctx.get("error_report", {})
                    if predicted and actual:
                        self.optimizer.online_optimize(predicted, actual, error_report)
                except Exception as e:
                    logger.error("ACTION_CB", f"模型调整回调异常: {e}", module="main")
            executor.register_callback("adjust_model", adjust_model_cb)
            executor.register_callback("optimize_params", adjust_model_cb)

    def _trigger_reactive_actions(self, drive_state, error_report, state_vector):
        """H2：基于驱力和误差主动触发真实运维动作（反应式自愈）"""
        executor = self.decision.executor
        from modules.decision import Action, ActionType

        # 触发条件1：安全驱力过低（系统危险）→ 发告警
        if drive_state.safety < 0.3:
            logger.warning("ONLINE_LOOP", f"安全驱力过低({drive_state.safety:.3f})，触发告警", module="main")
            alert_action = Action(
                ActionType.ALERT_NOTIFY, "低安全告警", "安全驱力过低",
                params={"message": f"系统安全驱力过低: {drive_state.safety:.3f}", "level": "warning"},
            )
            try:
                executor.execute(alert_action)
            except Exception as e:
                logger.error("ONLINE_LOOP", f"告警动作执行异常: {e}", module="main")

        # 触发条件2：误差过大（预测严重不准）→ 触发模型调整
        mae = error_report.get("mae", 0.0)
        if mae > 10.0:
            logger.warning("ONLINE_LOOP", f"预测误差过大(MAE={mae:.2f})，触发模型调整", module="main")
            adjust_action = Action(
                ActionType.ADJUST_MODEL, "误差驱动调参", "MAE过大触发优化",
                params={},
            )
            try:
                executor.execute(adjust_action)
            except Exception as e:
                logger.error("ONLINE_LOOP", f"模型调整动作执行异常: {e}", module="main")

        # 触发条件3：CPU持续过高 → 清理缓存
        cpu_val = float(state_vector.get(StateKeys.CPU_PERCENT) or 0.0) * 100
        if cpu_val > 80.0:
            logger.warning("ONLINE_LOOP", f"CPU过高({cpu_val:.1f}%)，触发缓存清理", module="main")
            clean_action = Action(
                ActionType.CLEAN_CACHE, "高CPU清理", "CPU过高触发缓存清理",
                params={},
            )
            try:
                executor.execute(clean_action)
            except Exception as e:
                logger.error("ONLINE_LOOP", f"缓存清理动作执行异常: {e}", module="main")

    @log_execution
    def _online_loop(self):
        """在线实时预测修正循环"""
        logger.info("ONLINE_LOOP", "在线循环线程启动", module="main")
        interval = 1.0 / self.config.main_loop_frequency
        logger.debug("ONLINE_LOOP", f"主循环间隔: {interval}s", module="main")

        while self.running and not self._shutdown_event.is_set():
            loop_start = time.time()
            self.loop_count += 1
            self._update_loop_heartbeat()  # M5：更新心跳
            logger.debug("ONLINE_LOOP", f"开始第 {self.loop_count} 轮在线循环", module="main")

            try:
                # 1. 环境感知 —— 采集状态
                logger.debug("ONLINE_LOOP", "步骤1: 环境感知", module="main")
                with perf_monitor.track("perceive"):
                    state_vector = self.perception.get_state_vector()
                logger.info("ONLINE_LOOP", f"状态向量采集完成: {len(state_vector)} 维", module="main",
                           extra={"state_vector": state_vector})

                # 2. 多层预测 —— 预测未来状态
                logger.debug("ONLINE_LOOP", "步骤2: 多层预测", module="main")
                _cpu = float(state_vector.get(StateKeys.CPU_PERCENT) or 0.0) * 100
                _mem = float(state_vector.get(StateKeys.MEMORY_PERCENT) or 0.0) * 100
                _disk = float(state_vector.get(StateKeys.DISK_PERCENT) or 0.0) * 100
                self.prediction.update({
                    MetricKeys.CPU: _cpu,
                    MetricKeys.MEMORY: _mem,
                    MetricKeys.DISK: _disk,
                })
                with perf_monitor.track("predict"):
                    pred_result = self.prediction.predict()
                logger.info("ONLINE_LOOP", f"预测完成, 层={pred_result.layer}, 置信度={pred_result.confidence:.4f}",
                           module="main", extra={"predicted": pred_result.predicted_values})

                # 3. 误差评估 —— 计算预测误差
                logger.debug("ONLINE_LOOP", "步骤3: 误差评估", module="main")
                actual = {
                    MetricKeys.CPU: _cpu,
                    MetricKeys.MEMORY: _mem,
                    MetricKeys.DISK: _disk,
                }
                with perf_monitor.track("error_eval"):
                    error_report = self.error_eval.evaluate(pred_result.predicted_values, actual)
                logger.info("ONLINE_LOOP", f"误差评估完成, MSE={error_report.get('mse', 0):.4f}, MAE={error_report.get('mae', 0):.4f}",
                           module="main", extra=error_report)

                # 4. 在线优化 —— 根据误差修正模型
                logger.debug("ONLINE_LOOP", "步骤4: 在线优化", module="main")
                if error_report.get("reconstruction_signal", False):
                    logger.warning("ONLINE_LOOP", "检测到重构信号，触发在线优化", module="main")
                with perf_monitor.track("optimize"):
                    opt_result = self.optimizer.online_optimize(pred_result.predicted_values, actual, error_report)
                logger.info("ONLINE_LOOP", f"在线优化完成, 类型={opt_result.optimization_type}", module="main",
                           extra={"opt_result": opt_result.to_dict()})

                # 5. 驱力计算 —— 计算内在动机
                logger.debug("ONLINE_LOOP", "步骤5: 驱力计算", module="main")
                executor_stats = self.decision.executor.get_execution_stats()
                with perf_monitor.track("drive_compute"):
                    drive_state = self.drive_engine.compute(
                        state_vector=state_vector,
                        tasks_completed=executor_stats.get("recent_completed", 0),
                        tasks_failed=executor_stats.get("recent_failed", 0),
                        prediction_error=error_report.get("mae", 0.0)
                    )
                logger.info("ONLINE_LOOP",
                           f"驱力计算完成: 安全={drive_state.safety:.4f}, 收益={drive_state.gain:.4f}, 能效={drive_state.efficiency:.4f}, 动机={drive_state.intrinsic_motivation:.4f}",
                           module="main", extra=drive_state.to_dict())

                # 6. 自主决策 —— 选择并执行动作
                logger.debug("ONLINE_LOOP", "步骤6: 自主决策", module="main")
                drives = {DriveKeys.SAFE: drive_state.safety, DriveKeys.GAIN: drive_state.gain, DriveKeys.EFF: drive_state.efficiency}
                # H2：决策前把上下文注入执行器，使内部动作回调能拿到真实数据
                self._inject_action_context(pred_result.predicted_values, actual, error_report, state_vector)
                with perf_monitor.track("decide_execute"):
                    action_result = self.decision.decide_and_execute(state_vector, drives)
                logger.info("ONLINE_LOOP", f"决策执行完成: {action_result.action.name}, 成功={action_result.success}",
                           module="main", extra={"action_result": action_result.to_dict()})

                # H2：当驱力安全值过低或误差过大时，主动触发真实运维动作
                self._trigger_reactive_actions(drive_state, error_report, state_vector)

                # 7. 记忆存储 —— 存储经验
                logger.debug("ONLINE_LOOP", "步骤7: 记忆存储", module="main")
                reward = drive_state.intrinsic_motivation
                next_state = self.perception.get_state_vector()
                self.memory.record_experience(
                    state=state_vector,
                    action=action_result.action.name,
                    reward=reward,
                    next_state=next_state,
                    td_error=error_report.get("mae", 0.0)
                )
                logger.info("ONLINE_LOOP", f"经验存储完成, 奖励={reward:.4f}", module="main")

                # 8. 预测模型在线学习
                logger.debug("ONLINE_LOOP", "步骤8: 预测模型在线学习", module="main")
                self.prediction.online_learn(actual)
                logger.info("ONLINE_LOOP", "预测模型在线学习完成", module="main")

                # 9. 检查配置热重载
                logger.debug("ONLINE_LOOP", "步骤9: 检查配置热重载", module="main")
                if self.loop_count % 10 == 0:
                    reloaded = config_manager.check_hot_reload()
                    if reloaded:
                        logger.info("ONLINE_LOOP", "配置已热重载", module="main")

                # 10. H3：告警检查 —— 用状态向量检查告警规则
                logger.debug("ONLINE_LOOP", "步骤10: 告警检查", module="main")
                if hasattr(self, "alert_manager") and self.alert_manager is not None:
                    try:
                        triggered_alerts = self.alert_manager.check(state_vector)
                        if triggered_alerts:
                            logger.info("ONLINE_LOOP", f"触发 {len(triggered_alerts)} 条告警", module="main",
                                       extra={"alerts": [a.rule_name for a in triggered_alerts]})
                    except Exception as e:
                        logger.warning("ONLINE_LOOP", f"告警检查异常: {e}", module="main")

                # H6：定期输出性能报告
                report_interval = self.config.logging.perf_report_interval_loops
                if report_interval > 0 and self.loop_count % report_interval == 0:
                    try:
                        report = perf_monitor.report()
                        summary = perf_monitor.get_summary()
                        logger.info("PERF_REPORT", "性能报告", module="main",
                                   extra={"summary": summary})
                        # 控制台直接打印（INFO级别可见）
                        print(report)
                    except Exception as e:
                        logger.warning("ONLINE_LOOP", f"性能报告生成异常: {e}", module="main")

                logger.debug("ONLINE_LOOP", f"第 {self.loop_count} 轮在线循环完成", module="main")

            except AgentException as e:
                # 已知异常类型，根据模块分类处理
                logger.error("ONLINE_LOOP", f"在线循环已知异常 [{e.__class__.__name__}]: {str(e)}", module="main",
                            extra={"error": str(e), "loop_count": self.loop_count, "degradable": e.degradable,
                                   "remediation": getattr(e, "remediation", "")})
                if not e.degradable:
                    logger.critical("ONLINE_LOOP", "遇到不可降级异常，停止在线循环", module="main")
                    break
                time.sleep(1.0)
                continue
            except Exception as e:
                # U6：未知异常保留完整堆栈，便于排查
                logger.exception("ONLINE_LOOP", f"在线循环未知异常 [{type(e).__name__}]: {str(e)}", module="main",
                            extra={"error": str(e), "loop_count": self.loop_count,
                                   "exception_type": type(e).__name__})
                time.sleep(1.0)
                continue

            # 控制循环频率
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                logger.debug("ONLINE_LOOP", f"本轮耗时 {elapsed:.4f}s, 休眠 {sleep_time:.4f}s", module="main")
                time.sleep(sleep_time)
            else:
                logger.warning("ONLINE_LOOP", f"本轮耗时 {elapsed:.4f}s 超过间隔 {interval}s", module="main")

        logger.info("ONLINE_LOOP", "在线循环线程结束", module="main",
                   extra={"total_loops": self.loop_count})

    @log_execution
    def _offline_loop(self):
        """离线复盘模型进化循环"""
        logger.info("OFFLINE_LOOP", "离线循环线程启动", module="main")
        interval_hours = self.config.evolution.idle_replay_interval_hours
        interval_seconds = interval_hours * 3600
        logger.debug("OFFLINE_LOOP", f"离线循环间隔: {interval_hours}h ({interval_seconds}s)", module="main")

        # 测试/开发环境可通过环境变量缩短离线循环间隔
        # 生产环境不设置此变量，使用配置文件的完整间隔
        test_interval = os.environ.get("AGENT_OFFLINE_LOOP_INTERVAL_SEC")
        if test_interval:
            try:
                interval_seconds = float(test_interval)
                logger.debug("OFFLINE_LOOP", f"使用环境变量覆盖离线循环间隔: {interval_seconds}s", module="main")
            except ValueError:
                pass

        while self.running and not self._shutdown_event.is_set():
            logger.debug("OFFLINE_LOOP", "等待离线循环触发", module="main")
            # 使用事件等待，支持快速退出
            if self._shutdown_event.wait(timeout=interval_seconds):
                break

            if not self.running:
                break

            logger.info("OFFLINE_LOOP", "开始离线复盘", module="main")
            try:
                # 1. 离线优化
                logger.debug("OFFLINE_LOOP", "步骤1: 离线优化", module="main")
                offline_result = self.optimizer.offline_optimize()
                logger.info("OFFLINE_LOOP", f"离线优化完成, 类型={offline_result.optimization_type}", module="main",
                           extra={"offline_result": offline_result.to_dict()})

                # 2. 记忆复盘
                logger.debug("OFFLINE_LOOP", "步骤2: 记忆复盘", module="main")
                replay_report = self.memory.offline_review()
                logger.info("OFFLINE_LOOP", f"记忆复盘完成, 最佳动作={replay_report.get('best_action', 'N/A')}",
                           module="main", extra=replay_report)

                # 3. 生成系统健康报告
                logger.debug("OFFLINE_LOOP", "步骤3: 生成健康报告", module="main")
                health_report = self._generate_health_report()
                logger.info("OFFLINE_LOOP", "系统健康报告生成完成", module="main",
                           extra={"health_report": health_report})

                logger.info("OFFLINE_LOOP", "离线复盘完成", module="main")

            except AgentException as e:
                logger.error("OFFLINE_LOOP", f"离线循环已知异常 [{e.__class__.__name__}]: {str(e)}", module="main",
                            extra={"error": str(e), "degradable": e.degradable,
                                   "remediation": getattr(e, "remediation", "")})
                if not e.degradable:
                    logger.critical("OFFLINE_LOOP", "遇到不可降级异常，停止离线循环", module="main")
                    break
            except Exception as e:
                # U6：未知异常保留完整堆栈
                logger.exception("OFFLINE_LOOP", f"离线循环未知异常 [{type(e).__name__}]: {str(e)}", module="main",
                            extra={"error": str(e), "exception_type": type(e).__name__})

        logger.info("OFFLINE_LOOP", "离线循环线程结束", module="main")

    @log_execution
    def _generate_health_report(self) -> dict:
        """生成系统健康报告"""
        logger.debug("HEALTH_REPORT", "开始生成健康报告", module="main")
        report = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self.start_time if self.start_time else 0,
            "total_loops": self.loop_count,
            "drive_summary": self.drive_engine.get_drive_summary(),
            "decision_status": self.decision.get_status(),
            "prediction_confidence": self.prediction.get_confidence_report(),
            "error_status": self.error_eval.get_history_stats(),
            "memory_stats": self.memory.get_stats(),
            "optimizer_history_count": len(self.optimizer.get_history()),
            # H6：性能监控摘要
            "performance_summary": perf_monitor.get_summary(),
        }
        logger.info("HEALTH_REPORT", "健康报告生成完成", module="main", extra=report)
        return report

    def get_performance_stats(self) -> Dict[str, Any]:
        """H6：获取性能监控统计

        Returns:
            包含 summary 和 operations 的字典
        """
        return {
            "summary": perf_monitor.get_summary(),
            "operations": perf_monitor.get_stats(),
            "slow_operations": perf_monitor.get_slow_operations(),
        }

    def get_performance_report(self) -> str:
        """H6：获取可读的性能报告字符串"""
        return perf_monitor.report()

    # ---------- M1：多Agent编排接口 ----------

    def enable_multi_agent(self, spawn_default: bool = True) -> List[str]:
        """M1：启用多Agent编排能力，创建子Agent集群

        Args:
            spawn_default: 是否创建默认4个专用子Agent

        Returns:
            创建的子Agent ID列表
        """
        self._orchestrator_enabled = True
        ids = []
        if spawn_default:
            ids = self.orchestrator.spawn_default_agents()
        logger.info("M1_ORCH", f"多Agent编排已启用, 子Agent: {ids}", module="main")
        return ids

    def disable_multi_agent(self) -> None:
        """M1：禁用多Agent编排，停止所有子Agent"""
        self._orchestrator_enabled = False
        self.orchestrator.stop_all()
        logger.info("M1_ORCH", "多Agent编排已禁用", module="main")

    def delegate_to_agent(self, task_type: str, payload: Dict[str, Any],
                          timeout: Optional[float] = None) -> Dict[str, Any]:
        """M1：向子Agent委派任务

        Args:
            task_type: 任务类型 (perceive/predict/decide/execute/analyze)
            payload: 任务参数
            timeout: 超时秒数

        Returns:
            委派结果字典
        """
        try:
            tt = TaskType(task_type)
        except ValueError:
            return {"success": False, "error": f"未知任务类型: {task_type}"}

        if not self._orchestrator_enabled:
            # 自动启用
            self.enable_multi_agent(spawn_default=True)

        with perf_monitor.track("delegate_to_agent"):
            result = self.orchestrator.delegate_task(tt, payload, timeout=timeout)
        return result.to_dict()

    def get_cluster_status(self) -> Dict[str, Any]:
        """M1：获取多Agent集群状态"""
        return self.orchestrator.get_cluster_status()

    def recover_cluster(self) -> int:
        """M1：自动恢复故障子Agent"""
        return self.orchestrator.auto_recover()

    # ---------- M2：任务分解与工作流接口 ----------

    def _register_workflow_handlers(self) -> None:
        """M2：注册工作流任务处理器，连接workflow_engine与orchestrator/模块"""
        # perceive任务：委派给感知子Agent或直接调用主感知模块
        def handle_perceive(payload: Dict[str, Any]) -> Dict[str, Any]:
            if self._orchestrator_enabled:
                result = self.orchestrator.delegate_task(TaskType.PERCEIVE, payload, timeout=3.0)
                return result.to_dict()
            # 回退：直接调用主感知模块
            return {"state": self.perception.get_state_vector(), "source": "main"}

        # predict任务
        def handle_predict(payload: Dict[str, Any]) -> Dict[str, Any]:
            if self._orchestrator_enabled:
                result = self.orchestrator.delegate_task(TaskType.PREDICT, payload, timeout=3.0)
                return result.to_dict()
            try:
                pred = self.prediction.predict()
                return {"predicted": pred.predicted_values, "source": "main"}
            except Exception as e:
                return {"error": str(e), "source": "main"}

        # decide任务
        def handle_decide(payload: Dict[str, Any]) -> Dict[str, Any]:
            if self._orchestrator_enabled:
                result = self.orchestrator.delegate_task(TaskType.DECIDE, payload, timeout=3.0)
                return result.to_dict()
            return {"decision": "noop", "source": "main"}

        # analyze任务
        def handle_analyze(payload: Dict[str, Any]) -> Dict[str, Any]:
            state = payload.get("state", self.perception.get_state_vector())
            return {"analysis": "completed", "state_snapshot": state, "source": "main"}

        # alert任务
        def handle_alert(payload: Dict[str, Any]) -> Dict[str, Any]:
            try:
                self.alert_manager.dispatch_alert(
                    rule_name="workflow_alert",
                    level="warning",
                    message=payload.get("message", "工作流触发告警"),
                    metric_value=payload.get("value", 0),
                    threshold=payload.get("threshold", 0),
                )
                return {"alerted": True}
            except Exception:
                # 回退：直接记录
                logger.warning("M2_WORKFLOW", f"工作流告警: {payload}", module="main")
                return {"alerted": True, "fallback": True}

        # clean_cache任务
        def handle_clean_cache(payload: Dict[str, Any]) -> Dict[str, Any]:
            from modules.decision import Action, ActionType
            action = Action(ActionType.CLEAN_CACHE, "工作流清理", "M2",
                            params=payload)
            result = self.decision.executor.execute(action)
            return {"success": result.success, "message": result.message}

        # optimize任务
        def handle_optimize(payload: Dict[str, Any]) -> Dict[str, Any]:
            try:
                actual = payload.get("actual", {})
                predicted = payload.get("predicted", {})
                error_report = self.error_eval.evaluate(predicted, actual)
                opt = self.optimizer.online_optimize(predicted, actual, error_report)
                return {"optimization_type": opt.optimization_type, "source": "main"}
            except Exception as e:
                return {"error": str(e)}

        # custom任务
        def handle_custom(payload: Dict[str, Any]) -> Dict[str, Any]:
            return {"executed": True, "payload": payload}

        for ttype, handler in [
            ("perceive", handle_perceive),
            ("predict", handle_predict),
            ("decide", handle_decide),
            ("analyze", handle_analyze),
            ("alert", handle_alert),
            ("clean_cache", handle_clean_cache),
            ("optimize", handle_optimize),
            ("custom", handle_custom),
        ]:
            self.workflow_engine.register_task_handler(ttype, handler)

        logger.info("M2_WORKFLOW", "工作流任务处理器已注册", module="main")

    def submit_goal(self, goal_type: str, description: str = "",
                    params: Optional[Dict[str, Any]] = None) -> str:
        """M2：提交高层目标，自动分解为工作流并执行

        Args:
            goal_type: 目标类型 (monitor_and_alert/optimize_resource/clean_and_recover/analyze_and_decide/custom)
            description: 目标描述
            params: 目标参数

        Returns:
            workflow_id
        """
        try:
            gt = GoalType(goal_type)
        except ValueError:
            gt = GoalType.CUSTOM

        goal = Goal(goal_type=gt, description=description, params=params or {})
        workflow = self.task_decomposer.decompose(goal)

        # 确保工作流引擎已启动
        if not self._workflow_engine_started:
            self.workflow_engine.start()
            self._workflow_engine_started = True

        wf_id = self.workflow_engine.submit_workflow(workflow)
        logger.info("M2_GOAL", f"目标已提交: {goal_type} -> 工作流 {wf_id}", module="main")
        return wf_id

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """M2：获取工作流状态"""
        return self.workflow_engine.get_workflow_status(workflow_id)

    def get_workflow_stats(self) -> Dict[str, Any]:
        """M2：获取工作流引擎统计"""
        stats = self.workflow_engine.get_stats()
        stats["decomposer_stats"] = self.task_decomposer.get_stats()
        return stats

    def wait_for_workflow(self, workflow_id: str, timeout: float = 30.0) -> Dict[str, Any]:
        """M2：等待工作流完成

        Args:
            workflow_id: 工作流ID
            timeout: 超时秒数

        Returns:
            工作流最终状态
        """
        start = time.time()
        while time.time() - start < timeout:
            status = self.workflow_engine.get_workflow_status(workflow_id)
            if status is None:
                return {"error": "工作流不存在"}
            state = status.get("state", "")
            if state in ("completed", "failed", "partial"):
                # M3：工作流完成后自动学习技能
                self._learn_skill_from_workflow(workflow_id, status)
                return status
            time.sleep(0.2)
        return {"error": "超时", "state": "timeout"}

    # ---------- M3：技能库与持续学习接口 ----------

    def _learn_skill_from_workflow(self, workflow_id: str, status: Dict[str, Any]) -> None:
        """M3：从完成的工作流中提炼技能

        第一性原理：成功的工作流模式应被提炼为可复用技能
        """
        try:
            tasks = status.get("tasks", {})
            if not tasks:
                return

            # 提取任务步骤
            steps = []
            task_types = set()
            total_elapsed = 0.0
            all_success = True
            for tid, t in tasks.items():
                steps.append(f"{t.get('name', tid)}: {t.get('task_type', '')}")
                task_types.add(t.get("task_type", ""))
                if t.get("state") != "success":
                    all_success = False
                if t.get("started_at") and t.get("completed_at"):
                    total_elapsed += t["completed_at"] - t["started_at"]

            # 为每种任务类型学习技能
            for tt in task_types:
                if tt:
                    self.skill_library.learn_from_workflow(
                        workflow_id=workflow_id,
                        task_type=tt,
                        steps=steps,
                        success=all_success,
                        elapsed=total_elapsed,
                    )
                    self._skills_learned_this_session += 1

            logger.info("M3_SKILL", f"从工作流 {workflow_id} 学习技能, 成功={all_success}",
                        module="main")
        except Exception as e:
            logger.warning("M3_SKILL", f"技能学习异常: {e}", module="main")

    def get_skill(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """M3：获取技能详情"""
        skill = self.skill_library.get(skill_id)
        return skill.to_dict() if skill else None

    def find_skills_for_task(self, task_type: str) -> List[Dict[str, Any]]:
        """M3：查找适合某任务类型的技能"""
        skills = self.skill_library.find_by_task_type(task_type)
        return [s.to_dict() for s in skills]

    def find_best_skill(self, task_type: str) -> Optional[Dict[str, Any]]:
        """M3：找到最适合某任务类型的技能"""
        skill = self.skill_library.find_best_skill(task_type)
        return skill.to_dict() if skill else None

    def record_skill_use(self, skill_id: str, success: bool, elapsed: float) -> None:
        """M3：记录技能使用结果（用于技能评估与进化）"""
        self.skill_library.record_use(skill_id, success, elapsed)

    def evolve_skill(self, skill_id: str, new_steps: List[str],
                     new_params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """M3：进化技能（增强自己的技能）"""
        skill = self.skill_library.evolve_skill(skill_id, new_steps, new_params)
        return skill.to_dict() if skill else None

    def get_skill_library_stats(self) -> Dict[str, Any]:
        """M3：获取技能库统计"""
        return self.skill_library.get_stats()

    def init_continual_learning(self, input_dim: int = 4, output_dim: int = 2) -> None:
        """M3：初始化持续学习控制器（防止技能遗忘）

        Args:
            input_dim: 输入维度（状态向量维度）
            output_dim: 输出维度（动作类别数）
        """
        try:
            from core.continual_learning import ContinualLearningController
            self._continual_learner = ContinualLearningController(
                input_dim=input_dim, output_dim=output_dim,
                strategy="ewc_replay",
            )
            logger.info("M3_CL", "持续学习控制器已初始化", module="main")
        except Exception as e:
            logger.warning("M3_CL", f"持续学习初始化失败: {e}", module="main")

    def init_meta_learning(self, input_dim: int = 4, output_dim: int = 2) -> None:
        """M3：初始化元学习控制器（学会快速适应新任务）"""
        try:
            from core.meta_learning import MetaLearningController
            self._meta_learner = MetaLearningController(
                input_dim=input_dim, output_dim=output_dim,
            )
            logger.info("M3_ML", "元学习控制器已初始化", module="main")
        except Exception as e:
            logger.warning("M3_ML", f"元学习初始化失败: {e}", module="main")

    def learn_task_continual(self, task_id: str, data: Any, labels: Any) -> Dict[str, Any]:
        """M3：持续学习一个新任务（不忘旧任务）"""
        if self._continual_learner is None:
            self.init_continual_learning()
        if self._continual_learner is None:
            return {"error": "持续学习未初始化"}
        try:
            result = self._continual_learner.learn_task(task_id, data, labels)
            logger.info("M3_CL", f"持续学习任务: {task_id}", module="main")
            return result
        except Exception as e:
            return {"error": str(e)}

    def meta_train(self, n_iterations: int = 10) -> Dict[str, Any]:
        """M3：元学习训练（学会学习）"""
        if self._meta_learner is None:
            self.init_meta_learning()
        if self._meta_learner is None:
            return {"error": "元学习未初始化"}
        try:
            result = self._meta_learner.meta_train(n_iterations=n_iterations)
            logger.info("M3_ML", f"元学习训练完成: {n_iterations}步", module="main")
            return result
        except Exception as e:
            return {"error": str(e)}

    # ---------- M4：长任务规划与自驱执行接口 ----------

    def plan_long_task(self, template_name: str, name: str = "",
                       description: str = "") -> str:
        """M4：规划长任务

        Args:
            template_name: 模板名 (system_optimization/continuous_monitoring/fault_recovery/capability_enhancement)
            name: 任务名
            description: 描述

        Returns:
            task_id
        """
        task = self.long_task_planner.plan_task(template_name, name, description)
        logger.info("M4_LT", f"长任务已规划: {task.task_id} [{template_name}], 子目标={len(task.subgoals)}",
                    module="main")
        return task.task_id

    def advance_long_task(self, task_id: Optional[str] = None,
                          drive_level: Optional[float] = None) -> Dict[str, Any]:
        """M4：自驱推进长任务 —— 执行下一个子目标

        第一性原理：驱力足够时自动推进，无需外部触发

        Args:
            task_id: 长任务ID，None则用当前活跃任务
            drive_level: 驱力水平[0,1]，None则从驱力引擎获取

        Returns:
            推进结果
        """
        tid = task_id or self._active_long_task_id
        if tid is None:
            return {"error": "无活跃长任务"}

        # 获取驱力水平
        if drive_level is None:
            try:
                drive_summary = self.drive_engine.get_drive_summary()
                # 用内在动机作为驱力水平
                drive_level = drive_summary.get("intrinsic_motivation", 0.5)
            except Exception:
                drive_level = 0.5

        # 子目标执行函数：通过M2工作流引擎执行
        def execute_subgoal(subgoal: SubGoal) -> Dict[str, Any]:
            wf_id = self.submit_goal(subgoal.goal_type, subgoal.description, subgoal.params)
            subgoal.workflow_id = wf_id
            return self.wait_for_workflow(wf_id, timeout=15)

        with perf_monitor.track("advance_long_task"):
            result = self.long_task_planner.advance_task(tid, drive_level, execute_subgoal)
        return result

    def self_drive_long_task(self, task_id: str, max_steps: int = 10,
                             drive_level: float = 0.8) -> Dict[str, Any]:
        """M4：自驱完成整个长任务 —— 自动推进直到完成或失败

        Args:
            task_id: 长任务ID
            max_steps: 最大推进次数
            drive_level: 驱力水平

        Returns:
            最终结果
        """
        self._active_long_task_id = task_id
        for i in range(max_steps):
            result = self.advance_long_task(task_id, drive_level)
            if result.get("completed"):
                logger.info("M4_LT", f"长任务自驱完成: {task_id}", module="main")
                self._active_long_task_id = None
                return result
            if result.get("failed"):
                logger.warning("M4_LT", f"长任务自驱失败: {task_id}", module="main")
                self._active_long_task_id = None
                return result
            if result.get("skipped"):
                # 驱力不足，等待
                time.sleep(0.5)
        return {"error": "超过最大步数", "steps": max_steps}

    def get_long_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """M4：获取长任务状态"""
        task = self.long_task_planner.get_task(task_id)
        return task.to_dict() if task else None

    def pause_long_task(self, task_id: str) -> bool:
        """M4：暂停长任务"""
        return self.long_task_planner.pause_task(task_id)

    def resume_long_task(self, task_id: str) -> bool:
        """M4：恢复长任务"""
        return self.long_task_planner.resume_task(task_id)

    def list_long_tasks(self) -> List[Dict[str, Any]]:
        """M4：列出所有长任务"""
        return [t.to_dict() for t in self.long_task_planner.list_tasks()]

    def get_long_task_stats(self) -> Dict[str, Any]:
        """M4：获取长任务统计"""
        return self.long_task_planner.get_stats()

    # ---------- M5：状态持久化与故障恢复接口 ----------

    def save_state(self) -> Optional[str]:
        """M5：保存当前Agent状态到快照

        Returns:
            快照文件路径，失败返回None
        """
        try:
            snapshot = AgentStateSerializer.serialize_full_agent(self)
            # M3/M4扩展：保存技能库和长任务状态
            snapshot.set_data("skill_library", self.skill_library.save())
            snapshot.set_data("long_tasks", [t.to_dict() for t in self.long_task_planner.list_tasks()])
            snapshot.set_data("loop_count", self.loop_count)
            path = self.persistence.save_snapshot(snapshot)
            logger.info("M5_PERSIST", f"状态已保存: {path}", module="main")
            return path
        except Exception as e:
            logger.error("M5_PERSIST", f"状态保存失败: {e}", module="main")
            return None

    def load_state(self, path: Optional[str] = None) -> bool:
        """M5：从快照恢复Agent状态

        Args:
            path: 快照路径，None则从最新快照恢复

        Returns:
            是否成功
        """
        try:
            if path:
                snapshot = self.persistence.load_snapshot(path)
            else:
                snapshot = self.persistence.restore_from_latest()
            if snapshot is None:
                logger.warning("M5_PERSIST", "无可用快照", module="main")
                return False

            success = AgentStateSerializer.deserialize_full_agent(self, snapshot)
            # 恢复loop_count
            self.loop_count = snapshot.get_data("loop_count", self.loop_count)
            logger.info("M5_PERSIST", f"状态已恢复, loop_count={self.loop_count}", module="main")
            return success
        except Exception as e:
            logger.error("M5_PERSIST", f"状态恢复失败: {e}", module="main")
            return False

    def list_snapshots(self) -> List[str]:
        """M5：列出所有可用快照"""
        return self.persistence.list_snapshots()

    def _start_watchdog(self) -> None:
        """M5：启动看门狗线程 —— 监控主循环健康"""
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="Watchdog", daemon=True
        )
        self._watchdog_thread.start()
        logger.info("M5_WATCHDOG", "看门狗已启动", module="main")

    def _watchdog_loop(self) -> None:
        """看门狗循环 —— 检测主循环卡死并触发恢复"""
        while self.running:
            time.sleep(5.0)
            if not self.running:
                break
            elapsed = time.time() - self._last_loop_heartbeat
            if elapsed > self._watchdog_timeout:
                logger.error("M5_WATCHDOG",
                             f"主循环卡死 {elapsed:.0f}s, 触发恢复",
                             module="main")
                if self._auto_recovery_enabled:
                    self._recover_from_stall()
                # 重置心跳避免重复触发
                self._last_loop_heartbeat = time.time()

    def _recover_from_stall(self) -> None:
        """M5：从卡死状态恢复"""
        logger.warning("M5_WATCHDOG", "执行卡死恢复", module="main")
        try:
            # 尝试保存当前状态
            self.save_state()
            # 重启主循环线程
            if self._main_thread and not self._main_thread.is_alive():
                logger.info("M5_WATCHDOG", "重启主循环线程", module="main")
                self._main_thread = threading.Thread(
                    target=self._online_loop, name="OnlineLoop-Recovered", daemon=True
                )
                self._main_thread.start()
        except Exception as e:
            logger.error("M5_WATCHDOG", f"恢复失败: {e}", module="main")

    def _update_loop_heartbeat(self) -> None:
        """M5：更新主循环心跳（每轮循环调用）

        U3：同时推送状态到Dashboard（如果已启用）
        """
        self._last_loop_heartbeat = time.time()
        # U3：推送状态到Dashboard
        if self._dashboard_enabled:
            try:
                self._push_dashboard_state()
            except Exception as e:
                logger.debug("DASHBOARD", f"状态推送失败: {e}", module="main")

    def _push_dashboard_state(self) -> None:
        """U3：推送当前状态到Dashboard"""
        from dashboard.dashboard_server import update_state
        state = self.perception.get_state_vector()
        update_state(
            cpu_percent=state.get(StateKeys.CPU_PERCENT, 0.0),
            memory_percent=state.get(StateKeys.MEMORY_PERCENT, 0.0),
            disk_percent=state.get(StateKeys.DISK_PERCENT, 0.0),
            loop_count=self.loop_count,
        )
        # 推送驱力状态
        try:
            drives = self.drive_engine.compute(state, 0, 0, 0.0)
            update_state(
                safety_drive=drives.safety,
                gain_drive=drives.gain,
                efficiency_drive=drives.efficiency,
            )
        except Exception:
            pass
        # 推送预测置信度
        try:
            pred = self.prediction.predict()
            update_state(prediction_confidence=pred.confidence)
        except Exception:
            pass

    def enable_dashboard(self, port: int = 8080) -> None:
        """U3：启用Dashboard监控面板

        Args:
            port: Dashboard端口
        """
        if self._dashboard_enabled:
            return
        from dashboard.dashboard_server import DashboardServer
        self.dashboard = DashboardServer(port=port)
        self.dashboard.start()
        self._dashboard_enabled = True
        logger.info("U3_DASHBOARD", f"Dashboard已启用: http://localhost:{port}", module="main")

    def disable_dashboard(self) -> None:
        """U3：禁用Dashboard"""
        if not self._dashboard_enabled:
            return
        if self.dashboard:
            self.dashboard.stop()
        self._dashboard_enabled = False
        logger.info("U3_DASHBOARD", "Dashboard已禁用", module="main")

    def set_auto_recovery(self, enabled: bool) -> None:
        """M5：启用/禁用自动恢复"""
        self._auto_recovery_enabled = enabled
        logger.info("M5_WATCHDOG", f"自动恢复 {'启用' if enabled else '禁用'}", module="main")

    def get_persistence_stats(self) -> Dict[str, Any]:
        """M5：获取持久化统计"""
        return {
            "snapshot_count": len(self.persistence.list_snapshots()),
            "snapshot_dir": self.persistence.snapshot_dir,
            "watchdog_running": (self._watchdog_thread is not None and
                                 self._watchdog_thread.is_alive()),
            "last_heartbeat_age": time.time() - self._last_loop_heartbeat,
            "auto_recovery_enabled": self._auto_recovery_enabled,
        }

    # ---------- M6：永续运行保障接口 ----------

    def enable_forever_running(self, health_check_interval: float = 30.0,
                               resource_cleanup_interval: float = 300.0) -> None:
        """M6：启用永续运行 —— 健康监控+资源回收+分级自愈

        第一性原理：永续=故障可自愈+资源可回收+健康可监控

        Args:
            health_check_interval: 健康检查间隔（秒）
            resource_cleanup_interval: 资源回收间隔（秒）
        """
        if self._forever_running_enabled:
            return
        # 重新配置间隔
        self.forever_runner._health_interval = health_check_interval
        self.forever_runner._cleanup_interval = resource_cleanup_interval
        self._forever_running_enabled = True
        # 若Agent已启动，立即启动永续运行器
        if self.running:
            self.forever_runner.attach(self)
            self.forever_runner.start()
        logger.info("M6_FOREVER", "永续运行已启用", module="main")

    def disable_forever_running(self) -> None:
        """M6：禁用永续运行"""
        if not self._forever_running_enabled:
            return
        self.forever_runner.stop()
        self._forever_running_enabled = False
        logger.info("M6_FOREVER", "永续运行已禁用", module="main")

    def get_health_report(self) -> Optional[Dict[str, Any]]:
        """M6：获取最新健康报告

        Returns:
            健康报告字典，包含level/score/issues/recommendations/metrics
        """
        # 确保forever_runner附加了agent（即使未启用永续运行也能查询健康）
        if self.forever_runner._agent is None:
            self.forever_runner.attach(self)
        # 主动触发一次健康检查，确保报告是最新的
        report = self.forever_runner.check_health()
        return report.to_dict()

    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """M6：获取健康历史记录"""
        return self.forever_runner.get_health_history(limit)

    def get_forever_stats(self) -> Dict[str, Any]:
        """M6：获取永续运行统计

        Returns:
            包含运行时长/重启次数/自愈次数/清理次数等
        """
        return self.forever_runner.get_stats()

    def trigger_self_heal(self) -> Dict[str, Any]:
        """M6：手动触发自愈（用于测试和紧急恢复）

        Returns:
            自愈前的健康报告
        """
        report = self.forever_runner.check_health()
        self.forever_runner._self_heal(report)
        return report.to_dict()

    def trigger_resource_cleanup(self) -> int:
        """M6：手动触发资源回收

        Returns:
            清理计数
        """
        self.forever_runner._cleanup_resources()
        return self.forever_runner._cleanup_count

    @log_execution
    def start(self):
        """启动Agent系统

        O2：统一启动顺序（按依赖关系从底层到上层）：
        1. 设置运行状态
        2. 启动感知模块（底层依赖，提供数据源）
        3. 启动在线循环线程（核心业务循环）
        4. 启动离线循环线程（复盘进化）
        5. 启动看门狗（监控主循环健康）
        6. 启动永续运行器（健康监控+资源回收，仅显式启用时）
        """
        logger.info("AGENT_START", "启动自驱Agent系统", module="main")
        self.running = True
        self._stopped = False
        self.start_time = time.time()
        self._shutdown_event.clear()
        logger.debug("AGENT_START", "运行状态已设置", module="main")

        # O2：启动感知模块（如果模块支持start方法）
        if hasattr(self.perception, "start") and callable(self.perception.start):
            try:
                self.perception.start()
                logger.debug("AGENT_START", "感知模块已启动", module="main")
            except Exception as e:
                logger.warning("AGENT_START", f"感知模块启动失败（降级为同步采集）: {e}", module="main")

        # 启动在线循环线程
        logger.debug("AGENT_START", "启动在线循环线程", module="main")
        self._main_thread = threading.Thread(target=self._online_loop, name="OnlineLoop", daemon=True)
        self._main_thread.start()
        logger.info("AGENT_START", "在线循环线程已启动", module="main")

        # 启动离线循环线程
        logger.debug("AGENT_START", "启动离线循环线程", module="main")
        self._offline_thread = threading.Thread(target=self._offline_loop, name="OfflineLoop", daemon=True)
        self._offline_thread.start()
        logger.info("AGENT_START", "离线循环线程已启动", module="main")

        # M5：启动看门狗
        self._last_loop_heartbeat = time.time()
        self._start_watchdog()

        # M6：启动永续运行器（仅在显式启用时）
        if self._forever_running_enabled:
            self.forever_runner.attach(self)
            self.forever_runner.start()
            logger.info("AGENT_START", "永续运行器已启动", module="main")

        logger.info("AGENT_START", "自驱Agent系统启动完成", module="main")

    @log_execution
    def stop(self):
        """停止Agent系统

        O2：统一关闭顺序（与启动顺序相反，从上层到底层）：
        1. 设置停止信号（running=False, shutdown_event.set()）
        2. 停止永续运行器（M6，最上层监控）
        3. 停止工作流引擎（M2）
        4. 停止多Agent集群（M1）
        5. 等待在线/离线循环线程结束
        6. 停止看门狗线程（M5）
        7. 关闭记忆系统
        8. 停止感知模块（底层依赖，最后关闭）
        9. 清理DI容器（如果存在）
        """
        # 幂等保护：防止重复调用导致资源重复释放
        if not self.running and self._stopped:
            logger.debug("AGENT_STOP", "stop()已调用过，跳过重复执行", module="main")
            return
        self._stopped = True

        logger.info("AGENT_STOP", "停止自驱Agent系统", module="main")
        self.running = False
        self._shutdown_event.set()
        logger.debug("AGENT_STOP", "停止信号已发送", module="main")

        # O2：停止永续运行器（M6，最上层监控）
        if self._forever_running_enabled and self.forever_runner is not None:
            logger.debug("AGENT_STOP", "停止永续运行器", module="main")
            self.forever_runner.stop()
            self._forever_running_enabled = False

        # U3：停止Dashboard
        if self._dashboard_enabled and self.dashboard is not None:
            logger.debug("AGENT_STOP", "停止Dashboard", module="main")
            self.dashboard.stop()
            self._dashboard_enabled = False

        # O2：停止工作流引擎（M2）
        if self._workflow_engine_started:
            logger.debug("AGENT_STOP", "停止工作流引擎", module="main")
            self.workflow_engine.stop()
            self._workflow_engine_started = False

        # O2：停止多Agent集群（M1）
        if self._orchestrator_enabled:
            logger.debug("AGENT_STOP", "停止多Agent集群", module="main")
            self.orchestrator.stop_all()
            self._orchestrator_enabled = False

        # 等待核心循环线程结束
        if self._main_thread and self._main_thread.is_alive():
            logger.debug("AGENT_STOP", "等待在线循环线程结束", module="main")
            self._main_thread.join(timeout=10.0)
            if self._main_thread.is_alive():
                logger.warning("AGENT_STOP", "在线循环线程未在10秒内结束", module="main")

        if self._offline_thread and self._offline_thread.is_alive():
            logger.debug("AGENT_STOP", "等待离线循环线程结束", module="main")
            self._offline_thread.join(timeout=10.0)
            if self._offline_thread.is_alive():
                logger.warning("AGENT_STOP", "离线循环线程未在10秒内结束", module="main")

        # O2：停止看门狗线程（M5）—— 原来未join，依赖daemon退出
        # 看门狗是daemon线程，设置running=False后会在下次循环检查时退出
        # 这里不需要显式join，因为daemon线程会随进程退出

        # 关闭记忆系统
        logger.debug("AGENT_STOP", "关闭记忆系统", module="main")
        self.memory.close()

        # O2：停止感知模块（底层依赖，最后关闭）
        if hasattr(self.perception, "stop") and callable(self.perception.stop):
            try:
                self.perception.stop()
                logger.debug("AGENT_STOP", "感知模块已停止", module="main")
            except Exception as e:
                logger.warning("AGENT_STOP", f"感知模块停止异常: {e}", module="main")

        # O2：清理DI容器（如果存在）
        if self._container is not None:
            try:
                if hasattr(self._container, "cleanup_all"):
                    self._container.cleanup_all()
                    logger.debug("AGENT_STOP", "DI容器已清理", module="main")
            except Exception as e:
                logger.warning("AGENT_STOP", f"DI容器清理异常: {e}", module="main")

        logger.info("AGENT_STOP", "自驱Agent系统已停止", module="main",
                   extra={"total_loops": self.loop_count,
                          "uptime": time.time() - self.start_time if self.start_time else 0})

    @log_execution
    def run_for_duration(self, duration_seconds: float = 30.0):
        """运行指定时长后自动停止（用于测试）"""
        logger.info("AGENT_RUN_FOR", f"计划运行 {duration_seconds}s", module="main")
        self.start()
        logger.debug("AGENT_RUN_FOR", "系统已启动，等待运行", module="main")
        time.sleep(duration_seconds)
        logger.info("AGENT_RUN_FOR", f"运行时间到达 {duration_seconds}s，准备停止", module="main")
        self.stop()
        logger.info("AGENT_RUN_FOR", "运行结束", module="main")


# 全局Agent实例
_agent_instance: SelfEvolvingAgent = None


def signal_handler(signum, frame):
    """信号处理 —— 优雅退出"""
    logger.info("SIGNAL", f"收到信号 {signum}，准备优雅退出", module="main")
    global _agent_instance
    if _agent_instance:
        _agent_instance.stop()
    sys.exit(0)


@log_execution
def main():
    """主入口函数

    U1：CLI框架重构 —— 委托给cli.py的cli_main()
    支持子命令：run / status / health / snapshot / config
    默认行为：无参数=显示帮助（而非30秒测试模式）
    """
    from cli import cli_main
    exit_code = cli_main()
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
