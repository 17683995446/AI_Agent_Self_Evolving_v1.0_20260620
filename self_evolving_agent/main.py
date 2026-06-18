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
from typing import Dict, Any, Optional

# 仅在直接运行时添加路径（避免副作用）
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.logger import logger, log_execution
from core.config import config_manager
from core.interfaces import StateKeys, MetricKeys, DriveKeys
from core.registry import ModuleRegistry, get_default_registry
from core.container import DIContainer
from core.exceptions import (
    AgentException, PerceptionException, PredictionException,
    DriveEngineException, DecisionException, ErrorEvalException,
    OptimizerException, MemoryException, ModuleInitException,
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
        self._main_thread = None
        self._offline_thread = None
        self._shutdown_event = threading.Event()
        logger.debug("AGENT_INIT", "运行状态初始化完成", module="main")

        # 统计
        self.loop_count = 0
        self.start_time = None
        logger.info("AGENT_INIT", "自驱Agent系统初始化完成", module="main")

    @log_execution
    def _online_loop(self):
        """在线实时预测修正循环"""
        logger.info("ONLINE_LOOP", "在线循环线程启动", module="main")
        interval = 1.0 / self.config.main_loop_frequency
        logger.debug("ONLINE_LOOP", f"主循环间隔: {interval}s", module="main")

        while self.running and not self._shutdown_event.is_set():
            loop_start = time.time()
            self.loop_count += 1
            logger.debug("ONLINE_LOOP", f"开始第 {self.loop_count} 轮在线循环", module="main")

            try:
                # 1. 环境感知 —— 采集状态
                logger.debug("ONLINE_LOOP", "步骤1: 环境感知", module="main")
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
                error_report = self.error_eval.evaluate(pred_result.predicted_values, actual)
                logger.info("ONLINE_LOOP", f"误差评估完成, MSE={error_report.get('mse', 0):.4f}, MAE={error_report.get('mae', 0):.4f}",
                           module="main", extra=error_report)

                # 4. 在线优化 —— 根据误差修正模型
                logger.debug("ONLINE_LOOP", "步骤4: 在线优化", module="main")
                if error_report.get("reconstruction_signal", False):
                    logger.warning("ONLINE_LOOP", "检测到重构信号，触发在线优化", module="main")
                opt_result = self.optimizer.online_optimize(pred_result.predicted_values, actual, error_report)
                logger.info("ONLINE_LOOP", f"在线优化完成, 类型={opt_result.optimization_type}", module="main",
                           extra={"opt_result": opt_result.to_dict()})

                # 5. 驱力计算 —— 计算内在动机
                logger.debug("ONLINE_LOOP", "步骤5: 驱力计算", module="main")
                executor_stats = self.decision.executor.get_execution_stats()
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
                action_result = self.decision.decide_and_execute(state_vector, drives)
                logger.info("ONLINE_LOOP", f"决策执行完成: {action_result.action.name}, 成功={action_result.success}",
                           module="main", extra={"action_result": action_result.to_dict()})

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

                logger.debug("ONLINE_LOOP", f"第 {self.loop_count} 轮在线循环完成", module="main")

            except AgentException as e:
                # 已知异常类型，根据模块分类处理
                logger.error("ONLINE_LOOP", f"在线循环已知异常 [{e.__class__.__name__}]: {str(e)}", module="main",
                            extra={"error": str(e), "loop_count": self.loop_count, "degradable": e.degradable})
                if not e.degradable:
                    logger.critical("ONLINE_LOOP", "遇到不可降级异常，停止在线循环", module="main")
                    break
                time.sleep(1.0)
                continue
            except Exception as e:
                # 未知异常，记录详细信息并继续
                logger.error("ONLINE_LOOP", f"在线循环未知异常: {str(e)}", module="main",
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

        # 为了测试，缩短离线循环间隔
        interval_seconds = min(interval_seconds, 60.0)
        logger.debug("OFFLINE_LOOP", f"实际离线循环间隔: {interval_seconds}s (测试模式)", module="main")

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
                            extra={"error": str(e), "degradable": e.degradable})
                if not e.degradable:
                    logger.critical("OFFLINE_LOOP", "遇到不可降级异常，停止离线循环", module="main")
                    break
            except Exception as e:
                logger.error("OFFLINE_LOOP", f"离线循环未知异常: {str(e)}", module="main",
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
        }
        logger.info("HEALTH_REPORT", "健康报告生成完成", module="main", extra=report)
        return report

    @log_execution
    def start(self):
        """启动Agent系统"""
        logger.info("AGENT_START", "启动自驱Agent系统", module="main")
        self.running = True
        self.start_time = time.time()
        self._shutdown_event.clear()
        logger.debug("AGENT_START", "运行状态已设置", module="main")

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

        logger.info("AGENT_START", "自驱Agent系统启动完成", module="main")

    @log_execution
    def stop(self):
        """停止Agent系统"""
        # 幂等保护：防止重复调用导致资源重复释放
        if not self.running and hasattr(self, '_stopped') and self._stopped:
            logger.debug("AGENT_STOP", "stop()已调用过，跳过重复执行", module="main")
            return
        self._stopped = True

        logger.info("AGENT_STOP", "停止自驱Agent系统", module="main")
        self.running = False
        self._shutdown_event.set()
        logger.debug("AGENT_STOP", "停止信号已发送", module="main")

        # 等待线程结束
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

        # 关闭记忆系统
        logger.debug("AGENT_STOP", "关闭记忆系统", module="main")
        self.memory.close()

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
    """主入口函数"""
    logger.info("MAIN", "===== 自驱永续进化预测修正Agent系统启动 =====", module="main")
    logger.info("MAIN", f"当前时间: {datetime.now().isoformat()}", module="main")
    logger.info("MAIN", f"配置文件: {config_manager.config_path}", module="main")
    logger.info("MAIN", f"日志目录: {config_manager.config.log_dir}", module="main")
    logger.info("MAIN", f"数据库路径: {config_manager.config.database_path}", module="main")
    logger.info("MAIN", f"模型快照目录: {config_manager.config.model_snapshot_dir}", module="main")

    # 注册信号处理
    logger.debug("MAIN", "注册信号处理器", module="main")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.debug("MAIN", "信号处理器注册完成", module="main")

    # 创建并启动Agent
    global _agent_instance
    _agent_instance = SelfEvolvingAgent()
    logger.debug("MAIN", "Agent实例创建完成", module="main")

    # 运行模式：命令行参数控制时长
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
            logger.info("MAIN", f"使用命令行参数运行时长: {duration}s", module="main")
            _agent_instance.run_for_duration(duration)
        except ValueError:
            logger.error("MAIN", f"无效的命令行参数: {sys.argv[1]}", module="main")
            _agent_instance.start()
            # 阻塞主线程
            logger.info("MAIN", "Agent正在运行，按 Ctrl+C 停止", module="main")
            while _agent_instance.running:
                time.sleep(1.0)
    else:
        # 默认运行30秒（测试模式）
        logger.info("MAIN", "使用默认运行时长: 30s (测试模式)", module="main")
        _agent_instance.run_for_duration(30.0)

    logger.info("MAIN", "===== 自驱永续进化预测修正Agent系统结束 =====", module="main")


if __name__ == "__main__":
    main()
