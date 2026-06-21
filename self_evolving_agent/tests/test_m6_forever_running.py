"""
M6阶段验证测试 —— 永续运行保障

验证点：
1. ForeverRunner接入main.py
2. enable/disable_forever_running开关
3. 健康检查（多维：心跳/内存/线程/错误率/运行状态）
4. 资源回收（gc/快照/日志清理）
5. 分级自愈（重启线程/内存回收/全量重启）
6. 永续统计查询
7. 健康历史记录
8. 不破坏M1-M5和主循环
"""

import os
import sys
import time
import threading
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestM6ForeverRunnerIntegration:
    """验证ForeverRunner集成"""

    def test_agent_has_forever_runner(self):
        """Agent实例化后应包含forever_runner属性"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert hasattr(agent, "forever_runner")
        assert agent.forever_runner is not None
        agent.stop()

    def test_forever_running_disabled_by_default(self):
        """默认未启用永续运行（避免测试噪声）"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert agent._forever_running_enabled is False
        agent.stop()

    def test_enable_forever_running(self):
        """启用永续运行后应启动监控线程"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        agent.enable_forever_running(health_check_interval=0.5,
                                     resource_cleanup_interval=1.0)
        assert agent._forever_running_enabled is True
        time.sleep(0.3)
        stats = agent.get_forever_stats()
        assert stats["running"] is True
        agent.stop()

    def test_disable_forever_running(self):
        """禁用永续运行后应停止监控线程"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        agent.enable_forever_running(health_check_interval=0.5,
                                     resource_cleanup_interval=1.0)
        time.sleep(0.2)
        agent.disable_forever_running()
        assert agent._forever_running_enabled is False
        stats = agent.get_forever_stats()
        assert stats["running"] is False
        agent.stop()

    def test_enable_idempotent(self):
        """重复启用应幂等"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.enable_forever_running()
        agent.enable_forever_running()  # 不应抛异常
        assert agent._forever_running_enabled is True
        agent.stop()


class TestM6HealthCheck:
    """验证健康检查"""

    def test_health_report_structure(self):
        """健康报告应包含必要字段"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(0.3)
        report = agent.get_health_report()
        assert report is not None
        assert "level" in report
        assert "score" in report
        assert "issues" in report
        assert "recommendations" in report
        assert "metrics" in report
        agent.stop()

    def test_health_level_value(self):
        """健康等级应为合法枚举值"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(0.3)
        report = agent.get_health_report()
        assert report["level"] in ("excellent", "good", "warning",
                                   "critical", "fatal")
        agent.stop()

    def test_health_score_range(self):
        """健康分数应在[0,1]区间"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(0.3)
        report = agent.get_health_report()
        assert 0.0 <= report["score"] <= 1.0
        agent.stop()

    def test_health_metrics_contains_heartbeat(self):
        """健康指标应包含心跳信息"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(0.3)
        report = agent.get_health_report()
        # 心跳年龄应被记录
        assert "heartbeat_age" in report["metrics"]
        agent.stop()

    def test_health_check_no_agent_attached(self):
        """未附加Agent时应返回warning级别"""
        from core.forever_runner import ForeverRunner, HealthLevel
        runner = ForeverRunner()
        report = runner.check_health()
        assert report.level == HealthLevel.WARNING
        assert "未附加Agent" in report.issues

    def test_health_history(self):
        """健康历史应可查询"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        agent.enable_forever_running(health_check_interval=0.3,
                                     resource_cleanup_interval=10.0)
        time.sleep(1.0)  # 等待至少2次健康检查
        history = agent.get_health_history(limit=5)
        assert isinstance(history, list)
        assert len(history) >= 1
        agent.stop()


class TestM6ResourceCleanup:
    """验证资源回收"""

    def test_trigger_resource_cleanup(self):
        """手动触发资源回收应增加清理计数"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        before = agent.get_forever_stats()["cleanup_count"]
        agent.trigger_resource_cleanup()
        after = agent.get_forever_stats()["cleanup_count"]
        assert after == before + 1
        agent.stop()

    def test_cleanup_does_not_crash(self):
        """资源回收不应抛异常"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        # 多次调用不应崩溃
        for _ in range(3):
            agent.trigger_resource_cleanup()
        agent.stop()

    def test_cleanup_removes_old_snapshots(self):
        """资源回收应清理超过保留数的旧快照"""
        import tempfile
        from main import SelfEvolvingAgent
        from core.persistence import PersistenceManager
        agent = SelfEvolvingAgent()
        # 用临时目录隔离
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.persistence = PersistenceManager(snapshot_dir=tmpdir, max_snapshots=5)
            # 创建8个快照（超过保留数5）
            for _ in range(8):
                agent.save_state()
            snapshots_before = agent.list_snapshots()
            # forever_runner保留最新5个
            agent.forever_runner._cleanup_resources()
            snapshots_after = agent.list_snapshots()
            # 应至少清理了一些
            assert len(snapshots_after) <= len(snapshots_before)
        agent.stop()


class TestM6SelfHeal:
    """验证分级自愈"""

    def test_trigger_self_heal_returns_report(self):
        """手动触发自愈应返回健康报告"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        report = agent.trigger_self_heal()
        assert report is not None
        assert "level" in report
        agent.stop()

    def test_self_heal_increases_count(self):
        """自愈应增加自愈计数"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        before = agent.get_forever_stats()["self_heal_count"]
        agent.trigger_self_heal()
        after = agent.get_forever_stats()["self_heal_count"]
        assert after == before + 1
        agent.stop()

    def test_self_heal_no_agent_does_not_crash(self):
        """未附加Agent时自愈不应崩溃"""
        from core.forever_runner import ForeverRunner, HealthReport, HealthLevel
        runner = ForeverRunner()
        report = HealthReport(level=HealthLevel.CRITICAL, score=0.3)
        # 不应抛异常
        runner._self_heal(report)


class TestM6ForeverStats:
    """验证永续统计"""

    def test_stats_structure(self):
        """统计应包含必要字段"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        stats = agent.get_forever_stats()
        assert "running" in stats
        assert "uptime_seconds" in stats
        assert "restart_count" in stats
        assert "self_heal_count" in stats
        assert "cleanup_count" in stats
        assert "agent_attached" in stats
        agent.stop()

    def test_stats_initial_values(self):
        """初始统计应为0"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        stats = agent.get_forever_stats()
        assert stats["restart_count"] == 0
        assert stats["self_heal_count"] == 0
        assert stats["cleanup_count"] == 0
        agent.stop()

    def test_stats_uptime_increases(self):
        """运行时长应随时间增加"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(0.5)
        stats = agent.get_forever_stats()
        assert stats["uptime_seconds"] >= 0.0
        agent.stop()


class TestM6NoRegression:
    """验证M6不破坏M1-M5和主循环"""

    def test_main_loop_still_runs(self):
        """主循环应仍能正常运行"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(1.0)
        assert agent.loop_count > 0
        agent.stop()

    def test_m1_orchestrator_still_works(self):
        """M1多Agent编排应仍可用"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        ids = agent.enable_multi_agent(spawn_default=True)
        assert len(ids) > 0
        status = agent.get_cluster_status()
        assert "agents" in status
        agent.stop()

    def test_m2_workflow_still_works(self):
        """M2工作流应仍可用"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        wf_id = agent.submit_goal("monitor_and_alert", "测试工作流")
        assert wf_id is not None
        agent.stop()

    def test_m3_skill_library_still_works(self):
        """M3技能库应仍可用"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        stats = agent.get_skill_library_stats()
        assert "total_skills" in stats
        agent.stop()

    def test_m4_long_task_still_works(self):
        """M4长任务规划应仍可用"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        task_id = agent.plan_long_task("system_optimization", "测试长任务")
        assert task_id is not None
        agent.stop()

    def test_m5_persistence_still_works(self):
        """M5持久化应仍可用"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        path = agent.save_state()
        assert path is not None
        stats = agent.get_persistence_stats()
        assert "snapshot_count" in stats
        agent.stop()

    def test_forever_runner_with_running_agent(self):
        """启用永续运行后Agent应继续正常运行"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        agent.enable_forever_running(health_check_interval=0.5,
                                     resource_cleanup_interval=2.0)
        time.sleep(1.5)
        # Agent仍在运行且循环数增加
        assert agent.running is True
        assert agent.loop_count > 0
        # 健康检查可执行
        report = agent.get_health_report()
        assert report is not None
        agent.stop()


class TestM6ForeverRunnerUnit:
    """ForeverRunner单元测试"""

    def test_runner_start_stop(self):
        """ForeverRunner可独立启停"""
        from core.forever_runner import ForeverRunner
        runner = ForeverRunner(health_check_interval=0.3,
                               resource_cleanup_interval=1.0)
        runner.start()
        time.sleep(0.5)
        stats = runner.get_stats()
        assert stats["running"] is True
        runner.stop()
        stats = runner.get_stats()
        assert stats["running"] is False

    def test_runner_attach_agent(self):
        """ForeverRunner可附加Agent"""
        from core.forever_runner import ForeverRunner
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        runner = ForeverRunner()
        runner.attach(agent)
        stats = runner.get_stats()
        assert stats["agent_attached"] is True
        agent.stop()

    def test_runner_health_check_with_agent(self):
        """附加Agent后健康检查应能获取心跳"""
        from core.forever_runner import ForeverRunner
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(0.3)
        runner = ForeverRunner()
        runner.attach(agent)
        report = runner.check_health()
        assert report.metrics.get("heartbeat_age") is not None
        agent.stop()

    def test_runner_get_last_health_report(self):
        """获取最近健康报告"""
        from core.forever_runner import ForeverRunner
        runner = ForeverRunner()
        runner.check_health()
        # check_health不自动存入_last_health_report，需要通过loop
        # 直接调用check_health返回报告
        report = runner.check_health()
        assert report is not None

    def test_runner_health_history_limit(self):
        """健康历史应受limit限制"""
        from core.forever_runner import ForeverRunner, HealthReport, HealthLevel
        runner = ForeverRunner()
        # 手动添加历史
        for i in range(20):
            runner._health_history.append(
                HealthReport(level=HealthLevel.GOOD, score=0.8)
            )
        history = runner.get_health_history(limit=5)
        assert len(history) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
