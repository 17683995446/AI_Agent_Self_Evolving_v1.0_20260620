"""
任务编排与工作流引擎测试 —— 验证DAG执行、依赖驱动、并行执行、失败重试

测试前置：
1. 拆解WorkflowEngine接口：register_task_handler/submit_workflow/start/stop/get_stats
2. 拆解Workflow接口：add_task/get_ready_tasks/get_task_order
3. 边界条件：无依赖、线性依赖、菱形依赖、失败重试、无处理器
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.workflow_engine import (
    WorkflowEngine, Workflow, WorkflowTask, TaskState, WorkflowState,
)


class TestWorkflowTask:
    """测试任务节点"""

    def test_defaults(self):
        """默认值"""
        t = WorkflowTask()
        assert t.task_id != ""
        assert t.state == TaskState.PENDING
        assert t.max_retries == 2

    def test_to_dict(self):
        """序列化"""
        t = WorkflowTask(name="test", task_type="echo")
        d = t.to_dict()
        assert d["name"] == "test"
        assert d["task_type"] == "echo"


class TestWorkflow:
    """测试工作流"""

    def setup_method(self):
        self.wf = Workflow(name="test_wf")

    def test_add_task(self):
        """添加任务"""
        t = WorkflowTask(name="t1")
        self.wf.add_task(t)
        assert t.task_id in self.wf.tasks

    def test_get_ready_tasks_no_deps(self):
        """无依赖任务全部就绪"""
        t1 = WorkflowTask(name="t1")
        t2 = WorkflowTask(name="t2")
        self.wf.add_task(t1)
        self.wf.add_task(t2)
        ready = self.wf.get_ready_tasks()
        assert len(ready) == 2

    def test_get_ready_tasks_with_deps(self):
        """有依赖时只有前置完成才就绪"""
        t1 = WorkflowTask(name="t1")
        t2 = WorkflowTask(name="t2", dependencies=[t1.task_id])
        self.wf.add_task(t1)
        self.wf.add_task(t2)
        ready = self.wf.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == t1.task_id

    def test_get_task_order_linear(self):
        """线性依赖拓扑排序"""
        t1 = WorkflowTask(name="t1")
        t2 = WorkflowTask(name="t2", dependencies=[t1.task_id])
        t3 = WorkflowTask(name="t3", dependencies=[t2.task_id])
        self.wf.add_task(t1)
        self.wf.add_task(t2)
        self.wf.add_task(t3)
        order = self.wf.get_task_order()
        assert order == [t1.task_id, t2.task_id, t3.task_id]

    def test_get_task_order_diamond(self):
        """菱形依赖拓扑排序"""
        t1 = WorkflowTask(name="t1")
        t2 = WorkflowTask(name="t2", dependencies=[t1.task_id])
        t3 = WorkflowTask(name="t3", dependencies=[t1.task_id])
        t4 = WorkflowTask(name="t4", dependencies=[t2.task_id, t3.task_id])
        self.wf.add_task(t1)
        self.wf.add_task(t2)
        self.wf.add_task(t3)
        self.wf.add_task(t4)
        order = self.wf.get_task_order()
        assert order[0] == t1.task_id
        assert order[-1] == t4.task_id


class TestWorkflowEngine:
    """测试工作流引擎"""

    def setup_method(self):
        self.engine = WorkflowEngine(max_workers=2)
        self.engine.register_task_handler("echo", lambda p: p.get("msg", ""))
        self.engine.register_task_handler("add", lambda p: p.get("a", 0) + p.get("b", 0))
        self.engine.register_task_handler("fail_once", self._fail_once)
        self.engine.start()

    def teardown_method(self):
        self.engine.stop()

    def _fail_once(self, payload):
        # 使用闭包变量跟踪尝试次数，不依赖payload副作用
        if not hasattr(self, '_fail_count'):
            self._fail_count = 0
        self._fail_count += 1
        if self._fail_count == 1:
            raise RuntimeError("first attempt fails")
        return "success"

    def test_single_task(self):
        """单任务工作流"""
        wf = Workflow(name="single")
        t = WorkflowTask(name="t1", task_type="echo", payload={"msg": "hello"})
        wf.add_task(t)
        wid = self.engine.submit_workflow(wf)
        time.sleep(0.5)

        status = self.engine.get_workflow_status(wid)
        assert status["state"] == "completed"
        assert status["tasks"][t.task_id]["result"] == "hello"

    def test_linear_dependency(self):
        """线性依赖执行"""
        wf = Workflow(name="linear")
        t1 = WorkflowTask(name="t1", task_type="add", payload={"a": 1, "b": 2})
        t2 = WorkflowTask(name="t2", task_type="add", payload={"a": 3, "b": 4},
                          dependencies=[t1.task_id])
        wf.add_task(t1)
        wf.add_task(t2)
        wid = self.engine.submit_workflow(wf)
        time.sleep(2.0)

        status = self.engine.get_workflow_status(wid)
        assert status["state"] == "completed"

    def test_parallel_execution(self):
        """并行执行无依赖任务"""
        wf = Workflow(name="parallel")
        t1 = WorkflowTask(name="t1", task_type="echo", payload={"msg": "a"})
        t2 = WorkflowTask(name="t2", task_type="echo", payload={"msg": "b"})
        t3 = WorkflowTask(name="t3", task_type="echo", payload={"msg": "c"})
        wf.add_task(t1)
        wf.add_task(t2)
        wf.add_task(t3)
        wid = self.engine.submit_workflow(wf)
        time.sleep(2.0)

        status = self.engine.get_workflow_status(wid)
        assert status["state"] == "completed"

    def test_retry_mechanism(self):
        """失败重试机制"""
        wf = Workflow(name="retry")
        t = WorkflowTask(name="t1", task_type="fail_once", payload={"attempt": 0},
                         max_retries=2, retry_delay_sec=0.1)
        wf.add_task(t)
        wid = self.engine.submit_workflow(wf)
        time.sleep(3.0)

        status = self.engine.get_workflow_status(wid)
        assert status["state"] == "completed"
        assert status["tasks"][t.task_id]["result"] == "success"

    def test_failure_no_retry(self):
        """重试耗尽后失败"""
        def always_fail(p):
            raise RuntimeError("fail")
        self.engine.register_task_handler("always_fail", always_fail)
        wf = Workflow(name="fail")
        t = WorkflowTask(name="t1", task_type="always_fail", max_retries=1, retry_delay_sec=0.1)
        wf.add_task(t)
        wid = self.engine.submit_workflow(wf)
        time.sleep(2.0)

        status = self.engine.get_workflow_status(wid)
        assert status["state"] == "failed"

    def test_missing_handler(self):
        """缺少处理器导致失败"""
        wf = Workflow(name="missing")
        t = WorkflowTask(name="t1", task_type="unknown")
        wf.add_task(t)
        wid = self.engine.submit_workflow(wf)
        time.sleep(2.0)

        status = self.engine.get_workflow_status(wid)
        assert status["state"] == "failed"

    def test_stats(self):
        """统计信息"""
        stats = self.engine.get_stats()
        assert "total_workflows" in stats


class TestWorkflowIntegration:
    """集成测试"""

    def test_complex_dag(self):
        """复杂DAG：感知->预测+分析->决策->执行"""
        engine = WorkflowEngine(max_workers=4)
        engine.register_task_handler("perceive", lambda p: {"cpu": 80, "mem": 60})
        # 依赖结果通过任务名注入，predict接收perceive的结果在perceive键下
        engine.register_task_handler("predict", lambda p: {"cpu_next": p.get("perceive", {}).get("cpu", 0) * 1.1})
        engine.register_task_handler("analyze", lambda p: {"risk": "high" if p.get("perceive", {}).get("cpu", 0) > 70 else "low"})
        engine.register_task_handler("decide", lambda p: "throttle" if p.get("predict", {}).get("cpu_next", 0) > 85 else "noop")
        engine.register_task_handler("execute", lambda p: {"action": p.get("decide", "noop"), "status": "done"})
        engine.start()

        wf = Workflow(name="monitor_pipeline")
        perceive = WorkflowTask(name="perceive", task_type="perceive")
        predict = WorkflowTask(name="predict", task_type="predict", dependencies=[perceive.task_id])
        analyze = WorkflowTask(name="analyze", task_type="analyze", dependencies=[perceive.task_id])
        decide = WorkflowTask(name="decide", task_type="decide", dependencies=[predict.task_id, analyze.task_id])
        execute = WorkflowTask(name="execute", task_type="execute", dependencies=[decide.task_id])

        wf.add_task(perceive)
        wf.add_task(predict)
        wf.add_task(analyze)
        wf.add_task(decide)
        wf.add_task(execute)

        wid = engine.submit_workflow(wf)
        time.sleep(3.0)

        status = engine.get_workflow_status(wid)
        assert status["state"] == "completed"
        engine.stop()
