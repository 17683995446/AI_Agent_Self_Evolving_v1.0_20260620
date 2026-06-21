"""
O3阶段验证测试 —— 异常秩序

验证点：
1. 异常类层次结构完整
2. 新增异常类（Workflow/Skill/Persistence/Forever/Alert/MultiAgent/LongTask）
3. 构造签名统一
4. error_eval.py使用ErrorEvalException而非ValueError
5. main.py导入所有异常类
6. 异常to_dict()方法
7. degradable属性正确
8. 不破坏现有功能
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestO3ExceptionHierarchy:
    """验证异常类层次结构"""

    def test_all_exceptions_inherit_agent_exception(self):
        """所有异常类应继承AgentException"""
        from core.exceptions import (
            AgentException, PerceptionException, PredictionException,
            DriveEngineException, DecisionException, ErrorEvalException,
            OptimizerException, MemoryException, ConfigurationException,
            ResourceExhaustedException, ModuleInitException,
            WorkflowException, SkillLibraryException, PersistenceException,
            ForeverRunnerException, AlertingException, MultiAgentException, LongTaskException,
        )
        for exc_class in [PerceptionException, PredictionException, DriveEngineException,
                          DecisionException, ErrorEvalException, OptimizerException,
                          MemoryException, ConfigurationException, ResourceExhaustedException,
                          ModuleInitException, WorkflowException, SkillLibraryException,
                          PersistenceException, ForeverRunnerException, AlertingException,
                          MultiAgentException, LongTaskException]:
            assert issubclass(exc_class, AgentException), \
                f"{exc_class.__name__} 应继承 AgentException"

    def test_new_exception_classes_exist(self):
        """O3新增的异常类应存在"""
        from core.exceptions import (
            WorkflowException, SkillLibraryException, PersistenceException,
            ForeverRunnerException, AlertingException, MultiAgentException, LongTaskException,
        )
        # 每个类都应可实例化
        for exc_class in [WorkflowException, SkillLibraryException, PersistenceException,
                          ForeverRunnerException, AlertingException, MultiAgentException,
                          LongTaskException]:
            exc = exc_class("测试异常")
            assert isinstance(exc, Exception)


class TestO3ExceptionModuleAttribute:
    """验证异常的module属性"""

    def test_perception_exception_module(self):
        from core.exceptions import PerceptionException
        exc = PerceptionException("test")
        assert exc.module == "perception"

    def test_workflow_exception_module(self):
        from core.exceptions import WorkflowException
        exc = WorkflowException("test")
        assert exc.module == "workflow"

    def test_skill_library_exception_module(self):
        from core.exceptions import SkillLibraryException
        exc = SkillLibraryException("test")
        assert exc.module == "skill_library"

    def test_persistence_exception_module(self):
        from core.exceptions import PersistenceException
        exc = PersistenceException("test")
        assert exc.module == "persistence"

    def test_forever_runner_exception_module(self):
        from core.exceptions import ForeverRunnerException
        exc = ForeverRunnerException("test")
        assert exc.module == "forever_runner"

    def test_alerting_exception_module(self):
        from core.exceptions import AlertingException
        exc = AlertingException("test")
        assert exc.module == "alerting"

    def test_multi_agent_exception_module(self):
        from core.exceptions import MultiAgentException
        exc = MultiAgentException("test")
        assert exc.module == "multi_agent"

    def test_long_task_exception_module(self):
        from core.exceptions import LongTaskException
        exc = LongTaskException("test")
        assert exc.module == "long_task"


class TestO3DegradableAttribute:
    """验证degradable属性"""

    def test_configuration_exception_not_degradable(self):
        from core.exceptions import ConfigurationException
        exc = ConfigurationException("test")
        assert exc.degradable is False

    def test_most_exceptions_degradable(self):
        from core.exceptions import (
            PerceptionException, WorkflowException, SkillLibraryException,
            PersistenceException, ForeverRunnerException, AlertingException,
        )
        for exc_class in [PerceptionException, WorkflowException, SkillLibraryException,
                          PersistenceException, ForeverRunnerException, AlertingException]:
            exc = exc_class("test")
            assert exc.degradable is True, \
                f"{exc_class.__name__} 应该是可降级的"


class TestO3ToDict:
    """验证to_dict()方法"""

    def test_to_dict_structure(self):
        from core.exceptions import WorkflowException
        exc = WorkflowException("测试消息", context={"key": "value"})
        d = exc.to_dict()
        assert d["type"] == "WorkflowException"
        assert d["message"] == "测试消息"
        assert d["module"] == "workflow"
        assert d["context"] == {"key": "value"}
        assert d["degradable"] is True

    def test_to_dict_with_context(self):
        from core.exceptions import PersistenceException
        exc = PersistenceException("保存失败", context={"path": "/tmp/test"})
        d = exc.to_dict()
        assert d["context"]["path"] == "/tmp/test"


class TestO3ErrorEvalUsesCustomException:
    """验证error_eval.py使用ErrorEvalException"""

    def test_error_eval_raises_custom_exception(self):
        """error_eval应抛出ErrorEvalException而非ValueError"""
        from modules.error_eval import ErrorEvaluator
        evaluator = ErrorEvaluator()
        from core.exceptions import ErrorEvalException
        with pytest.raises(ErrorEvalException):
            evaluator.compute_mse([], [])

    def test_error_eval_mae_raises_custom_exception(self):
        from modules.error_eval import ErrorEvaluator
        evaluator = ErrorEvaluator()
        from core.exceptions import ErrorEvalException
        with pytest.raises(ErrorEvalException):
            evaluator.compute_mae([1.0], [1.0, 2.0])

    def test_error_eval_no_value_error(self):
        """error_eval.py源码不应包含raise ValueError"""
        error_eval_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "modules", "error_eval.py"
        )
        with open(error_eval_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "raise ValueError" not in content, \
            "error_eval.py 不应使用 raise ValueError，应使用 ErrorEvalException"


class TestO3MainImports:
    """验证main.py导入所有异常类"""

    def test_main_imports_all_exceptions(self):
        """main.py应导入所有异常类"""
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main.py"
        )
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 检查新增异常类是否被导入
        for exc_name in ["WorkflowException", "SkillLibraryException",
                         "PersistenceException", "ForeverRunnerException",
                         "AlertingException", "MultiAgentException", "LongTaskException"]:
            assert exc_name in content, \
                f"main.py 应导入 {exc_name}"


class TestO3ExceptionContext:
    """验证异常上下文传递"""

    def test_exception_with_context(self):
        from core.exceptions import MultiAgentException
        ctx = {"agent_id": "agent-1", "task_type": "perceive"}
        exc = MultiAgentException("委派失败", context=ctx)
        assert exc.context == ctx
        assert exc.context["agent_id"] == "agent-1"

    def test_exception_default_context_empty(self):
        from core.exceptions import LongTaskException
        exc = LongTaskException("规划失败")
        assert exc.context == {}


class TestO3NoRegression:
    """验证O3不破坏现有功能"""

    def test_agent_init_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert agent is not None
        agent.stop()

    def test_main_loop_still_runs(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        import time
        time.sleep(0.8)
        assert agent.loop_count > 0
        agent.stop()

    def test_error_eval_normal_operation(self):
        """error_eval正常操作不应受影响"""
        from modules.error_eval import ErrorEvaluator
        evaluator = ErrorEvaluator()
        mse = evaluator.compute_mse([1.0, 2.0], [1.5, 2.5])
        assert mse > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
