"""
自动合并与Git版本控制集成测试 —— 验证分支创建、合并请求、冲突解决

测试前置：
1. 拆解GitRepository接口：init/create_branch/checkout/commit/merge/abort_merge
2. 拆解AutoMerger接口：create_agent_branch/submit_merge_request/process_merge_request/resolve_conflict
3. 边界条件：空仓库、重复分支、无冲突合并、冲突解决、中止合并
"""
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.git_merger import GitRepository, AutoMerger, MergeStatus


class TestGitRepository:
    """测试Git仓库管理"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.repo = GitRepository(self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_git_dir(self):
        """初始化创建.git目录"""
        assert os.path.exists(os.path.join(self.temp_dir, ".git"))

    def test_create_and_list_branches(self):
        """创建分支并列出"""
        # 先创建一个初始提交，否则无法创建分支
        with open(os.path.join(self.temp_dir, "init.txt"), "w") as f:
            f.write("init")
        self.repo.commit("init commit")

        assert self.repo.create_branch("feature-a")
        branches = self.repo.get_branches()
        assert "feature-a" in branches

    def test_checkout_branch(self):
        """切换分支"""
        with open(os.path.join(self.temp_dir, "init.txt"), "w") as f:
            f.write("init")
        self.repo.commit("init commit")
        self.repo.create_branch("branch-b")

        assert self.repo.checkout("branch-b") is True

    def test_commit_file(self):
        """提交文件"""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("hello")
        assert self.repo.commit("add test.txt", ["test.txt"]) is True

    def test_merge_no_conflict(self):
        """无冲突合并"""
        # 初始化
        with open(os.path.join(self.temp_dir, "base.txt"), "w") as f:
            f.write("base")
        self.repo.commit("base commit")

        # 创建分支并修改不同文件
        self.repo.create_branch("branch-no-conflict")
        with open(os.path.join(self.temp_dir, "branch.txt"), "w") as f:
            f.write("branch content")
        self.repo.commit("branch commit", ["branch.txt"])

        # 切回main合并
        self.repo.checkout("main")
        success, conflicts = self.repo.merge("branch-no-conflict")
        assert success is True
        assert conflicts == []

    def test_merge_with_conflict(self):
        """有冲突合并"""
        # 初始化
        with open(os.path.join(self.temp_dir, "shared.txt"), "w") as f:
            f.write("main line")
        self.repo.commit("base")

        # 创建分支并修改同一文件
        self.repo.create_branch("branch-conflict")
        with open(os.path.join(self.temp_dir, "shared.txt"), "w") as f:
            f.write("branch line")
        self.repo.commit("branch change", ["shared.txt"])

        # main也修改同一文件
        self.repo.checkout("main")
        with open(os.path.join(self.temp_dir, "shared.txt"), "w") as f:
            f.write("main modified")
        self.repo.commit("main change", ["shared.txt"])

        # 合并应该冲突
        success, conflicts = self.repo.merge("branch-conflict")
        assert success is False
        assert "shared.txt" in conflicts

        # 中止合并
        assert self.repo.abort_merge() is True

    def test_has_uncommitted(self):
        """检测未提交变更"""
        with open(os.path.join(self.temp_dir, "uncommitted.txt"), "w") as f:
            f.write("change")
        assert self.repo.has_uncommitted() is True


class TestAutoMerger:
    """测试自动合并Agent"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.merger = AutoMerger(self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_agent_branch(self):
        """创建Agent分支"""
        # 先初始化main分支
        with open(os.path.join(self.temp_dir, "init.txt"), "w") as f:
            f.write("init")
        self.merger.repo.commit("init")

        branch = self.merger.create_agent_branch("agent-1")
        assert branch == "agent/agent-1"
        assert branch in self.merger.repo.get_branches()

    def test_submit_merge_request(self):
        """提交合并请求"""
        mr_id = self.merger.submit_merge_request("agent-1", "agent/agent-1")
        assert mr_id.startswith("MR-")
        status = self.merger.get_merge_request_status(mr_id)
        assert status["status"] == "pending"

    def test_process_merge_request_success(self):
        """处理无冲突合并请求"""
        # 初始化
        with open(os.path.join(self.temp_dir, "base.txt"), "w") as f:
            f.write("base")
        self.merger.repo.commit("init")

        # 创建Agent分支并修改
        self.merger.create_agent_branch("agent-2")
        self.merger.write_file_and_commit("agent/agent-2", "agent2.txt", "content", "agent2 commit")

        # 提交并处理MR
        mr_id = self.merger.submit_merge_request("agent-2", "agent/agent-2")
        status = self.merger.process_merge_request(mr_id)
        assert status == MergeStatus.SUCCESS

        # 验证文件存在
        self.merger.repo.checkout("main")
        assert os.path.exists(os.path.join(self.temp_dir, "agent2.txt"))

    def test_process_merge_request_conflict(self):
        """处理有冲突合并请求"""
        # 初始化
        with open(os.path.join(self.temp_dir, "shared.txt"), "w") as f:
            f.write("base")
        self.merger.repo.commit("init")

        # Agent分支修改
        self.merger.create_agent_branch("agent-3")
        self.merger.write_file_and_commit("agent/agent-3", "shared.txt", "agent3", "agent3 commit")

        # main也修改
        self.merger.repo.checkout("main")
        with open(os.path.join(self.temp_dir, "shared.txt"), "w") as f:
            f.write("main")
        self.merger.repo.commit("main change", ["shared.txt"])

        # 提交并处理MR
        mr_id = self.merger.submit_merge_request("agent-3", "agent/agent-3")
        status = self.merger.process_merge_request(mr_id)
        assert status == MergeStatus.CONFLICT

    def test_resolve_conflict_ours(self):
        """使用ours策略解决冲突"""
        # 初始化
        with open(os.path.join(self.temp_dir, "shared.txt"), "w") as f:
            f.write("base")
        self.merger.repo.commit("init")

        # Agent分支修改
        self.merger.create_agent_branch("agent-4")
        self.merger.write_file_and_commit("agent/agent-4", "shared.txt", "agent4", "agent4 commit")

        # main也修改
        self.merger.repo.checkout("main")
        with open(os.path.join(self.temp_dir, "shared.txt"), "w") as f:
            f.write("main")
        self.merger.repo.commit("main change", ["shared.txt"])

        # 提交MR并处理冲突
        mr_id = self.merger.submit_merge_request("agent-4", "agent/agent-4")
        self.merger.process_merge_request(mr_id)

        # 解决冲突
        resolved = self.merger.resolve_conflict(mr_id, "ours")
        assert resolved is True

        status = self.merger.get_merge_request_status(mr_id)
        assert status["status"] == "success"

    def test_write_file_and_commit(self):
        """写入文件并提交"""
        with open(os.path.join(self.temp_dir, "init.txt"), "w") as f:
            f.write("init")
        self.merger.repo.commit("init")

        self.merger.create_agent_branch("agent-5")
        success = self.merger.write_file_and_commit("agent/agent-5", "modules/new.py", "x=1", "add new")
        assert success is True
        assert os.path.exists(os.path.join(self.temp_dir, "modules/new.py"))

    def test_stats(self):
        """统计信息"""
        self.merger.submit_merge_request("a1", "agent/a1")
        self.merger.submit_merge_request("a2", "agent/a2")
        stats = self.merger.get_stats()
        assert stats["total_mr"] == 2
        assert stats["success"] == 0
