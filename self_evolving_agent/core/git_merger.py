"""
自动合并与Git版本控制集成 —— 代码变更的自动化管理

设计原则（第一性原理）：
1. 代码即状态：每个Agent的代码变更视为状态变更
2. 分支隔离：每个Agent在独立分支开发，互不干扰
3. 自动合并：合并Agent负责冲突检测和解决
4. 审计追踪：所有变更可追溯，可回滚
"""

import os
import sys
import time
import subprocess
import tempfile
import shutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


class MergeStatus(Enum):
    """合并状态"""
    PENDING = "pending"
    SUCCESS = "success"
    CONFLICT = "conflict"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class BranchInfo:
    """分支信息"""
    branch_name: str
    agent_id: str
    created_at: float = field(default_factory=time.time)
    last_commit: str = ""
    commit_count: int = 0


@dataclass
class MergeRequest:
    """合并请求"""
    mr_id: str
    source_branch: str
    target_branch: str = "main"
    agent_id: str = ""
    status: MergeStatus = MergeStatus.PENDING
    created_at: float = field(default_factory=time.time)
    merged_at: Optional[float] = None
    conflict_files: List[str] = field(default_factory=list)
    test_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mr_id": self.mr_id,
            "source_branch": self.source_branch,
            "target_branch": self.target_branch,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "merged_at": self.merged_at,
            "conflict_files": self.conflict_files,
            "test_passed": self.test_passed,
        }


class GitRepository:
    """Git仓库管理 —— 封装git命令"""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self._ensure_repo()

    def _ensure_repo(self) -> None:
        """确保是git仓库，默认分支为main"""
        if not os.path.exists(os.path.join(self.repo_path, ".git")):
            self._run_git(["init"])
            self._run_git(["config", "user.email", "agent@self-evolving.ai"])
            self._run_git(["config", "user.name", "SelfEvolvingAgent"])
            # 创建初始提交以确保main分支存在
            init_file = os.path.join(self.repo_path, ".gitkeep")
            with open(init_file, "w") as f:
                f.write("")
            self._run_git(["add", ".gitkeep"])
            self._run_git(["commit", "-m", "init"])
            # 重命名为main（兼容旧版git默认master）
            self._run_git(["branch", "-M", "main"])

    def _run_git(self, args: List[str]) -> tuple:
        """运行git命令"""
        cmd = ["git", "-C", self.repo_path] + args
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return -1, "", str(e)

    def create_branch(self, branch_name: str, from_branch: str = "main") -> bool:
        """创建分支"""
        # 先切换到源分支
        self._run_git(["checkout", from_branch])
        code, _, _ = self._run_git(["checkout", "-b", branch_name])
        return code == 0

    def checkout(self, branch_name: str) -> bool:
        """切换分支"""
        code, _, _ = self._run_git(["checkout", branch_name])
        return code == 0

    def commit(self, message: str, files: Optional[List[str]] = None) -> bool:
        """提交变更"""
        if files:
            for f in files:
                self._run_git(["add", f])
        else:
            self._run_git(["add", "."])
        code, _, _ = self._run_git(["commit", "-m", message])
        return code == 0

    def get_branches(self) -> List[str]:
        """获取所有分支"""
        code, out, _ = self._run_git(["branch", "--format=%(refname:short)"])
        if code == 0:
            return [b.strip() for b in out.strip().split("\n") if b.strip()]
        return []

    def merge(self, source_branch: str, strategy: str = "recursive") -> tuple:
        """合并分支，返回(成功, 冲突文件列表)"""
        code, out, err = self._run_git(["merge", "--no-commit", "--no-ff", source_branch])
        if code == 0:
            return True, []

        # 检查冲突
        ccode, cout, _ = self._run_git(["diff", "--name-only", "--diff-filter=U"])
        conflicts = [c.strip() for c in cout.strip().split("\n") if c.strip()]
        return False, conflicts

    def abort_merge(self) -> bool:
        """中止合并"""
        code, _, _ = self._run_git(["merge", "--abort"])
        return code == 0

    def get_log(self, n: int = 5) -> List[Dict[str, str]]:
        """获取提交日志"""
        fmt = "%H|%s|%ci"
        code, out, _ = self._run_git(["log", f"--format={fmt}", f"-n{n}"])
        logs = []
        if code == 0:
            for line in out.strip().split("\n"):
                parts = line.split("|", 2)
                if len(parts) == 3:
                    logs.append({"hash": parts[0], "message": parts[1], "date": parts[2]})
        return logs

    def has_uncommitted(self) -> bool:
        """是否有未提交变更"""
        code, out, _ = self._run_git(["status", "--porcelain"])
        return len(out.strip()) > 0


class AutoMerger:
    """自动合并Agent —— 管理分支、合并请求、冲突解决"""

    def __init__(self, repo_path: str):
        self.repo = GitRepository(repo_path)
        self._branches: Dict[str, BranchInfo] = {}
        self._merge_requests: Dict[str, MergeRequest] = {}
        self._mr_counter = 0

    def create_agent_branch(self, agent_id: str) -> str:
        """为Agent创建开发分支"""
        branch_name = f"agent/{agent_id}"
        if branch_name not in self.repo.get_branches():
            self.repo.create_branch(branch_name)
        self._branches[branch_name] = BranchInfo(branch_name=branch_name, agent_id=agent_id)
        logger.info("MERGER", f"Agent分支创建: {branch_name}", module="git_merger")
        return branch_name

    def submit_merge_request(self, agent_id: str, source_branch: str) -> str:
        """提交合并请求"""
        self._mr_counter += 1
        mr_id = f"MR-{self._mr_counter:03d}"
        mr = MergeRequest(
            mr_id=mr_id,
            source_branch=source_branch,
            agent_id=agent_id,
        )
        self._merge_requests[mr_id] = mr
        logger.info("MERGER", f"合并请求提交: {mr_id} ({source_branch} -> main)", module="git_merger")
        return mr_id

    def process_merge_request(self, mr_id: str) -> MergeStatus:
        """处理合并请求"""
        mr = self._merge_requests.get(mr_id)
        if not mr:
            return MergeStatus.FAILED

        # 切换到main分支
        if not self.repo.checkout("main"):
            mr.status = MergeStatus.FAILED
            return mr.status

        # 尝试合并
        success, conflicts = self.repo.merge(mr.source_branch)

        if success:
            # 合并没有冲突，运行测试验证
            mr.test_passed = self._run_tests()
            if mr.test_passed:
                self.repo.commit(f"Merge {mr.source_branch}: {mr.agent_id}")
                mr.status = MergeStatus.SUCCESS
                mr.merged_at = time.time()
                logger.info("MERGER", f"合并成功: {mr_id}", module="git_merger")
            else:
                # 测试失败，回滚
                self.repo.abort_merge()
                mr.status = MergeStatus.FAILED
                logger.warning("MERGER", f"合并测试失败: {mr_id}", module="git_merger")
        else:
            # 有冲突
            mr.conflict_files = conflicts
            mr.status = MergeStatus.CONFLICT
            self.repo.abort_merge()
            logger.warning("MERGER", f"合并冲突: {mr_id} 文件: {conflicts}", module="git_merger")

        return mr.status

    def resolve_conflict(self, mr_id: str, resolution: str = "ours") -> bool:
        """解决冲突 —— 选择ours/theirs/manual"""
        mr = self._merge_requests.get(mr_id)
        if not mr or mr.status != MergeStatus.CONFLICT:
            return False

        self.repo.checkout("main")
        success, conflicts = self.repo.merge(mr.source_branch)

        if not success and conflicts:
            for f in conflicts:
                if resolution == "ours":
                    self.repo._run_git(["checkout", "--ours", f])
                elif resolution == "theirs":
                    self.repo._run_git(["checkout", "--theirs", f])
                self.repo._run_git(["add", f])

            self.repo.commit(f"Merge {mr.source_branch} (conflict resolved with {resolution})")
            mr.status = MergeStatus.SUCCESS
            mr.merged_at = time.time()
            logger.info("MERGER", f"冲突已解决: {mr_id} 策略={resolution}", module="git_merger")
            return True

        return False

    def _run_tests(self) -> bool:
        """运行测试验证 —— 简化版，实际可调用pytest"""
        # 在实际场景中，这里会运行完整的测试套件
        # 简化版：假设测试通过
        return True

    def get_merge_request_status(self, mr_id: str) -> Optional[Dict[str, Any]]:
        """获取合并请求状态"""
        mr = self._merge_requests.get(mr_id)
        return mr.to_dict() if mr else None

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        total = len(self._merge_requests)
        success = sum(1 for mr in self._merge_requests.values() if mr.status == MergeStatus.SUCCESS)
        conflict = sum(1 for mr in self._merge_requests.values() if mr.status == MergeStatus.CONFLICT)
        failed = sum(1 for mr in self._merge_requests.values() if mr.status == MergeStatus.FAILED)
        return {
            "total_mr": total,
            "success": success,
            "conflict": conflict,
            "failed": failed,
            "branches": len(self._branches),
        }

    def write_file_and_commit(self, branch: str, file_path: str,
                               content: str, message: str) -> bool:
        """在指定分支写入文件并提交"""
        if not self.repo.checkout(branch):
            return False

        full_path = os.path.join(self.repo.repo_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        return self.repo.commit(message, [file_path])
