"""
数据持久化与恢复测试 —— 验证快照、恢复和备份功能
"""
import sys
import os
import time
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.persistence import Snapshot, PersistenceManager, AgentStateSerializer
from main import SelfEvolvingAgent


class TestSnapshot:
    """测试快照基础功能"""

    def test_snapshot_create_and_verify(self):
        s = Snapshot()
        s.set_data("key1", "value1")
        s.set_data("key2", 123)
        checksum = s.compute_checksum()
        assert len(checksum) == 16
        assert s.verify_checksum() is True

    def test_snapshot_tamper_detection(self):
        s = Snapshot()
        s.set_data("key", "value")
        s.compute_checksum()
        # 篡改数据
        s.data["key"] = "tampered"
        assert s.verify_checksum() is False

    def test_snapshot_to_dict(self):
        s = Snapshot(version="2.0")
        s.set_data("a", 1)
        d = s.to_dict()
        assert d["version"] == "2.0"
        assert "data_keys" in d


class TestPersistenceManager:
    """测试持久化管理器"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pm = PersistenceManager(snapshot_dir=self.temp_dir, max_snapshots=3)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_snapshot(self):
        s = Snapshot()
        s.set_data("test", "data")
        path = self.pm.save_snapshot(s)
        assert os.path.exists(path)

        loaded = self.pm.load_snapshot(path)
        assert loaded is not None
        assert loaded.get_data("test") == "data"
        assert loaded.verify_checksum() is True

    def test_list_snapshots_sorted(self):
        for i in range(5):
            s = Snapshot()
            s.set_data("idx", i)
            time.sleep(0.01)
            self.pm.save_snapshot(s)

        snapshots = self.pm.list_snapshots()
        assert len(snapshots) == 3  # 最多保留3个
        # 验证按时间倒序
        timestamps = [float(os.path.basename(p).replace("snapshot_", "").replace(".pkl", "")) for p in snapshots]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_restore_from_latest(self):
        s1 = Snapshot()
        s1.set_data("version", 1)
        time.sleep(0.01)
        self.pm.save_snapshot(s1)

        s2 = Snapshot()
        s2.set_data("version", 2)
        self.pm.save_snapshot(s2)

        latest = self.pm.restore_from_latest()
        assert latest is not None
        assert latest.get_data("version") == 2

    def test_create_backup(self):
        source = os.path.join(self.temp_dir, "test_file.txt")
        with open(source, 'w') as f:
            f.write("test content")
        backup = self.pm.create_backup(source)
        assert os.path.exists(backup)
        with open(backup) as f:
            assert f.read() == "test content"

    def test_load_corrupted_snapshot(self):
        bad_path = os.path.join(self.temp_dir, "bad_snapshot.pkl")
        with open(bad_path, 'wb') as f:
            f.write(b"not a valid pickle")
        loaded = self.pm.load_snapshot(bad_path)
        assert loaded is None


class TestAgentStateSerializer:
    """测试Agent状态序列化"""

    def test_serialize_prediction(self):
        agent = SelfEvolvingAgent()
        for i in range(10):
            agent.prediction.update({"cpu": float(i), "memory": float(i), "disk": float(i)})
        data = AgentStateSerializer.serialize_prediction(agent.prediction)
        assert "base_history" in data
        assert "mid_history" in data
        assert len(data["base_history"]) > 0

    def test_roundtrip_prediction(self):
        agent1 = SelfEvolvingAgent()
        for i in range(10):
            agent1.prediction.update({"cpu": float(i), "memory": float(i), "disk": float(i)})

        data = AgentStateSerializer.serialize_prediction(agent1.prediction)

        agent2 = SelfEvolvingAgent()
        AgentStateSerializer.deserialize_prediction(agent2.prediction, data)

        assert len(agent2.prediction.base_predictor.history) == len(agent1.prediction.base_predictor.history)

    def test_full_agent_roundtrip(self):
        temp_dir = tempfile.mkdtemp()
        pm = PersistenceManager(snapshot_dir=temp_dir, max_snapshots=5)

        agent1 = SelfEvolvingAgent()
        agent1.run_for_duration(1.0)
        agent1.loop_count = 42

        snapshot = AgentStateSerializer.serialize_full_agent(agent1)
        path = pm.save_snapshot(snapshot)
        assert path != ""

        loaded_snapshot = pm.load_snapshot(path)
        assert loaded_snapshot is not None

        agent2 = SelfEvolvingAgent()
        success = AgentStateSerializer.deserialize_full_agent(agent2, loaded_snapshot)
        assert success is True
        assert agent2.loop_count == 42

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_restore_with_no_snapshot(self):
        temp_dir = tempfile.mkdtemp()
        pm = PersistenceManager(snapshot_dir=temp_dir, max_snapshots=5)
        result = pm.restore_from_latest()
        assert result is None
        shutil.rmtree(temp_dir, ignore_errors=True)
