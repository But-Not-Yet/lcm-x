"""Deterministic acceptance checks for the report-only maintainability inventory."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/maintainability/report.py"
BASELINE = SCRIPT.with_name("baseline.json")
spec = importlib.util.spec_from_file_location("maintainability_report", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def run(root, *arguments):
    # The subprocess timeout is an execution budget, not a timing assertion.
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments, "--format", "json"],
        cwd=root, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_repository_baseline():
    result = run(ROOT, "--baseline", str(BASELINE))
    assert result["metrics"]["warnings"] == []
    assert result["metrics"]["headline"]["engine_attributes"] > 0
    assert result["metrics"]["functions"]["compaction.py::CompactionMixin._compress_impl"]["cc"] > 15
    assert len(report.markdown(result).splitlines()) <= 60


def test_growth_and_baseline_round_trip(tmp_path):
    shutil.copyfile(ROOT / "reset_state.py", tmp_path / "reset_state.py")
    baseline = tmp_path / "baseline.json"
    original = run(tmp_path, "--baseline", str(baseline), "--write-baseline")
    assert original["metrics"]["warnings"] == []
    assert json.loads(baseline.read_text()) == original["metrics"]
    assert all(not value for value in original["changes"].values())
    assert run(tmp_path, "--baseline", str(baseline)) == original
    path = tmp_path / "reset_state.py"
    path.write_text(path.read_text() + "\ndef added_long_function():\n" + "    value = 1\n" * 81)
    grown = run(tmp_path, "--baseline", str(baseline))
    assert grown["changes"]["new_large_functions"] == [{
        "function": "reset_state.py::added_long_function",
        **grown["metrics"]["functions"]["reset_state.py::added_long_function"],
    }]
    assert json.loads(baseline.read_text()) == original["metrics"]
    rewritten = run(tmp_path, "--baseline", str(baseline), "--write-baseline")
    assert all(not value for value in rewritten["changes"].values())
    assert run(tmp_path, "--baseline", str(baseline)) == rewritten


def test_metric_contracts_and_growth(tmp_path):
    (tmp_path / "engine.py").write_text('''from .mix import StateMixin
class LCMEngine(StateMixin):
    def work(self, config):
        self._state = 1
        self._pair, self._other = 2, 3
        return getattr(config, "limit", 1)
''')
    (tmp_path / "mix.py").write_text('''class StateMixin:
    def reset(self):
        self._state: int = 0
class Unrelated:
    def reset(self):
        self._unrelated = 1
''')
    (tmp_path / "store.py").write_text('''from .engine import LCMEngine
from hermes_lcm import compaction
import hermes_lcm.tools
from . import command
SQL = "CREATE TABLE sample (id); ALTER TABLE sample ADD COLUMN name; CREATE UNIQUE INDEX idx ON sample(id)"
''')
    (tmp_path / "db_bootstrap.py").write_text('SQL = "CREATE TABLE schema_table (id)"\n')
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_example.py").write_text('''from hermes_lcm.store import _private
engine._state
self._store._db
unrelated._field
''')
    old = report.collect(tmp_path)
    assert old["warnings"] == []
    assert old["attributes"] == {
        "_state": ["engine.py", "mix.py"], "_pair": ["engine.py"], "_other": ["engine.py"],
    }
    assert len(old["imports"]) == 4
    assert len(old["config_getattr"]) == 1
    assert len(old["private_refs"]) == 3
    assert len(old["ddl_outside_schema"]) == 3
    assert old["headline"]["ddl_modules"] == 2
    (tmp_path / "mix.py").write_text('''class StateMixin:
    def reset(self):
        self._state = 0
        self._pair = 1
        self._new = 2
        if self._pair:
            self._new = 3
''')
    new = report.collect(tmp_path)
    changes = report.compare(new, old)
    assert changes["new_attributes"] == ["_new"]
    assert changes["new_writers"] == {"_pair": ["mix.py"]}
    assert changes["function_growth"] == [{
        "function": "mix.py::StateMixin.reset",
        "delta": {"loc": 4, "cc": 1, "branches": 1, "statements": 4},
    }]
    assert not changes["new_config_getattr"]
    # Moving source lines must not manufacture new sites.
    path = tmp_path / "engine.py"
    path.write_text("\n\n" + path.read_text())
    assert not report.compare(report.collect(tmp_path), new)["new_config_getattr"]


def test_new_complex_short_function(tmp_path):
    source = "def branchy(value):\n" + "    if value:\n        pass\n" * 16
    (tmp_path / "branchy.py").write_text(source)
    changes = report.compare(report.collect(tmp_path), {})
    assert changes["new_large_functions"][0]["cc"] == 17
    assert changes["new_large_functions"][0]["loc"] < 80


def test_incomplete_collection_is_report_only(tmp_path, monkeypatch):
    (tmp_path / "engine.py").write_text("def broken(:\n")
    result = run(tmp_path, "--baseline", str(tmp_path / "baseline.json"), "--write-baseline")
    assert result["metrics"]["warnings"]
    assert not (tmp_path / "baseline.json").exists()
    (tmp_path / "engine.py").write_text("def valid():\n    pass\n")
    monkeypatch.setenv("PATH", "")
    result = run(tmp_path, "--baseline", str(tmp_path / "baseline.json"), "--write-baseline")
    assert "Ruff metrics unavailable" in result["metrics"]["warnings"][0]
    assert not (tmp_path / "baseline.json").exists()
