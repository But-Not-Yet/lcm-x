"""Synthetic PMW cell: no real material, model calls or credentials."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

DECISION = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pmw_read", DECISION / "pmw_read.py")
pmw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pmw)
ARM = "synthetic"
WANT = dict(leaf_chunk_tokens=24000, leaf_target_max_tokens=2000, summary_timeout_ms=120000)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def cell(tmp_path):
    root, material = tmp_path / "out", tmp_path / "material"
    for seed in (1, 2):
        transcript = material / f"seed-{seed}" / "transcript.jsonl"
        transcript.parent.mkdir(parents=True)
        transcript.write_text("\n".join(json.dumps(dict(turn=t)) for t in range(20)))
        run = root / "lcmx-runs" / ARM / f"seed-{seed}" / f"d{seed}-r1"
        events = [dict(turn=t, tokens_at_trigger=100, tokens_after=10,
                       compress_wall_s=1, leaves=1, levels=[dict(depth=0, source_tokens=90)],
                       summariser_calls=[dict(latency_s=0.5)]) for t in range(4)]
        write(run / "summary.json", dict(status="DONE", threshold_tokens=100,
                                        stop_row_index=19, events=events, replay_wall_s=5))
        write(run / "config.json", dict(config=WANT))
        (run / "engine.log").write_text("")
        for cp in (176, 304):
            write(root / "decision" / "scores" / f"cp-{cp}" / f"{ARM}.seed-{seed}.d{seed}-r1.json",
                  dict(metrics=dict(facts_kept=dict(value=0.583))))
    return root, material


def invoke(cell, *extra):
    root, material = cell
    output = root / "read.json"
    code = pmw.main(["--run-root", str(root), "--material", str(material), "--arm", ARM,
                     "--seeds", "1", "2", "--json", str(output), *extra])
    return code, json.loads(output.read_text())


def change_runs(cell, change):
    for path in cell[0].glob("lcmx-runs/*/seed-*/*/summary.json"):
        summary = json.loads(path.read_text())
        change(summary)
        write(path, summary)


def test_pass(cell):
    code, result = invoke(cell)
    assert code == 0
    assert result["verdict"]["result"] == "PASS"
    assert all(result["verdict"]["bars"].values())
    pooled = result["pooled"][ARM]
    assert pooled["c_per_100"] == 20
    assert pooled["waited_turn_share"] == 0.2
    assert pooled["mean_turn_wait_s"] == 0.2
    assert pooled["freed_per_event"] == 90


@pytest.mark.parametrize("bar", range(1, 7))
def test_each_bar_fails_alone(cell, bar):
    if bar == 1:
        change_runs(cell, lambda s: s["events"].extend([dict(s["events"][0]), dict(s["events"][0])]))
    elif bar == 2:
        change_runs(cell, lambda s: [e.update(compress_wall_s=80) for e in s["events"]])
    elif bar == 3:
        # Sub-second compactions sum to waited turns without adding C events.
        change_runs(cell, lambda s: s["events"].extend([
            dict(s["events"][0], turn=t, compress_wall_s=0.6) for t in (4, 4, 5, 5)]))
    elif bar == 4:
        for path in cell[0].glob("decision/scores/cp-304/*.json"):
            write(path, dict(metrics=dict(facts_kept=dict(value=0.582))))
    elif bar == 5:
        for path in cell[0].glob("lcmx-runs/*/seed-*/*/engine.log"):
            path.write_text("survival fit applied (reason=budget, conversation=synthetic, dropped_rows=1)\n")
    else:
        change_runs(cell, lambda s: s["events"][0].update(compress_wall_s=121))
    code, result = invoke(cell)
    assert code == 1
    assert result["verdict"]["result"] == "FAIL"
    assert [int(k[0]) for k, v in result["verdict"]["bars"].items() if not v] == [bar]


@pytest.mark.parametrize("key", WANT)
def test_config_mismatch_is_invalid(cell, key):
    path = next(cell[0].glob("lcmx-runs/*/seed-*/*/config.json"))
    write(path, dict(config=dict(WANT, **{key: 1})))
    code, result = invoke(cell)
    assert code == 2
    assert result["verdict"]["result"] == "INVALID"
    assert not result["config_check"]["valid"]


def test_missing_run_is_invalid(cell):
    next(cell[0].glob("lcmx-runs/*/seed-*/*/summary.json")).unlink()
    code, result = invoke(cell)
    assert code == 2
    assert result["verdict"]["result"] == "INVALID"


@pytest.mark.parametrize("values,expected", [([], None), ([7], 7), (list(range(1, 11)), 9),
                                            ([11, 1, 10, 2, 9, 3, 8, 4, 7, 5, 6], 10)])
def test_nearest_rank_p90(values, expected):
    assert pmw.p90(values) == expected


@pytest.mark.parametrize("value", [0, None, "missing"])
def test_cp176_never_gates(cell, value):
    for path in cell[0].glob("decision/scores/cp-176/*.json"):
        if value == "missing":
            path.unlink()
        else:
            write(path, dict(metrics=dict(facts_kept=dict(value=value))))
    code, result = invoke(cell)
    assert code == 0
    assert result["verdict"]["complete"]
    assert result["pooled"][ARM]["facts_176_mean"] == (0 if value == 0 else None)


def test_declared_floor_and_config(cell):
    for path in cell[0].glob("lcmx-runs/*/seed-*/*/config.json"):
        write(path, dict(config=dict(leaf_chunk_tokens=8000, leaf_target_max_tokens=1000, summary_timeout_ms=60000)))
    args = ("--leaf-chunk-tokens", "8000", "--leaf-target-max-tokens", "1000", "--summary-timeout-ms", "60000")
    assert invoke(cell, *args, "--facts-floor", "0.583")[0] == 0
    assert invoke(cell, *args, "--facts-floor", "0.584")[0] == 1


def test_pre_crossing_and_reported_figures(cell):
    def add_preflight(summary):
        summary["events"].insert(0, dict(turn=-1, tokens_at_trigger=50, compress_wall_s=999,
                                       summariser_calls=[dict(exception=True, error="timeout")]))
    change_runs(cell, add_preflight)
    code, result = invoke(cell)
    assert code == 0
    pooled = result["pooled"][ARM]
    assert pooled["timeouts"] == pooled["failures"] == 2
    assert pooled["wall_p90"] == 1
    assert pooled["survival_fits"] == 0


@pytest.mark.parametrize("mode,expected", [("pass", 0), ("fail", 7), ("limit", 3)])
def test_runner_sequence_and_limit(tmp_path, mode, expected):
    kit = tmp_path / "kit"
    runner = kit / "s2" / "run_s_lcmx.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("import sys, time\nprint('HTTP 429' if " + repr(mode) + " == 'limit' else 'replay', flush=True)\n"
                      "if " + repr(mode) + " == 'limit': time.sleep(20)\n"
                      "sys.exit(7 if " + repr(mode) + " == 'fail' else 0)\n")
    decision = kit / "decision"
    decision.mkdir()
    for filename in ("score_decision.py", "pmw_read.py"):
        (decision / filename).write_text(
            "import os\nfrom pathlib import Path\n"
            "import sys\n"
            f"with (Path(os.environ['OUT']) / 'sequence').open('a') as f: f.write('{filename} ' + ' '.join(sys.argv[1:]) + '\\n')\n")
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "SYSTEMROOT")}
    # Product preflight uses a tiny clean repository; the kit itself is synthetic.
    product = tmp_path / "product"
    subprocess.run(["git", "init", "-q", str(product)], check=True)
    subprocess.run(["git", "-C", str(product), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-q", "--allow-empty", "-m", "fixture"], check=True)
    head = subprocess.check_output(["git", "-C", str(product), "rev-parse", "HEAD"], text=True).strip()
    env.update(PRODUCT_WT=str(product), PRODUCT_SHA=head, KIT_DIR=str(kit), MATERIAL=str(tmp_path / "material"),
               OUT=str(tmp_path / "out"), PY=sys.executable, GLM_API_KEY="synthetic-unused", SEEDS="1 2 3", PARALLEL="2",
               FACTS_FLOOR="0.772")
    result = subprocess.run(["bash", str(DECISION / "run_pmw_cell.sh")], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == expected, result.stderr
    sequence = Path(env["OUT"]) / "sequence"
    if mode == "pass":
        steps = sequence.read_text().splitlines()
        assert [line.split()[0] for line in steps] == ["score_decision.py", "pmw_read.py"]
        assert "--facts-floor 0.772" in steps[1]  # the declared floor reaches the reader
        assert len(list((Path(env["OUT"]) / "decision/logs").glob("*.log.wall"))) == 3
    else:
        assert not sequence.exists()
    if mode == "limit":
        assert (Path(env["OUT"]) / "STOP").exists()
        assert not (Path(env["OUT"]) / "decision/logs/s2-LCMX-fleet-rc3-d3-r1.log").exists()


@pytest.mark.parametrize("missing", ["PRODUCT_WT", "PRODUCT_SHA", "KIT_DIR", "MATERIAL", "OUT", "PY", "GLM_API_KEY",
                                     "FACTS_FLOOR"])
def test_runner_requires_environment(missing):
    env = dict(PATH=os.environ["PATH"])
    env.update({key: "synthetic-unused" for key in ("PRODUCT_WT", "PRODUCT_SHA", "KIT_DIR", "MATERIAL", "OUT", "PY", "GLM_API_KEY",
                                                    "FACTS_FLOOR")})
    del env[missing]
    result = subprocess.run(["bash", str(DECISION / "run_pmw_cell.sh")], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert missing in result.stderr
